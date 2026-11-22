"""Latency of one decision at batch size 1, on a fixed set of requests (docs/latency.md).

A latency set (data/latency/<version>.json, written by `jevarena-mm latency-set`) names the requests that every model
is timed on: per leaf of the selected tasks, a seeded sample of items, the sources of a leaf in turn and at most one
item per source recording (meta["group"]), so that no request reuses a medium that an earlier one decoded; warm-up
items apart from them; and closed-loop decision packets, copied with their screenshots from recorded episodes to the
media root.

`jevarena-mm latency` loads one model, answers the warm-up requests, then times every request of the set one at a
time, in a seeded order, and the first `repeat` of them a second time to show the noise. A request's latency is the
wall time of model.decide() (never decide_many): media decoding and sampling, every question of the request, and the
model's own calls. Per-request rows go to results/latency/<version>/<gpu>/<slug>.jsonl and the summary to
results/summary/latency/<gpu>/<slug>.json, <gpu> naming the kind of GPU it ran on (h100, a100; gpu_label), so the
same model can be timed on several: per input kind the median, p90, mean and a bootstrap interval of the median, GPU
seconds per 1,000 requests and the tokens the model wrote (output_tokens); the peak GPU memory, and how far the
probabilities are from the model's cached accuracy run (a configuration that differs from the accuracy run shows
there). "median_ms" is the median over the closed-loop decisions, the latency that realtime episodes charge
(envs.runner.realtime_latency reads it, for the kind of GPU asked for).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import dataio, env, recipes, runner
from .schema import DecisionRequest, Item

LOOP_QUOTA = {"gameworld": 50, "balrog_pixels": 30, "vsbench": 20}  # decision packets per suite
KINDS = ("short", "long", "av", "bench", "image", "audio", "loop")
READ_ONLY = {"d3", "clef", "onejev", "jev_omni", "raya", "xor", "random"}  # read the answer, write no text
AGREEMENT_FLOOR = 0.98  # share of questions whose top option matches the accuracy run; below it the run is flagged
PACKAGES = ("jevarena-mm", "torch", "vllm", "transformers", "sglang", "av", "numpy")


def kind_of(leaf: str, needs_audio: bool) -> str:
    """The input kind a leaf reports under: short, long, av (video leaves that need audio), bench, image, audio."""
    modality, domain = leaf.split(".")[0], leaf.split(".")[1].split("/")[0]
    if modality in ("vid", "lvid"):
        return "bench" if domain == "bench" else "av" if needs_audio else "short" if modality == "vid" else "long"
    return {"img": "image", "aud": "audio"}.get(modality, modality)


def media_seconds(request: DecisionRequest | dict) -> float | None:
    """Seconds of video and audio a request references (windows, else whole files); None without timed media."""
    refs = request.media() if isinstance(request, DecisionRequest) else DecisionRequest.model_validate(request).media()
    total, timed = 0.0, False
    for _, ref in refs:
        if ref.kind == "image":
            continue
        if ref.window is not None:
            total, timed = total + ref.window[1] - ref.window[0], True
        elif ref.info and ref.info.get("duration_s"):
            total, timed = total + float(ref.info["duration_s"]), True
    return round(total, 3) if timed else None


def candidates(task) -> list[dict]:
    """One row per item of a task: id, group, media seconds, audio need. From the built items, else from the recipe
    (its rows carry the item ids and their media), so a set can be drawn before Ego4D or Video-MME items are rebuilt."""
    rows = []
    for item in dataio.load_task(task):
        rows.append({"item_id": item.id, "group": item.meta.get("group") or item.id,
                     "media_s": media_seconds(item.request), "audio": "audio" in item.request.modalities()})
    if rows:
        return rows
    for row in recipes.load(task):
        request = DecisionRequest.model_validate(row["item"]["request"])
        rows.append({"item_id": row["id"], "group": (row["item"].get("meta") or {}).get("group") or row["id"],
                     "media_s": media_seconds(request), "audio": "audio" in request.modalities()})
    return rows


def build_set(tasks: list, per_leaf: int = 10, av_per_leaf: int = 20, warmup: int = 10, warmup_audio: int = 2,
              seed: int = 0, version: str = "v1", statuses: tuple[str, ...] = ("released",)) -> dict[str, Any]:
    """Draw the items of a latency set from the given task adapters (see the module docstring)."""
    by_leaf: dict[str, list] = defaultdict(list)
    for task in tasks:
        if task.status in statuses:
            by_leaf[task.leaf].append(task)
    pools: dict[str, dict[str, list[dict]]] = {}  # leaf -> task -> shuffled candidates
    for leaf, ts in sorted(by_leaf.items()):
        pools[leaf] = {}
        for task in sorted(ts, key=lambda t: t.name):
            rows = candidates(task)
            random.Random(f"{seed}:{task.name}").shuffle(rows)
            if rows:
                pools[leaf][task.name] = rows
    used: set[str] = set()  # groups taken, by the sample and the warm-up

    def draw(leaf: str, n: int) -> list[dict]:
        """n items of a leaf, its sources in turn, one per group; fewer when the leaf runs out."""
        picked, sources = [], list(pools[leaf])
        while len(picked) < n and sources:
            for name in list(sources):
                rows = pools[leaf][name]
                while rows and rows[0]["group"] in used:
                    rows.pop(0)
                if not rows:
                    sources.remove(name)
                    continue
                row = rows.pop(0)
                used.add(row["group"])
                picked.append({"kind": kind_of(leaf, row["audio"]), "leaf": leaf, "task": name, **row})
                if len(picked) == n:
                    break
        return picked

    audio_leaves = {leaf for leaf in pools if any(r["audio"] for rows in pools[leaf].values() for r in rows)}
    items = []
    for leaf in pools:
        items += draw(leaf, av_per_leaf if leaf in audio_leaves else per_leaf)

    def interleaved(leaves: list[str]) -> list[str]:
        """The leaves with their input kinds in turn (short, long, bench, ...), so the warm-up meets every kind."""
        by_kind: dict[str, list[str]] = defaultdict(list)
        for leaf in leaves:
            by_kind[kind_of(leaf, leaf in audio_leaves)].append(leaf)
        columns = [by_kind[k] for k in KINDS if by_kind[k]]
        return [c[i] for i in range(max(map(len, columns), default=0)) for c in columns if i < len(c)]

    warm = []  # one item per leaf in turn; a few audio requests warm an audio tower too
    for cycle, n in ((interleaved([l for l in pools if l not in audio_leaves]), warmup),
                     (interleaved([l for l in pools if l in audio_leaves]), warmup_audio)):
        taken = 0
        while taken < n and cycle:
            for leaf in list(cycle):
                got = draw(leaf, 1)
                if not got:
                    cycle.remove(leaf)
                    continue
                warm, taken = warm + got, taken + 1
                if taken == n:
                    break
    for entry in items + warm:
        entry.pop("audio", None)
    return {"version": version, "seed": seed, "per_leaf": per_leaf, "av_per_leaf": av_per_leaf,
            "created": time.strftime("%Y-%m-%d"), "items": items, "warmup": warm, "loop": []}


def sample_loop(attempts: Path, out: Path, quota: dict[str, int] | None = None, seed: int = 0,
                model_slug: str | None = None) -> list[dict]:
    """Copy closed-loop decision packets (decisions/NNNNN.json of recorded attempts) with their images to `out`, which
    lies under the media root, and return their set entries: per suite its quota, the environments in turn, at most
    one packet per (environment, seed), only answered steps with at least two candidates."""
    quota = quota or LOOP_QUOTA
    media_root = env.media_dir().resolve()
    out = out.resolve()
    if not out.is_relative_to(media_root):
        raise ValueError(f"{out} is not under the media root {media_root}")
    found: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))  # suite -> env -> packets
    for path in sorted(attempts.rglob("decisions/*.json")):
        parts = path.relative_to(attempts).parts
        if model_slug and model_slug not in parts:
            continue
        name = next((p.replace("__", "/") for p in parts if p.startswith("loop.game__")), None)
        if name and name.split("/")[1] in quota:
            found[name.split("/")[1]][name].append(path)
    rng, entries = random.Random(seed), []
    for suite, n in quota.items():
        envs = {name: rng.sample(paths, len(paths)) for name, paths in sorted(found[suite].items())}
        seen, taken = set(), 0
        while taken < n and envs:
            for name in list(envs):
                packet, path = None, None
                while envs[name] and packet is None:
                    path = envs[name].pop()
                    key = (name, next((p for p in path.parts if p.startswith("seed-")), ""))
                    record = json.loads(path.read_text(encoding="utf-8"))
                    request, reply = record.get("request") or {}, record.get("reply") or {}
                    request = saved_screenshots(request, path.parent.parent)
                    usable = (reply.get("status") == "ok" and len(request.get("candidates") or []) >= 2 and key not in seen
                              and all(Path(i["path"]).exists() for i in request.get("images") or []))
                    if usable:
                        packet = request
                        seen.add(key)
                if packet is None:
                    del envs[name]
                    continue
                folder = out / suite / f"{taken:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                images = []
                for i, image in enumerate(packet.get("images") or [], start=1):
                    source = Path(image["path"])
                    target = folder / f"image-{i}{source.suffix}"
                    shutil.copy2(source, target)
                    images.append({"path": target.name, "sha256": image["sha256"]})
                packet = {**packet, "images": images, "context": {}}
                (folder / "packet.json").write_text(json.dumps(packet, ensure_ascii=False) + "\n", encoding="utf-8")
                entries.append({"kind": "loop", "leaf": "/".join(name.split("/")[:2]), "env": name,
                                "packet": str((folder / "packet.json").relative_to(media_root)),
                                "candidates": len(packet["candidates"]), "images": len(images), "source": str(path)})
                taken += 1
                if taken == n:
                    break
    return entries


def saved_screenshots(packet: dict, attempt: Path) -> dict:
    """The packet with each screenshot gone since replaced by the attempt's saved copy with the same hash: GameWorld
    deletes its temporary screenshots (.screenshots_temp/) after the episode and keeps copies under native/."""
    if all(Path(image["path"]).exists() for image in packet.get("images") or []):
        return packet
    saved = {hashlib.sha256(p.read_bytes()).hexdigest(): str(p) for p in attempt.rglob("*.png")}
    return {**packet, "images": [{**image, "path": saved.get(image["sha256"], image["path"])} for image in packet["images"]]}


# ------------------------------------------------------------------------------------------------- the requests


class Requests:
    """Resolve set entries to DecisionRequests: items by id from their built task files, packets from the media root."""

    def __init__(self):
        self._tasks: dict[str, dict[str, Item]] = {}

    def item(self, entry: dict) -> Item:
        from .registry import tasks_under

        if entry["task"] not in self._tasks:
            (task,) = [t for t in tasks_under(entry["task"]) if t.name == entry["task"]]
            self._tasks[entry["task"]] = {i.id: i for i in dataio.load_task(task)}
        items = self._tasks[entry["task"]]
        if entry["item_id"] not in items:
            raise KeyError(f"{entry['item_id']} is not built (jevarena-mm build --leaf {entry['leaf']})")
        return items[entry["item_id"]]

    def request(self, entry: dict) -> DecisionRequest:
        if entry["kind"] != "loop":
            return self.item(entry).request
        from .serve import request_of

        path = env.media_dir() / entry["packet"]
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["images"] = [{**i, "path": str(path.parent / i["path"])} for i in packet.get("images") or []]
        return request_of(packet)


def media_files(latency_set: dict, requests: Requests | None = None) -> list[str]:
    """Paths under the media root that the set's requests read (media, marks, packets and their images)."""
    requests = requests or Requests()
    files: set[str] = set()
    for entry in latency_set["items"] + latency_set["warmup"]:
        for _, ref in requests.item(entry).request.media():
            files.add(ref.uri)
            if ref.marks is not None:
                files.add(ref.marks.uri)
    for entry in latency_set["loop"]:
        folder = Path(entry["packet"]).parent
        files.add(entry["packet"])
        packet = json.loads((env.media_dir() / entry["packet"]).read_text(encoding="utf-8"))
        files.update(str(folder / i["path"]) for i in packet.get("images") or [])
    return sorted(files)


def stage(latency_set: dict, to: Path) -> tuple[int, int]:
    """Copy the set's media files to `to` (the node's own disk), keeping their paths; returns (copied, bytes)."""
    copied, size, root = 0, 0, env.media_dir()
    for rel in media_files(latency_set):
        source, target = root / rel, to / rel
        if target.exists() and target.stat().st_size == source.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied, size = copied + 1, size + source.stat().st_size
    return copied, size


# ------------------------------------------------------------------------------------------------ the machine


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def visible_gpus() -> list[str]:
    """Indices of the GPUs this process may use: CUDA_VISIBLE_DEVICES when it lists indices, else every GPU."""
    listed = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if listed and all(part.strip().isdigit() for part in listed.split(",")):
        return [part.strip() for part in listed.split(",")]
    return [line.split(",")[0].strip() for line in _run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]).splitlines()]


def machine() -> dict[str, Any]:
    """The hardware and software a measurement ran on."""
    gpus = []
    for line in _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]).splitlines():
        index, name, memory, driver = (p.strip() for p in line.split(","))
        if index in visible_gpus():
            gpus.append({"index": index, "name": name, "memory_mib": int(memory), "driver": driver})
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"host": socket.gethostname(), "gpus": gpus, "cpu": cpu, "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "packages": versions,
            "commit": _run(["git", "-C", str(env.ROOT), "rev-parse", "HEAD"]),
            "env": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "JEVARENA_MM_DECODE_THREADS", "OMP_NUM_THREADS",
                                                "SLURM_JOB_ID", "CUDA_HOME", "NVCC_CCBIN", "DG_JIT_NVCC_COMPILER",
                                                "LD_LIBRARY_PATH", "TORCH_CUDNN_SDPA_DEPRIORITIZED") if k in os.environ}}


def gpu_label(gpus: list[dict] | None = None) -> str:
    """The kind of GPU a measurement runs on, in its file names: h100, a100, ... (the model number in the visible GPUs'
    names, e.g. "NVIDIA A100-PCIE-40GB" -> a100); cpu without one."""
    import re

    for gpu in machine()["gpus"] if gpus is None else gpus:
        found = re.search(r"\b([A-Z]{1,2}\d{2,3}[A-Z]?)\b", gpu["name"].upper().replace("-", " "))
        if found:
            return found.group(1).lower()
    return "cpu"


class MemoryWatch:
    """Peak memory.used of the visible GPUs (MiB, as nvidia-smi reports it), polled in a background thread; it sees
    every process on them, so engine processes (vLLM, SGLang) count too."""

    def __init__(self, interval_s: float = 0.5):
        self.interval, self.peak, self._stop = interval_s, {}, threading.Event()

    def _poll(self) -> None:
        devices = visible_gpus()
        while not self._stop.is_set():
            for line in _run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).splitlines():
                index, used = (p.strip() for p in line.split(","))
                if index in devices:
                    self.peak[index] = max(self.peak.get(index, 0), int(used))
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemoryWatch":
        if shutil.which("nvidia-smi"):
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()


# -------------------------------------------------------------------------------------------- the measurement


def output_tokens(model, usage: dict | None) -> int | None:
    """The tokens the model wrote before its answer was read. lm: and dgemma: report them (think_tokens: the
    thinking block; 0 at think=0, where the label probabilities are read at the answer position of the prompt itself);
    an adapter may report output_tokens. The adapters in READ_ONLY read the answer from a decision head or the label
    logits at the last prompt position and write no text: 0. None for any other adapter that reports neither."""
    for key in ("output_tokens", "think_tokens"):
        value = (usage or {}).get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0 if getattr(model, "kind", None) in READ_ONLY else None


def _decide(model, request: DecisionRequest) -> tuple[dict, float]:
    """One timed call; never raises (a crash is a failed request), except for a lost GPU context."""
    started = time.perf_counter()
    try:
        response = model.decide(request, 0)
    except Exception as error:  # noqa: BLE001 - recorded as a failed request
        if runner.gpu_lost(error):
            raise
        return {"status": "failed", "error": f"{type(error).__name__}: {str(error)[:300]}"}, (time.perf_counter() - started) * 1000
    elapsed = (time.perf_counter() - started) * 1000
    row = {"status": response.status, "error": response.error, "adapter_latency_ms": response.latency_ms,
           "input": (response.usage or {}).get("input"), "output_tokens": output_tokens(model, response.usage),
           "probs": {q: {o: round(p, 6) for o, p in a.probs.items()} for q, a in response.answers.items()}}
    return row, elapsed


def fit_entries(entries: list[dict], n: int) -> list[dict]:
    """Per input kind, the n requests with the most media seconds (loop: the most images, then candidates): the
    requests that set the peak memory."""
    out = []
    for kind in KINDS:
        rows = [e for e in entries if e["kind"] == kind]
        rows.sort(key=lambda e: (e.get("media_s") or 0, e.get("images", 0), e.get("candidates", 0)), reverse=True)
        out += rows[:n]
    return out


def takes(model, request: DecisionRequest) -> bool:
    return request.modalities() <= set(getattr(model, "modalities", ()) or ())


ENTRY_FIELDS = ("kind", "leaf", "task", "item_id", "env", "packet", "media_s")


def measure(model, latency_set: dict, kinds: set[str] | None = None, warmup: bool = True, repeat: int = 30,
            fit: int = 0, seed: int = 0, log=print) -> tuple[list[dict], dict[str, Any]]:
    """Time the set's requests on a model (see the module docstring); returns (rows, facts of the run). With fit, only
    the fit longest requests of each kind are timed (fit_entries), once."""
    requests = Requests()
    entries = [e for e in latency_set["items"] + latency_set["loop"] if kinds is None or e["kind"] in kinds]
    if fit:
        entries = fit_entries(entries, fit)
    else:
        random.Random(seed).shuffle(entries)  # kinds interleaved, so a drift over the run does not land on one kind
    facts: dict[str, Any] = {"machine": machine()}
    resolved, missing = {}, []  # every request before the weights load: a set the checkout cannot serve fails at once
    for entry in latency_set["warmup"] * warmup + entries:
        key = entry.get("item_id") or entry["packet"]
        try:
            resolved[key] = requests.request(entry)
        except (KeyError, FileNotFoundError, ValueError) as error:
            missing.append(f"{key}: {error}")
    if missing:
        raise SystemExit(f"{len(missing)} requests of the set cannot be read here, e.g. {missing[:3]}")

    def timed(entry: dict, **extra) -> dict:
        request = resolved[entry.get("item_id") or entry["packet"]]
        base = {k: entry[k] for k in ENTRY_FIELDS if k in entry}
        if not takes(model, request):
            return {**base, "status": "unsupported"}
        row, elapsed = _decide(model, request)
        return {**base, "questions": len(request.questions), "latency_ms": round(elapsed, 1), **extra, **row}

    with MemoryWatch() as watch:
        started = time.perf_counter()
        getattr(model, "load", lambda: None)()  # weights before the first request (lazy adapters load on it otherwise)
        facts["load_s"] = round(time.perf_counter() - started, 1)
        if warmup:
            facts["warmup_ms"] = [r["latency_ms"] for r in map(timed, latency_set["warmup"]) if "latency_ms" in r]
            log(f"  loaded in {facts['load_s']} s; {len(facts['warmup_ms'])} warm-up requests: {facts['warmup_ms']} ms")
        pairs = []
        for n, entry in enumerate(entries, start=1):
            pairs.append((entry, timed(entry)))
            if n % 50 == 0 or n == len(entries):
                log(f"  {n}/{len(entries)} requests timed")
        if repeat and not fit:  # the first answered requests once more: the run-to-run noise
            pairs += [(entry, timed(entry, repeat=True)) for entry, row in [p for p in pairs if p[1]["status"] == "ok"][:repeat]]
    facts["peak_memory_mib"] = watch.peak
    return [row for _, row in pairs], facts


def _stats(values: list[float], seed: int = 0) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    boot = np.median(np.random.default_rng(seed).choice(a, size=(1000, len(a)), replace=True), axis=1)
    return {"n": len(a), "median_ms": round(float(np.median(a)), 1), "p90_ms": round(float(np.percentile(a, 90)), 1),
            "mean_ms": round(float(a.mean()), 1),
            "median_ci95_ms": [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)]}


def _tokens(rows: list[dict]) -> dict[str, Any] | None:
    """Output tokens of the answered requests: mean and max per request (None when the adapter does not say)."""
    counts = [r.get("output_tokens") for r in rows if r["status"] == "ok"]
    if not counts or None in counts:
        return None
    return {"mean": round(sum(counts) / len(counts), 1), "max": max(counts)}


def consistency(model, rows: list[dict]) -> dict[str, Any]:
    """Compare the probabilities with the model's cached accuracy run (sample 0) for the same items."""
    cached: dict[str, dict[str, dict]] = {}
    agree = total = missing = 0
    diffs, changed_status = [], 0
    for row in rows:
        if row.get("repeat") or "item_id" not in row or row["status"] == "unsupported":
            continue
        if row["task"] not in cached:
            cached[row["task"]] = {r["item_id"]: r for r in runner.latest_rows(model.slug, row["task"]) if r["sample"] == 0}
        old = cached[row["task"]].get(row["item_id"])
        if old is None or runner.is_unsupported(old):
            missing += 1
            continue
        answers = old["response"].get("answers") or {}
        if (row["status"] == "ok") != (old["response"].get("status") == "ok"):
            changed_status += 1
            continue
        for question, probs in (row.get("probs") or {}).items():
            before = (answers.get(question) or {}).get("probs")
            if not before:
                continue
            total += 1
            agree += max(probs, key=probs.get) == max(before, key=before.get)
            diffs.append(max(abs(probs.get(o, 0.0) - before.get(o, 0.0)) for o in set(probs) | set(before)))
    out: dict[str, Any] = {"questions": total, "no_cached_row": missing, "status_changed": changed_status}
    if total:
        d = np.asarray(diffs)
        out.update({"top_option_agreement": round(agree / total, 4), "max_abs_diff_p50": round(float(np.median(d)), 5),
                    "max_abs_diff_p95": round(float(np.percentile(d, 95)), 5)})
        out["verdict"] = "same configuration" if agree / total >= AGREEMENT_FLOOR else "differs from the accuracy run"
    else:
        out["verdict"] = "no cached accuracy run to compare with (JEVARENA_MM_RUNS)"
    return out


def summarize(model, spec: str, latency_set: dict, rows: list[dict], facts: dict, gpus: int, seed: int = 0) -> dict:
    """The summary written to results/summary/latency_<slug>.json."""
    first = [r for r in rows if not r.get("repeat")]
    per_kind = {}
    for kind in KINDS:
        rs = [r for r in first if r["kind"] == kind]
        if not rs:
            continue
        ok = [r["latency_ms"] for r in rs if r["status"] == "ok"]
        entry = {"requests": len(rs), "failed": sum(r["status"] == "failed" for r in rs),
                 "unsupported": sum(r["status"] == "unsupported" for r in rs)}
        if ok:
            entry.update(_stats(ok, seed))
            entry["gpu_s_per_1k"] = round(entry["mean_ms"] * gpus, 1)  # mean s x GPUs x 1,000 requests
            entry["output_tokens"] = _tokens(rs)
        per_kind[kind] = entry
    once = {r.get("item_id", r.get("packet")): r for r in first}
    repeats = [r for r in rows if r.get("repeat") and r["status"] == "ok"]
    ratios = [r["latency_ms"] / once[key]["latency_ms"] for r in repeats
              for key in [r.get("item_id", r.get("packet"))] if (once.get(key, {}).get("latency_ms") or 0) > 0]
    loop = per_kind.get("loop", {})
    return {"model_id": model.model_id, "spec": spec, "set": latency_set["version"], "set_digest": set_digest(latency_set),
            "gpu": gpu_label(facts["machine"].get("gpus") or []), "gpus": gpus,
            "median_ms": loop.get("median_ms"), "per_kind": per_kind, "output_tokens": _tokens(first),
            "repeat": {"n": len(repeats),  # the second time over the first, per request
                       "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
                       "abs_change_p50": round(float(np.median(np.abs(np.asarray(ratios) - 1))), 3) if ratios else None},
            "consistency": consistency(model, rows), "load_s": facts.get("load_s"), "warmup_ms": facts.get("warmup_ms"),
            "peak_memory_mib": facts.get("peak_memory_mib"), "machine": facts["machine"],
            "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def fit_report(model, rows: list[dict], facts: dict) -> dict[str, Any]:
    """A fit test: which of the longest requests failed, and whether the accuracy run rejected the same ones."""
    cached: dict[str, dict] = {}
    new_failures, known = [], []
    for row in rows:
        if row["status"] != "failed":
            continue
        key = row.get("item_id", row.get("packet"))
        if "item_id" in row:
            if row["task"] not in cached:
                cached[row["task"]] = {r["item_id"]: r for r in runner.latest_rows(model.slug, row["task"]) if r["sample"] == 0}
            old = cached[row["task"]].get(row["item_id"])
            if old is not None and old["response"].get("status") == "failed":
                known.append(key)
                continue
        new_failures.append({"request": key, "error": row.get("error")})
    return {"model_id": model.model_id, "requests": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
            "failed_as_in_accuracy_run": known, "new_failures": new_failures,
            "peak_memory_mib": facts.get("peak_memory_mib"), "fits": not new_failures, "machine": facts["machine"]}


def export_runs(latency_set: dict, slugs: list[str], out: Path) -> dict[str, int]:
    """The cached accuracy rows a measurement compares with (each set item's latest row of sample 0), for these models,
    laid out as a runs folder (<out>/<slug>/<task>.jsonl). Where the accuracy caches are not at hand,
    JEVARENA_MM_RUNS=<out> gives `latency` its comparison, and its fit test the requests the accuracy run rejected."""
    wanted: dict[str, set[str]] = defaultdict(set)
    for entry in latency_set["items"]:
        wanted[entry["task"]].add(entry["item_id"])
    counts: dict[str, int] = {}
    for slug in slugs:
        counts[slug] = 0
        for task, ids in sorted(wanted.items()):
            rows = [r for r in runner.latest_rows(slug, task) if r["sample"] == 0 and r["item_id"] in ids]
            if rows:
                path = out / slug / (task.replace("/", "__") + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                counts[slug] += len(rows)
    return counts


def set_path(version: str) -> Path:
    return env.DATA / "latency" / f"{version}.json"


def rows_path(version: str, slug: str, fit: bool = False, gpu: str | None = None) -> Path:
    """results/latency/<version>/<gpu>/<slug>.jsonl (the fit test: <slug>.fit.jsonl), next to results/summary."""
    folder = env.SUMMARY.parent / "latency" / version / (gpu or "")
    return folder / (f"{slug}.fit.jsonl" if fit else f"{slug}.jsonl")


def summary_path(slug: str, gpu: str | None = None) -> Path:
    """results/summary/latency/<gpu>/<slug>.json; without a kind of GPU, the first layout (latency_<slug>.json)."""
    return env.SUMMARY / "latency" / gpu / f"{slug}.json" if gpu else env.SUMMARY / f"latency_{slug}.json"


def set_digest(latency_set: dict) -> str:
    """A short hash of the requests a set names, recorded with every measurement."""
    keys = [e.get("item_id") or e.get("packet") for e in latency_set["items"] + latency_set["loop"]]
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:12]
