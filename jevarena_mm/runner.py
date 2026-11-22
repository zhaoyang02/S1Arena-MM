"""Run a model over built items. Responses are cached per (model, item, sample),
so re-running only fills in what is missing and new models only cost their own calls.

An item that needs a modality the model does not take is not sent: it gets a row with status "unsupported" and is
left out of that model's scores. If the model later takes that modality, the item is called on the next run.

An item whose media files are missing or changed (media.check: not fetched, or no access to its source) is not sent
either, and gets no row: it is left for a run after `jevarena-mm fetch`, so it is neither scored nor counted as failed.

A run with --shard I/N writes its own file (<task>.shard-IofN.jsonl), so shards can run at once on several nodes of a
shared file system without appending to one file; every reader merges a task's files."""

from __future__ import annotations

import glob
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import env, media
from .schema import Item, ModelResponse


BATCH = 2000  # requests per decide_many call (local models)
Shard = tuple[int, int]  # (index, count)


def run_path(model_slug: str, task: str, shard: Shard | None = None) -> Path:
    """The file a run appends to: <task>.jsonl, or <task>.shard-IofN.jsonl for shard I of N (N > 1)."""
    stem = task.replace("/", "__")
    if shard and shard[1] > 1:
        stem += f".shard-{shard[0]}of{shard[1]}"
    return env.runs_dir() / model_slug / (stem + ".jsonl")


def run_files(model_slug: str, task: str) -> list[Path]:
    """Every file of a task's runs: the single-process file first, then the shard files by name."""
    folder, stem = env.runs_dir() / model_slug, task.replace("/", "__")
    main = folder / f"{stem}.jsonl"
    return ([main] if main.exists() else []) + sorted(folder.glob(f"{glob.escape(stem)}.shard-*.jsonl"))


def read_runs(model_slug: str, task: str) -> list[dict]:
    """The rows of all of a task's files. A line that does not parse (a write cut off when a job was killed) is
    skipped: its call is not recorded, so the next run makes it again."""
    rows = []
    for path in run_files(model_slug, task):
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"{path}:{n}: skipped an incomplete row", file=sys.stderr)
    return rows


def _append(path: Path, mode: str = "a", **kwargs):
    """Open a run file for appending, ending a cut-off last line first so the next row starts on its own line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            cut = f.read(1) != b"\n"
        if cut:
            with path.open("ab") as f:
                f.write(b"\n")
    return path.open(mode, **kwargs)


# 4xx statuses that reject the request itself (e.g. an input longer than the model's context). Other 4xx statuses
# (401 / 403 auth, 402 an empty gateway balance, 404, 408, 429) say nothing about the item and are retried.
REQUEST_REJECTED = (400, 413, 422)


def is_unsupported(row: dict) -> bool:
    return row["response"].get("status") == "unsupported"


def is_final(row: dict) -> bool:
    """A success, a reply that could not be used (its questions score 0; see reparse_task), or a request the model
    rejects outright (see REQUEST_REJECTED). Unsupported rows are not final: they are re-checked on every run.

    Only calls that got no reply because of transport, account, rate-limit or server errors are retried, so a
    model is never re-sampled because its first answer failed.
    """
    if is_unsupported(row):
        return False
    response = row["response"]
    error = response.get("error")
    if not error or response.get("raw_text") is not None or error.startswith("rejected: "):  # local models: input too long
        return True
    status = re.search(r"HTTP (\d{3})", error)
    return bool(status) and int(status.group(1)) in REQUEST_REJECTED


def supports(model, item: Item) -> bool:
    """True if the model takes every media kind the item references (models without a declaration take text only)."""
    return item.request.modalities() <= set(getattr(model, "modalities", ()) or ())


def unsupported_response(model, item: Item) -> ModelResponse:
    missing = sorted(item.request.modalities() - set(getattr(model, "modalities", ()) or ()))
    return ModelResponse(status="unsupported", usage={"missing_modalities": missing})


def _row(model, item: Item, sample: int, response: ModelResponse) -> dict:
    return {"item_id": item.id, "sample": sample, "model": model.model_id, "response": response.model_dump()}


def is_rejection(row: dict) -> bool:
    """A request the model refused (e.g. an input longer than its context): final, unless a run asks to retry them."""
    return (row["response"].get("error") or "").startswith("rejected: ")


def pending(model, task: str, items: list[Item], k: int,
            retry_rejected: bool = False) -> tuple[list[tuple[Item, int]], list[tuple[Item, int]]]:
    """(calls still to make, unsupported (item, sample) pairs not yet recorded as such). Items whose media are missing
    or changed are in neither and are reported (see the module docstring). With retry_rejected, items whose only final
    rows are rejections are called again (after an adapter change that makes them fit)."""
    rows = read_runs(model.slug, task)
    done = {(r["item_id"], r["sample"]) for r in rows if is_final(r) and not (retry_rejected and is_rejection(r))}
    marked = {(r["item_id"], r["sample"]) for r in rows if is_unsupported(r)}
    jobs, skipped, absent = [], [], 0
    for item in items:
        todo = [s for s in range(k) if (item.id, s) not in done]
        if not todo:
            continue
        if not supports(model, item):
            skipped += [(item, s) for s in todo if (item.id, s) not in marked]
        elif any(media.check(ref) for _, ref in item.request.media()):
            absent += 1
        else:
            jobs += [(item, s) for s in todo]
    if absent:
        print(f"  {task}: {absent} items not run, their media missing or changed (run `jevarena-mm fetch --leaf {task}`)",
              flush=True)
    return jobs, skipped


def _record_unsupported(model, task: str, skipped: list[tuple[Item, int]], shard: Shard | None = None) -> None:
    if not skipped:
        return
    with _append(run_path(model.slug, task, shard), encoding="utf-8") as out:
        for item, s in skipped:
            out.write(json.dumps(_row(model, item, s, unsupported_response(model, item)), ensure_ascii=False) + "\n")
    print(f"  {task}: {len(skipped)} unsupported (item, sample) pairs recorded, no calls", flush=True)


GPU_LOST = ("unspecified launch failure", "illegal memory access", "illegal instruction", "misaligned address",
            "device-side assert")


def gpu_lost(error: Exception) -> bool:
    """A CUDA error that leaves the process's GPU context unusable (e.g. after an Xid fault): every later call fails in
    seconds, so the run ends and the next one, in a clean process, resumes."""
    text = str(error)
    return "CUDA error" in text and any(sign in text for sign in GPU_LOST)


def run_task(model, task: str, items: list[Item], k: int = 1, workers: int = 8,
             budget_usd: float | None = None, retry_rejected: bool = False, shard: Shard | None = None) -> int:
    """Query the model k times per item for anything not already cached; returns calls made.

    With budget_usd, calls not yet started are cancelled once this run's spend reaches it (calls in flight finish).
    With shard, rows go to that shard's own file (run_path); what is cached is read from all files.
    """
    jobs, skipped = pending(model, task, items, k, retry_rejected)
    _record_unsupported(model, task, skipped, shard)
    if not jobs:
        return 0
    started = time.time()
    if hasattr(model, "decide_many"):  # local models answer a whole batch at once
        return sum(run_tasks_batched(model, [(task, items)], k, retry_rejected, shard).values())
    lock = threading.Lock()
    made, spent, lost = 0, 0.0, ""
    # One unbuffered append per row, so a killed run leaves at most one cut-off line (skipped by read_runs).
    with _append(run_path(model.slug, task, shard), "ab", buffering=0) as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(model.decide, item.request, s): (item, s) for item, s in jobs}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            item, s = futures[future]
            try:
                response = future.result()
            except Exception as error:  # a crash is recorded like any other failed call
                if gpu_lost(error):  # later calls would fail in seconds: cancel them, keep the answers made, then stop
                    lost = lost or f"GPU context lost, ending the run: {type(error).__name__}: {str(error)[:300]}"
                    for other in futures:
                        other.cancel()
                    continue  # not recorded: the item stays pending for the next run
                response = ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}")
            with lock:
                out.write((json.dumps(_row(model, item, s, response), ensure_ascii=False) + "\n").encode("utf-8"))
            made += 1
            spent += response.cost_usd or 0.0
            if budget_usd is not None and spent >= budget_usd:
                cancelled = sum(f.cancel() for f in futures)
                if cancelled:
                    print(f"  {task}: budget reached (${spent:.2f}), {cancelled} calls not started", flush=True)
                budget_usd = None
            if made % 100 == 0 or made == len(jobs):
                print(f"  {task}: {made}/{len(jobs)} calls, {time.time() - started:.0f}s", flush=True)
    if lost:
        raise SystemExit(lost)
    return made


def run_tasks_batched(model, tasks: list[tuple[str, list[Item]]], k: int = 1, retry_rejected: bool = False,
                      shard: Shard | None = None) -> dict[str, int]:
    """Local models: pool the uncached calls of many tasks into shared decide_many batches of BATCH requests, so
    small tasks do not each pay for a half-empty batch. Rows go to each task's own cache file; returns calls per task."""
    jobs = []
    for task, items in tasks:
        todo, skipped = pending(model, task, items, k, retry_rejected)
        _record_unsupported(model, task, skipped, shard)
        jobs += [(task, item, s) for item, s in todo]
    calls: dict[str, int] = {}
    started = time.time()
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        try:
            responses = model.decide_many([(item.request, s) for _, item, s in chunk])
        except Exception as error:  # a crashed batch is recorded like failed calls and retried next time
            if gpu_lost(error):
                raise SystemExit(f"GPU context lost, ending the run: {type(error).__name__}: {str(error)[:300]}") from error
            responses = [ModelResponse(error=f"{type(error).__name__}: {str(error)[:300]}") for _ in chunk]
        rows: dict[str, list[str]] = {}
        for (task, item, s), response in zip(chunk, responses):
            rows.setdefault(task, []).append(json.dumps(_row(model, item, s, response), ensure_ascii=False) + "\n")
            calls[task] = calls.get(task, 0) + 1
        for task, lines in rows.items():
            with _append(run_path(model.slug, task, shard), encoding="utf-8") as out:
                out.writelines(lines)
        print(f"  batch {start + len(chunk)}/{len(jobs)} calls over {len(rows)} tasks, {time.time() - started:.0f}s", flush=True)
    return calls


def reparse_task(model, task: str, items: list[Item]) -> int:
    """Re-read the stored replies of failed calls with the current parser; no new calls.

    A fixed row is appended with "reparsed": true, so latest_rows picks it over the failure.
    """
    requests = {item.id: item.request for item in items}
    fixed = []
    for row in latest_rows(model.slug, task):
        response = row["response"]
        if row["item_id"] not in requests or not response.get("error") or not response.get("raw_text"):
            continue
        new = model.reparse(ModelResponse(**response), requests[row["item_id"]])
        if new is not None and not new.error:
            fixed.append({**row, "response": new.model_dump(), "reparsed": True})
    if fixed:
        with _append(run_path(model.slug, task), encoding="utf-8") as out:
            for row in fixed:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(fixed)


def _rank(row: dict) -> int:
    if is_unsupported(row):
        return 1
    return 3 if not row["response"].get("error") else 2 if is_final(row) else 0


def latest_rows(model_slug: str, task: str) -> list[dict]:
    """One row per (item, sample): the most recent successful response, else the most recent final failure (a reply
    that could not be used or a request the model rejects, see is_final), else the most recent unsupported mark, else
    the most recent failure. A transport error recorded after a final row does not hide it: run_task never calls such
    an item again."""
    best: dict[tuple, dict] = {}
    for row in read_runs(model_slug, task):
        key = (row["item_id"], row["sample"])
        if key not in best or _rank(row) >= _rank(best[key]):
            best[key] = row
    return list(best.values())
