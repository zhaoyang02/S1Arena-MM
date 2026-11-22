"""Play closed-loop episodes and score them.

Episodes are cached per (model, environment, seed) in results/episodes/, so re-running only plays
the missing seeds. Each row keeps the raw outcome and a compact trace of the decisions.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..env import ROOT
from ..schema import ModelResponse

EPISODES = ROOT / "results" / "episodes"
ACCOUNT_ERROR = re.compile(r"HTTP 40[1-4]\b")  # auth, an empty gateway balance, unknown model


class GatewayRefused(RuntimeError):
    """The gateway refused a call for account reasons, so the episode says nothing about the model."""


def episode_path(model_slug: str, env_name: str) -> Path:
    return EPISODES / model_slug / (env_name.replace("/", "__") + ".jsonl")


def read_episodes(model_slug: str, env_name: str) -> list[dict]:
    path = episode_path(model_slug, env_name)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


LATENCY_TABLE = ROOT / "results" / "latency.json"


def latest_episodes(model_slug: str, env_name: str, mode: str | None = None, max_decisions: int | None = None,
                    latency_ms: float | None = None, version: int | None = None, latency_gpu: str | None = None) -> list[dict]:
    """One row per seed (the latest), among episodes played in the given mode, decision cap, (realtime) latency and harness
    version (rows written before versions were recorded count as version 1). A realtime latency also selects the kind
    of GPU it was measured on (latency_gpu; None for a latency from no measurement), so two measurements that round
    to the same latency stay apart."""
    best: dict[int, dict] = {}
    for row in read_episodes(model_slug, env_name):
        if mode is not None and row.get("mode") != mode:
            continue
        if version is not None and row.get("env_version", 1) != version:
            continue
        if max_decisions is not None and row.get("max_decisions") != max_decisions:
            continue
        if latency_ms is not None and row.get("mode") == "realtime" and (row.get("assumed_latency_ms") != latency_ms
                                                                       or row.get("latency_gpu") != latency_gpu):
            continue
        best[row["seed"]] = row
    return [best[s] for s in sorted(best)]


def model_latency_ms(model) -> float:
    """The fixed latency charged to a model in realtime mode.

    Fixed on first use: the median latency of the model's cached static runs, rounded to 10 ms, stored in
    results/latency.json and reused from then on, so episodes stay comparable as more runs accumulate.
    """
    table = json.loads(LATENCY_TABLE.read_text(encoding="utf-8")) if LATENCY_TABLE.exists() else {}
    if model.model_id in table:
        return float(table[model.model_id])
    values = []
    for path in (ROOT / "results" / "runs" / model.slug).glob("*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                latency = json.loads(line)["response"].get("latency_ms")
                if latency:
                    values.append(latency)
    if not values:
        raise ValueError(f"no cached runs for {model.model_id}; pass --latency-ms")
    return _store_latency(model.model_id, round(float(np.median(values)), -1))


def realtime_latency(model, gpu: str | None = None) -> tuple[float, str | None]:
    """The fixed latency a realtime episode charges the model, and the kind of GPU it was timed on: the median over the
    closed-loop decisions of the model's latency measurement (`jevarena-mm latency`, docs/latency.md), rounded to
    10 ms. A kind of GPU named (h100, a100) needs that measurement (ValueError without it); none named takes the H100
    one, else one without a kind of GPU (results/summary/latency_<slug>.json), else model_latency_ms (kind None)."""
    from .. import latency

    for kind in [gpu] if gpu else ["h100", None]:
        path = latency.summary_path(model.slug, kind)
        median = json.loads(path.read_text(encoding="utf-8")).get("median_ms") if path.exists() else None
        if median:
            return round(float(median), -1), kind
    if gpu:
        raise ValueError(f"no {gpu} latency measurement of {model.model_id} (jevarena-mm latency, docs/latency.md)")
    return model_latency_ms(model), None


def _store_latency(model_id: str, value: float) -> float:
    """Fix a model's latency in results/latency.json unless a parallel run fixed one first; returns the stored value.
    Runs on several nodes write the table, so a lock file created exclusively guards it: on many shared file systems
    flock is not shared across nodes (e.g. an NFS v3 home, or Lustre mounted with localflock)."""
    lock = LATENCY_TABLE.with_name(LATENCY_TABLE.name + ".lock")
    for _ in range(1200):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 60:  # left by a run that died holding it
                    lock.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            time.sleep(0.1)
    else:
        raise TimeoutError(f"{lock} held for over two minutes")
    try:
        table = json.loads(LATENCY_TABLE.read_text(encoding="utf-8")) if LATENCY_TABLE.exists() else {}
        table.setdefault(model_id, value)
        tmp = LATENCY_TABLE.with_name(f"{LATENCY_TABLE.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(LATENCY_TABLE)  # readers without the lock see the old table or the new one, never a partial one
        return float(table[model_id])
    finally:
        lock.unlink(missing_ok=True)


def _choose(response: ModelResponse, request, env) -> tuple[dict[str, str], float, bool]:
    """Argmax option per question; the environment's default for questions left unanswered."""
    default = env.default_choice(request)
    choice, confidence, failed = {}, 1.0, bool(response.error)
    for name in request.questions:
        answer = response.answers.get(name)
        if answer and answer.probs:
            option = max(answer.probs, key=answer.probs.get)
            choice[name] = option
            confidence = min(confidence, answer.probs[option])
        else:
            choice[name], failed = default[name], True
    return choice, confidence, failed


def play_episode(model, env, seed: int, mode: str, latency_ms: float, max_decisions: int | None = None,
                 latency_gpu: str | None = None) -> dict[str, Any]:
    env.reset(seed)
    limit = max_decisions or env.max_decisions
    hold = round(latency_ms / 1000 / env.control_dt) if mode == "realtime" else 0
    decisions = failures = 0
    latencies, costs, trace = [], [], []
    started = time.time()
    try:
        while not env.done() and decisions < limit:
            if env.agents:  # one independent call per active agent, decided in parallel
                requests = {agent: env.observe_agent(agent) for agent in env.active_agents()}
                with ThreadPoolExecutor(max_workers=max(1, len(requests))) as pool:
                    responses = dict(zip(requests, pool.map(lambda r: model.decide(r, 0), requests.values())))
                picked = {agent: _choose(responses[agent], requests[agent], env) for agent in requests}
                choice = {agent: p[0] for agent, p in picked.items()}
                confidence = min((p[1] for p in picked.values()), default=1.0)
                failed = any(p[2] for p in picked.values())  # a step fails if any car's decision failed
                replies = list(responses.values())
            else:
                request = env.observe()
                response = model.decide(request, 0)
                choice, confidence, failed = _choose(response, request, env)
                replies = [response]
            failures += failed
            for reply in replies:  # why a decision failed, for the log (an episode keeps only the count)
                if reply.error:
                    print(f"  {env.name} seed {seed}: decision error: {reply.error[:160]}", file=sys.stderr, flush=True)
            refused = next((r.error for r in replies if r.error and ACCOUNT_ERROR.search(r.error)), None)
            if refused:
                raise GatewayRefused(refused[:300])
            for response in replies:
                if response.latency_ms:
                    latencies.append(response.latency_ms)
                if response.cost_usd is not None:
                    costs.append(response.cost_usd)
            for _ in range(hold):  # the world moves on while the model decides
                if env.done():
                    break
                env.idle()
            if env.done():
                break
            if env.agents:
                env.act_all(choice)
            else:
                env.act(choice)
            decisions += 1
            trace.append({"c": choice, "p": round(confidence, 3), **({"x": int(failed)} if failed else {})})
        outcome = env.outcome()
    finally:
        env.close()
    return {
        "seed": seed,
        "model": model.model_id,
        "mode": mode,
        "env_version": getattr(env, "version", 1),
        "assumed_latency_ms": latency_ms if mode == "realtime" else None,
        **({"latency_gpu": latency_gpu} if mode == "realtime" else {}),
        "hold_steps": hold,
        "max_decisions": limit,
        "outcome": outcome,
        "decisions": decisions,
        "failed_decisions": failures,
        "latency_p50_ms": float(np.median(latencies)) if latencies else None,
        "cost_usd": float(sum(costs)) if costs else None,
        "wall_s": round(time.time() - started, 1),
        "trace": trace,
    }


def run_env(model, env_cls, seeds: list[int], mode: str, latency_ms: float, workers: int = 4, max_decisions: int | None = None,
            latency_gpu: str | None = None) -> int:
    """Play every seed not yet cached for this model, environment and mode (realtime: at this latency, measured on this
    kind of GPU); returns episodes played."""
    name = env_cls.name
    if getattr(env_cls, "recorded_only", False):  # an upstream suite: played through its bridge when installed
        if max_decisions:
            print(f"  {name}: --max-decisions does not apply; the upstream harness sets the episode length", flush=True)
        return env_cls.play(model, seeds, workers, mode=mode, latency_ms=latency_ms, latency_gpu=latency_gpu)
    missing = set(env_cls.modalities) - set(getattr(model, "modalities", ()) or ())
    if missing:  # nothing is recorded: the environment is unsupported for this model, not failed
        print(f"  {name}: unsupported for {model.model_id} (needs {', '.join(sorted(missing))}), not played", flush=True)
        return 0
    path = episode_path(model.slug, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = max_decisions or env_cls.max_decisions
    done = {row["seed"] for row in latest_episodes(model.slug, name, mode, cap, latency_ms if mode == "realtime" else None, env_cls.version,
                                                   latency_gpu) if "outcome" in row}
    todo = [s for s in seeds if s not in done]
    if not todo:
        return 0
    lock = threading.Lock()
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(play_episode, model, env_cls(), s, mode, latency_ms, max_decisions, latency_gpu): s for s in todo}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                row = future.result()
            except GatewayRefused as error:  # nothing is recorded; the seed is played on the next run
                print(f"  {name} seed {seed}: not recorded, {error}", flush=True)
                continue
            except Exception as error:  # a crashed episode is recorded, not retried silently
                row = {"seed": seed, "model": model.model_id, "mode": mode, "env_version": env_cls.version,
                       "error": f"{type(error).__name__}: {str(error)[:300]}"}
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            if "error" in row:
                print(f"  {name} seed {seed}: {row['error']}", flush=True)
            else:
                outcome = row["outcome"]
                print(f"  {name} seed {seed}: score={outcome.get('score')} success={outcome.get('success')} decisions={row['decisions']}", flush=True)
    return len(todo)


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Raw results over episodes: success rate (when defined), mean score, decisions, failures, latency, cost.
    Unsupported episodes (the model cannot take the environment's inputs or actions) are counted apart; episodes that
    ended in a technical error count as crashed, never as losses."""
    unsupported = sum(1 for r in rows if "unsupported" in r)
    ok = [r for r in rows if "outcome" in r]
    if not ok:
        return {"episodes": 0, "crashed": len(rows) - unsupported, "unsupported": unsupported}
    scores = [r["outcome"].get("score") for r in ok if r["outcome"].get("score") is not None]
    successes = [r["outcome"]["success"] for r in ok if r["outcome"].get("success") is not None]
    decisions = sum(r["decisions"] for r in ok)
    latencies = [r["latency_p50_ms"] for r in ok if r.get("latency_p50_ms")]
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    return {
        "episodes": len(ok),
        "crashed": len(rows) - len(ok) - unsupported,
        "unsupported": unsupported,
        "success_rate": float(np.mean(successes)) if successes else None,
        "score_mean": float(np.mean(scores)) if scores else None,
        "score_se": float(statistics.stdev(scores) / len(scores) ** 0.5) if len(scores) > 1 else None,
        "decisions_mean": decisions / len(ok),
        "failed_decision_rate": sum(r["failed_decisions"] for r in ok) / decisions if decisions else None,
        "latency_p50_ms": float(np.median(latencies)) if latencies else None,
        "cost_per_episode_usd": float(np.mean(costs)) if costs else None,
    }
