"""Closed-loop suites run by their own upstream harness (GameWorld, BALROG, VS-Bench).

The upstream code owns the game loop and the score; the model answers each step as a decision request built from the
native screenshots and the environment's own text, with the official legal actions as options. S1Arena-MM registers
one Environment per game or task, from the suite list in data/loop/suites.json, so that cards, `env-score` and sweep
reports cover them, and holds their recorded episodes.

Playing: when a suite's checkout and Python are configured (JEVARENA_MM_<SUITE> and JEVARENA_MM_<SUITE>_PYTHON, in the
environment or .env), `jevarena-mm play` serves the model (jevarena_mm/serve.py) and runs one bridge process per
episode inside the suite's Python (jevarena_mm/bridges/); otherwise the suite has recorded episodes only.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .. import env
from ..registry import register_env
from . import runner as env_runner
from .base import Environment

_SERVERS: dict[str, tuple[Any, str]] = {}  # model id -> (DecisionServer, url): one served copy per model and process
_APPEND = threading.Lock()


def served(model) -> str:
    """The URL of a decision server for this model, started on first use and kept for the process."""
    if model.model_id not in _SERVERS:
        from ..serve import DecisionServer

        server = DecisionServer(model, log=env.CACHE / "decision_calls" / f"{model.slug}.jsonl")
        _SERVERS[model.model_id] = (server, server.__enter__())
    return _SERVERS[model.model_id][1]


def close_servers() -> None:
    """Stop the decision servers and drop their models. A model still held when the interpreter exits keeps its engine
    process (e.g. vLLM's engine core) waiting for requests, and the exit waits for that process: closed here, the
    engine is shut down by its own finalizer once the model is collected."""
    for server, _ in _SERVERS.values():
        server.__exit__(None, None, None)
        getattr(server.model, "close", lambda: None)()  # e.g. vLLM: shut the engine-core process down now
    _SERVERS.clear()


def bridge_path() -> Path:
    """A directory holding only the jevarena_mm package, for the suites' PYTHONPATH (keeps the repository's other
    top-level folders, such as data/, from shadowing the suites' modules)."""
    path = env.CACHE / "bridge_path"
    link = path / "jevarena_mm"
    if not link.exists():
        path.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(env.ROOT / "jevarena_mm")
        except FileExistsError:  # another episode or shard on this checkout made it first
            pass
    return path


STALL_S = int(os.environ.get("JEVARENA_MM_BRIDGE_STALL_S", 1800))


def last_change(folder: Path) -> float:
    """The newest modification time of the files in a folder; files the bridge removes meanwhile are skipped (a
    temporary file that vanishes between the listing and its stat would otherwise end the whole play run with
    FileNotFoundError)."""
    newest = 0.0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                continue
    try:
        return max(newest, folder.stat().st_mtime)
    except OSError:
        return newest or time.time()


def run_watched(command: list[str], attempt: Path, variables: dict, log, limit: float | None = None) -> int | str:
    """Run one bridge in its own session; end it with everything it started (browser, game server) when it runs past
    `limit` seconds or nothing in its attempt folder has changed for STALL_S seconds (a live bridge writes its log and
    one file per decision; on a file system that refuses writes, a bridge would otherwise wait forever). Returns the
    exit code, or why it was ended."""
    process = subprocess.Popen(command, cwd=attempt, env=variables, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    started = time.time()
    while True:
        try:
            return process.wait(timeout=min(60, STALL_S))
        except subprocess.TimeoutExpired:
            newest = last_change(attempt)
            stalled, over = time.time() - newest > STALL_S, limit is not None and time.time() - started > limit
            if stalled or over:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return f"stopped after {STALL_S} s without progress" if stalled else f"killed after {limit:.0f} s"


class UpstreamEnvironment(Environment):
    """An environment whose episodes are played by an upstream harness; S1Arena-MM holds its card and results."""

    upstream = ""  # harness and revision, e.g. "GameWorld @ 3c26bda"
    recorded_only = True  # run_env hands such environments to play() instead of its own loop
    realtime_ok = False  # whether its bridge can charge each decision a fixed latency (play --realtime)
    unsupported: str | None = None  # why no Jev-style classifier can play it (e.g. free-text actions)
    blocked: str | None = None  # why the pinned upstream cannot run it (its episodes are not replayed)
    modalities = frozenset({"image"})
    mode = "paused"
    details: dict[str, Any] = {}
    suite = ""  # bridge module and configuration key, e.g. "gameworld"
    setting = ""  # JEVARENA_MM_<SETTING> names the checkout, JEVARENA_MM_<SETTING>_PYTHON its interpreter
    extra_env: dict[str, str] = {}

    def reset(self, seed: int) -> None:
        raise NotImplementedError(f"{self.name} is played by its upstream harness ({self.upstream}); see docs/envs.md")

    def card(self) -> dict[str, Any]:
        return {**super().card(), "upstream": self.upstream, "recorded_only": self.recorded_only, "unsupported": self.unsupported,
                "blocked": self.blocked}

    def bridge_args(self, seed: int) -> list[str]:
        raise NotImplementedError

    @classmethod
    def installation(cls) -> tuple[Path, Path] | None:
        """(checkout, python) of the suite on this machine, or None."""
        env.load_env()
        checkout = os.environ.get(f"JEVARENA_MM_{cls.setting}")
        python = os.environ.get(f"JEVARENA_MM_{cls.setting}_PYTHON")
        if checkout and python and Path(checkout).is_dir() and Path(python).exists():
            return Path(checkout), Path(python)
        return None

    @classmethod
    def play(cls, model, seeds: list[int], workers: int = 1, mode: str = "paused", latency_ms: float = 0.0,
             latency_gpu: str | None = None) -> int:
        """Play the seeds not yet recorded for this model, environment version and mode (realtime: at this latency,
        measured on this kind of GPU); returns episodes recorded. Realtime needs a suite whose bridge can charge the
        latency (realtime_ok)."""
        name, cap = cls.name, cls.max_decisions
        latency = latency_ms if mode == "realtime" else None
        if mode == "realtime" and not cls.realtime_ok:
            print(f"  {name}: no realtime protocol (the suite waits for every move), not played", flush=True)
            return 0
        done = {row["seed"] for row in env_runner.latest_episodes(model.slug, name, mode, cap, latency, cls.version, latency_gpu)
                if "outcome" in row or "unsupported" in row}
        todo = [s for s in seeds if s not in done]
        if not todo:
            return 0
        if cls.unsupported:
            for seed in todo:
                cls.record(model, seed, unsupported=cls.unsupported, mode=mode, assumed_latency_ms=latency,
                           **({"latency_gpu": latency_gpu} if mode == "realtime" else {}))
            print(f"  {name}: unsupported ({cls.unsupported[:80]}); {len(todo)} episodes recorded as such", flush=True)
            return len(todo)
        if cls.blocked:
            print(f"  {name}: blocked upstream, not played ({cls.blocked[:100]})", flush=True)
            return 0
        installed = cls.installation()
        if installed is None:
            print(f"  {name}: recorded episodes only; set JEVARENA_MM_{cls.setting} and JEVARENA_MM_{cls.setting}_PYTHON to play it", flush=True)
            return 0
        missing = set(cls.modalities) - set(getattr(model, "modalities", ()) or ())
        if missing:
            print(f"  {name}: unsupported for {model.model_id} (needs {', '.join(sorted(missing))}), not played", flush=True)
            return 0
        url = served(model)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(lambda seed: cls.play_one(model, seed, url, installed, mode, latency_ms, latency_gpu), todo))
        return len(todo)

    @classmethod
    def episode_timeout_s(cls, latency_ms: float = 0.0) -> int:
        """How long one episode may run before its bridge is killed (a technical failure, retried by the next play):
        6 s per decision of the cap (1,000 where the upstream sets the length), at least 45 min; a realtime episode also
        gets the latency it charges each decision of the cap, the game time it adds by design."""
        cap = cls.max_decisions or 1000
        return max(2700, 6 * cap) + round(cap * latency_ms / 1000)

    @classmethod
    def play_one(cls, model, seed: int, url: str, installed: tuple[Path, Path], mode: str = "paused",
                 latency_ms: float = 0.0, latency_gpu: str | None = None) -> dict:
        checkout, python = installed
        timing = {"mode": mode, "assumed_latency_ms": latency_ms if mode == "realtime" else None,
                  **({"latency_gpu": latency_gpu} if mode == "realtime" else {})}
        root = env.attempts_dir()  # a file system without a file-count quota, when set
        episodes = root / model.slug / cls.name.replace("/", "__") if root else env_runner.episode_path(model.slug, cls.name).with_suffix("")
        attempt = episodes / f"seed-{seed:03d}" / f"attempt-{int(time.time() * 1000)}"
        attempt.mkdir(parents=True, exist_ok=True)
        command = [str(python), "-m", f"jevarena_mm.bridges.{cls.suite}", "--upstream", str(checkout),
                   *cls().bridge_args(seed), "--endpoint", url, "--out", str(attempt),
                   *(["--latency-ms", f"{latency_ms:g}"] if mode == "realtime" else [])]
        # The suites' temporary files (Chromium's profile and cache among them) go to the node's own disk: under the
        # results folder on a shared (Lustre) file system, GameWorld's Wolf3D loads its scripts so slowly that its
        # loader races and the game never leaves "loading"
        tmp = tempfile.mkdtemp(prefix="jevarena-mm-")
        variables = {**os.environ, "PYTHONPATH": str(bridge_path()), "TMPDIR": tmp}
        for key, value in cls.extra_env.items():  # "$NAME" copies another variable when it is set
            if not value.startswith("$"):
                variables[key] = value
            elif os.environ.get(value[1:]):
                variables[key] = os.environ[value[1:]]
        started = time.time()
        limit = cls.episode_timeout_s(latency_ms if mode == "realtime" else 0.0)
        with (attempt / "bridge.log").open("w") as log:
            try:
                code = run_watched(command, attempt, variables, log, limit)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        result_path = attempt / "outcome.json"
        why = code if isinstance(code, str) else f"exited with {code}"
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "technical_failure", "error": f"bridge {why} without an outcome; see {attempt}/bridge.log"}
        source = {"attempt": str(attempt.relative_to(env.ROOT)) if attempt.is_relative_to(env.ROOT) else str(attempt)}
        wall = round(time.time() - started, 1)
        if result["status"] == "completed":
            outcome = {"success": result.get("success"), "score": result.get("score"), **(result.get("details") or {})}
            row = cls.record(model, seed, outcome=outcome, decisions=result.get("decisions"),
                             failed_decisions=result.get("failed_decisions", 0), wall_s=wall, played_from=source, **timing)
        elif result["status"] == "unsupported":
            row = cls.record(model, seed, unsupported=result.get("error", "unsupported"), played_from=source, **timing)
        else:
            row = cls.record(model, seed, error=result.get("error", "technical failure"), wall_s=wall, played_from=source, **timing)
        status = "completed" if "outcome" in row else "unsupported" if "unsupported" in row else "failed"
        print(f"  {cls.name} seed {seed}: {status}" + (f" score={row['outcome'].get('score')} success={row['outcome'].get('success')}"
                                                       if "outcome" in row else f" ({row.get('error') or row.get('unsupported')})"[:160]), flush=True)
        return row

    @classmethod
    def record(cls, model, seed: int, **fields) -> dict:
        row = {"seed": seed, "model": model.model_id, "mode": "paused", "env_version": cls.version, "assumed_latency_ms": None,
               "hold_steps": 0, "max_decisions": cls.max_decisions, "trace": [], **fields}
        path = env_runner.episode_path(model.slug, cls.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _APPEND, path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


class GameWorldEnvironment(UpstreamEnvironment):
    suite, setting = "gameworld", "GAMEWORLD"
    realtime_ok = True  # the game pauses while the model decides, then runs for the latency before the action lands
    extra_env = {"PLAYWRIGHT_BROWSERS_PATH": "$JEVARENA_MM_GAMEWORLD_BROWSERS"}
    source = "GameWorld (2025)"
    url = "https://github.com/gameworld-project/gameworld"
    license = "See the repository license"
    tier = "B"
    criteria_source = "Each game's official semantic controls as the options; official task targets and evaluator"
    score_name = "progress"
    notes = ("Paused main protocol, at most 100 agent steps; the current screenshot and up to two previous ones plus the "
             "task text. Episode i is the game's task i+1 at upstream seed 42; score is GameWorld's normalized task progress.")

    def bridge_args(self, seed: int) -> list[str]:
        return ["--game", self.name.rsplit("/", 1)[1], "--task", self.details["tasks"][seed],
                "--seed", str(self.details.get("upstream_seed", 42))]


class BalrogEnvironment(UpstreamEnvironment):
    suite, setting = "balrog", "BALROG"
    extra_env = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1", "OMP_NUM_THREADS": "2",
                 "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONWARNINGS": "ignore"}
    source = "BALROG (Paglieri et al., 2025)"
    url = "https://github.com/balrog-ai/BALROG"
    license = "See the repository license"
    tier = "A"
    criteria_source = "Official BALROG actions and instructions; naive agent with 1 image and 16 text turns of history"
    score_name = "progression"
    notes = "Score is BALROG's progression (0 to 1); success where the suite defines it. Episode i is the i-th seed listed in the suite record."

    def bridge_args(self, seed: int) -> list[str]:
        return ["--env", self.details["suite"], "--task", self.details["task"], "--seed", str(self.details["seeds"][seed]),
                "--episode-idx", str(seed)]


class VSBenchEnvironment(UpstreamEnvironment):
    suite, setting = "vsbench", "VSBENCH"
    extra_env = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    source = "VS-Bench (Xu et al., 2025)"
    url = "https://github.com/zelaix/VS-Bench"
    license = "See the repository license"
    tier = "A"
    criteria_source = "Official VS-Bench prompts per role, all legal actions as options"
    score_name = "return"
    notes = ("Native images (1 to 4) plus per-role text; the model plays the prompt-agent roles, upstream agents the others. "
             "Score is the mean return of the model's roles; success is a win where the environment declares a winner. "
             "max_decisions 0 = the upstream episode limit.")

    def bridge_args(self, seed: int) -> list[str]:
        # Version 1 is the official seeding the episodes were first recorded with; version 2 isolates the RNG streams.
        return ["--env", self.name.rsplit("/", 1)[1], "--episode", str(seed), "--rng", "legacy" if self.version == 1 else "isolated"]


SUITES = {"gameworld": GameWorldEnvironment, "balrog_pixels": BalrogEnvironment, "vsbench": VSBenchEnvironment}


def register_suites() -> list[type]:
    """Register one environment per entry of data/loop/suites.json."""
    path = env.DATA / "loop" / "suites.json"
    if not path.exists():
        return []
    classes = []
    for suite, record in json.loads(path.read_text(encoding="utf-8")).items():
        base = SUITES[suite]
        for name, attrs in record["envs"].items():
            title = "".join(part.title() for part in name.replace("-", "_").split("_"))
            cls = type(f"{base.__name__.removesuffix('Environment')}{title}", (base,), {
                "name": f"loop.game/{suite}/{name}", "leaf": f"loop.game/{suite}", "upstream": record["upstream"],
                "episodes": attrs["episodes"], "max_decisions": attrs["max_decisions"], "version": attrs.get("version", 1),
                "unsupported": attrs.get("unsupported"), "blocked": attrs.get("blocked"), "details": attrs.get("details", {})})
            classes.append(register_env(cls))
    return classes


register_suites()
