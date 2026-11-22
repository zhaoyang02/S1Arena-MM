"""Submit a sweep plan to Slurm: one job array of scripts/slurm/run_shards.slurm per model (`jevarena-mm sweep-submit`).

Each model of the plan may carry its cluster settings in a `slurm` table (defaults in `[defaults.slurm]`):

    [[models]]
    spec = "clef:Cloudflare/clef@gpus=2"
    slurm = { gpus_per_shard = 2, fill = "raya", array = 2, time = "24:00:00" }

gpus_per_shard: GPUs one shard process uses (default 1); fill: a model run on the GPUs a node has beyond its whole
shards (run_shards.slurm's JV_FILL_MODEL); array: nodes (array tasks) of the model (default 1); time: the time limit
of each (default 48:00:00); name: the job name (default: jv- and the model id). The plan's tasks and environments become JV_LEAVES and JV_ENVS, its k, limit and episodes
the run and play arguments. Without --go the sbatch commands are printed, not run. Submitting again resumes: the
runs and episodes already cached are not made again.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .env import ROOT
from .registry import make_model
from .sweep import Plan

SCRIPT = ROOT / "scripts" / "slurm" / "run_shards.slurm"


def commands(plan: Plan, partition: str, account: str = "", site_env: str = "", log_dir: str = "",
             only: str | None = None, extra: list[str] | None = None) -> list[tuple[dict[str, str], list[str]]]:
    """(environment, sbatch argv) per model of the plan."""
    out = []
    logs = Path(log_dir or ROOT / "results" / "slurm").resolve()
    for mp in plan.models:
        cfg = mp.slurm
        if int(cfg.get("array", 1)) == 0:  # runs only as another model's fill: a second shard scheme would repeat calls
            continue
        run_args = [f"--k {mp.k}"] if mp.k != 1 else []
        run_args += [f"--workers {mp.workers}"] if mp.workers != 8 else []  # threads preparing media (and calling)
        run_args += [f"--limit {plan.limit}"] if plan.limit else []
        play_args = [f"--workers {mp.env_workers}"] + ([f"--episodes {plan.episodes}"] if plan.episodes else [])
        env = {
            "JV_MODEL": mp.spec,
            "JV_GPUS_PER_SHARD": str(cfg.get("gpus_per_shard", 1)),
            "JV_FILL_MODEL": cfg.get("fill", ""),
            "JV_LEAVES": " ".join(plan.tasks) if mp.static and only != "envs" else "",
            "JV_ENVS": " ".join(plan.envs) if mp.envs and only != "static" else "",
            "JV_RUN_ARGS": " ".join(run_args),
            "JV_PLAY_ARGS": " ".join(play_args),
            "JV_REPO": str(ROOT),
            "JV_LOG_DIR": str(logs),
        }
        if site_env:
            env["JV_SITE_ENV"] = str(Path(site_env).resolve())
        name = cfg.get("name") or "jv-" + re.sub(r"[^A-Za-z0-9.]+", "-", make_model(mp.spec).model_id.split("/")[-1])[:24]
        argv = ["sbatch", "-p", partition, *(["-A", account] if account else []), "-t", str(cfg.get("time", "48:00:00")),
                f"--array=0-{int(cfg.get('array', 1)) - 1}", "-J", name, "-o", str(logs / "%x.%A_%a.out"), *(extra or []),
                str(SCRIPT)]
        out.append((env, argv))
    return out


def submit(plan: Plan, go: bool = False, **kwargs) -> None:
    """Print each model's sbatch command; with go, run it (settings travel in the environment, which sbatch exports:
    --export=A=x,B=y would split specs that hold commas)."""
    import os

    for env, argv in commands(plan, **kwargs):
        line = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(argv)
        if not go:
            print(line)
            continue
        Path(env["JV_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
        print(f"{env['JV_MODEL']}: {(result.stdout.strip().splitlines() or [''])[-1]}{result.stderr.strip()[-300:]}")
