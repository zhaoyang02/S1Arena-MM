#!/usr/bin/env python3
"""A latency plan on a machine without Slurm: the local counterpart of `jevarena-mm latency-plan --go`.

Runs the plan's models one after another, each through scripts/slurm/latency.slurm with the variables latency-plan
passes (JV_SPEC, JV_GPUS, JV_SET, JV_ENV, JV_KINDS, JV_GPU_LABEL, JV_SITE_ENV) and the plan's time limit, so each stages the media,
runs the fit test and then the set as a job would. Every model runs on the first GPU of --gpus; a two-GPU spec adds the
first of the others that no process holds (JV_CUDA_DEVICES). A model whose one-GPU fit test fails (exit 3) runs its
plan entry's `fallback` spec on two GPUs. Logs and status.tsv (one line per run: time, spec, kinds, exit code, minutes)
go to --logs; a spec already run with exit 0 is skipped.

  python scripts/slurm/latency_local.py sweeps/latency_h100_sxm.toml --site-env scripts/slurm/site.env --dry-run
  python scripts/slurm/latency_local.py sweeps/latency_h100_sxm.toml --gpus 0,1,2,3 --models raya
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def seconds(limit: str) -> int:
    h, m, s = (int(x) for x in limit.split(":"))
    return h * 3600 + m * 60 + s


def free_gpus(candidates: list[int]) -> list[int]:
    """The candidates on which no process runs."""
    uuids = dict(line.split(", ") for line in subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines())
    busy = {int(uuids[u]) for u in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                                                  capture_output=True, text=True).stdout.split() if u in uuids}
    return [g for g in candidates if g not in busy]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", nargs="?", default=str(REPO / "sweeps" / "latency_h100.toml"))
    parser.add_argument("--models", nargs="*", default=[], help="only these specs of the plan")
    parser.add_argument("--kinds", default="", help="only these input kinds, e.g. loop (written apart)")
    parser.add_argument("--site-env", default=str(REPO / "scripts" / "slurm" / "site.env"))
    parser.add_argument("--gpus", default="0,1", help="GPU for every model, then the candidates for a second one")
    parser.add_argument("--logs", default=str(REPO / ".cache" / "latency_logs"))
    parser.add_argument("--force", action="store_true", help="also run specs already run with exit 0")
    parser.add_argument("--dry-run", action="store_true", help="print what would run")
    args = parser.parse_args()

    plan = tomllib.loads(Path(args.plan).read_text(encoding="utf-8"))
    models = [m for m in plan["models"] if not args.models or m["spec"] in args.models]
    if args.models and len(models) != len(args.models):
        raise SystemExit(f"not in the plan: {sorted(set(args.models) - {m['spec'] for m in models})}")
    kinds = args.kinds or plan.get("kinds", "")
    gpus_given = [int(g) for g in args.gpus.split(",")]
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    status = logs / "status.tsv"
    done = {tuple(line.split("\t")[1:3]) for line in (status.read_text().splitlines() if status.exists() else [])
            if line.split("\t")[3] == "0"}

    def run(spec: str, gpus: int, m: dict) -> int:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", spec)[:80] + (f".{kinds.replace(',', '-')}" if kinds else "")
        devices = gpus_given[:1] + free_gpus(gpus_given[1:])[:gpus - 1]
        if len(devices) < gpus:
            print(f"{time.strftime('%F %T')} {spec}: no free GPU for its {gpus} GPUs, skipped", flush=True)
            return 4
        variables = {"JV_SPEC": spec, "JV_GPUS": str(gpus), "JV_SET": plan.get("set", ""), "JV_ENV": m.get("env", "jev"),
                     "JV_SITE_ENV": args.site_env, "JV_REPO": str(REPO), "JV_CUDA_DEVICES": ",".join(map(str, devices)),
                     **({"JV_KINDS": kinds} if kinds else {}),
                     **({"JV_GPU_LABEL": plan["gpu_label"]} if plan.get("gpu_label") else {})}
        limit = m.get("time", plan.get("time", "04:00:00"))
        print(f"{time.strftime('%F %T')} {spec} (env {variables['JV_ENV']}, GPU {variables['JV_CUDA_DEVICES']}, {limit}) "
              f"-> {logs / name}.log", flush=True)
        if args.dry_run:
            return 0
        started = time.time()
        with open(logs / f"{name}.log", "w") as log:
            code = subprocess.run(["timeout", "--signal=TERM", "--kill-after=120", str(seconds(limit)), "bash",
                                   str(REPO / "scripts" / "slurm" / "latency.slurm")], cwd=REPO,
                                  env={**os.environ, **variables}, stdout=log, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - started) / 60
        with open(status, "a") as out:
            out.write(f"{time.strftime('%F %T')}\t{spec}\t{kinds}\t{code}\t{minutes:.1f}\n")
        print(f"  exit {code} after {minutes:.0f} min" + ("  (fit test failed)" if code == 3 else ""), flush=True)
        return code

    for m in models:
        if {(m["spec"], kinds), (m.get("fallback"), kinds)} & done and not args.force:
            print(f"skip (done): {m['spec']}")
            continue
        if run(m["spec"], int(m.get("gpus", 1)), m) == 3 and m.get("fallback"):
            run(m["fallback"], 2, m)


if __name__ == "__main__":
    main()
