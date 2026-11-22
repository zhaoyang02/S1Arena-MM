"""Streaming commands share the main CLI without importing optional runtimes."""

import asyncio
from pathlib import Path


def cmd_prepare(args):
    from .prepare import prepare

    if args.per_task < 0:
        raise ValueError("per-task must be nonnegative; 0 selects all eligible cases")
    manifest = prepare(args.source, args.out, args.per_task)
    print(f"Prepared {len(manifest['cases'])} episodes in {args.out / 'manifest.json'}")


def cmd_run(args):
    from .runner import run

    asyncio.run(run(args))


def cmd_score(args):
    from .report import score

    score(args)


def add_commands(sub):
    p = sub.add_parser(
        "stream-prepare",
        help="prepare full audio streaming episodes from Full-Duplex-Bench's sets (BetterUp's CANDOR folders, v1.5 ZIPs)",
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--per-task", type=int, default=20, help="cases per task; 0 = all eligible"
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "stream-run",
        help="run persistent audio I/O diagnostics (separate from offline run/play)",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["gemini", "qwen", "jev", "controller", "minicpm"], required=True)
    p.add_argument(
        "--model",
        help="explicit Gemini Live ID; Qwen uses the pinned backend's served name",
    )
    p.add_argument("--env", default="st.audio", help="streaming environment prefix")
    p.add_argument(
        "--limit", type=int, default=0, help="first N cases per category; 0 = all"
    )
    p.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument(
        "--url",
        default="ws://127.0.0.1:8099/v1/realtime?duplex=1",
        help="Qwen duplex endpoint",
    )
    p.add_argument("--resume-from", type=Path, action="append", default=[], help="reuse first successful matching episodes; retain runtime strata")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "stream-score",
        help="score local streaming traces with observation deadline clipping (no model calls)",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="fresh report directory")
    p.set_defaults(func=cmd_score)
