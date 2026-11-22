"""Explicit streaming execution; immutable inputs and fresh run directories."""

import asyncio
import hashlib
import importlib.metadata
import os
from datetime import datetime, timezone
from pathlib import Path

from .catalog import PROTOCOL, TASKS, select_cases
from .manifest import digest, load_manifest, write_json


async def run(args):
    manifest = load_manifest(args.manifest, verify_audio=True)
    cases = select_cases(manifest, args.env, args.limit)
    if args.workers not in range(1, 5) or (
        args.backend in ("qwen", "jev", "controller", "minicpm") and args.workers != 1
    ):
        raise ValueError("Use 1–4 workers for Gemini, exactly 1 for Qwen or Jev")
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.out.exists():
        raise ValueError(
            "Use a new output directory; prior attempts are never overwritten or retried"
        )

    from ..env import load_env

    load_env()
    client = None
    controller_service = None
    if args.backend == "gemini":
        if not args.model:
            raise ValueError("Gemini requires an explicit --model Live model ID")
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment")
        from google import genai
        from .gemini import CONFIG, Episode

        client = genai.Client(api_key=key)
        model, config = args.model, CONFIG
        versions = {"google-genai": importlib.metadata.version("google-genai")}
    else:
        if args.backend == "controller":
            from .controller import CONFIG, MODEL, Episode
        elif args.backend == "minicpm":
            from .minicpm import CONFIG, MODEL, Episode
        elif args.backend == "jev":
            from .jev_controller import CONFIG, MODEL, Episode
        else:
            from .qwen import CONFIG, MODEL, Episode

        if args.model and args.model != MODEL:
            raise ValueError(f"This transport expects {MODEL}")
        model, config = MODEL, CONFIG
        versions = {"websockets": importlib.metadata.version("websockets")}
        if args.backend == "jev":
            import json
            from urllib.request import urlopen

            with urlopen("http://127.0.0.1:8767/health", timeout=5) as response:
                controller_service = json.load(response)
            if (
                controller_service.get("protocol") != "jev-external-voice-v1"
                or controller_service.get("checkpoint_revision")
                != "5addda86ddee081a68fb067477ea100c221b8917"
                or not controller_service.get("ready")
            ):
                raise ValueError("Unexpected Jev controller service identity")
            for name in ("silero-vad", "onnxruntime", "torch"):
                versions[name] = importlib.metadata.version(name)
            config = {
                **config,
                "controller": {
                    "protocol": "jev-external-voice-v1",
                    "checkpoint_revision": "5addda86ddee081a68fb067477ea100c221b8917",
                    "voice_revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
                    "vad_threshold": 0.5,
                    "silence_ms": 500,
                    "padding_ms": 300,
                    "option_order": "sha256(case_id:decision_number), label-independent",
                    "scope": "Jev controller with external Qwen voice; not native Jev speech",
                },
            }

    args.out.mkdir(parents=True, exist_ok=False)
    progress = {
        "protocol": PROTOCOL,
        "backend": args.backend,
        "model": model,
        "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest(args.manifest),
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "runtime": versions,
        "controller_service": controller_service,
        "workers": args.workers,
        "automatic_retries": 0,
        "case_ids": [c["id"] for c in cases],
        "state": "running",
        "results": [],
    }

    if args.backend == "controller":
        from .controller import identity
        controller_service = identity()
        progress["controller_service"] = controller_service
        progress["model"] = controller_service["model_id"] + " + external Qwen voice"
        progress["config"] = {**config, "controller": controller_service}

    from .resume import reuse
    reused = reuse(getattr(args, 'resume_from', []), progress, args.out)
    progress["results"].extend(reused)
    reused_ids = {row["id"] for row in reused}

    def snapshot(state):
        progress["state"] = state
        finished = {r["id"] for r in progress["results"]}
        progress["pending_ids"] = [c["id"] for c in cases if c["id"] not in finished]
        write_json(args.out / "progress.json", progress)

    async def group(selected):
        async with semaphore:
            for case in selected:
                if case["id"] in reused_ids:
                    continue
                if stop.is_set():
                    return
                episode = Episode(case, args.manifest.parent, args.out / case["id"])
                if client is None:
                    result = await episode.run(args.url)
                else:
                    result = await episode.run(client, model)
                result["env"] = TASKS[case["category"]]
                progress["results"].append(result)
                snapshot("running")
                print(f"{result['env']} {result['id']}: {result['status']}", flush=True)
                if result["status"] != "completed":
                    stop.set()
                    raise RuntimeError(
                        "Streaming episode failed; see the local episode trace"
                    )

    semaphore = asyncio.Semaphore(args.workers)
    stop = asyncio.Event()
    groups = [[c for c in cases if c["category"] == category] for category in TASKS]
    jobs = []
    state = "interrupted"
    snapshot("running")
    try:
        jobs = [asyncio.create_task(group(g)) for g in groups if g]
        await asyncio.gather(*jobs)
        state = "completed"
    except Exception:
        state = "failed"
        raise
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        snapshot(state)
        if client is not None:
            await client.aio.aclose()
    return progress
