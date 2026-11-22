"""Validate the frozen local manifest before opening any model connection."""

import hashlib
import json
import math
import re
from pathlib import Path

from .catalog import TASKS
from .gemini import pcm_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path, *, verify_audio=False):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "audio-live-v1":
        raise ValueError("Expected an audio-live-v1 manifest")
    seen = set()
    for case in manifest["cases"]:
        ident = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in seen:
            raise ValueError("Unsafe or duplicate case ID")
        seen.add(ident)
        category = case["category"]
        if category not in TASKS:
            raise ValueError(f"Unknown streaming category: {category}")
        mode = "stream" if category.startswith("candor_") else "reactive"
        if case["mode"] != mode:
            raise ValueError("Category/mode mismatch")
        specs = [case["audio"]]
        if mode == "reactive":
            specs.append(case["user_event"])
            if case["trigger_after_audible_s"] != 1.0:
                raise ValueError(
                    "This protocol requires a one-second audible playback trigger"
                )
        for spec in specs:
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Audio path must stay inside the manifest directory")
            duration = spec["duration_s"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid audio duration")
            if not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
                raise ValueError("Invalid audio fingerprint")
            if verify_audio:
                pcm = pcm_file(path.parent, spec)
                if abs(len(pcm) / 32000 - duration) > 1 / 16000:
                    raise ValueError("Audio duration disagrees with the manifest")
        for event in case["gold_events"]:
            a, b = event["interval_s"]
            expected = "turn" if category == "candor_turn_taking" else "pause"
            if (
                mode != "stream"
                or event["type"] != expected
                or not 0 <= a < b <= case["audio"]["duration_s"]
            ):
                raise ValueError("Invalid held-out event")
    if not seen:
        raise ValueError("Empty streaming manifest")
    return manifest


def write_json(path, value):
    """Atomic progress snapshots; raw episode logs remain append-only."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
