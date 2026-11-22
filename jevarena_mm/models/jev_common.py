"""Helpers shared by the Jev-family adapters (OneJev, Jev-Omni): reading a request's text and media, sampling video at
an adapter's own rate, and the sampling note these models were evaluated with in the recorded runs.

Recorded presentation: for the items of the recorded OneJev and Jev-Omni runs, recorded_presentation.json.gz holds the
media-timing note those runs printed, which comes in several variants (1 fps, prediction prefix, last permitted frame,
none), and, for boolean questions, the per-item Yes/No order shown to Jev-Omni. The adapters use it, so they rebuild
those runs' inputs exactly; any other item gets the generated note and a Yes/No order drawn from the request's hash,
the same per-item shuffle."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .. import media
from ..schema import DecisionRequest, MediaRef, Question


def state_text(request: DecisionRequest) -> str:
    """The text of a request's state: string values in order, other non-media values as JSON."""
    parts = []
    for value in request.state.values():
        if isinstance(value, MediaRef) or (isinstance(value, list) and value and all(isinstance(v, MediaRef) for v in value)):
            continue
        parts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return "\n".join(parts)


IMAGE_MARKER = re.compile(r"<image:(\d+)>")


def placed_images(text: str, count: int) -> bool:
    """True if the text already marks each of `count` images once, in order, as <image:k> (the suites' format)."""
    return count > 0 and IMAGE_MARKER.findall(text) == [str(i) for i in range(1, count + 1)]


def split_at_images(text: str) -> list[str]:
    """The text pieces around the <image:k> markers (one more piece than markers)."""
    return IMAGE_MARKER.split(text)[::2]


def refs(request: DecisionRequest, kind: str) -> list[MediaRef]:
    return [ref for _, ref in request.media() if ref.kind == kind]


def options(question: Question, order: list[str] | None = None) -> list[tuple[str, str]]:
    """(option id, option text) in presentation order; boolean questions read as Yes / No, or in the given id order."""
    if question.type == "choice":
        return list(question.criteria.items())
    if question.type == "score":
        return [(str(i), text) for i, text in enumerate(question.criteria)]
    texts = {"true": "Yes", "false": "No"}
    return [(option, texts[option]) for option in (order if order and sorted(order) == ["false", "true"] else ["true", "false"])]


RECORDED = Path(__file__).with_name("recorded_presentation.json.gz")
_RECORDED: dict | None = None


def request_key(request: DecisionRequest) -> str:
    """A request's key in recorded_presentation.json.gz: sha256 of its JSON, 16 hex digits."""
    return hashlib.sha256(request.model_dump_json().encode()).hexdigest()[:16]


def recorded(request: DecisionRequest) -> dict:
    """How the recorded runs presented this request ({"note": text, "order": boolean option ids}); {} otherwise."""
    global _RECORDED
    if _RECORDED is None:
        _RECORDED = json.loads(gzip.decompress(RECORDED.read_bytes()))["entries"] if RECORDED.exists() else {}
    return _RECORDED.get(request_key(request), {})


@dataclass
class View:
    ref: MediaRef
    frames: list[media.Frame]
    fps: float


def sample_views(request: DecisionRequest, fps: float, min_frames: int) -> list[View]:
    """Every video of the request at `fps` (a grid from the window start, nearest frames), with the rules of the
    recorded runs: a window that gives fewer than min_frames frames is read at min_frames / window seconds (frames at
    the window start and at equal fractions of it, as for short prediction prefixes), and where the recorded runs kept
    the last permitted observation, the last frame before the window end is added after the grid."""
    keep_last = "final permitted observation" in recorded(request).get("note", "")
    views = []
    for ref in refs(request, "video"):
        rate = fps
        frames = media.frames(ref, fps=rate)
        start, end = ref.window or (0.0, float((ref.info or {}).get("duration_s") or 0.0))
        if len(frames) < min_frames and end > start:
            rate = min_frames / (end - start)
            frames = media.frames(ref, fps=rate)
        if keep_last and end > start:
            last = media.frames(ref, timestamps=[end])
            if last and (not frames or last[-1].t > frames[-1].t):
                frames = frames + last[-1:]
        views.append(View(ref, frames, rate))
    return views


def window_of(view: View) -> tuple[float, float]:
    if view.ref.window:
        return view.ref.window
    duration = (view.ref.info or {}).get("duration_s")
    return 0.0, float(duration) if duration else (view.frames[-1].t if view.frames else 0.0)


def timing_note(request: DecisionRequest, views: list[View]) -> str:
    """The media-timing note after the state text: as the recorded runs printed it, else generated."""
    entry = recorded(request)
    return entry["note"] if "note" in entry else sampling_note(views)


def state_with_note(request: DecisionRequest, views: list[View]) -> str:
    """The state text with the media-timing note: after the text, or where the recorded runs put it."""
    text, note = state_text(request), timing_note(request, views)
    if not note:
        return text
    at = recorded(request).get("note_at")
    if at is None:
        return text + "\n" + note
    lines = text.split("\n")
    return "\n".join(lines[:at] + [note] + lines[at:])


def boolean_order(request: DecisionRequest) -> list[str]:
    """Boolean option ids in the order shown: the recorded runs' order, else a per-item order fixed by the request's
    hash (about half No first), so a model with a position bias is not always asked the same way."""
    order = recorded(request).get("order")
    if order:
        return order
    return ["true", "false"] if hashlib.sha256(b"boolean-order:" + request_key(request).encode()).digest()[0] % 2 == 0 else ["false", "true"]


def gpu_to_ourselves(torch) -> bool:
    """True when no other process computes on this process's GPU (so an out-of-memory error is the input's own doing,
    not contention); False when another process is there or it cannot be told."""
    import subprocess

    try:
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        listing = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return False
    pids = [int(pid) for gpu, pid in (line.split(",") for line in listing.splitlines() if line.strip())
            if gpu.strip().removeprefix("GPU-") == uuid.removeprefix("GPU-")]
    return all(pid == os.getpid() for pid in pids)


def prediction_prefix(request: DecisionRequest) -> bool:
    """The recorded runs treated this window as the history before a prediction cutoff (at least two frames)."""
    note = recorded(request).get("note", "")
    return "Prediction prefix:" in note or "prediction cutoff" in note


def sampling_note(views: list[View]) -> str:
    """The generated media-timing note after the state text (the recorded runs' wording): each video's sampling grid."""
    if not views:
        return ""
    rates = {view.fps for view in views}
    rate = views[0].fps if len(rates) == 1 else None
    step = "k seconds" if rate == 1 else f"k/{rate:g} seconds" if rate else "k/rate seconds"
    lines = [f"Media timing: each video is a {rate:g}fps material; video index k maps to target source time "
             f"window_start + {step}. Actual nearest source frames can differ slightly. Video views have independent clocks."
             if rate else
             "Media timing: each video is sampled on a regular grid from its window start (rates below). Actual nearest "
             "source frames can differ slightly. Video views have independent clocks."]
    for index, view in enumerate(views, start=1):
        start, end = window_of(view)
        lines.append(f"View {index}: source window [{start!r}, {end!r}) seconds; real window duration {end - start!r}s; "
                     f"{view.fps:g}fps container duration {len(view.frames) / view.fps!r}s.")
    return "\n".join(lines)


def pil(array):
    """A frame array as a PIL image."""
    from PIL import Image

    return Image.fromarray(array)


def model_dir(env_var: str | None, repo: str, revision: str) -> Path:
    """Local snapshot of a pinned model: $<env_var> (when one is named), else the Hugging Face Hub at that revision."""
    path = os.environ.get(env_var) if env_var else None
    if path:
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision))
