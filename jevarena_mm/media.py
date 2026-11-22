"""Resolve, hash, probe and decode media references.

No sampling policy lives here. Model adapters choose frame rates, frame caps, resolutions and audio rates; these
helpers only decode what they are asked for, so every adapter decodes the same way:

    media.image(ref)                                    -> PIL.Image (RGB)
    media.frames(ref, fps=1.0, max_frames=64)           -> [Frame(t, array)], inside ref.window
    media.audio(ref, sample_rate=16000, mono=True)      -> (float32 array, sample rate), inside ref.window

Video and audio decoding needs PyAV (`pip install -e .[video]` or `.[audio]`); images need Pillow (`.[image]`).
Closed-loop environments put live observations into requests with inline_image(), which keeps the picture in memory
and references it as "mem:<sha256>".
"""

from __future__ import annotations

import atexit
import bisect
import hashlib
import io
import json
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import env
from .schema import Marks, MediaRef

INLINE_LIMIT = 512  # in-memory observations kept at once (oldest dropped first)
_INLINE: OrderedDict[str, Any] = OrderedDict()
_SHA_LOCK = threading.Lock()


@dataclass
class Frame:
    t: float  # seconds on the file's own timeline
    array: np.ndarray  # height x width x 3, uint8, RGB


def path_of(ref: MediaRef) -> Path:
    if ref.uri.startswith("mem:"):
        raise ValueError(f"{ref.uri} is an in-memory observation, not a file")
    return env.media_dir() / ref.uri


_SHA_CACHE: dict[str, list] | None = None
_SHA_DIRTY = 0
SHA_FLUSH_EVERY = 200  # new hashes between writes of the cache file


def _sha_cache_path() -> Path:
    return env.CACHE / "media_sha256.json"


def _sha_cache() -> dict[str, list]:
    global _SHA_CACHE
    if _SHA_CACHE is None:
        path = _sha_cache_path()
        _SHA_CACHE = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _SHA_CACHE


def flush_sha_cache() -> None:
    """Write the sha256 cache to .cache/media_sha256.json (also done every SHA_FLUSH_EVERY new hashes and at exit).
    Several processes may hash at once (`fetch` runs its scripts in parallel): each merges the file's entries into its
    own and writes through a temporary file of its own, so the file is always one process's complete table."""
    global _SHA_DIRTY
    with _SHA_LOCK:
        if _SHA_CACHE is None or not _SHA_DIRTY:
            return
        path = _sha_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):  # replaced or being replaced by another process: keep ours
            on_disk = {}
        for key, hit in on_disk.items():
            _SHA_CACHE.setdefault(key, hit)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(_SHA_CACHE), encoding="utf-8")
        tmp.replace(path)
        _SHA_DIRTY = 0


atexit.register(flush_sha_cache)


def sha256_file(path: Path) -> str:
    """sha256 of a file, cached by (path, size, mtime) in .cache/media_sha256.json so large media is hashed once."""
    global _SHA_DIRTY
    path = Path(path).resolve()
    stat = path.stat()
    key, stamp = str(path), [stat.st_size, stat.st_mtime_ns]
    with _SHA_LOCK:
        hit = _sha_cache().get(key)
        if hit and hit[:2] == stamp:
            return hit[2]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    value = digest.hexdigest()
    with _SHA_LOCK:
        _sha_cache()[key] = stamp + [value]
        _SHA_DIRTY += 1
        flush = _SHA_DIRTY >= SHA_FLUSH_EVERY
    if flush:
        flush_sha_cache()
    return value


def open_media(path):
    """av.open for reading, with the container's and streams' metadata tags decoded leniently: a tag cut inside a
    multi-byte character (a MeetingBank file) made av.open itself fail; tags never change the decoded frames or audio."""
    import av

    try:
        return av.open(str(path), metadata_errors="replace")
    except TypeError:  # PyAV 19 dropped the argument
        return av.open(str(path))


def probe_file(path: Path, kind: str) -> dict[str, Any]:
    """Facts about a file that adapters may use to plan sampling."""
    if kind == "image":
        from PIL import Image

        with Image.open(path) as im:
            return {"width": im.width, "height": im.height}
    import av

    with open_media(path) as container:
        info: dict[str, Any] = {}
        if container.duration:
            info["duration_s"] = round(container.duration / av.time_base, 3)
        if kind == "video" and container.streams.video:
            stream = container.streams.video[0]
            if stream.average_rate:
                info["fps"] = round(float(stream.average_rate), 3)
            info["width"], info["height"] = stream.codec_context.width, stream.codec_context.height
        info["has_audio"] = bool(container.streams.audio)
        if container.streams.audio:
            stream = container.streams.audio[0]
            info["sample_rate"] = stream.codec_context.sample_rate
            info["channels"] = stream.codec_context.channels
        return info


def make_ref(kind: str, uri: str, window: tuple[float, float] | None = None, probe: bool = True) -> MediaRef:
    """A reference to $JEVARENA_MM_MEDIA/<uri>, with its sha256 and (with probe) its facts."""
    path = env.media_dir() / uri
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path} (run `jevarena-mm fetch` for its task)")
    info = probe_file(path, kind) if probe else None
    return MediaRef(kind=kind, uri=uri, sha256=sha256_file(path), window=window, info=info)


def check(ref: MediaRef) -> str | None:
    """None if the file (and its marks file, if any) is present and matches its sha256; otherwise a short reason."""
    try:
        path = path_of(ref)
    except ValueError as error:
        return str(error)
    if not path.is_file():
        return "missing"
    if sha256_file(path) != ref.sha256:
        return "sha256 mismatch"
    if ref.marks is not None:
        marks_path = env.media_dir() / ref.marks.uri
        if not marks_path.is_file():
            return "marks missing"
        if sha256_file(marks_path) != ref.marks.sha256:
            return "marks sha256 mismatch"
    return None


_MARKS: OrderedDict[str, dict[int, tuple[float, float, float, float]]] = OrderedDict()


def load_marks(marks: Marks) -> dict[int, tuple[float, float, float, float]]:
    """frame index -> box (x1, y1, x2, y2) of a marks file."""
    if marks.sha256 not in _MARKS:
        data = json.loads((env.media_dir() / marks.uri).read_text(encoding="utf-8"))
        _MARKS[marks.sha256] = {int(row[0]): tuple(map(float, row[1:5])) for row in data["boxes"]}
        while len(_MARKS) > 64:
            _MARKS.popitem(last=False)
    return _MARKS[marks.sha256]


def draw_marks(array: np.ndarray, box: tuple[float, float, float, float], style: str) -> np.ndarray:
    """Draw one marking onto a frame (style "red-outline-3px": a 3-pixel red rectangle on the box's pixels)."""
    from PIL import Image, ImageDraw

    if style != "red-outline-3px":
        raise ValueError(f"unknown marks style {style!r}")
    x1, y1, x2, y2 = box
    image = Image.fromarray(array)
    ImageDraw.Draw(image).rectangle((math.floor(x1), math.floor(y1), math.ceil(x2) - 1, math.ceil(y2) - 1),
                                    outline=(255, 0, 0), width=3)
    return np.asarray(image)


def inline_image(picture: Any) -> MediaRef:
    """An in-memory image (PIL image or height x width x 3 uint8 array) as a "mem:<sha256>" reference."""
    from PIL import Image

    im = picture if isinstance(picture, Image.Image) else Image.fromarray(np.asarray(picture, dtype=np.uint8))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    _INLINE[sha] = im
    _INLINE.move_to_end(sha)
    while len(_INLINE) > INLINE_LIMIT:
        _INLINE.popitem(last=False)
    return MediaRef(kind="image", uri=f"mem:{sha}", sha256=sha, info={"width": im.width, "height": im.height})


def image(ref: MediaRef):
    """The referenced image as an RGB PIL image."""
    from PIL import Image

    if ref.kind != "image":
        raise ValueError(f"expected an image reference, got {ref.kind}")
    if ref.uri.startswith("mem:"):
        if ref.sha256 not in _INLINE:
            raise KeyError(f"in-memory image {ref.sha256[:12]} is no longer held (INLINE_LIMIT={INLINE_LIMIT})")
        return _INLINE[ref.sha256].convert("RGB")
    with Image.open(path_of(ref)) as im:
        return im.convert("RGB")


def _targets(start: float, end: float, fps: float | None, max_frames: int | None, timestamps: list[float] | None) -> list[float] | None:
    if timestamps is not None:
        return sorted(float(t) for t in timestamps)
    if fps:
        count = max(1, math.ceil((end - start) * fps - 1e-9))
        return [start + i / fps for i in range(count)]
    if max_frames:
        step = (end - start) / max_frames
        return [start + (i + 0.5) * step for i in range(max_frames)]
    return None  # every frame


def _thin(frames: list[Frame], max_frames: int | None) -> list[Frame]:
    if not max_frames or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    return [frames[round(i * (len(frames) - 1) / (max_frames - 1))] for i in range(max_frames)]


class _Unordered(Exception):
    """A frame's timestamp went backwards: the file needs the sorted (two-pass) selection."""


def _open_video(ref: MediaRef, fps, max_frames, timestamps):
    import av

    if ref.kind != "video":
        raise ValueError(f"expected a video reference, got {ref.kind}")
    container = open_media(path_of(ref))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = DECODE_THREADS
    if ref.window:
        start, end = ref.window
    else:
        start = 0.0
        end = container.duration / av.time_base if container.duration else math.inf
    if timestamps is None and fps is None and max_frames and math.isinf(end):
        container.close()
        raise ValueError("max_frames without fps needs a window or a known duration")
    if start > 0 and ref.marks is None:  # marks are indexed from the first frame, so a marked video is read from there
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
    return container, stream, start, end, _targets(start, end, fps, max_frames, timestamps)


def _frame(ref: MediaRef, t: float, frame, index: int) -> Frame:
    array = frame.to_ndarray(format="rgb24")
    if ref.marks is not None:
        box = load_marks(ref.marks).get(index)
        if box is not None:
            array = draw_marks(array, box, ref.marks.style)
    return Frame(t=round(t, 6), array=array)


def _stream_select(ref: MediaRef, fps, max_frames, timestamps) -> list[Frame]:
    """One pass, for files whose frame timestamps only increase (almost all files)."""
    container, stream, start, end, targets = _open_video(ref, fps, max_frames, timestamps)
    chosen: dict[float, Frame] = {}

    def take(t: float, frame, index: int) -> None:
        if t not in chosen:
            chosen[t] = _frame(ref, t, frame, index)

    with container:
        prev = None  # (t, frame, index): the last decoded frame inside the window
        last = -math.inf
        i, index = 0, -1  # index: presentation index from the first frame (meaningful when decoding from the start)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            index += 1
            t = float(frame.time)
            if t < last - 1e-9:
                raise _Unordered
            last = t
            if t < start - WINDOW_EPS_S:
                continue
            if t >= end - WINDOW_EPS_S:
                break
            if targets is None:
                take(t, frame, index)
            else:
                while i < len(targets) and targets[i] <= t + 1e-9:
                    if prev is not None and targets[i] - prev[0] <= t - targets[i] + 1e-9:  # ties: the earlier frame
                        take(*prev)
                    else:
                        take(t, frame, index)
                    i += 1
                if i == len(targets):
                    break
            prev = (t, frame, index)
        if targets is not None and i < len(targets) and prev is not None:  # targets past the last frame
            take(*prev)
    return [chosen[t] for t in sorted(chosen)]


# FFmpeg's threads per decoder (its own default is one per core: with many files decoded at once, thousands of threads
# thrash); threaded decoding gives the same frames for any count.
DECODE_THREADS = int(os.environ.get("JEVARENA_MM_DECODE_THREADS", "4"))
SEEK_GAP_S = 5.0  # targets at least this far apart on average are reached by seeking, not by decoding every frame
# Checked to give the whole-stream decode's frames, pixel for pixel, on 56 MP4 files of 17 datasets (two AVI files did
# not: their indexed keyframes depend on earlier frames), so other containers decode the whole stream.
SEEK_FRAMES = True
SEEK_CONTAINERS = {".mp4", ".m4v", ".mov"}
SEEK_MARKED = True  # marked videos by seeking: same frames and marks as the whole-stream decode on 24 PIE reads


def _seek_select(ref: MediaRef, fps, max_frames, timestamps) -> list[Frame] | None:
    """Sparse targets: the frames _stream_select returns (the nearest in-window frame to each target, the earlier on a
    tie, the last one for targets past it), decoding from the keyframe before each far target instead of every frame
    of a long window. None (the caller decodes the whole stream) for dense targets, containers outside
    SEEK_CONTAINERS, a seek that lands past its target, or timestamps that go backwards. Marked videos (marks count
    frames from the file's start) are read so only at a constant frame rate: the file's frame count matches its
    duration and every decoded timestamp sits on the grid of its first frame, so a frame's index is its time on it."""
    if ref.kind != "video" or path_of(ref).suffix.lower() not in SEEK_CONTAINERS:
        return None
    try:
        container, stream, start, end, targets = _open_video(ref, fps, max_frames, timestamps)
    except ValueError:
        return None
    if not targets or math.isinf(end) or (end - start) / len(targets) < SEEK_GAP_S:
        container.close()
        return None
    rate = float(stream.average_rate or stream.guessed_rate or 25)
    origin = None  # marked videos: the first frame's time, index 0 of the constant-rate grid
    if ref.marks is not None and not SEEK_MARKED:
        container.close()
        return None
    if ref.marks is not None:
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if not (stream.average_rate and stream.frames and duration and abs(stream.frames - duration * rate) <= 1.0):
            container.close()
            return None
        first = next((f for f in container.decode(stream) if f.time is not None), None)  # _open_video does not seek
        if first is None:
            container.close()
            return None
        origin = float(first.time)
        container.seek(0, stream=stream, backward=True)  # back to the first frame for the read itself
    chosen: dict[float, Frame] = {}

    def take(t: float, frame) -> None:
        if t not in chosen:
            chosen[t] = _frame(ref, t, frame, round((t - origin) * rate) if origin is not None else -1)

    try:
        with container:
            decoded = container.decode(stream)
            prev, last, i = None, -math.inf, 0
            fresh = sought = None  # the target of the last seek (fresh: until its first frame is checked)
            while i < len(targets):
                if targets[i] - max(last, start) > SEEK_GAP_S and sought != targets[i]:  # far: from the keyframe before it
                    container.seek(int(max(targets[i] - 2.0 / rate, 0.0) / stream.time_base), stream=stream, backward=True)
                    decoded = container.decode(stream)
                    prev, last, fresh = None, -math.inf, targets[i]
                    sought = fresh  # once per target: a keyframe further back than SEEK_GAP_S is decoded forward from
                frame = next(decoded, None)
                if frame is None:
                    break
                if frame.time is None:
                    continue
                t = float(frame.time)
                if fresh is not None:
                    if t >= fresh - 1e-9:  # the seek landed past the target: its previous frame is unknown
                        return None
                    fresh = None
                if t < last - 1e-9:
                    return None
                if origin is not None and abs((t - origin) * rate - round((t - origin) * rate)) > 0.01:
                    return None  # off the constant-rate grid: indices by time would be wrong
                last = t
                if t < start - WINDOW_EPS_S:
                    continue
                if t >= end - WINDOW_EPS_S:
                    break
                while i < len(targets) and targets[i] <= t + 1e-9:
                    if prev is not None and targets[i] - prev[0] <= t - targets[i] + 1e-9:  # ties: the earlier frame
                        take(*prev)
                    else:
                        take(t, frame)
                    i += 1
                prev = (t, frame)
            if i < len(targets) and prev is not None:  # targets past the last frame
                take(*prev)
    except av_errors():
        return None
    return [chosen[t] for t in sorted(chosen)]


def av_errors() -> tuple[type, ...]:
    import av

    return (av.error.FFmpegError, OSError, ValueError)


UNORDERED_SLACK_S = 5.0  # how far past the window end a backwards timestamp may still land inside it
WINDOW_EPS_S = 1e-6  # windows are stored to the microsecond, so a bound may sit up to 0.5 us past a frame's timestamp


def _sorted_select(ref: MediaRef, fps, max_frames, timestamps) -> list[Frame]:
    """Two passes, for files whose timestamps go backwards: frames are ordered by timestamp (equal timestamps: the
    earliest decoded), the frames nearest the targets are chosen, then only those are decoded to pixels."""
    container, stream, start, end, targets = _open_video(ref, fps, max_frames, timestamps)
    with container:
        seen = []  # (t, decode order)
        for order, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                continue
            t = float(frame.time)
            if t >= end + UNORDERED_SLACK_S:
                break
            seen.append((t, order))
    ranked = sorted(seen)
    rank = {order: position for position, (_, order) in enumerate(ranked)}  # presentation index (marks)
    unique: list[tuple[float, int]] = []
    for t, order in ranked:
        if start - WINDOW_EPS_S <= t < end - WINDOW_EPS_S and (not unique or t > unique[-1][0]):
            unique.append((t, order))
    if targets is None:
        picks = unique
    else:
        times = [t for t, _ in unique]
        picks = []
        for target in targets:
            j = bisect.bisect_left(times, target - 1e-9)
            options = [k for k in (j - 1, j) if 0 <= k < len(unique)]
            if options:
                picks.append(unique[min(options, key=lambda k: (abs(times[k] - target), k))])
    wanted = {order: t for t, order in picks}
    out: dict[float, Frame] = {}
    container, stream, *_ = _open_video(ref, fps, max_frames, timestamps)
    with container:
        for order, frame in enumerate(container.decode(stream)):
            if order in wanted and wanted[order] not in out:
                out[wanted[order]] = _frame(ref, wanted[order], frame, rank[order])
            if order >= max(wanted, default=-1):
                break
    return [out[t] for t in sorted(out)]


def frames(ref: MediaRef, fps: float | None = None, max_frames: int | None = None,
           timestamps: list[float] | None = None) -> list[Frame]:
    """Decoded frames inside the reference's window [start, end), in timestamp order.

    timestamps: the frame nearest each given time (the earlier one on a tie); fps: one frame every 1/fps seconds from
    the window start; neither: every frame (or, with max_frames alone, max_frames frames at the centres of equal
    spans). max_frames then keeps an evenly spaced subset. A frame is never returned twice, so asking for more frames
    than the file holds returns every distinct frame once. Marks declared on the reference are drawn onto the returned
    frames. Files whose timestamps go backwards are read twice, ordering frames by timestamp.
    """
    try:
        chosen = _seek_select(ref, fps, max_frames, timestamps) if SEEK_FRAMES else None
        if chosen is None:
            chosen = _stream_select(ref, fps, max_frames, timestamps)
    except _Unordered:
        chosen = _sorted_select(ref, fps, max_frames, timestamps)
    return _thin(chosen, max_frames)


AUDIO_BAD_PACKETS = 50  # undecodable packets an audio read skips before it fails


def audio(ref: MediaRef, sample_rate: int = 16000, mono: bool = True) -> tuple[np.ndarray, int]:
    """The audio track inside the reference's window, resampled: float32, shape (samples,) when mono, else
    (channels, samples). Works for audio files and for the sound track of a video file."""
    import av

    if ref.kind not in ("audio", "video"):
        raise ValueError(f"expected an audio or video reference, got {ref.kind}")
    with open_media(path_of(ref)) as container:
        if not container.streams.audio:
            raise ValueError(f"{ref.uri} has no audio stream")
        stream = container.streams.audio[0]
        channels = 1 if mono else stream.codec_context.channels
        layout = "mono" if mono else stream.codec_context.layout.name
        resampler = av.AudioResampler(format="flt", layout=layout, rate=sample_rate)
        start, end = ref.window or (0.0, math.inf)
        if start > 0:
            container.seek(int(start * av.time_base), backward=True)
        chunks, first, bad = [], None, 0
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:  # e.g. a partial frame right after a seek: skipped, as ffmpeg does
                bad += 1
                if bad > AUDIO_BAD_PACKETS:
                    raise
                continue
            done = False
            for frame in frames:
                if first is None:
                    first = float(frame.time or 0.0)
                if frame.time is not None and float(frame.time) >= end:
                    done = True
                    break
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1, channels))
            if done:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1, channels))
    samples = np.concatenate(chunks) if chunks else np.zeros((0, channels), dtype=np.float32)
    skip = max(0, round((start - (first or 0.0)) * sample_rate))
    samples = samples[skip:]
    if not math.isinf(end):
        samples = samples[: round((end - start) * sample_rate)]
    samples = samples.astype(np.float32)
    return (samples[:, 0] if mono else samples.T), sample_rate
