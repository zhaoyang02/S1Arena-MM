"""Shared fixtures: an isolated workspace (data, runs, cache, media) and tiny synthetic media files."""

from __future__ import annotations

import math
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from jevarena_mm import env, media, registry
from jevarena_mm.envs import runner as env_runner


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every path the harness writes to into tmp_path; media root is tmp_path/media."""
    monkeypatch.setattr(env, "DATA", tmp_path / "data")
    monkeypatch.setattr(env, "DATA_LOCAL", tmp_path / "data_local")
    monkeypatch.setattr(env, "RUNS", tmp_path / "results" / "runs")
    monkeypatch.setattr(env, "SUMMARY", tmp_path / "results" / "summary")
    monkeypatch.setattr(env, "CACHE", tmp_path / ".cache")
    monkeypatch.setattr(env_runner, "EPISODES", tmp_path / "results" / "episodes")
    monkeypatch.setattr(media, "_SHA_CACHE", None)
    monkeypatch.setattr(media, "_SHA_DIRTY", 0)
    monkeypatch.setenv("JEVARENA_MM_MEDIA", str(tmp_path / "media"))
    monkeypatch.setenv("JEVARENA_MM_RUNS", str(tmp_path / "results" / "runs"))  # a .env setting must not leak in
    # nor the developer's .env at all: env.media_dir() and the CLI load it into os.environ for good, and its suite paths
    # made a later test play the installed GameWorld
    monkeypatch.setattr(env, "_loaded", True)
    (tmp_path / "media").mkdir()
    return tmp_path


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Tests may register tasks and environments; they are dropped afterwards."""
    registry.discover()
    monkeypatch.setattr(registry, "TASKS", dict(registry.TASKS))
    monkeypatch.setattr(registry, "ENVS", dict(registry.ENVS))
    return registry


def make_video(path: Path, seconds: float = 4.0, fps: int = 10, size: int = 32) -> Path:
    """A lossless video whose frame i has every pixel set to (8 * i) % 256; frame i starts at i / fps seconds."""
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv444p"
        stream.time_base = Fraction(1, fps)
        for i in range(round(seconds * fps)):
            array = np.full((size, size, 3), (8 * i) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_mp4(path: Path, seconds: float = 60.0, fps: int = 10, size: int = 32, gop: int = 25) -> Path:
    """An MPEG-4 Part 2 video in MP4 with a keyframe every `gop` frames (frame i: every pixel near (8 * i) % 256)."""
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop
        stream.options = {"qscale": "1"}
        for i in range(round(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.full((size, size, 3), (8 * i) % 256, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_wav(path: Path, seconds: float = 2.0, rate: int = 8000, hz: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(seconds * rate)) / rate
    samples = (0.5 * np.sin(2 * math.pi * hz * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.tobytes())
    return path


def make_png(path: Path, size: int = 16, value: int = 200) -> Path:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(path)
    return path
