"""Paths and secrets. Secrets come from .env (never committed)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_LOCAL = ROOT / "data_local"  # built items we may not redistribute
RUNS = ROOT / "results" / "runs"  # raw model responses (gitignored); $JEVARENA_MM_RUNS moves them (runs_dir)
SUMMARY = ROOT / "results" / "summary"
CACHE = ROOT / ".cache"

_loaded = False


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ without overriding existing variables."""
    global _loaded
    if _loaded:
        return
    path = path or Path(os.environ.get("JEVARENA_MM_ENV_FILE", ROOT / ".env"))
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if "HF_API_KEY" in os.environ:
        os.environ.setdefault("HF_TOKEN", os.environ["HF_API_KEY"])
    _loaded = True


def media_dir() -> Path:
    """Root of all media files: $JEVARENA_MM_MEDIA (also read from .env), else media/ in the repository (gitignored)."""
    load_env()
    return Path(os.environ.get("JEVARENA_MM_MEDIA") or ROOT / "media")


def attempts_dir() -> Path | None:
    """Where closed-loop episode attempts keep their files (requests, replies, upstream logs; hundreds per episode):
    $JEVARENA_MM_ATTEMPTS, else next to the episode rows in results/episodes/ (None)."""
    load_env()
    value = os.environ.get("JEVARENA_MM_ATTEMPTS")
    return Path(value) if value else None


def runs_dir() -> Path:
    """Root of the raw model responses: $JEVARENA_MM_RUNS (also read from .env), else results/runs/ (gitignored)."""
    load_env()
    return Path(os.environ.get("JEVARENA_MM_RUNS") or RUNS)
