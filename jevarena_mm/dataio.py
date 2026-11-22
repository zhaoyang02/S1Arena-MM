"""Reading and writing built items (one JSONL file per task).

Task "vid.basic/temporal/ssv2" is stored at data/vid/basic/temporal/ssv2.jsonl; each <modality>/<domain> folder holds
a manifest.json with the task cards, counts and file hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import env
from .schema import Item


def _parts(name: str) -> tuple[str, str]:
    modality, dot, rest = name.partition(".")
    if not dot or not rest:
        raise ValueError(f"expected <modality>.<domain>/..., got {name!r}")
    return modality, rest


def task_path(task: str, redistributable: bool = True) -> Path:
    modality, rest = _parts(task)
    return (env.DATA if redistributable else env.DATA_LOCAL) / modality / f"{rest}.jsonl"


def domain_dir(domain: str) -> Path:
    """data/<modality>/<domain> for a domain such as "vid.basic"."""
    modality, rest = _parts(domain)
    return env.DATA / modality / rest


def write_items(path: Path, items: list[Item]) -> None:
    """Write via a temporary file, so a run reading the task meanwhile never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    tmp.replace(path)


def read_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as f:
        return [Item.model_validate_json(line) for line in f if line.strip()]


def load_task(task) -> list[Item]:
    """Built items for a task adapter instance; empty if it has not been built yet."""
    path = task_path(task.name, task.redistributable)
    return read_items(path) if path.exists() else []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, rows: list[dict], known: set[str] | None = None) -> None:
    """Merge rows into root/manifest.json by task name; entries of tasks not rebuilt are kept,
    except tasks that are no longer registered (when `known` is given)."""
    path = root / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    new = {row["name"]: row for row in rows}
    merged = [new.pop(row["name"], row) for row in old] + list(new.values())
    merged = [row for row in merged if known is None or row["name"] in known]
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
