"""Image tasks: catalog.json holds one task card per source, each registered as a frozen task; no media or models are
loaded on discovery."""

from __future__ import annotations

import json
from pathlib import Path

from ...registry import register_task
from ..frozen import FrozenTask

DIRECTORY = Path(__file__).parent


def catalog() -> list[dict]:
    return json.loads((DIRECTORY / "catalog.json").read_text(encoding="utf-8"))


class ExportedImage(FrozenTask):
    status = "preview"


def task_class(spec: dict) -> type[ExportedImage]:
    return type("Image_" + spec["name"].replace("/", "_").replace(".", "_"), (ExportedImage,), dict(spec))


def register_domain(domain: str) -> None:
    for spec in catalog():
        if spec["name"].startswith(f"img.{domain}/"):
            register_task(task_class(spec))
