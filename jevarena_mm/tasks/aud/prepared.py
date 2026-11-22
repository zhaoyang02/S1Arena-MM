"""Audio tasks: catalog.json holds one task card per source, and the sources export.json lists are registered as frozen
tasks; no media or models are loaded on discovery."""

from __future__ import annotations

import json
from pathlib import Path

from ...registry import register_task
from ..frozen import FrozenTask

DIRECTORY = Path(__file__).parent


def catalog() -> list[dict]:
    return json.loads((DIRECTORY / "catalog.json").read_text(encoding="utf-8"))


class PreparedAudio(FrozenTask):
    redistributable = False
    media_root = "audio_delivery"
    status = "released"
    task_id = ""  # the task's folder in the prepared audio delivery (A01, ...), read by audio_delivery.py
    source_task = ""  # this source's records in that folder: their ids start with it, "/" written as "."
    diagnostic_only = False  # True: a diagnostic task, exported only on request and reported apart

    def card(self):
        return {**super().card(), "diagnostic_only": self.diagnostic_only}


def task_class(spec: dict) -> type[PreparedAudio]:
    return type("Audio_" + spec["name"].replace("/", "_").replace(".", "_"), (PreparedAudio,), dict(spec))


def register_domain(domain: str) -> None:
    # Catalog entries that export.json does not list (sources not in this release) are not registered.
    exported = json.loads((DIRECTORY / "export.json").read_text(encoding="utf-8"))
    for spec in catalog():
        if spec["name"].startswith(f"aud.{domain}/") and spec["name"] in exported["tasks"]:
            register_task(task_class(spec))
