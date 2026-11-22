"""Base for tasks whose items are committed instead of built from the source dataset on each machine.

A frozen task carries the task card. Its items are committed under data/, or, when the source's annotations may not be
redistributed, as a recipe that `build` fills in from the user's own copy of them; `build` only checks the items.
"""

from __future__ import annotations

from .base import TaskAdapter


class FrozenTask(TaskAdapter):
    frozen = True
    pipeline = ""  # where the items were exported from: provenance for maintainers, not part of the card
