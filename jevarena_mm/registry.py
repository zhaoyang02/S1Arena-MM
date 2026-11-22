"""Plugin registries. Adding a task or a model means adding one module; nothing else changes.

Tasks live under jevarena_mm/tasks/<modality>/<domain>/, closed-loop environments under jevarena_mm/envs/<domain>/,
models under jevarena_mm/models/. Every module in those packages is imported on discovery, so a decorated class
registers itself.

Names: a leaf is "<modality>.<domain>/<leaf>" (e.g. "vid.basic/temporal"), a task or environment is "<leaf>/<source>"
(e.g. "vid.basic/temporal/ssv2"). A prefix selects by whole components: "vid", "vid.basic" and "vid.basic/temporal"
all select that task; "vid" does not select "lvid.*".
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys

TASKS: dict[str, type] = {}
MODELS: dict[str, type] = {}
ENVS: dict[str, type] = {}
_discovered = False

MODALITIES = ("img", "vid", "lvid", "aud", "av", "st", "loop")
NAME = re.compile(r"^(?P<leaf>(?:" + "|".join(MODALITIES) + r")\.[a-z][a-z0-9_]*/[a-z0-9_]+)/[a-z0-9][a-z0-9_.\-]*$")


def _check_name(cls: type, kind: str) -> None:
    match = NAME.match(cls.name or "")
    if not match:
        raise ValueError(f"{kind} name {cls.name!r} is not <modality>.<domain>/<leaf>/<source> (modality in {MODALITIES})")
    if cls.leaf != match["leaf"]:
        raise ValueError(f"{kind} {cls.name!r} declares leaf {cls.leaf!r}; expected {match['leaf']!r}")


def register_task(cls: type) -> type:
    _check_name(cls, "task")
    if cls.name in TASKS:
        raise ValueError(f"task name duplicated: {cls.name!r}")
    TASKS[cls.name] = cls
    return cls


def register_env(cls: type) -> type:
    _check_name(cls, "environment")
    if cls.name in ENVS:
        raise ValueError(f"environment name duplicated: {cls.name!r}")
    ENVS[cls.name] = cls
    return cls


def register_model(cls: type) -> type:
    if not cls.kind or cls.kind in MODELS:
        raise ValueError(f"model kind missing or duplicated: {cls.kind!r}")
    MODELS[cls.kind] = cls
    return cls


SERVERS = "jevarena_mm.models.servers"  # model servers (run as their own processes), not adapters: never imported here


def discover() -> None:
    global _discovered
    if _discovered:
        return
    import jevarena_mm.envs
    import jevarena_mm.models
    import jevarena_mm.tasks

    for package in (jevarena_mm.tasks, jevarena_mm.envs, jevarena_mm.models):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=lambda name: None):
            if info.name.startswith(SERVERS):
                continue
            try:
                importlib.import_module(info.name)
            except Exception as error:  # one broken module must not hide every other task
                print(f"warning: skipped {info.name} ({type(error).__name__}: {error})", file=sys.stderr)
    _discovered = True


def matches(name: str, prefix: str) -> bool:
    prefix = prefix.strip("/")
    return not prefix or name == prefix or name.startswith(prefix + "/") or name.startswith(prefix + ".")


def envs_under(prefix: str) -> list[type]:
    """Environment classes selected by the prefix."""
    discover()
    return [ENVS[n] for n in sorted(ENVS) if matches(n, prefix)]


def tasks_under(prefix: str) -> list:
    """Instantiate every task selected by the prefix."""
    discover()
    return [TASKS[n]() for n in sorted(TASKS) if matches(n, prefix)]


def make_model(spec: str):
    """Build a model from a spec: "<kind>", "<kind>@<options>" or "<kind>:<argument>[@<options>]",
    e.g. "random", "onejev@fps=1", "llm:openai/gpt-4.1-mini". The adapter receives everything after the kind."""
    discover()
    match = re.match(r"[a-z0-9_]+", spec)
    kind = match.group(0) if match else ""
    if kind not in MODELS:
        raise KeyError(f"unknown model kind {kind!r}; known: {sorted(MODELS)}")
    arg = spec[len(kind):].removeprefix(":")
    return MODELS[kind](arg) if arg else MODELS[kind]()
