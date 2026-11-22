"""Task status, versions and releases.

Every task has a status. "released" tasks are frozen: they form the main tables, and their items change only with a
new task version. "preview" tasks ship with the same files and can be run, but are reported apart and may still
change. A new task starts as preview (TaskAdapter.status).

A release is a git tag vMAJOR.MINOR.PATCH over the committed items and manifests. `jevarena-mm check` compares the
released tasks with the last tag: a released task whose item file changed while its version did not is an error, and
so is a released task that is no longer registered, unless data/retired.json names it with the reason.
`jevarena-mm release` checks that `check` passes, the tree is clean and the version is new, writes the snapshot and
tags.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from typing import Any

from . import dataio, env, metrics, runner

STATUSES = ("released", "preview")
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
TABLE_METRICS = ("accuracy", "maj", "brier", "ece", "coverage", "latency_p50_ms", "cost_per_1k_usd", "ndcg", "pair_accuracy")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=env.ROOT, capture_output=True, text=True, check=True).stdout


def release_tags() -> list[str]:
    """Release tags, oldest first; none outside a git repository."""
    try:
        tags = _git("tag", "--list", "v*").split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return sorted((t for t in tags if TAG.match(t)), key=lambda t: tuple(map(int, TAG.match(t).groups())))


def released_manifest(tag: str) -> dict[str, dict]:
    """Manifest rows of every task at a tag, by name."""
    rows: dict[str, dict] = {}
    for path in _git("ls-tree", "-r", "--name-only", tag, "--", "data").splitlines():
        if path.endswith("/manifest.json"):
            rows.update({row["name"]: row for row in json.loads(_git("show", f"{tag}:{path}"))})
    return rows


def retired() -> dict[str, str]:
    """Released tasks withdrawn on purpose: data/retired.json, {"task name": "reason"}."""
    path = env.DATA / "retired.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def version_problems(tasks: list, rows: dict[str, dict], leaf: str = "") -> tuple[str | None, list[str]]:
    """(last tag, problems): released tasks of the last tag whose items changed without a version bump, or that are no
    longer registered (renamed or removed) without an entry in data/retired.json. `leaf` limits the check."""
    from .registry import matches

    tags = release_tags()
    if not tags:
        return None, []
    base, problems = released_manifest(tags[-1]), []
    names, withdrawn = {t.name for t in tasks}, retired()
    for name, old in sorted(base.items()):
        if old.get("status") == "released" and name not in names and matches(name, leaf) and name not in withdrawn:
            problems.append(f"{name}: released in {tags[-1]} but no longer registered; a renamed or removed released task "
                            "must be listed in data/retired.json with the reason (and noted in CHANGELOG.md)")
    for task in tasks:
        old, row = base.get(task.name), rows.get(task.name)
        if not old or old.get("status") != "released" or row is None:
            continue
        before = old.get("version", 1)
        if task.version < before:
            problems.append(f"{task.name}: version {task.version} is lower than {before} in {tags[-1]}")
        elif task.version == before and row.get("sha256") != old.get("sha256"):
            problems.append(f"{task.name}: items changed since {tags[-1]}, where the task was released, but its version is "
                            f"still {before} (set version = {before + 1} and note the change in CHANGELOG.md)")
    return tags[-1], problems


def task_results(models: list, tasks: list, limit: int = 0, k: int | None = None) -> dict[str, dict[str, dict]]:
    """task -> model id -> task metrics, from the run caches (no calls)."""
    out: dict[str, dict[str, dict]] = {}
    for task in tasks:
        items = dataio.load_task(task)[: limit or None]
        keep = {i.id for i in items}
        for model in models:
            rows = [r for r in runner.latest_rows(model.slug, task.name) if r["item_id"] in keep and r["sample"] < (k or 10**9)]
            if rows:
                ids = {r["item_id"] for r in rows}
                out.setdefault(task.name, {})[model.model_id] = metrics.task_metrics(task, [i for i in items if i.id in ids], rows)
    return out


def leaderboard(models: list, tasks: list, limit: int = 0, k: int | None = None) -> dict[str, Any]:
    """Main table over released tasks (a leaf is the mean of its released tasks, the overall score the mean over leaves,
    median for latency) and the preview tasks apart, each on its own."""
    import numpy as np

    results = task_results(models, tasks, limit, k)
    names = [m.model_id for m in models]
    agg = lambda key, values: float(np.median(values)) if key == "latency_p50_ms" else float(np.mean(values))
    by_leaf: dict[str, dict[str, dict[str, list[float]]]] = {}
    preview: dict[str, dict[str, dict]] = {}
    for task in tasks:
        for name, res in results.get(task.name, {}).items():
            if task.status != "released":
                preview.setdefault(task.name, {})[name] = {k: res[k] for k in TABLE_METRICS if res.get(k) is not None}
                continue
            cell = by_leaf.setdefault(task.leaf, {}).setdefault(name, {k: [] for k in TABLE_METRICS})
            for key in TABLE_METRICS:
                if res.get(key) is not None:
                    cell[key].append(res[key])
    per_leaf = {leaf: {n: {k: agg(k, v) for k, v in cell.items() if v} for n, cell in d.items()} for leaf, d in by_leaf.items()}
    overall = {}
    for n in names:
        leaves = [d[n] for d in per_leaf.values() if "accuracy" in d.get(n, {})]  # unsupported leaves have no scores
        if leaves:
            overall[n] = {k: agg(k, [leaf[k] for leaf in leaves if k in leaf]) for k in TABLE_METRICS if any(k in leaf for leaf in leaves)}
            overall[n]["leaves"] = len(leaves)
            overall[n]["unsupported_leaves"] = sum(1 for d in per_leaf.values() if n in d and "accuracy" not in d[n])
    return {"models": names, "per_leaf": per_leaf, "overall": overall, "preview": preview}


def _cell(value: dict | None, key: str = "accuracy") -> str:
    if value is None:
        return "-"
    return f"{value[key]:.3f}" if key in value else "unsupported"


def leaderboard_markdown(board: dict[str, Any], title: str) -> str:
    names = board["models"]
    head = lambda first: [f"| {first} | " + " | ".join(f"`{n}`" for n in names) + " |", "|---" * (len(names) + 1) + "|"]
    lines = [f"# {title}", "", "Accuracy from cached results. The main table covers released tasks only: a leaf is the mean "
             "of its released tasks, the overall score the mean over leaves. Preview tasks may still change and are "
             "listed apart. `-` = not run.", "", "## Released", "", *head("Leaf")]
    for leaf in sorted(board["per_leaf"]):
        lines.append(f"| {leaf} | " + " | ".join(_cell(board["per_leaf"][leaf].get(n)) for n in names) + " |")
    lines.append("| **Overall** | " + " | ".join(_cell(board["overall"].get(n)) for n in names) + " |")
    lines += ["", "## Preview (not in the overall score)", "", *head("Task")]
    for task in sorted(board["preview"]):
        lines.append(f"| {task} | " + " | ".join(_cell(board["preview"][task].get(n)) for n in names) + " |")
    return "\n".join(lines) + "\n"


def release(version: str, tasks: list, envs: list, check_problems: list[str], dry_run: bool = False) -> int:
    """Check the release; then write docs/releases/<version>.json, move the Unreleased changelog entries under the
    version, commit and tag. Returns 0 on success, 1 when blocked."""
    problems = list(check_problems)
    tags = release_tags()
    if not TAG.match(version):
        problems.append(f"{version!r} is not vMAJOR.MINOR.PATCH")
    elif tags and TAG.match(version).groups() and tuple(map(int, TAG.match(version).groups())) <= tuple(map(int, TAG.match(tags[-1]).groups())):
        problems.append(f"{version} is not newer than the last release {tags[-1]}")
    try:
        dirty = [line for line in _git("status", "--porcelain").splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        dirty = ["not a git repository"]
    if dirty:
        problems.append(f"working tree not clean ({len(dirty)} paths, e.g. {dirty[0].strip()}); commit first")
    released = [t for t in tasks if t.status == "released"]
    for problem in problems:
        print(f"BLOCKED {problem}")
    print(f"{version}: {len(released)} released and {len(tasks) - len(released)} preview tasks: "
          f"{'ready' if not problems else f'{len(problems)} problem(s)'}"
          + (" (dry run, nothing written)" if dry_run else ""))
    if problems or dry_run:
        return 1 if problems else 0
    rows = {}
    for domain in sorted({t.name.split("/")[0] for t in tasks}):
        path = dataio.domain_dir(domain) / "manifest.json"
        rows.update({r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {})
    date = dt.date.today().isoformat()
    snapshot = {"version": version, "date": date, "code_commit": _git("rev-parse", "HEAD").strip(),
                "tasks": [{k: rows.get(t.name, {}).get(k) for k in ("name", "leaf", "status", "version", "items", "path", "sha256",
                                                                     "redistributable", "modalities")} for t in tasks],
                "envs": [{"name": e.name, "version": getattr(e, "version", 1), "episodes": e.episodes,
                          "unsupported": getattr(e, "unsupported", None), "blocked": getattr(e, "blocked", None)} for e in envs]}
    out = env.ROOT / "docs" / "releases" / f"{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    changelog = env.ROOT / "CHANGELOG.md"
    if changelog.exists():
        text = changelog.read_text(encoding="utf-8")
        changelog.write_text(text.replace("## Unreleased\n", f"## Unreleased\n\n## {version} ({date})\n", 1), encoding="utf-8")
    paths = [str(p.relative_to(env.ROOT)) for p in (out, changelog) if p.exists()]
    _git("add", "--", *paths)
    _git("commit", "-q", "-m", f"Release {version}")
    _git("tag", "-a", version, "-m", f"S1Arena-MM {version}: {len(released)} released and {len(tasks) - len(released)} preview tasks")
    print(f"tagged {version}; wrote {', '.join(paths)}")
    print("Next: git push origin main --tags, then publish the items and media of this tag to the Hugging Face dataset.")
    return 0
