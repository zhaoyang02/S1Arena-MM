"""Task status, the version rule against the last release tag, and the release command (in a scratch git repository)."""

import json
import subprocess

import pytest

from conftest import make_png
from jevarena_mm import cli, dataio, env, media, release
from jevarena_mm.registry import make_model, register_task
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question
from jevarena_mm.tasks.base import TaskAdapter

pytest.importorskip("PIL")


class Stable(TaskAdapter):
    name, leaf, status = "img.basic/toy/stable", "img.basic/toy", "released"
    source, license, tier, media_root, frozen = "Toy pictures", "CC0", "C", "toy", True


class Draft(TaskAdapter):
    name, leaf, status = "img.basic/draft/new", "img.basic/draft", "preview"
    source, license, tier, media_root, frozen = "Toy pictures", "CC0", "C", "toy", True


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=env.ROOT, check=True, capture_output=True, text=True).stdout


def write_items(task, bright: int = 2) -> None:
    items = []
    for n in range(3):
        make_png(env.ROOT / "media" / "toy" / f"{n}.png", value=40 * n)
        items.append(Item(leaf=task.leaf, task=task.name,
                          request=DecisionRequest(state={"image": media.make_ref("image", f"toy/{n}.png")},
                                                  questions={"bright": Question(type="boolean", instructions="Is it bright?")}),
                          gold={"bright": Gold(label="true" if n == bright else "false")}))
    dataio.write_items(dataio.task_path(task.name), items)


@pytest.fixture
def repo(workspace, clean_registry, monkeypatch):
    """A git repository holding two toy tasks (one released, one preview), built and committed."""
    clean_registry.TASKS.clear()
    clean_registry.ENVS.clear()
    for task in (Stable, Draft):
        register_task(task)
    monkeypatch.setattr(env, "ROOT", workspace)
    monkeypatch.setattr(Stable, "version", 1)
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (workspace / ".gitignore").write_text("media/\nresults/runs/\n.cache/\n")
    (workspace / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n- First tasks.\n")
    write_items(Stable)
    write_items(Draft)
    cli.main(["build", "--leaf", "img"])
    git("add", "-A")
    git("commit", "-q", "-m", "tasks")
    return workspace


def problems() -> list[str]:
    return cli.consistency_problems("")[0]


def test_released_items_change_only_with_a_new_version(repo):
    assert problems() == [] and release.release_tags() == []
    git("tag", "-a", "v0.1.0", "-m", "first")

    write_items(Stable, bright=1)  # the released task's items change ...
    write_items(Draft, bright=1)  # ... and so do the preview task's, which is allowed
    cli.main(["build", "--leaf", "img"])
    found = problems()
    assert len(found) == 1 and "img.basic/toy/stable: items changed since v0.1.0" in found[0] and "version = 2" in found[0]

    Stable.version = 2
    cli.main(["build", "--leaf", "img"])
    assert problems() == []
    Stable.version = 0
    assert any("lower than 1" in p for p in problems())


def test_a_released_task_cannot_disappear_silently(repo, clean_registry):
    git("tag", "-a", "v0.1.0", "-m", "first")
    del clean_registry.TASKS[Stable.name]  # renamed or removed after the release
    cli.main(["build", "--leaf", "img"])  # drops its manifest entry
    assert any("img.basic/toy/stable: released in v0.1.0 but no longer registered" in p for p in problems())
    assert cli.consistency_problems("img.basic/draft")[0] == []  # outside the checked leaf
    (env.DATA / "retired.json").write_text(json.dumps({Stable.name: "merged into another task"}))
    assert problems() == []


def test_status_must_be_known(repo, monkeypatch):
    monkeypatch.setattr(Draft, "status", "beta")
    assert any("status 'beta'" in p for p in problems())


def test_leaderboard_reports_preview_apart(repo):
    cli.main(["run", "--model", "random", "--leaf", "img"])
    board = release.leaderboard([make_model("random")], [Stable(), Draft()])
    assert list(board["per_leaf"]) == ["img.basic/toy"] and list(board["preview"]) == ["img.basic/draft/new"]
    assert board["overall"]["test/random"]["leaves"] == 1


def test_release_gates_and_tag(repo, capsys):
    (repo / "notes.txt").write_text("uncommitted\n")
    with pytest.raises(SystemExit):
        cli.main(["release", "v0.1.0"])
    assert "working tree not clean" in capsys.readouterr().out
    (repo / "notes.txt").unlink()

    cli.main(["release", "v0.1.0", "--dry-run"])
    assert "ready (dry run, nothing written)" in capsys.readouterr().out and release.release_tags() == []

    cli.main(["release", "v0.1.0"])
    assert release.release_tags() == ["v0.1.0"] and git("status", "--porcelain") == ""
    snapshot = json.loads((repo / "docs" / "releases" / "v0.1.0.json").read_text())
    assert {t["name"]: t["status"] for t in snapshot["tasks"]} == {Stable.name: "released", Draft.name: "preview"}
    assert "## Unreleased\n\n## v0.1.0 (" in (repo / "CHANGELOG.md").read_text()

    with pytest.raises(SystemExit):
        cli.main(["release", "v0.1.0", "--dry-run"])
    assert "not newer than the last release v0.1.0" in capsys.readouterr().out
