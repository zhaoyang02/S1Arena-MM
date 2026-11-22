import copy
import json

import pytest

from conftest import make_wav
from jevarena_mm import audio_delivery, cli, dataio, env, media, recipes
from jevarena_mm.registry import tasks_under
from jevarena_mm.schema import content_id
from jevarena_mm.tasks.aud.prepared import catalog, task_class


@pytest.fixture
def delivery(workspace, monkeypatch):
    root = workspace / "delivery"
    spec = {**next(s for s in catalog() if s["task_id"] == "A21"), "redistributable": False}  # the recipe path
    path = make_wav(root / "media" / "clip.wav")
    ident = "audio.emotion.cremad:original"
    sha = audio_delivery.digest(path)
    row = {"id": ident, "request": {"state": {"audio": {
        "path": "media/clip.wav", "sha256": sha, "sample_rate": 8000, "channels": 1, "duration_s": 2}},
        "questions": {"emotion": {"type": "choice", "instructions": "Which emotion?",
                                  "criteria": {"neutral": "Neutral", "angry": "Angry"}}}}}
    target = {"id": ident, "gold": {"emotion": {"label": "neutral", "dist": {"neutral": 0.75, "angry": 0.25}}}}
    provenance = {"id": ident, "delivery_track": "default", "restrictions": [], "original_split": "test",
                  "transcript": "PRIVATE PROVENANCE", "speaker": "actor"}
    group = {"id": ident, "connected_group": "connected-recording", "source_group": "recording",
             "exact_audio_sha256": sha}
    folder = root / "default" / "A21"
    folder.mkdir(parents=True)
    for kind, value in (("inputs", row), ("targets", target), ("provenance", provenance)):
        (folder / f"{kind}.jsonl").write_text(json.dumps(value) + "\n")
    (root / "split_groups.jsonl").write_text(json.dumps(group) + "\n")
    (root / "manifest.json").write_text('{"version":"test"}\n')
    metadata = workspace / "metadata"
    metadata.mkdir()
    monkeypatch.setattr(audio_delivery, "DIRECTORY", metadata)
    monkeypatch.setattr(audio_delivery, "catalog", lambda: [spec])
    monkeypatch.setattr(env, "ROOT", workspace)
    return root, spec, row, target, provenance, group


def test_export_rebuild_and_full_runner_flow(delivery, clean_registry, monkeypatch, capsys):
    root, spec, row, target, provenance, group = delivery
    exported = audio_delivery.export_delivery(root)
    task = task_class(spec)()
    monkeypatch.setitem(clean_registry.TASKS, task.name, type(task))
    for key, value in exported["sources"].items():
        monkeypatch.setitem(recipes.SOURCES, key, recipes.Source(**value))
    audio_delivery.attach_delivery(root)
    recipe = recipes.path(task).read_text()
    assert "PRIVATE PROVENANCE" not in recipe and "Which emotion?" not in recipe
    assert '"neutral": 0.75' not in recipe  # source questions, labels and votes stay outside Git
    assert not dataio.task_path(task.name, False).exists()
    assert recipes.media_of(task)[0].kind == "audio"
    cli.main(["build", "--leaf", task.name])
    items = dataio.load_task(task)
    assert len(items) == 1
    item = items[0]
    assert item.id == content_id(item) and item.id != row["id"]
    assert item.meta["factory_id"] == row["id"] and item.meta["group"] == group["connected_group"]
    assert item.gold["emotion"].dist == target["gold"]["emotion"]["dist"]
    assert item.meta["audio_provenance"] == provenance
    assert item.request.modalities() == {"audio"}
    assert media.check(item.request.media()[0][1]) is None
    cli.main(["verify", "--leaf", task.name])
    cli.main(["run", "--model", "random", "--leaf", task.name])
    cli.main(["run", "--model", "random@modalities=image", "--leaf", task.name])
    cli.main(["score", "--models", "random", "random@modalities=image", "--leaf", task.name])
    assert "unsupported (1 items)" in capsys.readouterr().out
    assert cli.consistency_problems(task.name)[0] == []
    assert "| audio | 1 |" in cli.cards_text(task.leaf)
    # A pristine checkout can pass check from recipes without the restricted items or delivery.
    dataio.task_path(task.name, False).unlink()
    (env.media_dir() / "audio_delivery").unlink()
    assert cli.consistency_problems(task.name)[0] == []
    rebuilt, missing = recipes.rebuild(task)
    assert not rebuilt and missing


def test_reproducible_export_and_no_annotation_mutation(delivery):
    root, spec, *_ = delivery
    exported = audio_delivery.export_delivery(root)
    first = recipes.path(task_class(spec)()).read_bytes()
    assert audio_delivery.export_delivery(root) == exported
    assert recipes.path(task_class(spec)()).read_bytes() == first
    dest = audio_delivery.attach_delivery(root)
    assert audio_delivery.attach_delivery(root) == dest
    (root / "default/A21/targets.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audio_delivery.attach_delivery(root)


@pytest.mark.parametrize("change", ["restricted", "conditional", "group", "path", "ids"])
def test_rejects_unadmitted_or_invalid_records(delivery, change):
    _, spec, row, target, provenance, group = copy.deepcopy(delivery)
    if change == "restricted":
        provenance["restrictions"] = ["quarantined"]
    elif change == "conditional":
        provenance["delivery_track"] = "conditional"
    elif change == "group":
        group["exact_audio_sha256"] = "0" * 64
    elif change == "path":
        row["request"]["state"]["audio"]["path"] = "../secret.wav"
    else:
        target["id"] = "other"
    with pytest.raises(ValueError):
        audio_delivery.convert(spec, row, target, provenance, group)


def test_changed_question_or_media_invalidates_content_id(delivery):
    _, spec, row, target, provenance, group = delivery
    original = audio_delivery.convert(spec, row, target, provenance, group)
    changed = copy.deepcopy(row)
    changed["request"]["questions"]["emotion"]["instructions"] += " Listen carefully."
    assert audio_delivery.convert(spec, changed, target, provenance, group).id != original.id
    changed = copy.deepcopy(row)
    changed["request"]["state"]["audio"]["sha256"] = "0" * 64
    assert audio_delivery.convert(spec, changed, target, provenance, {**group, "exact_audio_sha256": "0" * 64}).id != original.id


def test_no_empty_or_conditional_sources_registered():
    tasks = tasks_under("aud")
    assert len(tasks) == 21 and len({t.leaf for t in tasks}) == 19
    assert all(t.status == "released" and t.frozen for t in tasks)
    assert not any(t.task_id in ("A17", "A20") or t.source_task.endswith("mmau_pro") for t in tasks)
    # committed items for tasks whose annotations may be redistributed, recipes for the others, never both
    for t in tasks:
        assert dataio.task_path(t.name).exists() == t.redistributable != recipes.path(t).exists(), t.name


def test_redistributable_task_is_exported_as_items(delivery, clean_registry):
    root, spec, row, *_ = delivery
    spec["redistributable"] = True
    audio_delivery.export_delivery(root)
    task = task_class(spec)()
    assert not recipes.path(task).exists()
    items = dataio.load_task(task)
    assert len(items) == 1 and items[0].meta["factory_id"] == row["id"] and items[0].id == content_id(items[0])


def test_duplicate_ids_rejected(delivery):
    root, *_ = delivery
    path = root / "default/A21/inputs.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate id"):
        audio_delivery.export_delivery(root)


def test_unknown_recipe_source_is_reported(delivery, monkeypatch):
    root, spec, *_ = delivery
    audio_delivery.export_delivery(root)
    task = task_class(spec)()
    monkeypatch.delitem(recipes.SOURCES, "audio_a21_inputs", raising=False)
    with pytest.raises(KeyError):
        recipes.rebuild(task)  # a pin is required; never silently substitute data
