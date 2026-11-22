import json

import pytest

from conftest import make_png
from jevarena_mm import cli, dataio, env, media, recipes
from jevarena_mm.registry import register_task
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question
from jevarena_mm.tasks.base import TaskAdapter

pytest.importorskip("PIL")


class Restricted(TaskAdapter):
    name = "img.basic/toyrecipe/frozen"
    leaf = "img.basic/toyrecipe"
    source = "Toy pictures with licensed labels"
    license = "toy license"
    tier = "C"
    media_root = "toy"
    frozen = True
    redistributable = False


@pytest.fixture
def restricted(workspace, clean_registry, monkeypatch):
    """A task of two items whose gold labels come from an annotation file in the OSCC format, shipped as a recipe."""
    register_task(Restricted)
    labels = {"u1": True, "u2": False}
    annotations = workspace / "media" / "toy" / "annotations.json"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(json.dumps({"clips": [{"unique_id": k, "state_change": v} for k, v in labels.items()]}))
    monkeypatch.setitem(recipes.SOURCES, "ego4d_oscc", recipes.Source("toy/annotations.json", media.sha256_file(annotations), "toy"))
    recipes._table.cache_clear()
    rows, items = [], []
    for n, key in enumerate(labels):
        make_png(workspace / "media" / "toy" / f"{n}.png", value=40 * n)
        item = Item(leaf=Restricted.leaf, task=Restricted.name,
                    request=DecisionRequest(state={"image": media.make_ref("image", f"toy/{n}.png")},
                                            questions={"q": Question(type="boolean", instructions="Changed?")}),
                    gold={"q": Gold(label="true" if labels[key] else "false")})
        data = item.model_dump(mode="json")
        data["id"], data["gold"]["q"]["label"] = "", f"⟦ego4d_oscc:state_change:{key}⟧"
        rows.append({"id": item.id, "item": data})
        items.append(item)
    path = recipes.path(Restricted())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return items


def test_build_rebuilds_items_from_the_recipe(restricted):
    assert {ref.uri for ref in recipes.media_of(Restricted())} == {"toy/0.png", "toy/1.png"}  # fetch can run first
    cli.main(["build", "--leaf", "img.basic/toyrecipe"])
    built = dataio.read_items(dataio.task_path(Restricted.name, redistributable=False))
    assert sorted(i.id for i in built) == sorted(i.id for i in restricted)
    assert not dataio.task_path(Restricted.name).exists()  # nothing written where it would be committed


def test_fetch_wants_every_recipe_file_after_a_partial_build(restricted):
    dataio.write_items(dataio.task_path(Restricted.name, redistributable=False), restricted[:1])  # e.g. built before keys
    assert {ref.uri for ref in recipes.media_of(Restricted())} == {"toy/0.png", "toy/1.png"}


def test_check_names_a_partial_rebuild(restricted):
    cli.main(["build", "--leaf", "img.basic/toyrecipe"])
    dataio.write_items(dataio.task_path(Restricted.name, redistributable=False), restricted[:1])
    problems, _ = cli.consistency_problems(Restricted.name)
    assert problems == [f"{Restricted.name}: 1 of 2 items rebuilt from the recipe; run `jevarena-mm fetch --leaf "
                        f"{Restricted.name}`, then `jevarena-mm build --leaf {Restricted.name}`"]
    assert "| 2 |" in cli.cards_text("img.basic/toyrecipe")  # cards describe it from the manifest, as everywhere


def test_missing_annotations_leave_items_out(restricted, capsys):
    (env.media_dir() / "toy" / "annotations.json").unlink()
    cli.main(["build", "--leaf", "img.basic/toyrecipe"])
    out = capsys.readouterr().out
    assert "2 of 2 items need toy/annotations.json" in out and "jevarena-mm fetch" in out
    assert not dataio.task_path(Restricted.name, redistributable=False).exists()


def test_a_wrong_annotation_is_refused(restricted, monkeypatch):
    annotations = env.media_dir() / "toy" / "annotations.json"
    annotations.write_text(json.dumps({"clips": [{"unique_id": "u1", "state_change": False},
                                                 {"unique_id": "u2", "state_change": False}]}))
    monkeypatch.setitem(recipes.SOURCES, "ego4d_oscc", recipes.Source("toy/annotations.json", media.sha256_file(annotations), "toy"))
    recipes._table.cache_clear()
    with pytest.raises(ValueError, match="differs from the recipe"):
        recipes.rebuild(Restricted())


def test_nlq_quarter_rule():
    assert recipes.quarter([(10.0, 20.0), (12.0, 19.0)], 100.0) == "first_quarter"
    assert recipes.quarter([(75.0, 99.0)], 100.0) == "fourth_quarter"
    with pytest.raises(ValueError):
        recipes.quarter([(20.0, 30.0)], 100.0)
