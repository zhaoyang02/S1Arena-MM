import pytest

from conftest import make_png
from jevarena_mm import cli, dataio, env, hub, media
from jevarena_mm.registry import register_task
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question
from jevarena_mm.tasks.base import TaskAdapter

pytest.importorskip("PIL")


class Toy(TaskAdapter):
    name = "img.basic/toyhub/frozen"
    leaf = "img.basic/toyhub"
    source = "Toy pictures"
    license = "CC0"
    tier = "C"
    media_root = "toy"
    frozen = True


@pytest.fixture
def toy(workspace, clean_registry, monkeypatch):
    """A frozen task with one hosted file (toy/a/0.png) and one that is not (toy/b/1.png)."""
    register_task(Toy)
    monkeypatch.setattr(hub, "NOT_HOSTED", {"toy/b/*": "toy terms"})
    monkeypatch.setattr(hub, "FETCHED_FROM", {"toy/b/*": "the toy owner"})
    monkeypatch.setattr(hub, "NOT_HOSTED_FOLDERS", set())
    items = []
    for n, sub in enumerate(("a", "b")):
        make_png(workspace / "media" / "toy" / sub / f"{n}.png", value=40 * n)
        items.append(Item(leaf=Toy.leaf, task=Toy.name,
                          request=DecisionRequest(state={"image": media.make_ref("image", f"toy/{sub}/{n}.png")},
                                                  questions={"q": Question(type="boolean", instructions="Bright?")}),
                          gold={"q": Gold(label="false")}, meta={"source": Toy.source, "license": Toy.license}))
    dataio.write_items(dataio.task_path(Toy.name), items)
    return workspace


def test_not_hosted_patterns():
    assert hub.not_hosted_key("vsi-bench/videos/scannet_scene0001_00.mp4") == "vsi-bench/*scannet_*"
    assert hub.not_hosted_key("vsi-bench/acquired-v2/scannetpp_ab12.mp4") == "vsi-bench/*scannetpp_*"
    assert hub.not_hosted_key("vsi-bench/videos/arkitscenes_1.mp4") is None
    assert hub.not_hosted_key("charades/videos/3IRHH.mp4") == "charades/*"
    assert hub.not_hosted_key("mvbench/star/Charades_v1_480/X.mp4") == "mvbench/star/*"
    assert hub.not_hosted_key("mvbench/clevrer/video_validation/v.mp4") is None
    assert {"charades", "ego4d", "videomme"} <= hub.NOT_HOSTED_FOLDERS and "mvbench" not in hub.NOT_HOSTED_FOLDERS
    assert hub.partly_hosted("mvbench") and hub.partly_hosted("vsi-bench")
    assert not hub.partly_hosted("charades") and not hub.partly_hosted("ami")


def test_fetch_downloads_hosted_folders_then_runs_the_script_for_the_rest(toy, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "fetch", lambda folders, dest: calls.append((folders, dest)))
    script = toy / "repo" / "scripts" / "fetch" / "toy.py"
    script.parent.mkdir(parents=True)
    script.write_text("import pathlib, sys\npathlib.Path(sys.argv[1], 'ran').write_text('ok')\n")
    monkeypatch.setattr(env, "ROOT", toy / "repo")
    cli.main(["fetch", "--leaf", "img.basic/toyhub"])
    assert calls == [(["toy"], env.media_dir())]
    assert (env.media_dir() / "toy" / "ran").read_text() == "ok"


def test_card_lists_hosted_and_missing_media(toy):
    text = hub.card([Toy()], {"toy": (1, 2_000_000_000)})
    assert text.startswith("---\n") and "extra_gated_fields:" in text
    assert "| `toy` | Image | Toy pictures | CC0 | 1 | 2.00 GB |" in text
    assert "| `toy/b/*` | Toy pictures | toy terms | the toy owner |" in text
    assert set(hub.references([Toy()])) == {"toy/a/0.png", "toy/b/1.png"}


def test_tasks_not_hosted_in_a_shared_folder(toy, monkeypatch):
    """Files named by content hash share a folder; a task's files can be not hosted while the folder is."""
    monkeypatch.setattr(hub, "NOT_HOSTED", {})
    monkeypatch.setattr(hub, "NOT_HOSTED_TASKS", {Toy.name: "toy task terms"})
    monkeypatch.setattr(hub, "FETCHED_FROM_TASKS", {Toy.name: "the toy owner"})
    hub.task_files.cache_clear()
    try:
        assert hub.not_hosted_key("toy/a/0.png") == Toy.name and hub.not_hosted_key("other/x.png") is None
        assert hub.partly_hosted("toy")
        text = hub.card([Toy()], {})
        assert f"| `{Toy.name}` (its files in `toy`) | Toy pictures | toy task terms | the toy owner |" in text
    finally:
        hub.task_files.cache_clear()
