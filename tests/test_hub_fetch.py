"""hub.fetch: files on Xet come in one download group per folder, the others one by one, present files are kept."""

import sys
import types
from types import SimpleNamespace

import pytest

from jevarena_mm import hub

FILES = {"imgs/a.jpg": (5, "x1"), "imgs/b.jpg": (7, "x2"), "imgs/c.JPG": (3, None),  # c is kept in git, not on Xet
         "vids/d.mp4": (11, "x4"), "other/e.jpg": (2, "x5")}


@pytest.fixture
def hub_fakes(monkeypatch):
    calls = {"group": [], "one": [], "snapshot": []}
    fail_after = {"n": None}

    class Group:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def start_download_file(self, info, path):
            if fail_after["n"] is not None and len(calls["group"]) >= fail_after["n"]:
                raise OSError("connection reset")
            calls["group"].append(info.hash)
            with open(path, "wb") as f:
                f.write(b"\0" * info.size)

    xet = types.ModuleType("hf_xet")
    xet.XetFileInfo = lambda h, size: SimpleNamespace(hash=h, size=size)
    xet.XetSession = lambda: SimpleNamespace(new_file_download_group=lambda **kw: Group())
    monkeypatch.setitem(sys.modules, "hf_xet", xet)

    class Api:
        def dataset_info(self, repo, revision=None):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_tree(self, repo, repo_type, revision, recursive):
            return [SimpleNamespace(path=p, size=s, xet_hash=h) for p, (s, h) in FILES.items()]

    def one(repo, path, repo_type, revision, local_dir):
        calls["one"].append(path)
        (local_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (local_dir / path).write_bytes(b"\0" * FILES[path][0])

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", one)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: calls["snapshot"].append(k))
    return calls, fail_after


def test_xet_files_in_groups_git_files_one_by_one_present_files_kept(hub_fakes, tmp_path):
    calls, _ = hub_fakes
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "a.jpg").write_bytes(b"\0" * 5)  # already there, same size
    hub.fetch(["imgs", "vids"], tmp_path)
    assert calls["group"] == ["x2", "x4"] and calls["one"] == ["imgs/c.JPG"] and not calls["snapshot"]
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == {
        "imgs/a.jpg", "imgs/b.jpg", "imgs/c.JPG", "vids/d.mp4"}  # other/ was not asked for, no .incomplete left


def test_a_stopped_group_keeps_what_it_delivered(hub_fakes, tmp_path):
    calls, fail_after = hub_fakes
    fail_after["n"] = 1  # the imgs group gets a.jpg, then the connection drops before b.jpg and d.mp4
    hub.fetch(["imgs", "vids"], tmp_path)
    assert calls["group"] == ["x1"] and sorted(calls["one"]) == ["imgs/b.jpg", "imgs/c.JPG", "vids/d.mp4"]
    assert (tmp_path / "vids" / "d.mp4").stat().st_size == 11 and not list(tmp_path.rglob("*.incomplete"))


def test_without_xet_groups_snapshot_download_fetches_all(hub_fakes, tmp_path, monkeypatch):
    calls, _ = hub_fakes
    monkeypatch.setitem(sys.modules, "hf_xet", types.ModuleType("hf_xet"))  # an hf_xet without XetSession
    hub.fetch(["imgs"], tmp_path)
    assert calls["snapshot"] and calls["snapshot"][0]["allow_patterns"] == ["imgs/*"] and not calls["group"]


def test_latency_packets_are_the_files_the_sets_name():
    files = hub.latency_files()  # data/latency/v1.json: 100 closed-loop packets and their 234 screenshots
    assert len(files) == len(set(files)) == 334 and all(f.startswith("latency/v1/loop/") for f in files)
    assert sum(f.endswith("/packet.json") for f in files) == 100


def test_mmau_is_hosted_but_its_iemocap_and_voxceleb_clips():
    """The fetch script gets only MMAU's IEMOCAP and VoxCeleb clips (80 files of the reasoning task)."""
    mmau = [n for n in hub.task_files().values() if "mmau" in n]
    assert len(mmau) == 80 and set(mmau) == {"aud.know/reasoning/mmau_mini"}
