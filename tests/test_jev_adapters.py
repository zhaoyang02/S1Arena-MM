"""CPU checks of the Jev-family adapters' input building (no weights are loaded)."""

import pytest

from conftest import make_video
from jevarena_mm import media
from jevarena_mm.models import jev_common
from jevarena_mm.models.jev_omni import bin_centres
from jevarena_mm.models.onejev import OneJev
from jevarena_mm.registry import make_model
from jevarena_mm.schema import DecisionRequest, MediaRef, Question

pytest.importorskip("av")


def test_specs_name_their_sampling_policy():
    assert make_model("onejev").model_id == "OmniJev/OneJev-27B@fps=1,min_frames=2"
    assert make_model("jev_omni").model_id == "akhilaaa3/Jev-Omni@fps=1,frames=16"
    assert make_model("onejev@fps=2,gpu_layers=40").model_id == "OmniJev/OneJev-27B@fps=2,min_frames=2"  # placement is not policy


def test_official_bin_centres():
    assert bin_centres(5, 16) == [0, 1, 2, 3, 4]
    picked = bin_centres(100, 16)
    assert len(picked) == 16 and picked[0] == 3 and picked[-1] == 96


def test_sampling_note_matches_the_recorded_prompt():
    ref = MediaRef(kind="video", uri="ssv2/videos/163230.webm", sha256="a" * 64, window=(0.0, 3.666))
    frames = [media.Frame(t=float(i), array=None) for i in range(4)]
    note = jev_common.sampling_note([jev_common.View(ref, frames, 1.0)])
    assert note == ("Media timing: each video is a 1fps material; video index k maps to target source time window_start + k "
                    "seconds. Actual nearest source frames can differ slightly. Video views have independent clocks.\n"
                    "View 1: source window [0.0, 3.666) seconds; real window duration 3.666s; 1fps container duration 4.0s.")


def test_short_windows_get_at_least_two_frames(workspace):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=2, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv", window=(1.0, 1.4))
    request = DecisionRequest(state={"text": "t", "video": ref}, questions={"q": Question(type="boolean", instructions="?")})
    (view,) = jev_common.sample_views(request, fps=1.0, min_frames=2)
    assert round(view.fps, 6) == 5.0 and [round(f.t, 2) for f in view.frames] == [1.0, 1.2]  # 2 / window seconds


def test_onejev_request_payload(workspace):
    make_video(workspace / "media" / "toy" / "a.mkv", seconds=2, fps=10)
    make_video(workspace / "media" / "toy" / "b.mkv", seconds=2, fps=10)
    views = [media.make_ref("video", "toy/a.mkv", window=(0.0, 2.0)), media.make_ref("video", "toy/b.mkv", window=(0.0, 2.0))]
    request = DecisionRequest(state={"text": "Two views.", "videos": views},
                              questions={"answer": Question(type="boolean", instructions="Is it a foul?")})
    model = OneJev()
    payload = model.payload(request, jev_common.sample_views(request, model.fps, model.min_frames), [])
    assert payload["questions"]["q"] == {"type": "noul", "instructions": "Is it a foul?", "criteria": {"true": "Yes", "false": "No"}}
    state = payload["state"]
    assert state.startswith("Two views.\nMedia timing: each video is a 1fps material")
    assert "View 2: source window [0.0, 2.0)" in state
    assert state.endswith("BEGIN VIDEO VIEW 2; independent timeline\n<video:2>\nEND VIDEO VIEW 2\n")
    score = DecisionRequest(state={"text": "x"}, questions={"answer": Question(type="score", instructions="?", criteria=["low", "high"])})
    assert model.payload(score, [], [])["questions"]["q"]["criteria"] == ["low", "high"]


class PromptStub:
    """Stands in for Jev-Omni's official helper: returns what it would format."""

    def _prompt(self, state, question, texts):
        return {"state": state, "texts": texts}


def recorded_entry(monkeypatch, request, **entry):
    monkeypatch.setattr(jev_common, "_RECORDED", {jev_common.request_key(request): entry})


def test_recorded_presentation_is_reproduced(workspace, monkeypatch):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=3, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv", window=(0.0, 2.95))
    request = DecisionRequest(state={"text": "Part one.\nPart two.", "video": ref},
                              questions={"q": Question(type="boolean", instructions="Is it?")})
    omni = make_model("jev_omni")
    omni.helper = PromptStub()
    plain = omni.content(request)[0][-1]["text"]
    assert sorted(plain["texts"]) == ["No", "Yes"] and plain["state"].startswith("Part one.\nPart two.\nMedia timing: each video")

    recorded_entry(monkeypatch, request, note="View 1: recorded note; final permitted observation", note_at=1,
                   order=["false", "true"], omni_timestamps="Official sampled source timestamps (seconds), per independent view: [[0.0]]")
    content, used = omni.content(request)
    built = content[-1]["text"]
    assert built["texts"] == ["No", "Yes"] and [o for o, _ in omni.options(request, request.questions["q"])] == ["false", "true"]
    assert built["state"] == ("Part one.\nView 1: recorded note; final permitted observation\nPart two.\n"
                              "Official sampled source timestamps (seconds), per independent view: [[0.0]]")
    assert used["views"][0]["timestamps_s"] == [0.0, 1.0, 2.0, 2.9]  # the last frame before the window end is kept
    payload = OneJev().payload(request, jev_common.sample_views(request, 1.0, 2), [])
    assert payload["state"].startswith("Part one.\nView 1: recorded note; final permitted observation\nPart two.\nBEGIN VIDEO VIEW 1")


def test_prediction_prefixes_give_jev_omni_two_frames(workspace, monkeypatch):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=2, fps=10)
    request = DecisionRequest(state={"text": "t", "video": media.make_ref("video", "toy/clip.mkv", window=(0.0, 1.0))},
                              questions={"q": Question(type="boolean", instructions="?")})
    omni = make_model("jev_omni")
    omni.helper = PromptStub()
    assert omni.content(request)[1]["views"][0]["timestamps_s"] == [0.0]
    recorded_entry(monkeypatch, request, note="View 1: source history [0.0, 1.0) seconds; prediction cutoff 1.0s is excluded.")
    assert omni.content(request)[1]["views"][0]["timestamps_s"] == [0.0, 0.5]


def test_recorded_presentation_covers_the_exported_items():
    """Every committed item of a task still at version 1 (so with recorded runs) has an entry; the keys follow the
    request's JSON form. Later versions and tasks built later have no recorded runs."""
    from jevarena_mm import dataio
    from jevarena_mm.registry import tasks_under

    jev_common._RECORDED = None  # read the shipped table
    items = [item for task in tasks_under("vid") + tasks_under("lvid")
             if task.redistributable and task.version == 1 and "s1-arena release" in getattr(task, "pipeline", "")
             for item in dataio.load_task(task)]
    missing = [item.id for item in items if not jev_common.recorded(item.request)]
    assert len(items) > 9000 and missing == []  # 9,720 since SoccerNet-MVFoul became a recipe


def test_boolean_order_without_a_record_is_fixed_per_item_and_balanced():
    question = {"q": Question(type="boolean", instructions="?")}
    requests = [DecisionRequest(state={"text": f"item {n}"}, questions=question) for n in range(400)]
    orders = [jev_common.boolean_order(r) for r in requests]
    assert orders == [jev_common.boolean_order(r) for r in requests]  # the same item is always asked the same way
    no_first = sum(order == ["false", "true"] for order in orders)
    assert 160 <= no_first <= 240 and all(sorted(order) == ["false", "true"] for order in orders)


def test_out_of_memory_is_a_final_rejection(monkeypatch):
    from jevarena_mm import runner

    class OutOfMemoryError(RuntimeError):
        pass

    class Torch:
        class cuda:
            @staticmethod
            def empty_cache():
                pass

    model = make_model("jev_omni")
    model.torch = Torch
    monkeypatch.setattr(model, "_decide", lambda request: (_ for _ in ()).throw(OutOfMemoryError("CUDA out of memory")))
    request = DecisionRequest(state={"text": "t"}, questions={"q": Question(type="boolean", instructions="?")})
    monkeypatch.setattr(jev_common, "gpu_to_ourselves", lambda torch: True)
    response = model.decide(request)
    assert response.status == "failed" and response.error.startswith("rejected: out of GPU memory")
    assert runner.is_final({"item_id": "x", "sample": 0, "response": response.model_dump()})
    monkeypatch.setattr(jev_common, "gpu_to_ourselves", lambda torch: False)  # contention: an ordinary failure, retried
    with pytest.raises(OutOfMemoryError):
        model.decide(request)


def test_onejev_sizes_are_pinned_without_the_hub(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", None)  # a Hub call would fail (offline jobs)
    assert make_model("onejev:OmniJev/OneJev-9B").model_id.endswith(",revision=59429e8")
    assert make_model("onejev:OmniJev/OneJev-4B").model_id.endswith(",revision=c88e186")
    assert make_model("onejev:OmniJev/OneJev-0.8B").model_id.endswith(",revision=c3939d8")
    assert make_model("onejev:OmniJev/OneJev-9B@revision=0123456789abcdef").model_id.endswith(",revision=0123456")
    with pytest.raises(ValueError, match="revision"):
        make_model("onejev:OmniJev/OneJev-2B")


def test_model_dir_reads_its_variable_only_when_named(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setenv("ONEJEV_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision: f"hub/{repo}@{revision}")
    assert jev_common.model_dir("ONEJEV_MODEL_DIR", "OmniJev/OneJev-27B", "r") == tmp_path
    assert str(jev_common.model_dir(None, "OmniJev/OneJev-9B", "r")) == "hub/OmniJev/OneJev-9B@r"


def test_xor_compat_driver_is_a_path_prefix(monkeypatch):
    """On a host driver before R580 the image's CUDA 13 compat libraries lead LD_LIBRARY_PATH, as their own entry."""
    from types import SimpleNamespace

    from jevarena_mm.models import xor

    for version, expected in (("570.86.15", xor.COMPAT + ":"), ("580.65.06", "")):
        monkeypatch.setattr(xor.subprocess, "run", lambda *a, v=version, **k: SimpleNamespace(stdout=f"{v}\n"))
        assert xor._compat_libraries() == expected


def test_xor_binds_where_the_snapshot_links_lead(tmp_path):
    """A hub cache whose repo blobs link on into a store at the cache root, and a file kept elsewhere."""
    from jevarena_mm.models import xor

    cache = tmp_path / "hub"; snap = cache / "models--x--y" / "snapshots" / "abc"; snap.mkdir(parents=True)
    (cache / "blobs" / "ce").mkdir(parents=True); (cache / "blobs" / "ce" / "w").write_text("w")
    (cache / "models--x--y" / "blobs").mkdir(); (cache / "models--x--y" / "blobs" / "h").symlink_to("../../blobs/ce/w")
    (snap / "model.safetensors").symlink_to("../../blobs/h")
    elsewhere = tmp_path / "other"; elsewhere.mkdir(); (elsewhere / "t.json").write_text("{}")
    (snap / "tokenizer.json").symlink_to(elsewhere / "t.json")
    assert xor._bound_folders(snap) == [str(cache), str(elsewhere.resolve())]


def test_xor_sends_one_request_at_a_time_and_stops_on_a_dead_server():
    from types import SimpleNamespace

    import pytest

    model = make_model("xor")
    assert model.slots._value == 1 and make_model("xor@concurrency=4").slots._value == 4
    assert make_model("xor@concurrency=4").model_id == model.model_id  # serving, not policy
    model.processes = [SimpleNamespace(poll=lambda: None, returncode=None)]
    model._raise_if_dead()  # alive: nothing
    model.processes.append(SimpleNamespace(poll=lambda: -9, returncode=-9))
    with pytest.raises(SystemExit, match="exited with -9"):
        model._raise_if_dead()
