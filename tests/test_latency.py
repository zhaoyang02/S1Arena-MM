"""Latency sets and the batch-size-1 measurement (jevarena_mm/latency.py), on toy tasks with the random model."""

import hashlib
import json

import pytest

from conftest import make_png, make_wav
from jevarena_mm import cli, dataio, env, latency, media, runner
from jevarena_mm.registry import make_model, register_task
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question
from jevarena_mm.tasks.base import TaskAdapter

pytest.importorskip("PIL")


class ToyA(TaskAdapter):
    name, leaf, status = "vid.basic/toy/a", "vid.basic/toy", "released"


class ToyB(TaskAdapter):
    name, leaf, status = "vid.basic/toy/b", "vid.basic/toy", "released"


class Meeting(TaskAdapter):
    name, leaf, status = "lvid.dialog/toy/meet", "lvid.dialog/toy", "released"


class Draft(TaskAdapter):
    name, leaf, status = "vid.know/toy/draft", "vid.know/toy", "preview"


def write_items(task, n: int, group_size: int = 1, audio: bool = False) -> list[Item]:
    """n items, each referencing its own picture (and a sound track); every group_size items share one group."""
    items = []
    for i in range(n):
        stem = f"toy/{task.name.rsplit('/', 1)[1]}-{i}"
        make_png(env.media_dir() / f"{stem}.png", value=10 + i)
        state = {"image": media.make_ref("image", f"{stem}.png", probe=False)}
        if audio:
            make_wav(env.media_dir() / f"{stem}.wav", seconds=1.0 + i)
            state["audio"] = media.make_ref("audio", f"{stem}.wav")
        items.append(Item(leaf=task.leaf, task=task.name,
                          request=DecisionRequest(state=state, questions={"q": Question(type="boolean", instructions="Is it so?")}),
                          gold={"q": Gold(label="true")}, meta={"group": f"{task.name}:{i // group_size}"}))
    dataio.write_items(dataio.task_path(task.name), items)
    return items


@pytest.fixture
def toy(workspace, clean_registry):
    clean_registry.TASKS.clear()
    for task in (ToyA, ToyB, Meeting, Draft):
        register_task(task)
    write_items(ToyA, 8, group_size=2)  # 4 groups
    write_items(ToyB, 6)
    write_items(Meeting, 6, audio=True)
    write_items(Draft, 4)
    return workspace


def test_a_set_takes_sources_in_turn_one_item_per_group(toy):
    s = latency.build_set([ToyA(), ToyB(), Meeting(), Draft()], per_leaf=5, av_per_leaf=3, warmup=2, warmup_audio=1)
    short = [e for e in s["items"] if e["leaf"] == "vid.basic/toy"]
    assert [e["task"] for e in short] == ["vid.basic/toy/a", "vid.basic/toy/b"] * 2 + ["vid.basic/toy/a"]
    assert {e["kind"] for e in short} == {"short"}
    assert [e["kind"] for e in s["items"] if e["leaf"] == "lvid.dialog/toy"] == ["av"] * 3  # av_per_leaf
    assert not any(e["leaf"] == "vid.know/toy" for e in s["items"])  # preview tasks stay out
    groups = [e["group"] for e in s["items"] + s["warmup"]]
    assert len(groups) == len(set(groups))  # warm-up items are apart, no medium twice
    assert [w["kind"] for w in s["warmup"]] == ["short", "short", "av"]
    assert s == latency.build_set([ToyA(), ToyB(), Meeting(), Draft()], per_leaf=5, av_per_leaf=3, warmup=2,
                                  warmup_audio=1) | {"created": s["created"]}  # seeded


def test_measure_times_each_request_once_and_checks_the_accuracy_run(toy):
    s = latency.build_set([ToyA(), ToyB(), Meeting()], per_leaf=4, av_per_leaf=2, warmup=1, warmup_audio=0)
    model = make_model("random@modalities=image")  # no audio: the meeting items are unsupported, not timed
    for task in (ToyA, ToyB):
        runner.run_task(model, task.name, dataio.load_task(task()))  # the "accuracy run" to compare with
    rows, facts = latency.measure(model, s, repeat=2, log=lambda *_: None)
    first = [r for r in rows if not r.get("repeat")]
    assert len(first) == len(s["items"]) and sum(bool(r.get("repeat")) for r in rows) == 2
    assert {r["status"] for r in first if r["kind"] == "av"} == {"unsupported"}
    assert all(r["latency_ms"] >= 0 and r["questions"] == 1 and r["output_tokens"] == 0 for r in first if r["kind"] == "short")
    summary = latency.summarize(model, "random@modalities=image", s, rows, facts, gpus=2)
    short = summary["per_kind"]["short"]
    assert short["n"] == 4 and short["gpu_s_per_1k"] == pytest.approx(short["mean_ms"] * 2, abs=0.11)
    assert short["output_tokens"] == summary["output_tokens"] == {"mean": 0.0, "max": 0}
    assert summary["per_kind"]["av"] == {"requests": 2, "failed": 0, "unsupported": 2}
    assert summary["consistency"]["top_option_agreement"] == 1.0
    assert summary["consistency"]["verdict"] == "same configuration"
    assert summary["median_ms"] is None and summary["repeat"]["n"] == 2  # no closed-loop packets in this set


def test_output_tokens_are_what_the_adapter_wrote():
    lm, head, other = (type("M", (), {"kind": kind})() for kind in ("lm", "clef", "new"))
    assert latency.output_tokens(lm, {"think_tokens": 0, "candidate_mass": {}}) == 0  # think=0: read at the prompt
    assert latency.output_tokens(lm, {"think_tokens": 412}) == 412
    assert latency.output_tokens(head, {"input": {}}) == 0  # a decision head writes nothing
    assert latency.output_tokens(other, {"output_tokens": 3}) == 3 and latency.output_tokens(other, None) is None
    assert latency._tokens([{"status": "ok", "output_tokens": 0}, {"status": "ok", "output_tokens": None}]) is None


def test_fit_test_takes_the_longest_requests_of_each_kind(toy):
    s = latency.build_set([ToyA(), Meeting()], per_leaf=4, av_per_leaf=6, warmup=0, warmup_audio=0)
    longest = latency.fit_entries(s["items"], 2)
    assert [e["media_s"] for e in longest if e["kind"] == "av"] == [6.0, 5.0]  # the 6 s and 5 s sound tracks
    rows, facts = latency.measure(make_model("random"), s, fit=2, log=lambda *_: None)
    assert len(rows) == 4 and not any(r.get("repeat") for r in rows)  # 2 per kind (short, av), once each
    assert latency.fit_report(make_model("random"), rows, facts)["fits"]


def test_loop_packets_are_copied_with_their_screenshots(workspace):
    shot = make_png(workspace / "attempts" / "shot.png")
    sha = hashlib.sha256(shot.read_bytes()).hexdigest()
    for seed in range(3):
        folder = workspace / "attempts" / "m" / "loop.game__gameworld__01_2048" / f"seed-{seed:03d}" / "attempt-1" / "decisions"
        folder.mkdir(parents=True)
        packet = {"protocol": "jevarena-mm-decision-v1", "state": "Board <image:1>", "question": "Which move?",
                  "candidates": [{"id": "a000", "text": "left"}, {"id": "a001", "text": "right"}],
                  "images": [{"path": str(shot), "sha256": sha}]}
        (folder / "00000.json").write_text(json.dumps({"request": packet, "reply": {"status": "ok", "selected": "a000"}}))
    entries = latency.sample_loop(workspace / "attempts", env.media_dir() / "latency" / "v1" / "loop", {"gameworld": 2})
    assert len(entries) == 2 and {e["env"] for e in entries} == {"loop.game/gameworld/01_2048"}
    shot.unlink()  # the copies stand alone
    request = latency.Requests().request(entries[0])
    assert request.modalities() == {"image"} and set(request.questions["action"].criteria) == {"a000", "a001"}
    s = {"version": "v1", "items": [], "warmup": [], "loop": entries}
    rows, _ = latency.measure(make_model("random"), s, repeat=0, log=lambda *_: None)
    assert latency.summarize(make_model("random"), "random", s, rows, {"machine": {}}, 1)["median_ms"] is not None


def test_gameworld_packets_take_the_screenshots_the_attempt_kept(workspace):
    gone = workspace / "tmp" / "frame_000001.png"  # GameWorld's temporary screenshot, deleted after the episode
    attempt = workspace / "attempts" / "m" / "loop.game__gameworld__01_2048" / "seed-000" / "attempt-1"
    kept = make_png(attempt / "native" / "agent_0" / "artifacts" / "screenshots" / "step_000001.png")
    (attempt / "decisions").mkdir(parents=True)
    packet = {"protocol": "jevarena-mm-decision-v1", "state": "Board <image:1>", "question": "Which move?",
              "candidates": [{"id": "a000", "text": "left"}, {"id": "a001", "text": "right"}],
              "images": [{"path": str(gone), "sha256": hashlib.sha256(kept.read_bytes()).hexdigest()}]}
    (attempt / "decisions" / "00000.json").write_text(json.dumps({"request": packet, "reply": {"status": "ok"}}))
    entries = latency.sample_loop(workspace / "attempts", env.media_dir() / "latency" / "v1" / "loop", {"gameworld": 1})
    assert len(entries) == 1 and latency.Requests().request(entries[0]).modalities() == {"image"}


def test_the_kind_of_gpu_comes_from_its_name():
    names = [("NVIDIA H100 PCIe", "h100"), ("NVIDIA A100-PCIE-40GB", "a100"), ("NVIDIA A100-SXM4-80GB", "a100"),
             ("NVIDIA GH200 480GB", "gh200"), ("NVIDIA L40S", "l40s")]
    for name, kind in names:
        assert latency.gpu_label([{"name": name}]) == kind
    assert latency.gpu_label([]) == "cpu"


def test_cli_draws_stages_and_measures(toy, tmp_path):
    cli.main(["latency-set", "--leaf", "vid", "--per-leaf", "3", "--warmup", "1", "--out", str(tmp_path / "set.json")])
    s = json.loads((tmp_path / "set.json").read_text())
    assert len(s["items"]) == 3
    cli.main(["latency-stage", "--set", str(tmp_path / "set.json"), "--to", str(tmp_path / "local")])
    staged = sorted(str(p.relative_to(tmp_path / "local")) for p in (tmp_path / "local").rglob("*") if p.is_file())
    assert staged == latency.media_files(s) and len(staged) == 4  # the 3 items and the warm-up item, nothing else
    cli.main(["latency", "--model", "random", "--set", str(tmp_path / "set.json"), "--repeat", "1", "--gpus", "1"])
    summary = json.loads(latency.summary_path(make_model("random").slug, latency.gpu_label()).read_text())
    assert summary["per_kind"]["short"]["n"] == 3 and summary["set_digest"] == latency.set_digest(s)
    cli.main(["latency", "--model", "random", "--set", str(tmp_path / "set.json"), "--repeat", "0", "--kinds", "short"])
    path = latency.summary_path(make_model("random").slug, latency.gpu_label())
    assert json.loads(path.read_text()) == summary  # a run of some kinds is written apart, the measurement stays
    assert path.with_name(path.stem + ".short.json").exists()


def test_the_accuracy_rows_a_set_needs_go_to_a_runs_folder(toy, tmp_path, monkeypatch):
    cli.main(["latency-set", "--leaf", "vid", "--per-leaf", "3", "--warmup", "1", "--out", str(tmp_path / "set.json")])
    s = json.loads((tmp_path / "set.json").read_text())
    cli.main(["run", "--model", "random", "--leaf", "vid"])  # the accuracy run: every item of the toy tasks
    cli.main(["latency-runs", "--set", str(tmp_path / "set.json"), "--models", "random", "--out", str(tmp_path / "runs")])
    slug = make_model("random").slug
    copied = {(f.stem.replace("__", "/"), r["item_id"])  # a runs folder: <slug>/<task>.jsonl
              for f in (tmp_path / "runs" / slug).glob("*.jsonl") for r in map(json.loads, f.open())}
    assert copied == {(e["task"], e["item_id"]) for e in s["items"]}  # the set's items only, not the warm-up or the rest
    monkeypatch.setenv("JEVARENA_MM_RUNS", str(tmp_path / "runs"))  # a machine without the caches reads them there
    assert {r["item_id"] for r in runner.latest_rows(slug, s["items"][0]["task"])} >= {s["items"][0]["item_id"]}


def test_a_plan_names_each_models_software(tmp_path, capsys):
    plan = tmp_path / "plan.toml"
    plan.write_text('set = "s.json"\ngpu_label = "h100-sxm"\n[[models]]\nspec = "lm:Qwen/Qwen3.5-4B"\nenv = "vllm"\n'
                    '[[models]]\nspec = "raya"\n')
    cli.main(["latency-plan", str(plan), "-p", "gpu-h100"])
    lm, raya = capsys.readouterr().out.splitlines()
    assert "JV_ENV=vllm " in lm and "JV_ENV" not in raya and "latency.slurm" in raya
    assert "JV_GPU_LABEL=h100-sxm" in lm and "JV_GPU_LABEL=h100-sxm" in raya  # a plan's label for every job
