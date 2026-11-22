import hashlib
import json

import pytest

from jevarena_mm import env, metrics, runner
from jevarena_mm.models.base import ModelAdapter
from jevarena_mm.schema import Answer, DecisionRequest, Gold, Item, ModelResponse, Question


def make_item(kind: str, n: int) -> Item:
    """An item with one media file, written under the workspace's media root (the runner skips missing media)."""
    uri = f"toy/{kind}_{n}.{'png' if kind == 'image' else 'mkv'}"
    path = env.media_dir() / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{kind} {n}".encode())
    ref = {"kind": kind, "uri": uri, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return Item(
        leaf="vid.basic/temporal",
        task="vid.basic/temporal/toy",
        request=DecisionRequest(state={"media": ref, "n": str(n)},
                                questions={"q": Question(type="boolean", instructions="Is it so?")}),
        gold={"q": Gold(label="true")},
    )


class Fake(ModelAdapter):
    kind = "fake"
    native_probs = True

    def __init__(self, modalities):
        self.modalities = frozenset(modalities)
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "test/fake"

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        self.calls += 1
        return ModelResponse(answers={"q": Answer(probs={"true": 0.9, "false": 0.1})})


def test_unsupported_items_are_marked_not_called_and_left_out(workspace):
    items = [make_item("image", 0), make_item("image", 1), make_item("video", 2)]
    image_only = Fake({"image"})
    assert runner.run_task(image_only, "vid.basic/temporal/toy", items) == 2
    assert image_only.calls == 2
    rows = runner.latest_rows(image_only.slug, "vid.basic/temporal/toy")
    assert sorted(r["response"]["status"] for r in rows) == ["ok", "ok", "unsupported"]

    # a rerun makes no calls and adds no duplicate marks
    assert runner.run_task(image_only, "vid.basic/temporal/toy", items) == 0
    assert len(runner.read_runs(image_only.slug, "vid.basic/temporal/toy")) == 3

    result = metrics.score(items, rows)
    assert result["n_questions"] == 2 and result["accuracy"] == 1.0
    assert result["coverage"] == 2 / 3 and result["unsupported_items"] == 1
    assert result["error_rate"] == 0.0

    # once the model takes video, only the formerly unsupported item is called
    both = Fake({"image", "video"})
    assert runner.run_task(both, "vid.basic/temporal/toy", items) == 1
    result = metrics.score(items, runner.latest_rows(both.slug, "vid.basic/temporal/toy"))
    assert result["n_questions"] == 3 and result["coverage"] == 1.0


def test_text_only_models_take_text_items(workspace):
    text_item = Item(
        leaf="vid.basic/temporal", task="vid.basic/temporal/toy",
        request=DecisionRequest(state={"text": "no media"}, questions={"q": Question(type="boolean", instructions="?")}),
        gold={"q": Gold(label="false")},
    )
    model = Fake(set())
    assert runner.run_task(model, "vid.basic/temporal/toy", [text_item, make_item("audio", 3)]) == 1
    result = metrics.score([text_item], runner.latest_rows(model.slug, "vid.basic/temporal/toy"))
    assert result["accuracy"] == 0.0 and result["coverage"] == 1.0


def test_failures_still_score_zero(workspace):
    items = [make_item("image", 0)]

    class Broken(Fake):
        def decide(self, request, sample=0):
            raise RuntimeError("model-side error")

    model = Broken({"image"})
    runner.run_task(model, "vid.basic/temporal/toy", items)
    result = metrics.score(items, runner.latest_rows(model.slug, "vid.basic/temporal/toy"))
    assert result["accuracy"] == 0.0 and result["unanswered"] == 1 and result["coverage"] == 1.0


def test_rejections_are_final_unless_a_run_retries_them(workspace):
    items = [make_item("image", 0)]

    class TooLong(Fake):
        def decide(self, request, sample=0):
            self.calls += 1
            return ModelResponse(error="rejected: prompt longer than the context")

    model = TooLong({"image"})
    runner.run_task(model, "vid.basic/temporal/toy", items)
    assert runner.run_task(model, "vid.basic/temporal/toy", items) == 0  # final: not called again

    fixed = Fake({"image"})  # the same model id, after an adapter change that makes the item fit
    assert runner.run_task(fixed, "vid.basic/temporal/toy", items) == 0
    assert runner.run_task(fixed, "vid.basic/temporal/toy", items, retry_rejected=True) == 1
    result = metrics.score(items, runner.latest_rows(fixed.slug, "vid.basic/temporal/toy"))
    assert result["accuracy"] == 1.0  # the answer outranks the earlier rejection
    assert runner.run_task(fixed, "vid.basic/temporal/toy", items, retry_rejected=True) == 0


TASK = "vid.basic/temporal/toy"


def shard_of(items: list[Item], index: int, count: int) -> list[Item]:
    return [item for n, item in enumerate(sorted(items, key=lambda i: i.id)) if n % count == index]


class BatchFake(Fake):
    def decide_many(self, requests):
        return [self.decide(request, sample) for request, sample in requests]


@pytest.mark.parametrize("cls", [Fake, BatchFake])
def test_shards_write_their_own_files_and_readers_merge_them(workspace, cls):
    items = [make_item("video", n) for n in range(5)] + [make_item("audio", 5)]
    model = cls({"video"})
    for index in range(3):
        runner.run_task(model, TASK, shard_of(items, index, 3), shard=(index, 3))
    assert model.calls == 5
    assert [p.name for p in runner.run_files(model.slug, TASK)] == [
        f"vid.basic__temporal__toy.shard-{i}of3.jsonl" for i in range(3)]
    rows = runner.latest_rows(model.slug, TASK)
    assert sorted(r["item_id"] for r in rows) == sorted(i.id for i in items)
    assert sum(runner.is_unsupported(r) for r in rows) == 1

    # resuming in one process, or with another shard count, finds every call cached in the shard files
    assert runner.run_task(model, TASK, items) == 0
    assert sum(runner.run_task(model, TASK, shard_of(items, i, 2), shard=(i, 2)) for i in range(2)) == 0
    assert model.calls == 5 and len(runner.read_runs(model.slug, TASK)) == 6
    assert metrics.score(items, rows)["coverage"] == 5 / 6


def test_single_process_runs_keep_one_file(workspace):
    items = [make_item("video", n) for n in range(3)]
    model = Fake({"video"})
    runner.run_task(model, TASK, items[:2])
    runner.run_task(model, TASK, items, shard=(0, 1))  # 0/1 is the whole task: same file as no shard
    assert [p.name for p in runner.run_files(model.slug, TASK)] == ["vid.basic__temporal__toy.jsonl"]
    assert model.calls == 3


def test_a_row_cut_off_by_a_killed_job_is_skipped_and_called_again(workspace, capsys):
    items = [make_item("video", n) for n in range(2)]
    model = Fake({"video"})
    runner.run_task(model, TASK, items[:1], shard=(0, 2))
    path = runner.run_path(model.slug, TASK, (0, 2))
    with path.open("ab") as f:  # the job was killed while writing item 1's row
        f.write(f'{{"item_id": "{items[1].id}", "sample": 0, "model": "test/fa'.encode())
    assert len(runner.read_runs(model.slug, TASK)) == 1
    assert "skipped an incomplete row" in capsys.readouterr().err
    assert runner.run_task(model, TASK, items, shard=(0, 2)) == 1
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[2])["item_id"] == items[1].id  # the new row starts on its own line
    assert {r["item_id"] for r in runner.latest_rows(model.slug, TASK)} == {i.id for i in items}


def test_runs_dir_follows_the_environment(workspace, monkeypatch):
    other = workspace / "check_runs"
    monkeypatch.setenv("JEVARENA_MM_RUNS", str(other))
    model = Fake({"video"})
    runner.run_task(model, TASK, [make_item("video", 0)])
    assert runner.run_files(model.slug, TASK) == [other / "test__fake" / "vid.basic__temporal__toy.jsonl"]
    assert not (workspace / "results" / "runs").exists()


def test_a_lost_gpu_ends_the_run_instead_of_failing_every_call(workspace):
    from jevarena_mm import runner

    assert runner.gpu_lost(RuntimeError("CUDA error: unspecified launch failure\nSearch for `cudaErrorLaunchFailure`"))
    assert not runner.gpu_lost(RuntimeError("CUDA out of memory. Tried to allocate 3.28 GiB"))  # the input's own failure
    assert not runner.gpu_lost(ValueError("no frames in the window"))

    class Lost(Fake):
        def decide(self, request, sample=0):
            import time

            time.sleep(0.02)  # a model call takes time: the calls not yet started can still be cancelled
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("CUDA error: unspecified launch failure")
            return ModelResponse(answers={"q": Answer(probs={"true": 0.9, "false": 0.1})})

    model = Lost({"video"})
    items = [make_item("video", n) for n in range(40)]
    with pytest.raises(SystemExit, match="GPU context lost"):
        runner.run_task(model, "vid.basic/temporal/toy", items, workers=1)
    rows = runner.read_runs(model.slug, "vid.basic/temporal/toy")
    assert len(rows) == 2 and model.calls < 10  # the answers made are kept; the rest is not called, nor recorded


def test_items_whose_media_are_missing_are_not_run_until_fetched(workspace, capsys):
    """Not fetched (or no access to the source): not called, no row, so neither scored nor a failure."""
    items = [make_item("image", 0), make_item("image", 1)]
    absent = env.media_dir() / items[1].request.state["media"].uri
    data = absent.read_bytes()
    absent.unlink()
    model = Fake({"image"})
    assert runner.run_task(model, "vid.basic/temporal/toy", items) == 1 and model.calls == 1
    assert "1 items not run, their media missing or changed" in capsys.readouterr().out
    assert [r["item_id"] for r in runner.read_runs(model.slug, "vid.basic/temporal/toy")] == [items[0].id]
    absent.write_bytes(data)  # fetched
    assert runner.run_task(model, "vid.basic/temporal/toy", items) == 1 and model.calls == 2
