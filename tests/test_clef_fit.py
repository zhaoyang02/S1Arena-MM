"""CPU checks of clef-omni's frame fitting (no weights): a 1 fps grid, thinned evenly until the record fits."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from conftest import make_video
from jevarena_mm import media
from jevarena_mm.registry import make_model
from jevarena_mm.schema import DecisionRequest, Question

pytest.importorskip("av")


class Values:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def softmax(self, dim):
        exps = [math.exp(v) for v in self.values]
        return Values([e / sum(exps) for e in exps])

    def tolist(self):
        return self.values


class FakeRelease:
    """encode_record: 100 tokens of text, 1 per 160 audio samples, `pair` tokens per pair of video frames."""
    MAX_VIDEO_FRAME_PIXELS = 256 * 32 * 32

    def __init__(self, pair):
        self.pair, self.records = pair, []

    def encode_record(self, tokenizer, record, max_length, processor):
        self.records.append(record)
        length = 100 + sum(len(a) // 160 for a in record.get("audio", []))
        length += sum(math.ceil(len(v) / 2) * self.pair for v in record.get("videos", []))
        ids = [(q["type"] == "noul" and ("true", "false")) or tuple(q["criteria"]) for q in record["questions"].values()]
        questions = [SimpleNamespace(question_id=name, option_ids=o) for name, o in zip(record["questions"], ids)]
        return SimpleNamespace(input_ids=[0] * length, questions=questions)

    def collate_records(self, records, pad, device):
        return records


def omni(workspace, pair):
    model = make_model("clef:Cloudflare/clef-omni")
    model.jsm, model.max_length = FakeRelease(pair), 2000
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    model.input_device = "cpu"
    model.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(OutOfMemoryError=MemoryError))
    model.model = lambda batch: [[Values([0.0, 1.0])]]
    model.load = lambda: None
    return model


def test_the_id_names_the_policy():
    assert make_model("clef:Cloudflare/clef-omni").model_id == "Cloudflare/clef-omni@fps=1,fit=64000"
    assert make_model("clef:Cloudflare/clef-omni@fps=2").model_id == "Cloudflare/clef-omni@fps=2,fit=64000"
    assert make_model("clef").model_id == "Cloudflare/clef-flash@video-default"  # unchanged


def test_frames_are_a_1fps_grid_thinned_to_fit(workspace):
    make_video(workspace / "media" / "toy" / "long.mkv", seconds=60, fps=10, size=32)
    video = media.make_ref("video", "toy/long.mkv")
    question = {"q": Question(type="boolean", instructions="Is it so?")}

    model = omni(workspace, pair=10)  # 30 pairs at 1 fps: 400 tokens, fits 2000
    response = model.decide(DecisionRequest(state={"video": video}, questions=question))
    view = response.usage["input"][0]
    assert response.error is None and view["grid"] == 60 and view["frames"] == 60
    assert view["timestamps_s"][:3] == [0.0, 1.0, 2.0]
    assert "videos" not in model.jsm.records[0]  # the first encode measures everything else

    model = omni(workspace, pair=100)  # all 60 frames would be 3,100 tokens: thinned, evenly
    response = model.decide(DecisionRequest(state={"video": video}, questions=question))
    view = response.usage["input"][0]
    assert response.error is None and view["grid"] == 60 and 2 <= view["frames"] < 60
    assert 100 + math.ceil(view["frames"] / 2) * 100 <= 2000
    stamps = view["timestamps_s"]
    assert stamps[-1] > 50  # spread over the window, not its start

    model = omni(workspace, pair=5000)  # not even two frames fit
    response = model.decide(DecisionRequest(state={"video": video}, questions=question))
    assert response.error.startswith("rejected:")
