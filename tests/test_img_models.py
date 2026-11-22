"""The image adapters (models/qev.py) and the image scoring (metrics_img.py): specs, what is sent, how replies and
failures are scored."""

import re
import sys
import types

import pytest

from jevarena_mm import metrics_img
from jevarena_mm.models import qev
from jevarena_mm.registry import make_model
from jevarena_mm.schema import DecisionRequest, Item


def request(**state):
    return DecisionRequest.model_validate({
        "state": state,
        "questions": {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "A", "b": "B", "c": "C"}},
                      "ok": {"type": "boolean", "instructions": "Is it a bird?"}}})


def image_ref(name):
    return {"kind": "image", "uri": f"toy/{name}.png", "sha256": name * 64, "info": {"content": "photo"}}


def test_specs_are_kept_as_model_ids():
    for spec in ("qev_onejev@model=OneJev-27B,img=auto,debias=1", "qev@model=Qwen3.5-9B,img=auto,debias=1",
                 "qev_readout@model=Qwen2.5-Omni-7B,img=auto,debias=1", "s1_d3@model=d3,img=auto,debias=1"):
        model = make_model(spec)
        assert model.model_id == spec
        assert model.modalities == frozenset({"image"})
    with pytest.raises(ValueError):
        make_model("qev:Qwen3.5-9B")


# Specs with recorded image runs, stored under the spec as written: each row carries it, and its folder is the spec
# with every character outside [A-Za-z0-9@=,._-] replaced by "_". The adapters keep it as the model id, so the
# recorded runs are the cached runs of these specs.
RECORDED = ("qev_readout@model=Qwen3-VL-30B-A3B-Instruct,img=auto,debias=1",
            "qev_dgemma@model=diffusiongemma-26B-A4B-it-AWQ-INT4,img=auto,debias=1",
            "qev_dgemma@model=diffusiongemma-26B-A4B-it,img=auto,debias=1")


def test_recorded_specs_keep_their_model_ids():
    for spec in RECORDED:
        assert re.sub(r"[^A-Za-z0-9@=,._-]", "_", spec) == spec  # the folder of its recorded runs
        model = make_model(spec)
        assert model.model_id == spec
        assert model.modalities == frozenset({"image"})


def test_state_placeholders_follow_presentation_order():
    req = request(text="Compare <image 2> with the first.", image_1=image_ref("a"), image_2=image_ref("b"),
                  gallery=[image_ref("c"), image_ref("d")])
    state, refs = qev.to_qev(qev.as_request(req))
    # image_2 is named in the text, so it is placed there and its key dropped; image_1 keeps its key's position
    assert state == {"text": "Compare <image:1> with the first.", "image_1": "<image:2>", "gallery": ["<image:3>", "<image:4>"]}
    assert [r["uri"] for r in refs] == ["toy/b.png", "toy/a.png", "toy/c.png", "toy/d.png"]


def test_image_policy():
    assert qev.long_edge("auto", 1, "PNG") == 2560
    assert qev.long_edge("auto", 1, "JPEG") == 1536
    assert qev.long_edge("auto", 1, "PNG", "photo") == 1536
    assert qev.long_edge("auto", 3, "JPEG") == 1024
    assert qev.long_edge("auto", 9, "JPEG") == 512
    assert qev.long_edge("800", 1, "PNG") == 800


def test_decide_reads_the_reply_into_a_model_response(monkeypatch):
    sent = {}

    class Reply:
        model = "OneJev-27B"
        usage = {"prompt_tokens": 10}
        answers = {"q": types.SimpleNamespace(raw={"type": "choice", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}),
                   "ok": types.SimpleNamespace(raw={"type": "noul", "noul": 0.9})}

    class Client:
        def __init__(self, url, timeout):
            sent["url"] = url

        def system_one(self, **kwargs):
            sent.update(kwargs)
            return Reply()

    fake = types.ModuleType("qev.client")
    fake.Client, fake.Choice, fake.Noul, fake.Score = Client, *(lambda *a: a,) * 3
    monkeypatch.setitem(sys.modules, "qev", types.ModuleType("qev"))
    monkeypatch.setitem(sys.modules, "qev.client", fake)
    model = make_model("qev_onejev@model=OneJev-27B,img=auto,debias=1,url=http://node:8200")
    response = model.decide(request(text="no image"))
    assert sent["url"] == "http://node:8200" and sent["debias"] == 1 and sent["media"] == []
    assert response.status == "ok"
    assert response.answers["q"].probs == {"a": 0.2, "b": 0.7, "c": 0.1}
    assert response.answers["ok"].probs == pytest.approx({"true": 0.9, "false": 0.1})
    assert response.usage["input"] == {"images": [], "policy": "img=auto"}

    Reply.model = "OneJev-9B"
    assert make_model("qev_onejev@model=OneJev-27B").decide(request(text="x")).status == "failed"


def item(task, leaf, n_questions=1):
    questions = {f"q{i}": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}}
                 for i in range(n_questions)}
    return Item.model_validate({"leaf": leaf, "task": task, "request": {"state": {"text": task}, "questions": questions},
                                "gold": {name: {"label": "a"} for name in questions},
                                "meta": {"n": n_questions}})


def row(it, answers, status="ok", sample=0):
    return {"item_id": it.id, "sample": sample, "response": {"status": status, "error": None if status == "ok" else "x",
                                                            "answers": {q: {"probs": p} for q, p in answers.items()}}}


def test_failures_score_an_all_zero_distribution():
    it = item("img.basic/x/t", "img.basic/x")
    m = metrics_img.score_items([metrics_img.as_item(it)], metrics_img.runs_of([row(it, {}, status="failed")]))
    assert m["avg@1"] == 0.0 and m["brier"] == 1.0 and m["coverage"] == 1.0
    m = metrics_img.score_items([metrics_img.as_item(it)], {})
    assert m["covered"] == 0 and m["avg@1"] is None  # not run: neither covered nor failed


def test_items_weigh_the_same_and_leaves_pool_their_tasks():
    one = item("img.basic/x/t1", "img.basic/x")             # 1 question, answered right
    two = item("img.basic/x/t2", "img.basic/x", n_questions=2)  # 2 questions, one right
    other = item("img.std/y/t3", "img.std/y")                 # a standard benchmark: not in the overall score
    rows = [row(one, {"q0": {"a": 1.0}}), row(two, {"q0": {"a": 1.0}, "q1": {"b": 1.0}}), row(other, {"q0": {"b": 1.0}})]
    out = metrics_img.score_model([metrics_img.as_item(i) for i in (one, two, other)], metrics_img.runs_of(rows))
    assert out["leaf"]["img.basic/x"]["avg@1"] == pytest.approx((1 + 0.5) / 2)  # per item; per question it would be 2/3
    assert out["overall"]["avg@1"] == pytest.approx(0.75) and out["overall"]["leaves"] == 1
    assert out["domain"]["img.std"]["avg@1"] == 0.0


def test_replies_are_renormalized_and_accept_merges():
    it = Item.model_validate({"leaf": "img.struct/d", "task": "img.struct/d/t",
                              "request": {"state": {"text": "?"}, "questions": {"q": {
                                  "type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": "B", "c": "C"}}}},
                              "gold": {"q": {"label": "a", "accept": ["a", "b"]}}})
    rows = [row(it, {"q": {"a": 0.1, "b": 0.7, "c": 0.2}})]
    m = metrics_img.score_items([metrics_img.as_item(it)], metrics_img.runs_of(rows))
    assert m["avg@1"] == 1.0 and m["brier"] == pytest.approx(0.08)
    rows = [row(it, {"q": {"a": 0.05, "b": 0.35, "c": 0.1}})]  # sums to 0.5: scored as 0.1 / 0.7 / 0.2
    assert metrics_img.score_items([metrics_img.as_item(it)], metrics_img.runs_of(rows))["brier"] == pytest.approx(0.08)
