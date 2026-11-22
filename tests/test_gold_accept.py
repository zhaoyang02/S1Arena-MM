"""gold.accept: alternatives counted as correct, merged with the label for the Brier score, absent from items without it."""

import json

import pytest

from jevarena_mm.metrics import score
from jevarena_mm.schema import Item


def make(accept=None):
    gold = {"label": "a"} | ({"accept": accept} if accept is not None else {})
    return Item.model_validate({
        "leaf": "img.struct/document_qa", "task": "img.struct/document_qa/demo",
        "request": {"state": {"text": "Which answer?"},
                    "questions": {"q": {"type": "choice", "instructions": "Pick one.",
                                        "criteria": {"a": "A", "b": "B", "c": "C"}}}},
        "gold": {"q": gold}})


def row(item, probs):
    return {"item_id": item.id, "sample": 0, "response": {"answers": {"q": {"probs": probs}}}}


def test_unset_accept_keeps_item_bytes():
    item = make()
    assert "accept" not in json.loads(item.model_dump_json())["gold"]["q"]
    assert item.id == make().id


def test_accept_changes_the_id_and_is_validated():
    assert make(["b"]).id != make().id
    assert make(["a", "b"]).gold["q"].correct() == {"a", "b"}  # the image export repeats the label in accept
    with pytest.raises(ValueError):
        make([])
    with pytest.raises(ValueError):
        make(["z"])  # not an option


def test_accepted_alternative_scores_as_correct():
    item = make(["b"])
    result = score([item], [row(item, {"a": 0.1, "b": 0.7, "c": 0.2})])
    assert result["accuracy"] == 1.0
    # label and accepted alternative are one outcome: (0.8 - 1)^2 + (0.2 - 0)^2
    assert result["brier"] == pytest.approx(0.08)


def test_without_accept_the_alternative_is_wrong():
    item = make()
    result = score([item], [row(item, {"a": 0.1, "b": 0.7, "c": 0.2})])
    assert result["accuracy"] == 0.0
    assert result["brier"] == pytest.approx((0.1 - 1) ** 2 + 0.7 ** 2 + 0.2 ** 2)
