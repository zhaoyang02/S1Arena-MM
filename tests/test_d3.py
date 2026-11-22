"""d3 adapter: questions as d3 types them and its answers as distributions over option ids (no model needed)."""

import pytest

from jevarena_mm.models.d3 import d3_probs, d3_question, split_map
from jevarena_mm.schema import Question


def test_d3_types_questions_and_reads_its_answers():
    choice = Question(type="choice", instructions="Which?", criteria={"a": "left", "b": "right"})
    assert d3_question(choice) == {"type": "choice", "instructions": "Which?", "criteria": {"a": "left", "b": "right"}}
    assert d3_probs(choice, {"type": "choice", "choice": "b", "probabilities": {"a": 0.25, "b": 0.75}}) == {"a": 0.25, "b": 0.75}

    boolean = Question(type="boolean", instructions="Is it?")
    assert d3_question(boolean) == {"type": "noul", "instructions": "Is it?"}  # d3's noul answer is P(true)
    assert d3_probs(boolean, {"type": "noul", "noul": 0.8}) == pytest.approx({"true": 0.8, "false": 0.2})

    score = Question(type="score", instructions="How much?", criteria=["low", "mid", "high"])
    assert d3_question(score) == {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}
    answer = {"type": "score", "score": 1.0, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}}
    assert d3_probs(score, answer) == {"0": 0.2, "1": 0.6, "2": 0.2}


def test_d3_split_map_keeps_inputs_and_readout_side_on_gpu0():
    device_map = split_map(64, 2)
    layers = [device_map[f"language_model.layers.{i}"] for i in range(64)]
    assert layers == [0] * 30 + [1] * 34  # in order; GPU 0 also holds the embeddings and the vision tower
    assert {device_map[k] for k in ("visual", "language_model.embed_tokens", "language_model.rotary_emb", "language_model.norm")} == {0}
