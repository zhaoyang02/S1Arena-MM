"""Prompt layout of the lm: and raya adapters (no model needed)."""

import pytest

from jevarena_mm.models import local_lm
from jevarena_mm.models.local_lm import LocalLM
from jevarena_mm.models.raya import PLACEHOLDERS, Raya
from jevarena_mm.schema import DecisionRequest, MediaRef, Question

QUESTIONS = {"action": Question(type="choice", instructions="Which?", criteria={"a": "left", "b": "right"})}


def image(n: int) -> MediaRef:
    return MediaRef(kind="image", uri=f"toy/{n}.png", sha256="c" * 64)


class EchoTokenizer:
    chat_template = "set"

    def apply_chat_template(self, messages, **kwargs):
        return messages


def test_lm_places_images_where_the_text_marks_them():
    packet = DecisionRequest(state={"text": "Before <image:1> middle <image:2> end", "images": [image(1), image(2)]},
                             questions=QUESTIONS)
    assert local_lm.state_text(packet) == "Input:\nBefore <image:1> middle <image:2> end"
    model = object.__new__(LocalLM)
    model._harmony = model._thinking = False
    model._tok = EchoTokenizer()
    content = model._prompt("system", local_lm.state_text(packet) + "\n\nWhich?", False, ["image", "image"])[1]["content"]
    assert content == [{"type": "text", "text": "Input:\nBefore "}, {"type": "image"}, {"type": "text", "text": " middle "},
                       {"type": "image"}, {"type": "text", "text": " end\n\nWhich?"}]

    # without markers (every offline task) the images come first and the state stays JSON
    plain = DecisionRequest(state={"images": [image(1)]}, questions=QUESTIONS)
    assert local_lm.placed_text(plain) is None
    assert local_lm.state_text(plain).startswith('Images, in the order shown: "<image 1>"\n\nInput:\n{')
    assert model._prompt("system", "text", False, ["image"])[1]["content"] == [{"type": "image"}, {"type": "text", "text": "text"}]
    # markers that do not place every image once, in order, are text
    shuffled = DecisionRequest(state={"text": "<image:2> then <image:1>", "images": [image(1), image(2)]}, questions=QUESTIONS)
    assert local_lm.placed_text(shuffled) is None


def test_raya_places_image_placeholders_where_the_text_marks_them():
    packet = DecisionRequest(state={"text": "Before <image:1> middle <image:2> end", "images": [image(1), image(2)]},
                             questions=QUESTIONS)
    tag = PLACEHOLDERS["image"]
    assert Raya.lead(packet, tag + tag) == f"Before {tag} middle {tag} end\n\n"
    plain = DecisionRequest(state={"text": "a state", "images": [image(1)]}, questions=QUESTIONS)
    assert Raya.lead(plain, tag) == f"{tag}\n\na state\n\n"


def test_lm_model_id_names_a_set_context():
    model = object.__new__(LocalLM)
    model.hf_id, model.mode, model.think, model.norm, model.effort = "Qwen/Qwen3-Omni-30B-A3B-Thinking", "label", 0, "sum", None
    model.quant = model.dequant = None
    model.modalities, model.video, model.max_frames = frozenset({"image", "video", "audio"}), "native", 0
    model.maxlen = None  # the checkpoint's own context: ids of runs made without maxlen= stay as they were
    assert model.model_id == "Qwen/Qwen3-Omni-30B-A3B-Thinking@label-think0-video"
    model.maxlen = 65536  # sets the Omni video budget and which requests fit
    assert model.model_id == "Qwen/Qwen3-Omni-30B-A3B-Thinking@label-think0-ctx65536-video"


def test_lm_pplayers_needs_one_count_per_stage():
    with pytest.raises(ValueError, match="pplayers"):
        LocalLM("Qwen/Qwen3-Omni-30B-A3B-Thinking@pp=3,pplayers=12+36")


def test_a_null_audio_config_is_no_audio_tower():
    from jevarena_mm.models.local_lm import media_kinds

    vision = {"model_type": "gemma4", "vision_config": {"model_type": "gemma4_vision"}}
    assert media_kinds(vision | {"audio_config": None}) == {"image", "video"}  # gemma-4-26B-A4B-it
    assert media_kinds(vision | {"audio_config": {"model_type": "gemma4_audio"}}) == {"image", "video", "audio"}
