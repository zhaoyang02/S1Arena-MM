"""Raya (yuyu199741/raya-decision-v1): a System-One decision model on Qwen3.5-2B that reads text, images and video and
reads the option labels at the last position (masked projection), run as its model card runs it.

As the card does it:
- No chat template. The prompt is the media placeholders (<|vision_start|><|image_pad|><|vision_end|>, ... video_pad
  ...), a blank line, the text, the question, the options as "A. text" lines and "Answer with exactly one label from the
  options above."
- The logits at the last position of the labels' tokens (" A", " B", ...), softmaxed over the labels, are the answer.
  Booleans are the options "A. Yes" / "B. No", score levels are options in level order. Past 26 options, two-letter
  labels the tokenizer keeps whole (" AA", ...); a question whose labels are not all single tokens is rejected.
- Images and video go through the processor of Qwen/Qwen3.5-2B (the checkpoint ships none). Video is read as the card
  reads a clip, num_frames=8 with cap_pixels_per_frame=True: 8 frames at arange(0, frames, frames / 8) of the window
  (transformers' sample_frames), decoded here (so marks are drawn) and passed with their metadata (do_sample_frames
  False), so the processor sizes them and writes their timestamps as it would for the clip.
The text is the request's state text (its string values; media are the placeholders), then the question. A text that
places each of its images with <image:k> (the closed-loop suites' packets, see README) has the image placeholders there.

Spec: raya[@frames=<n>]; needs one CUDA GPU.
"""

from __future__ import annotations

import itertools
import string
import threading
import time

from .. import media
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from . import jev_common
from .base import ModelAdapter, split_options
from .local_lm import source_fps, window

REPO, REVISION = "yuyu199741/raya-decision-v1", "036f2fb7fb775e95246327200a532a24394b07bd"
PROCESSOR, PROCESSOR_REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
CLOSING = "Answer with exactly one label from the options above."
PLACEHOLDERS = {"image": "<|vision_start|><|image_pad|><|vision_end|>", "video": "<|vision_start|><|video_pad|><|vision_end|>"}


@register_model
class Raya(ModelAdapter):
    kind = "raya"
    modalities = frozenset({"image", "video"})
    native_probs = True
    probability_source = "label-token logits at the last position (masked projection)"
    _one_at_a_time = threading.Lock()

    def __init__(self, arg: str = "") -> None:
        repo, options = split_options(arg)
        self.repo = repo or REPO
        self.frames = int(options.get("frames", 8))
        self.model = None

    @property
    def model_id(self) -> str:
        return f"{self.repo}@frames={self.frames}"

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        self.torch = torch
        revision = REVISION if self.repo == REPO else None
        self.model = AutoModelForImageTextToText.from_pretrained(self.repo, revision=revision, dtype=torch.bfloat16).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=revision)
        self.processor = AutoProcessor.from_pretrained(PROCESSOR, revision=PROCESSOR_REVISION)
        pairs = ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))
        self.letters = list(string.ascii_uppercase) + [p for p in pairs if len(self.tokenizer.encode(" " + p, add_special_tokens=False)) == 1]

    def _video(self, ref):
        import numpy as np

        start, end = window(ref)
        rate = source_fps(ref)
        total = max(1, int(round((end - start) * rate)))
        count = min(self.frames, total)
        wanted = [int(i) for i in np.arange(0, total, total / count)][:count]
        frames = media.frames(ref, timestamps=[start + i / rate for i in wanted])
        indices = [max(0, int(round((f.t - start) * rate))) for f in frames]
        metadata = {"fps": rate, "frames_indices": indices, "total_num_frames": total, "duration": end - start}
        return np.stack([f.array for f in frames]), metadata, [round(f.t, 3) for f in frames]

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        with self._one_at_a_time:
            self.load()
        started = time.perf_counter()
        images, videos, metas, used, heads = [], [], [], [], []  # decoded outside the lock, by the runner's threads in parallel
        for _, ref in request.media():
            if ref.kind == "image":
                images.append(media.image(ref))
                heads.append(PLACEHOLDERS["image"])
                used.append({"image": True})
            else:
                array, metadata, stamps = self._video(ref)
                videos.append(array)
                metas.append(metadata)
                heads.append(PLACEHOLDERS["video"])
                used.append({"frames": len(stamps), "timestamps_s": stamps})
        lead = self.lead(request, "".join(heads))
        answers, errors = {}, []
        for name, question in request.questions.items():
            with self._one_at_a_time:  # one GPU model: forward passes take turns
                probs = self._answer(question, lead, images, videos, metas)
            if isinstance(probs, str):
                errors.append(f"{name}: {probs}")
            else:
                answers[name] = Answer(probs=probs)
        latency = (time.perf_counter() - started) * 1000
        error = "rejected: " + "; ".join(errors) if errors else None
        return ModelResponse(answers=answers, latency_ms=round(latency, 1), usage={"input": used}, cost_usd=0.0, error=error)

    @staticmethod
    def lead(request: DecisionRequest, heads: str) -> str:
        """The prompt before the question: the media placeholders, a blank line and the text; a text that places each
        of its images once, in order, with <image:k> (the closed-loop suites' packets) has them there instead."""
        text = jev_common.state_text(request)
        kinds = [ref.kind for _, ref in request.media()]
        if kinds and set(kinds) == {"image"} and jev_common.placed_images(text, len(kinds)):
            pieces = jev_common.split_at_images(text)
            return pieces[0] + "".join(PLACEHOLDERS["image"] + piece for piece in pieces[1:]) + "\n\n"
        return f"{heads}\n\n" + (f"{text}\n\n" if text else "")

    def _answer(self, question: Question, lead: str, images, videos, metas) -> dict[str, float] | str:
        torch = self.torch
        options = jev_common.options(question)  # (option id, text); booleans as Yes / No
        if len(options) > len(self.letters):
            return f"{len(options)} options: more than single-token labels"
        labels = self.letters[:len(options)]
        lines = [f"{label}. {option_text}" for label, (_, option_text) in zip(labels, options)]
        prompt = lead + f"{question.instructions}\n\n" + "\n".join(lines) + f"\n\n{CLOSING}"
        kwargs = {}
        if videos:
            kwargs.update(videos=videos, video_metadata=metas, do_sample_frames=False, cap_pixels_per_frame=True)
        if images:
            kwargs["images"] = images
        inputs = (self.processor(text=[prompt], return_tensors="pt", **kwargs) if kwargs
                  else self.tokenizer(prompt, return_tensors="pt")).to("cuda")
        ids = [self.tokenizer.encode(" " + label, add_special_tokens=False)[0] for label in labels]
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits[ids], dim=-1).tolist()
        return {option_id: p for (option_id, _), p in zip(options, probs)}
