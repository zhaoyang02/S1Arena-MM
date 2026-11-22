"""Random distributions. Only for testing the pipeline; never reported on the leaderboard.

"random" takes every modality; "random@modalities=image+video" takes only those, to exercise the unsupported path.
"""

from __future__ import annotations

import hashlib
import random

from ..registry import register_model
from ..schema import MEDIA_KINDS, Answer, DecisionRequest, ModelResponse
from .base import ModelAdapter, split_options


@register_model
class RandomModel(ModelAdapter):
    kind = "random"
    native_probs = True
    probability_source = "random"

    def __init__(self, arg: str = ""):
        _, options = split_options(arg)
        self.options = options
        kinds = options.get("modalities")
        self.modalities = frozenset(kinds.split("+") if kinds else MEDIA_KINDS)
        unknown = self.modalities - set(MEDIA_KINDS)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}")

    @property
    def model_id(self) -> str:
        return "test/random" + (f"@modalities={'+'.join(sorted(self.modalities))}" if "modalities" in self.options else "")

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        seed = hashlib.sha256((request.model_dump_json() + str(sample)).encode()).hexdigest()
        rng = random.Random(seed)
        answers = {}
        for name, question in request.questions.items():
            weights = [rng.random() for _ in question.option_ids()]
            total = sum(weights)
            answers[name] = Answer(probs={o: w / total for o, w in zip(question.option_ids(), weights)})
        return ModelResponse(answers=answers, latency_ms=0.0, cost_usd=0.0)
