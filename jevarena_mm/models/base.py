"""Base class for model adapters.

To add a model family, subclass ModelAdapter in a new module under
jevarena_mm/models/, set `kind` and `modalities`, implement decide(), and decorate the class with
@register_model. decide() receives exactly what the model may see (a
DecisionRequest) and returns a probability distribution per question over that
question's option ids. Metrics only ever read those distributions, so every
model is scored the same way.

Media sampling belongs to the adapter: frame rate, frame cap, resolution, audio rate and chunking of long media are
options of the model spec (e.g. "onejev@fps=1"), part of model_id, and recorded per call in usage["input"]. Two
policies of one model are two entries with separate caches. jevarena_mm.media offers shared decoders.
"""

from __future__ import annotations

from ..schema import DecisionRequest, ModelResponse


def split_options(arg: str) -> tuple[str, dict[str, str]]:
    """"org/model@fps=1,max_frames=64" -> ("org/model", {"fps": "1", "max_frames": "64"}); "@fps=1" -> ("", {...})."""
    base, _, opts = arg.partition("@")
    options = {}
    for part in filter(None, opts.split(",")):
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"model option {part!r} is not key=value")
        options[key.strip()] = value.strip()
    return base, options


class ModelAdapter:
    kind: str = ""  # registry key used in model specs, e.g. "onejev" or "llm"
    modalities: frozenset[str] = frozenset()  # media kinds the model takes ("image", "video", "audio"); text always
    native_probs: bool = False  # True if the model returns probabilities itself
    probability_source: str = ""  # how probabilities are obtained, shown on the leaderboard

    @property
    def model_id(self) -> str:
        """Unique per model and sampling policy, e.g. "onejev@fps=1"; runs are cached under it."""
        raise NotImplementedError

    @property
    def slug(self) -> str:
        return self.model_id.replace("/", "__").replace(":", "_").replace("@", "_at_").replace(",", "_").replace("=", "-")

    def supports(self, request: DecisionRequest) -> bool:
        """True if the model takes every media kind the request references."""
        return request.modalities() <= set(self.modalities)

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        raise NotImplementedError

    def reparse(self, response: ModelResponse, request: DecisionRequest) -> ModelResponse | None:
        """Re-read a stored response with the current parser (models that answer in free text).

        Returns None when the model has nothing to re-read, e.g. structured APIs such as Jev.
        """
        return None
