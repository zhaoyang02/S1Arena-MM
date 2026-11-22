"""The item format every task produces and every model consumes.

A benchmark item is a DecisionRequest (what the model sees) plus gold answers
and metadata (what the model never sees). Question types mirror the typed
decisions of System-One models:

- choice:  pick one option; criteria maps option id -> description
- boolean: probability that a statement is true; option ids are "true"/"false"
- score:   ordered levels; criteria is the ordered list, option ids are "0".."n-1"

Multi-label tasks use several boolean questions in one request.

Multimodal extension: a value in `state` may be a MediaRef (one image, video or audio file) or an ordered list of
MediaRefs (multi-view video, image sequences). The media kinds a request references are exactly the modalities it
needs; a model that does not take one of them is marked unsupported instead of being called.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

SCHEMA_VERSION = "mm-0.1"
MAX_OPTIONS = 255
MAX_LEVELS = 10  # score questions: Jev's API accepts at most 10 levels
BOOLEAN_OPTIONS = ("true", "false")
MEDIA_KINDS = ("image", "video", "audio")

QuestionType = Literal["choice", "boolean", "score"]
MediaKind = Literal["image", "video", "audio"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _relative_uri(value: str) -> str:
    if not value or value.startswith("/") or ".." in value.split("/") or "\\" in value:
        raise ValueError(f"uri must be a relative path under the media root: {value!r}")
    return value


class Marks(BaseModel):
    """Per-frame markings drawn onto a video before any model sees it (e.g. a red outline around a target).

    The file at uri (under the media root) is JSON: {"boxes": [[frame_index, x1, y1, x2, y2], ...]}, where
    frame_index counts frames in presentation order from the first frame of the video file. media.frames() draws them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str
    sha256: str
    style: Literal["red-outline-3px"] = "red-outline-3px"

    @field_validator("uri")
    @classmethod
    def _check_uri(cls, value: str) -> str:
        return _relative_uri(value)

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value


class MediaRef(BaseModel):
    """A pointer to one media file. The file itself never travels inside an item.

    uri: "<dataset>/<path as in the source release>" under $JEVARENA_MM_MEDIA, or "mem:<sha256>" for an in-memory
         observation of a closed-loop environment (see media.inline_image).
    window: [start_s, end_s) on the file's presentation timeline, i.e. the timestamps decoders report for its frames
            or samples (a file whose first frame has PTS 0.17 s starts its timeline at 0.17); video and audio only.
            None means the whole file.
    info: facts about the file (duration_s, fps, width, height, sample_rate, channels, has_audio), so an adapter can
          plan its sampling without decoding.
    marks: markings drawn onto every decoded frame (video only), part of what the model sees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MediaKind
    uri: str
    sha256: str
    window: tuple[float, float] | None = None
    info: dict[str, Any] | None = None
    marks: Marks | None = None

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("uri")
    @classmethod
    def _check_uri(cls, value: str) -> str:
        return value if value.startswith("mem:") else _relative_uri(value)

    @model_validator(mode="after")
    def _check_window(self) -> "MediaRef":
        if self.window is not None:
            if self.kind == "image":
                raise ValueError("an image takes no window")
            start, end = self.window
            if not 0 <= start < end:
                raise ValueError(f"window must satisfy 0 <= start < end, got {self.window}")
        if self.marks is not None and self.kind != "video":
            raise ValueError("marks apply to video only")
        return self


def is_media(value: Any) -> bool:
    """True for a dict that declares itself a media reference (a "kind" among MEDIA_KINDS plus a "uri")."""
    return isinstance(value, dict) and value.get("kind") in MEDIA_KINDS and "uri" in value


def _parse_media(value: Any) -> Any:
    if isinstance(value, MediaRef):
        return value
    if is_media(value):
        return MediaRef.model_validate(value)
    if isinstance(value, list) and value and all(isinstance(v, MediaRef) or is_media(v) for v in value):
        return [_parse_media(v) for v in value]
    return value


class Question(BaseModel):
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_OPTIONS:
                raise ValueError(f"choice questions need 2..{MAX_OPTIONS} options given as a dict")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= MAX_LEVELS:
                raise ValueError(f"score questions need an ordered list of 2..{MAX_LEVELS} levels")
        elif self.criteria is not None:
            raise ValueError("boolean questions take no criteria; put clarifications in the instructions")
        return self

    def option_ids(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(BOOLEAN_OPTIONS)


class DecisionRequest(BaseModel):
    """Everything the model is allowed to see. Key order of state is presentation order."""

    state: dict[str, Any]
    questions: dict[str, Question]

    @field_validator("state", mode="before")
    @classmethod
    def _media(cls, state: Any) -> Any:
        if isinstance(state, dict):
            return {key: _parse_media(value) for key, value in state.items()}
        return state

    def media(self) -> list[tuple[str, MediaRef]]:
        """Every media reference in state order, with its state key (lists are flattened in order)."""
        out = []
        for key, value in self.state.items():
            if isinstance(value, MediaRef):
                out.append((key, value))
            elif isinstance(value, list):
                out += [(key, v) for v in value if isinstance(v, MediaRef)]
        return out

    def modalities(self) -> set[str]:
        """The media kinds this request needs; text is always taken and not listed."""
        return {ref.kind for _, ref in self.media()}


class Gold(BaseModel):
    """A hard label, a soft label distribution, or both.

    accept: option ids that count as correct besides the label (e.g. answer variants of a document question); it may
    repeat the label. Scoring merges them with the label into one outcome. Left out of the serialized item when unset,
    so items without it keep their bytes and ids.
    """

    label: str | None = None
    dist: dict[str, float] | None = None
    accept: list[str] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Gold":
        if self.label is None and self.dist is None:
            raise ValueError("gold needs a label or a distribution")
        if self.dist is not None:
            total = sum(self.dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"gold distribution sums to {total}, not 1")
        if self.accept is not None and (self.label is None or not self.accept):
            raise ValueError("gold.accept needs a label and at least one option")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        data = handler(self)
        if data.get("accept") is None:
            data.pop("accept", None)
        return data

    def correct(self) -> set[str]:
        """The option ids counted as correct: the hard label and any accepted alternatives."""
        return {self.hard(), *(self.accept or [])}

    def hard(self) -> str:
        if self.label is not None:
            return self.label
        return max(self.dist, key=self.dist.get)

    def target(self, options: list[str]) -> dict[str, float]:
        """Target distribution for proper scoring (Brier): soft if available, else one-hot."""
        if self.dist is not None:
            return {o: float(self.dist.get(o, 0.0)) for o in options}
        return {o: float(o == self.label) for o in options}


class Item(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str = ""
    leaf: str
    task: str
    request: DecisionRequest
    gold: dict[str, Gold]
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Item":
        for name, gold in self.gold.items():
            if name not in self.request.questions:
                raise ValueError(f"gold for unknown question {name!r}")
            options = self.request.questions[name].option_ids()
            if gold.label is not None and gold.label not in options:
                raise ValueError(f"gold label {gold.label!r} is not an option of {name!r}")
            if gold.dist is not None and not set(gold.dist) <= set(options):
                raise ValueError(f"gold distribution of {name!r} has unknown options")
            if gold.accept is not None and not set(gold.accept) <= set(options):
                raise ValueError(f"gold.accept of {name!r} has unknown options")
        if any(ref.uri.startswith("mem:") for _, ref in self.request.media()):
            raise ValueError("a benchmark item cannot reference in-memory media")
        if not self.id:
            self.id = content_id(self)
        return self


def content_id(item: Item) -> str:
    """Stable id derived from content, so any change to an item changes its id.

    The request includes every media reference with its sha256, so a changed media file also changes the id.
    """
    payload = json.dumps(
        {
            "task": item.task,
            "request": item.request.model_dump(mode="json"),
            "gold": {k: v.model_dump() for k, v in item.gold.items()},
        },
        ensure_ascii=False,
    )  # key order is kept on purpose: presenting options in another order is another item
    return item.task.replace("/", ".") + ":" + hashlib.sha256(payload.encode()).hexdigest()[:12]


class Answer(BaseModel):
    """A model's answer to one question: a distribution over that question's option ids."""

    probs: dict[str, float]


class ModelResponse(BaseModel):
    """status: "ok"; "failed" (any response with an error; its questions score 0); "unsupported" (the request needs a
    modality the model does not take: no call is made and the item is left out of that model's scores)."""

    answers: dict[str, Answer] = Field(default_factory=dict)
    status: Literal["ok", "failed", "unsupported"] = "ok"
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None  # usage["input"]: what the adapter fed the model (frames, fps, audio seconds)
    cost_usd: float | None = None
    error: str | None = None
    raw_text: str | None = None  # the model's reply as written, kept for auditing (LLMs)

    @model_validator(mode="after")
    def _status(self) -> "ModelResponse":
        if self.error and self.status == "ok":
            self.status = "failed"
        return self


def normalize(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    """Restrict to known options, clip negatives, renormalize; uniform if nothing usable."""
    clean = {o: max(0.0, float(probs.get(o, 0.0) or 0.0)) for o in options}
    total = sum(clean.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: p / total for o, p in clean.items()}
