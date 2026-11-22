"""d3 (vllm-sr/d3): the 27B multimodal decision model of Decision 3.0, vLLM Semantic Router's decision models (Qwen3.8-27B
with a 255-way answer-code readout), run through its own runtime as its card runs it: AutoModel with
trust_remote_code at a pinned revision, then system_one(state, questions, images, videos). Each question is one forward
pass that scores the options' answer codes at the last prompt position; no text is generated.

As the card does it:
- The request is already a System One request: the state, with its media named by their placeholders
  (render.media_slots; d3 shows a JSON state as compact JSON), and the questions as d3 types them: choice with its
  criteria, boolean as noul, score with its ordered levels. The answer is d3's probability per option id (noul: its
  P(true)).
- Images: PIL images in placeholder order; d3 puts them before the text and reads each at up to 1.6 MP.
- Video: read as d3 reads a video file, the window being the whole video: 2 frames per second, at least 4 and at most
  32 spread evenly over the window's frames (the runtime's video_sample_indices), decoded here (so marks are drawn)
  and passed as the runtime's own Video (frames, their source frame numbers and the source rate), so d3 sizes them
  (at most 0.2 MP each) and writes their timestamps as it would for the file.
- Audio: not taken; items with audio are unsupported.
- What d3 refuses is rejected (final): a question over its input limit (max_length_exceeded) or one it cannot take,
  and a request whose videos exceed its 16,384 video tokens.

Spec: d3[:<repo>][@device=<torch device>,gpus=<n>]. Its runtime runs on one device (its loader refuses a device map
over several): one 80 GB GPU (H100; it peaks at about 62-65 GiB), or the CPU for a functional check. gpus=2 runs it
on two 40 GB GPUs instead: the backbone is loaded over cuda:0 and cuda:1 (split_map; accelerate moves the hidden
states between the layers) while the runtime keeps its inputs, the vision tower, the embeddings, the final norm and the
readout on cuda:0, so the same steps run on the same numbers. On NVIDIA GPUs the runtime takes its plain path either
way (its fused kernels and CUDA graphs are for ROCm). gpus is not part of the model id.
"""

from __future__ import annotations

import sys
import threading
import time

from .. import media, render
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse, Question
from .base import ModelAdapter, split_options
from .local_lm import source_fps, window

REPO, REVISION = "vllm-sr/d3", "fc2d74bd48ec29ae42d4330a543913f7858e3f40"
REJECTED = "rejected: "


def split_map(layers: int, gpus: int) -> dict[str, int]:
    """A device map of d3's Qwen3.5 backbone over gpus GPUs: the vision tower, the embeddings, the rotary embedding
    and the final norm on GPU 0 (with the runtime's inputs and readout), the decoder layers in order over the GPUs,
    two fewer on GPU 0, which also holds the embeddings (~2.5 GB) and the vision tower (~0.9 GB)."""
    counts = [layers // gpus] * gpus
    counts[-1] += layers - sum(counts)
    shift = min(2, counts[0] - 1) if gpus > 1 else 0
    counts[0], counts[1 % gpus] = counts[0] - shift, counts[1 % gpus] + shift
    device_map = {"visual": 0, "language_model.embed_tokens": 0, "language_model.rotary_emb": 0, "language_model.norm": 0}
    layer = 0
    for gpu, count in enumerate(counts):
        for _ in range(count):
            device_map[f"language_model.layers.{layer}"] = gpu
            layer += 1
    return device_map


def d3_question(question: Question) -> dict:
    """A question as d3 types it: choice with its criteria, boolean as noul, score with its ordered levels."""
    if question.type == "choice":
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.criteria)}
    if question.type == "boolean":
        return {"type": "noul", "instructions": question.instructions}
    return {"type": "score", "instructions": question.instructions, "criteria": list(question.criteria)}


def d3_probs(question: Question, answer: dict) -> dict[str, float]:
    """d3's answer as a distribution over the question's option ids."""
    if question.type == "boolean":
        yes = float(answer["noul"])
        return {"true": yes, "false": 1.0 - yes}
    return {option: float(answer["probabilities"][option]) for option in question.option_ids()}


@register_model
class D3(ModelAdapter):
    kind = "d3"
    modalities = frozenset({"image", "video"})
    native_probs = True
    probability_source = "the model's answer-code readout at the last prompt position"
    _one_at_a_time = threading.Lock()

    def __init__(self, arg: str = "") -> None:
        repo, options = split_options(arg)
        self.repo = repo or REPO
        self.device = options.get("device")  # None: the runtime's default, cuda:0 (else the CPU)
        self.gpus = int(options.get("gpus", 1))  # >1: the backbone split over that many GPUs; not part of the model id
        if self.gpus > 1 and self.device:
            raise ValueError(f"d3: gpus={self.gpus} runs on cuda:0 .. cuda:{self.gpus - 1}; give no device")
        self.model = None

    @property
    def model_id(self) -> str:
        return self.repo

    def load(self) -> None:
        with self._one_at_a_time:
            if self.model is not None:
                return
            from transformers import AutoModel

            revision = REVISION if self.repo == REPO else None
            if self.gpus > 1:
                self._load_split(revision)
                return
            self.model = AutoModel.from_pretrained(self.repo, revision=revision, trust_remote_code=True,
                                                   **({"device": self.device} if self.device else {}))
            self.runtime = sys.modules[type(self.model.runtime).__module__]  # d3_runtime as the repository ships it

    def _load_split(self, revision: str | None) -> None:
        """The repository's runtime, with only its backbone load given split_map instead of one device."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        runtime_class = get_class_from_dynamic_module("d3_runtime.D3", self.repo, revision=revision)
        runtime = sys.modules[runtime_class.__module__]
        stock, gpus = runtime.backbone_class, self.gpus

        def split(kind: str):
            backbone = stock(kind)

            class Split:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    layers = backbone.config_class.from_pretrained(path).text_config.num_hidden_layers
                    return backbone.from_pretrained(path, **{**kwargs, "device_map": split_map(layers, gpus)})

            return Split

        runtime.backbone_class = split
        try:
            self.model = runtime_class.from_pretrained(self.repo, revision=revision, device="cuda:0")
        finally:
            runtime.backbone_class = stock
        self.runtime = runtime

    def _video(self, ref):
        """The window as the runtime samples a video file (video_sample_indices over the window's frames)."""
        start, end = window(ref)
        rate = source_fps(ref)
        total = max(1, int(round((end - start) * rate)))
        wanted = self.runtime.video_sample_indices(total, rate)
        frames = media.frames(ref, timestamps=[start + i / rate for i in wanted])
        if not frames:
            raise ValueError(f"{ref.uri}: no frames in the window")
        indices = [min(total - 1, max(0, int(round((f.t - start) * rate)))) for f in frames]
        video = self.runtime._video([f.array for f in frames], indices, rate, total)
        return video, {"frames": len(frames), "timestamps_s": [round(f.t, 3) for f in frames]}

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        self.load()
        started = time.perf_counter()
        state, slots = render.media_slots(request)
        images, videos, used = [], [], []
        try:  # decoded outside the lock, by the runner's threads in parallel
            for name, ref in slots:
                if ref.kind == "image":
                    images.append(media.image(ref))
                    used.append({"placeholder": name, "image": True})
                else:
                    video, info = self._video(ref)
                    videos.append(video)
                    used.append({"placeholder": name, **info})
        except (OSError, ValueError, KeyError, RuntimeError, MemoryError) as error:  # this request only, retried next run
            return ModelResponse(error=f"media: {type(error).__name__}: {error}"[:300], usage={"input": used})
        questions = {name: d3_question(question) for name, question in request.questions.items()}
        try:
            with self._one_at_a_time:  # one model: requests take turns on it
                result = self.model.system_one(state=state, questions=questions, images=images or None, videos=videos or None)
        except ValueError as error:  # refused as a whole, e.g. videos over its 16,384 video tokens
            return ModelResponse(error=(REJECTED + f"{type(error).__name__}: {error}")[:300], usage={"input": used})
        answers, errors = {}, []
        for name, question in request.questions.items():
            answer = result["answers"].get(name) or {"error": "missing", "message": "no answer"}
            if "error" in answer:
                errors.append(f"{name}: {answer['error']}: {answer.get('message', '')}")
            else:
                answers[name] = Answer(probs=d3_probs(question, answer))
        latency = (time.perf_counter() - started) * 1000
        usage = {"input": used, "input_tokens": (result.get("usage") or {}).get("input_tokens")}
        error = (REJECTED + "; ".join(errors))[:300] if errors else None
        return ModelResponse(answers=answers, error=error, usage=usage, latency_ms=round(latency, 1), cost_usd=0.0)
