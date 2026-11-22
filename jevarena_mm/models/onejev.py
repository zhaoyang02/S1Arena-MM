"""OneJev-27B (OmniJev/OneJev-27B): a System-One decision model that reads images and video and returns its own
probability distribution, run through its official engine (`qev`, github.com/OmniJev/OneJev).

Spec: onejev[@fps=<rate>,min_frames=<n>,gpu_layers=<n>]
- fps (default 1): each video is read on a grid of this rate from its window start (nearest frames), and every grid
  frame is passed to the model as video at that rate;
- min_frames (default 2): a window that yields fewer frames is read at doubled rates (the official video processor
  rejects a single frame);
- gpu_layers (default 32): decoder layers kept on the GPU, the rest offloaded to CPU; not part of the model id.
- gpus (default 1): with gpus > 1 the decoder layers are spread evenly over that many GPUs and nothing is offloaded
  (gpu_layers is then unused); not part of the model id.
- Other OneJev sizes (onejev:OmniJev/OneJev-9B, -4B, -0.8B): pinned at SIZE_REVISIONS (the model id carries their
  first 7 hex digits); a size not listed there needs revision=<commit>, so that offline jobs need no Hub call.
  OneJev-27B keeps REVISION.

Weights: $ONEJEV_MODEL_DIR (OneJev-27B only; other sizes always come from the Hub cache), else the pinned revision from
the Hugging Face Hub. Engine: $ONEJEV_ENGINE_DIR (a checkout of the pinned revision), else an installed `qev` package.
Needs one BF16-capable CUDA GPU and `pip install -e .[jev]`.

Prompt: the state text, the media-timing note of the recorded runs (jev_common.sampling_note) and one
"BEGIN VIDEO VIEW k" block per video, as in those runs; images become <image:k> placeholders,
unless the text already places them (game screenshots).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .. import media
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse
from . import jev_common
from .base import ModelAdapter, split_options

REPO = "OmniJev/OneJev-27B"
REVISION = "aec969d68e7e8e8b0df051e065c04304c0cd7163"
SIZE_REVISIONS = {  # the other sizes, pinned at their Hub commits of 2026-10-11
    "OmniJev/OneJev-9B": "59429e80aa6cbb82c49d4ff8802359044a084eee",
    "OmniJev/OneJev-4B": "c88e18653ceb7a8770716287f55fdefc79d6b588",
    "OmniJev/OneJev-0.8B": "c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8",
}
ENGINE_REVISION = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
TYPES = {"choice": "choice", "boolean": "noul", "score": "score"}


@contextmanager
def unrounded_choice():
    """Read the engine's choice probabilities before it rounds them to 6 decimals (needed for long option lists,
    where the rounded values may not sum to 1). The response itself is not changed."""
    engine = importlib.import_module("qev.engine")
    answers = importlib.import_module("qev.answers")
    original = engine.make_choice
    captured: dict[str, float] = {}

    def assemble(option_ids, probabilities):
        response = original(option_ids, probabilities)
        if (len(option_ids) == len(probabilities) and len(set(option_ids)) == len(option_ids)
                and all(math.isfinite(p) and p >= 0 for p in probabilities) and sum(probabilities) > 0):
            captured.update(zip(option_ids, answers._normalize(probabilities), strict=True))
        return response

    engine.make_choice = assemble
    try:
        yield captured
    finally:
        engine.make_choice = original


def cpu_offload_map(layers: int, gpu_layers: int) -> dict[str, str | int]:
    devices: dict[str, str | int] = {"model.visual": 0, "model.language_model.embed_tokens": 0,
                                     "model.language_model.norm": 0, "model.language_model.rotary_emb": 0, "lm_head": "cpu"}
    devices.update({f"model.language_model.layers.{i}": 0 if i < gpu_layers else "cpu" for i in range(layers)})
    return devices


def spread_map(layers: int, gpus: int) -> dict[str, str | int]:
    """Every module on a GPU: the decoder layers in equal consecutive parts over `gpus` GPUs, the rest on GPU 0."""
    # The final norm stays on GPU 0 with the engine's slot readout (_slot_w): dispatch hooks move the last layer's output there.
    devices: dict[str, str | int] = {"model.visual": 0, "model.language_model.embed_tokens": 0,
                                     "model.language_model.norm": 0, "model.language_model.rotary_emb": 0, "lm_head": 0}
    devices.update({f"model.language_model.layers.{i}": i * gpus // layers for i in range(layers)})
    return devices


@register_model
class OneJev(ModelAdapter):
    kind = "onejev"
    _one_at_a_time = threading.Lock()
    modalities = frozenset({"image", "video"})
    native_probs = True
    probability_source = "native distribution"

    def __init__(self, arg: str = ""):
        repo, options = split_options(arg)
        self.repo = repo or REPO
        self.fps = float(options.get("fps", 1))
        self.min_frames = int(options.get("min_frames", 2))
        self.gpu_layers = int(options.get("gpu_layers", 32))
        self.gpus = int(options.get("gpus", 1))
        self.revision = options.get("revision") or (REVISION if self.repo == REPO else SIZE_REVISIONS.get(self.repo))
        if not self.revision:
            raise ValueError(f"{self.repo}: no pinned revision; pass revision=<commit> in the spec")
        self.engine = None

    @property
    def model_id(self) -> str:
        model_id = f"{self.repo}@fps={self.fps:g},min_frames={self.min_frames}"
        return model_id if self.repo == REPO else model_id + f",revision={self.revision[:7]}"

    def load(self) -> None:
        if self.engine is not None:
            return
        import torch
        from accelerate import dispatch_model

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("OneJev needs a BF16-capable CUDA GPU")
        engine_dir = os.environ.get("ONEJEV_ENGINE_DIR")
        if engine_dir and str(Path(engine_dir).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(engine_dir).resolve()))
        module = importlib.import_module("qev.mm_engine")
        self.schema = importlib.import_module("qev.schema")
        self.torch = torch
        path = jev_common.model_dir("ONEJEV_MODEL_DIR" if self.repo == REPO else None, self.repo, self.revision)
        started = time.perf_counter()
        engine = module.MMDecisionEngine(str(path), device="cpu", dtype="bfloat16", head_dtype="float32",
                                         fork_mode="sequential", cuda_graphs=False, gpu_preprocess=False)
        layers = engine.model.config.text_config.num_hidden_layers
        devices = cpu_offload_map(layers, self.gpu_layers) if self.gpus <= 1 else spread_map(layers, self.gpus)
        # DeltaNet reads conv1d.weight directly, bypassing conv1d's offload hook.
        dispatch_model(engine.model, device_map=devices, main_device=0, preload_module_classes=["Qwen3_5GatedDeltaNet"])
        engine.device = torch.device("cuda:0")
        engine._slot_w = engine._slot_w.to(engine.device)
        if engine._slot_b is not None:
            engine._slot_b = engine._slot_b.to(engine.device)
        torch.cuda.synchronize()
        self.engine = engine
        self.load_seconds = time.perf_counter() - started

    def payload(self, request: DecisionRequest, views: list[jev_common.View], images: list) -> dict:
        """The official SystemOneRequest for one question."""
        (_, question), = request.questions.items()
        state = jev_common.state_with_note(request, views)
        if not jev_common.placed_images(state, len(images)):  # images placed in the text keep their places
            for index, _ in enumerate(images, start=1):
                state += f"\n<image:{index}>"
        for index, _ in enumerate(views, start=1):
            state += f"\nBEGIN VIDEO VIEW {index}; independent timeline\n<video:{index}>\nEND VIDEO VIEW {index}\n"
        ids = jev_common.options(question)
        criteria = ({"true": "Yes", "false": "No"} if question.type == "boolean"
                    else [text for _, text in ids] if question.type == "score" else dict(ids))
        return {"model": REPO, "state": state,
                "questions": {"q": {"type": TYPES[question.type], "instructions": question.instructions, "criteria": criteria}}}

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        with self._one_at_a_time:  # one GPU model: runner threads take turns
            try:
                return self._decide(request)
            except Exception as error:  # an input too large for the card is rejected once, not retried on every run
                if type(error).__name__ != "OutOfMemoryError":
                    raise
                self.torch.cuda.empty_cache()
                if not jev_common.gpu_to_ourselves(self.torch):  # another process may have taken the memory: retry later
                    raise
                return ModelResponse(error=f"rejected: out of GPU memory on this input ({str(error)[:200]})")

    def _decide(self, request: DecisionRequest) -> ModelResponse:
        if len(request.questions) != 1:
            return ModelResponse(error="rejected: this adapter answers one question per request")
        self.load()
        (name, question), = request.questions.items()
        views = jev_common.sample_views(request, self.fps, self.min_frames)
        images = [media.image(ref) for ref in jev_common.refs(request, "image")]
        items = [{"type": "image", "image": image} for image in images]
        items += [{"type": "video", "frames": [jev_common.pil(f.array) for f in view.frames], "fps": view.fps} for view in views]
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        long_choice = question.type == "choice" and len(question.criteria) > 20
        with (unrounded_choice() if long_choice else nullcontext({})) as unrounded:
            response, _ = self.engine.decide(self.schema.SystemOneRequest.model_validate(self.payload(request, views, images)),
                                             debug=False, debias=1, media=items)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        answer = response.answers["q"].model_dump(mode="json")
        if question.type == "boolean":
            p_yes = float(answer["noul"])
            probs = {"true": p_yes, "false": 1.0 - p_yes}
        else:
            raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
            probs = unrounded if unrounded and abs(math.fsum(raw.values()) - 1) > 1e-9 else raw
        usage = {"input": {"views": [{"frames": len(v.frames), "fps": v.fps, "timestamps_s": [f.t for f in v.frames]} for v in views],
                           "images": len(images)},
                 "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        return ModelResponse(answers={name: Answer(probs=probs)}, latency_ms=round(latency, 1), usage=usage, cost_usd=0.0)
