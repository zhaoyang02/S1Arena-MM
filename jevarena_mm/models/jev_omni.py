"""Jev-Omni (akhilaaa3/Jev-Omni): a System-One decision model that reads images, video and audio and returns its own
probability distribution, run with the model repository's own helper (jev_omni.py) and decision head.

Spec: jev_omni[@fps=<rate>,frames=<n>,gpus=<n>]
- fps (default 1) and frames (default 16): each video is read on a grid of `fps` from its window start, then the
  model's official sampler keeps `frames` frames at the centres of equal bins of that grid (as in the recorded runs);
- gpus (default 1): GPUs the weights are spread over; not part of the model id.
- head (default: the repository's own head.pt): a Hugging Face repository holding another 256-slot decision head for
  the same backbone revision as decision_head.pt (e.g. ferdinandl007/jev-omni-general-head-v3), loaded in place of
  head.pt as its card loads it; part of the model id. Heads in HEAD_REVISIONS load at that commit (not part of the id).
Audio is passed whole (16 kHz mono, not truncated or chunked); an input longer than the model's context is rejected.

Weights: $JEV_OMNI_MODEL_DIR, else the pinned revision from the Hugging Face Hub. Needs CUDA and `pip install -e .[jev]`.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time

from .. import media
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse
from . import jev_common
from .base import ModelAdapter, split_options

REPO = "akhilaaa3/Jev-Omni"
REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
SAMPLE_RATE = 16000
HEAD_REVISIONS = {"ferdinandl007/jev-omni-general-head-v3": "10e061b811f4add247dc0cbf06a67cb5c0233ad8"}  # 2026-09-30


def bin_centres(total: int, count: int) -> list[int]:
    """The official sampler's frame indices: centres of `count` equal bins over `total` frames, without repeats."""
    return sorted({int(round((total - 1) * (k + 0.5) / count)) for k in range(count)})


@register_model
class JevOmni(ModelAdapter):
    kind = "jev_omni"
    _one_at_a_time = threading.Lock()
    modalities = frozenset({"image", "video", "audio"})
    native_probs = True
    probability_source = "native distribution"

    def __init__(self, arg: str = ""):
        repo, options = split_options(arg)
        self.repo = repo or REPO
        self.fps = float(options.get("fps", 1))
        self.frames = int(options.get("frames", 16))
        self.gpus = int(options.get("gpus", 1))
        self.head_repo = options.get("head")
        self.model = None

    @property
    def model_id(self) -> str:
        return f"{self.repo}@fps={self.fps:g},frames={self.frames}" + (f",head={self.head_repo}" if self.head_repo else "")

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        import transformers

        if not torch.cuda.is_available():
            raise RuntimeError("Jev-Omni needs a CUDA GPU")
        path = jev_common.model_dir("JEV_OMNI_MODEL_DIR", self.repo, REVISION)
        started = time.perf_counter()
        spec = importlib.util.spec_from_file_location("jev_omni_helper", path / "jev_omni.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
        model_class = getattr(transformers, config.architectures[0])
        placement = {"device_map": "cuda"}
        if self.gpus > 1:
            placement = {"device_map": "balanced", "max_memory": {
                i: int(torch.cuda.get_device_properties(i).total_memory * 0.45) for i in range(self.gpus)}}
        model = model_class.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True, **placement).eval()
        _, decoder = helper._find_backbone(model)
        self.input_device = model.get_input_embeddings().weight.device
        self.head_device = decoder.norm.weight.device
        decision = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
        head = helper._Head256(decision["hidden_size"]).to(self.head_device).eval()
        weights = path / "head.pt"
        if self.head_repo:  # another head for this backbone revision, as its card loads it
            from huggingface_hub import hf_hub_download

            revision = HEAD_REVISIONS.get(self.head_repo)
            card = json.loads(open(hf_hub_download(self.head_repo, "config.json", revision=revision)).read())
            if card.get("base_revision") not in (None, REVISION):
                raise ValueError(f"{self.head_repo} was trained on backbone revision {card.get('base_revision')}, not {REVISION}")
            weights = hf_hub_download(self.head_repo, card.get("checkpoint_file", "decision_head.pt"), revision=revision)
        head.load_state_dict(torch.load(weights, map_location=self.head_device, weights_only=True))
        self.processor = transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
        self.classifier = helper.JevOmni(model, head, self.processor, decoder, device=str(self.input_device))
        self.helper, self.head, self.torch, self.model = helper, head, torch, model
        self.load_seconds = time.perf_counter() - started

    def content(self, request: DecisionRequest) -> tuple[list[dict], dict]:
        """The chat content: audio (if any), images, the sampled frames of each video, then the official prompt. Images
        that the text places itself (<image:k>, game screenshots) are interleaved with the text instead."""
        (_, question), = request.questions.items()
        # The recorded runs gave Jev-Omni the frames prepared for both models: at least two for prediction prefixes.
        grids = jev_common.sample_views(request, self.fps, 2 if jev_common.prediction_prefix(request) else 1)
        picked = [[view.frames[i] for i in bin_centres(len(view.frames), self.frames)] for view in grids]
        content: list[dict] = []
        used: dict = {"views": [{"grid_frames": len(view.frames), "fps": view.fps, "frames": len(frames),
                                 "timestamps_s": [f.t for f in frames]} for view, frames in zip(grids, picked)]}
        for ref in jev_common.refs(request, "audio"):
            samples, _ = media.audio(ref, sample_rate=SAMPLE_RATE, mono=True)
            start = ref.window[0] if ref.window else 0.0
            seconds = len(samples) / SAMPLE_RATE
            content += [{"type": "text", "text": (
                f"The supplied native audio contains the entire permitted audio observation: source clock "
                f"[{start:.6f}, {start + seconds:.6f}) seconds, duration {seconds:.6f} seconds. No transcript is supplied. "
                "Audio and video source clocks are recorded separately; precise cross-stream synchronization is not asserted.")},
                {"type": "audio", "audio": samples}]
            used.setdefault("audio_s", []).append(round(seconds, 3))
        images = [media.image(ref) for ref in jev_common.refs(request, "image")]
        used["images"] = len(images)
        text = jev_common.state_text(request)
        if jev_common.placed_images(text, len(images)) and not grids:
            # Game screenshots placed in the text: text and images interleaved, then the prompt without a state.
            for piece, image in zip(jev_common.split_at_images(text), images + [None]):
                content.append({"type": "text", "text": piece})
                if image is not None:
                    content.append({"type": "image", "image": image})
            texts = [option_text for _, option_text in self.options(request, question)]
            content.append({"type": "text", "text": self.helper._prompt("", question.instructions, texts)})
            return content, used
        content += [{"type": "image", "image": image} for image in images]
        if len(picked) == 1:
            content += [{"type": "image", "image": jev_common.pil(f.array)} for f in picked[0]]
        else:
            for index, frames in enumerate(picked, start=1):
                content.append({"type": "text", "text": (
                    f"View {index} of {len(picked)}: {len(frames)} chronological frames from one complete camera clip. "
                    "This view has its own timeline; do not align its frames with other views.")})
                content += [{"type": "image", "image": jev_common.pil(f.array)} for f in frames]
        state = jev_common.state_with_note(request, grids)
        recorded_stamps = jev_common.recorded(request).get("omni_timestamps")
        if recorded_stamps:  # the same frames, numbered from the stream start as the recorded runs printed them
            state += "\n" + recorded_stamps
        elif picked:
            state += ("\nOfficial sampled source timestamps (seconds), per independent view: "
                      + str([[f.t for f in frames] for frames in picked]))
        texts = [text for _, text in self.options(request, question)]
        content.append({"type": "text", "text": self.helper._prompt(state, question.instructions, texts)})
        return content, used

    @staticmethod
    def options(request: DecisionRequest, question) -> list[tuple[str, str]]:
        """Options in the order shown; boolean questions in the recorded or per-item order (jev_common.boolean_order)."""
        return jev_common.options(question, jev_common.boolean_order(request) if question.type == "boolean" else None)

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
        if len(request.questions) > 1 and request.modalities() == {"audio"}:
            # Audio multi-label items keep their original question IDs; the head reads one question at a time.
            parts = {name: self._decide(request.model_copy(update={"questions": {name: question}}))
                     for name, question in request.questions.items()}
            errors = [f"{name}: {part.error}" for name, part in parts.items() if part.error]
            return ModelResponse(
                answers={name: answer for part in parts.values() for name, answer in part.answers.items()},
                error="; ".join(errors) if errors else None,
                latency_ms=sum(part.latency_ms or 0 for part in parts.values()), cost_usd=0.0,
                usage={"audio_question_protocol": "independent-native-head-v1",
                       "questions": {name: part.usage for name, part in parts.items()}},
            )
        if len(request.questions) != 1:
            return ModelResponse(error="rejected: this adapter answers one question per request")
        self.load()
        torch = self.torch
        (name, question), = request.questions.items()
        ids = [option for option, _ in self.options(request, question)]
        started = time.perf_counter()
        content, used = self.content(request)
        audio = any(part["type"] == "audio" for part in content)
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", enable_thinking=False,
                **({"processor_kwargs": {"audio_kwargs": {"sampling_rate": SAMPLE_RATE, "truncation": False},
                                         "text_kwargs": {"truncation": False}}} if audio else {}))
            tokens = int(inputs["input_ids"].shape[-1])
            limit = self.model.config.text_config.max_position_embeddings
            if tokens > limit:
                return ModelResponse(error=f"rejected: {tokens} input tokens exceed the context of {limit}", usage={"input": used})
            inputs = {k: v.to(self.input_device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.input_device)
                      for k, v in inputs.items()}
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
            self.classifier._capture.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.model(**inputs, use_cache=False, **self.classifier._extra)
                hidden = self.classifier._capture["hidden"]
                values = self.head(hidden.to(self.head_device), torch.tensor([len(ids)], device=self.head_device))[0, :len(ids)].softmax(-1)
            probs = dict(zip(ids, values.float().cpu().tolist(), strict=True))
        latency = (time.perf_counter() - started) * 1000
        peak = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**20
        return ModelResponse(answers={name: Answer(probs=probs)}, latency_ms=round(latency, 1), cost_usd=0.0,
                             usage={"input": {**used, "tokens": tokens}, "peak_gpu_memory_mib": round(peak, 1)})

