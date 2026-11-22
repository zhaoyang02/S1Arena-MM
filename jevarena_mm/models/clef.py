"""Clef (Cloudflare/clef-flash, Cloudflare/clef, Cloudflare/clef-omni): System-One joint-schema decision models, run
with each repository's own joint_schema_model.py (encode_record, collate_records, the model's forward) as their cards
run them; probabilities are the softmax of the head's logits per question.

The request is the card's record: "state" is the request's state with each medium replaced by its placeholder ("<video
1>", render.media_slots), "questions" the Jev/SystemOne questions (boolean -> noul), and the media go in "images",
"videos" and "audio", which the repository's encoder places before the state.
- clef-flash, clef (Qwen3-VL processor): each video as the checkpoint's video processor reads a whole video by default
  (fps 2, 4 to 768 frames spread over the window, its size budget), decoded here and passed with their metadata
  (do_sample_frames=False) so that the processor sizes them and writes their timestamps as for the clip.
- clef-omni: frames on a grid of fps=1 (default; the repository's load_video reads 2 fps, which overflows the context
  past ~4.6 min) from the window start, each scaled to at most 256x32x32 pixels (even sides) as load_video does, an
  even count; when the record would exceed the release's context (64,000 tokens), the most frames that fit, evenly
  spaced over the grid, so the state is never cut (the benchmark's rule: a 1 fps grid, frames capped by the context). The
  audio the state references at 16 kHz mono.
Images always go before the state: the repository's encoder (_encode_media) places every medium there and takes no
position, so a text that places images with <image:k> (the closed-loop suites' packets) keeps those markers as text
and the images come first, in the same order.
gpus=N (default 1) spreads the backbone over N GPUs (device_map "balanced"); the head and the inputs go to the GPU of
the backbone's output (the head indexes the last hidden states with the input ids); with one GPU the repository's
load_release_model is used as is.

Spec: clef:<repo>[@gpus=N]; default repo Cloudflare/clef-flash; clef-omni also takes fps=<rate> (part of its id).
An out-of-memory error is a final rejection only when no other process computes on the GPU (jev_common.gpu_to_ourselves).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path

from .. import media, render
from . import jev_common
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse
from .base import ModelAdapter, split_options
from .local_lm import source_fps, window

REPO = "Cloudflare/clef-flash"
REVISIONS = {  # the releases as run, pinned (weights, joint head and joint_schema_model.py)
    "Cloudflare/clef-flash": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
    "Cloudflare/clef": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
    "Cloudflare/clef-omni": "0db1cd2607d76a7bdb2a382f659e7b313079f84b",
}
TYPES = {"boolean": "noul", "choice": "choice", "score": "score"}


class _TooLong(ValueError):
    """A medium whose tokens certainly exceed the model's max_length."""


OMNI_MAX_LENGTH = 64000  # clef-omni's DEFAULT_MAX_LENGTH (checked when the release loads)
MIN_FRAMES = 2


@register_model
class Clef(ModelAdapter):
    kind = "clef"
    native_probs = True
    probability_source = "native joint-schema head"
    _one_at_a_time = threading.Lock()

    def __init__(self, arg: str = "") -> None:
        repo, options = split_options(arg)
        self.repo = repo or REPO
        self.gpus = int(options.get("gpus", 1))
        self.omni = "omni" in self.repo.lower()
        self.modalities = frozenset({"image", "video", "audio"} if self.omni else {"image", "video"})
        self.fps = float(options.get("fps", 1))  # clef-omni only
        self.model = None

    @property
    def model_id(self) -> str:
        return f"{self.repo}@fps={self.fps:g},fit={OMNI_MAX_LENGTH}" if self.omni else f"{self.repo}@video-default"

    # ---------- loading ----------

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        path = Path(snapshot_download(self.repo, revision=REVISIONS.get(self.repo)))
        spec = importlib.util.spec_from_file_location(f"joint_schema_model_{abs(hash(str(path)))}", path / "joint_schema_model.py")
        jsm = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = jsm  # its dataclasses look their module up there
        spec.loader.exec_module(jsm)
        self.jsm = jsm
        self.max_length = int(getattr(jsm, "DEFAULT_MAX_LENGTH", 16384))
        if self.omni and self.max_length != OMNI_MAX_LENGTH:
            raise ValueError(f"{self.repo}: the release's context is {self.max_length}, the model id says {OMNI_MAX_LENGTH}")
        if self.gpus <= 1:
            model, self.processor = jsm.load_release_model(path, device="cuda")
            self.input_device = next(model.parameters()).device
        else:  # inputs go where the last hidden states come out: the release's head indexes those with the input ids
            model, self.processor, self.input_device = self._load_spread(path)
        video = self.processor.video_processor
        self.video_defaults = {"fps": float(video.fps or 2), "min_frames": int(video.min_frames), "max_frames": int(video.max_frames)}
        self.model = model  # last: a load that fails part way is tried again by the next call

    def _load_spread(self, path: Path):
        """The repository's load_release_model with the backbone spread over self.gpus GPUs."""
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor

        torch, jsm = self.torch, self.jsm
        memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.85) for i in range(self.gpus)}
        if self.omni:
            config = AutoConfig.from_pretrained(path)
            config.enable_audio_output = False
            backbone = transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                path, config=config, dtype=torch.bfloat16, device_map="balanced", max_memory=memory).thinker
        else:
            backbone = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, device_map="balanced", max_memory=memory)
        backbone.config.use_cache = False
        out_device = next(p.device for n, p in backbone.named_parameters() if n.endswith("norm.weight") and "layers" not in n
                          and "visual" not in n and "audio" not in n)
        if backbone.lm_head.weight.device != out_device:  # the head multiplies the last hidden states by its weights
            raise RuntimeError(f"lm_head on {backbone.lm_head.weight.device}, the final norm on {out_device}")
        head = jsm.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
        head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
        head = head.to(device=out_device, dtype=torch.bfloat16)
        return jsm.ClefModel(backbone, head).eval(), AutoProcessor.from_pretrained(path), out_device

    # ---------- media ----------

    def _video_default(self, ref):
        """Qwen3-VL processor default: num = duration x fps clamped to [min_frames, max_frames] and the frames there are,
        at linspace(0, frames - 1, num).round() of the window."""
        import numpy as np

        d = self.video_defaults
        start, end = window(ref)
        rate = source_fps(ref)
        total = max(1, int(round((end - start) * rate)))
        count = min(max(int(total / rate * d["fps"]), d["min_frames"]), d["max_frames"], total)
        wanted = np.linspace(0, total - 1, count).round().astype(int).tolist()
        frames = media.frames(ref, timestamps=[start + i / rate for i in wanted])
        indices = [max(0, int(round((f.t - start) * rate))) for f in frames]
        metadata = {"fps": rate, "frames_indices": indices, "total_num_frames": total, "duration": end - start}
        return np.stack([f.array for f in frames]), metadata, [round(f.t, 3) for f in frames]

    def _video_omni(self, ref, count: int | None = None):
        """clef-omni's frames: a grid of self.fps from the window start (at most `count` of them, evenly spaced), each
        scaled to at most MAX_VIDEO_FRAME_PIXELS (even sides) as the repository's load_video does, an even count."""
        import numpy as np
        from PIL import Image

        cap = float(self.jsm.MAX_VIDEO_FRAME_PIXELS)
        frames = media.frames(ref, fps=self.fps, max_frames=count)
        out = []
        for frame in frames:
            height, width = frame.array.shape[:2]
            scale = min(1.0, math.sqrt(cap / (width * height)))
            size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
            out.append(np.asarray(Image.fromarray(frame.array).resize(size, Image.BILINEAR)) if size != (width, height) else frame.array)
        if len(out) % 2:
            out.append(out[-1])
        return np.stack(out), [round(f.t, 3) for f in frames]

    def _omni_pair_tokens(self, ref) -> int:
        """Video tokens per pair of frames (the processor's 32-pixel patches of a frame scaled as _video_omni scales it)."""
        cap, info = float(self.jsm.MAX_VIDEO_FRAME_PIXELS), ref.info or {}
        width, height = float(info.get("width") or 512), float(info.get("height") or 512)
        scale = min(1.0, math.sqrt(cap / (width * height)))
        return max(1, round(height * scale / 32)) * max(1, round(width * scale / 32))

    # ---------- decisions ----------

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        with self._one_at_a_time:
            self.load()
        started = time.perf_counter()  # media decoded outside the lock, by the runner's threads in parallel
        state, slots = render.media_slots(request)
        images, videos, audio, metas, used = [], [], [], [], []
        if self.omni:
            return self._decide_omni(request, state, slots, started)
        try:
            self._collect(slots, images, videos, audio, metas, used)
        except _TooLong as error:  # what encode_record would refuse, known before decoding
            return ModelResponse(error=f"rejected: {error}", usage={"input": used})
        return self._decide_record(request, state, images, videos, audio, metas, used, started)

    def _decide_omni(self, request, state, slots, started) -> ModelResponse:
        """clef-omni: the most frames per video (evenly spaced over its grid) whose record fits the context uncut."""
        refs = [ref for _, ref in slots if ref.kind == "video"]
        grids = [max(1, math.ceil((window(ref)[1] - window(ref)[0]) * self.fps - 1e-9)) for ref in refs]
        pair = [self._omni_pair_tokens(ref) for ref in refs]
        base = len(self._encode(request, state, slots, {name: 0 for name, ref in slots if ref.kind == "video"})[0].input_ids)
        budget = self.max_length - base - 16 * len(refs)  # 16: each video's own special tokens, generously
        need = sum(math.ceil(n / 2) * p for n, p in zip(grids, pair))
        scale = min(1.0, budget / need) if need else 1.0
        counts = [max(MIN_FRAMES, min(n, math.floor(n * scale))) for n in grids]
        for _ in range(6):
            named = {name: c for (name, ref), c in zip(((n, r) for n, r in slots if r.kind == "video"), counts)}
            encoded, images, videos, audio, used = self._encode(request, state, slots, named)
            for entry, n in zip((u for u in used if "frames" in u), grids):
                entry["grid"] = n
            if len(encoded.input_ids) <= self.max_length:
                return self._decide_record(request, state, images, videos, audio, [], used, started, encoded=encoded)
            if all(c <= MIN_FRAMES for c in counts):
                break
            counts = [max(MIN_FRAMES, math.floor(c * 0.9)) for c in counts]
        return ModelResponse(error=f"rejected: {len(encoded.input_ids)} tokens with {MIN_FRAMES} frames per video; "
                                   f"maximum is {self.max_length}", usage={"input": used})

    def _encode(self, request, state, slots, counts):
        """(record encoded without a length limit, images, videos, audio, used) with counts[name] frames per video
        (0: the video left out, to measure everything else)."""
        images, videos, audio, used = [], [], [], []
        kept = [(name, ref) for name, ref in slots if not (ref.kind == "video" and counts.get(name) == 0)]
        self._collect(kept, images, videos, audio, [], used, counts)
        record = self._record(request, state, images, videos, audio, [])
        encoded = self.jsm.encode_record(self.processor.tokenizer, record, max_length=10**9, processor=self.processor)
        return encoded, images, videos, audio, used

    def _collect(self, slots, images, videos, audio, metas, used, counts=None) -> None:
        for name, ref in slots:
            if ref.kind == "image":
                images.append(media.image(ref))
                used.append({"placeholder": name, "image": True})
            elif ref.kind == "audio":
                wave, rate = media.audio(ref, sample_rate=int(getattr(self.jsm, "AUDIO_SAMPLE_RATE", 16000)))
                audio.append(wave)
                used.append({"placeholder": name, "audio_s": round(len(wave) / rate, 3)})
            elif self.omni:
                array, stamps = self._video_omni(ref, (counts or {}).get(name))
                videos.append(array)
                used.append({"placeholder": name, "frames": len(stamps), "timestamps_s": stamps})
            else:
                array, metadata, stamps = self._video_default(ref)
                videos.append(array)
                metas.append(metadata)
                used.append({"placeholder": name, "frames": len(stamps), "timestamps_s": stamps})

    def _decide_record(self, request, state, images, videos, audio, metas, used, started, encoded=None) -> ModelResponse:
        torch = self.torch
        if encoded is None:
            try:
                encoded = self.jsm.encode_record(self.processor.tokenizer, self._record(request, state, images, videos, audio, metas),
                                                 max_length=self.max_length, processor=self.processor)
            except ValueError as error:  # longer than the model's context: final
                return ModelResponse(error=f"rejected: {str(error)[:250]}", usage={"input": used})
        try:
            with self._one_at_a_time, torch.inference_mode():  # one GPU model: forward passes take turns
                batch = self.jsm.collate_records([encoded], self.processor.tokenizer.pad_token_id, self.input_device)
                logits = self.model(batch)[0]
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            if not jev_common.gpu_to_ourselves(torch):  # another process may hold the memory: an ordinary failure, retried
                raise
            return ModelResponse(error=f"rejected: out of GPU memory on this input ({str(error)[:200]})", usage={"input": used})
        answers = {}
        for question, question_logits in zip(encoded.questions, logits):
            probs = question_logits.float().softmax(-1).tolist()
            answers[question.question_id] = Answer(probs=dict(zip(question.option_ids, probs)))
        latency = (time.perf_counter() - started) * 1000
        return ModelResponse(answers=answers, latency_ms=round(latency, 1), usage={"input": used}, cost_usd=0.0)

    def _record(self, request, state, images, videos, audio, metas) -> dict:
        record = {"state": state, "questions": {}}
        if images:
            record["images"] = images
        if videos:
            record["videos"] = videos
        if audio:
            record["audio"] = audio
        if metas:
            record["media_kwargs"] = {"video_metadata": metas, "do_sample_frames": False}
        for name, question in request.questions.items():
            entry = {"type": TYPES[question.type], "instructions": question.instructions}
            if question.type != "boolean":
                entry["criteria"] = question.criteria
            record["questions"][name] = entry
        return record
