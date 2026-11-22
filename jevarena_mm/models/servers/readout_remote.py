"""Letter-logit readout for checkpoints that ship their own modeling code (trust_remote_code).

The same readout as HFReadoutEngine (qev's prompt, the checkpoint's chat template, the option letters' logits at the
last position), loaded through the checkpoint's own classes and fed images the way the checkpoint's card does. One
branch per forward (their processors do not all batch safely).

  minicpmo  openbmb/MiniCPM-o-4_5, MiniCPM-o-2_6: each image is the text "(<image>./</image>)" where the state refers
            to it (model.chat() turns an image part into that tag and joins parts with newlines); the processor turns
            tags into the sliced vision tokens (pixel_values, tgt_sizes, image_bound); get_vllm_embedding() builds the
            input embeddings and the language model (Qwen3 / Qwen2) reads them. Vision only: init_audio / init_tts off.
  phi4mm    microsoft/Phi-4-multimodal-instruct: each image is "<|image_N|>" where the state refers to it (the card's
            prompt format), the text-only chat template, forward(**processor(text, images)) with the vision LoRA the
            processor's input_mode selects.
  nemotron  nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16: HF-style processor and chat template with image content
            parts (enable_thinking=False); forward(pixel_values, input_ids, image_flags, ..) with image_flags = 1 per
            image. The Mamba layers run their torch fallback here (no mamba_ssm / causal_conv1d kernels).
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

from qev import media as mmedia
from qev.engine import EngineInfo
from qev.prompt import DEFAULT_STYLE, build_messages

from jevarena_mm.models.servers.readout_engines import HFReadoutEngine, assistant_prefix_policy, log

PLACEHOLDER = re.compile(r"<image:(\d+)>")


class RemoteReadoutEngine(HFReadoutEngine):
    """HFReadoutEngine whose model and processor come from the checkpoint's own code."""

    template_kwargs_default: dict = {}
    forward_extra: dict = {"use_cache": False, "return_dict": True}

    def __init__(self, model_path: str, dtype: str = "bfloat16", max_memory: dict | None = None,
                 batch_tokens: int = 24_000, style=DEFAULT_STYLE, **_) -> None:
        import torch
        import transformers
        from qev.calibrate import Calibration

        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, dtype)
        self.calibration = Calibration()
        self.max_branch_tokens, self.max_request_tokens = 32768, 1 << 20
        self.fork_token_budget, self.fork_tolerance = batch_tokens, 5e-2
        self.batch_tokens = batch_tokens
        self.style = style
        self.head_dtype = "float32"
        self._lock = threading.Lock()
        self._want_mass = False
        self._media, self._loaded = [], ([], [], [])
        t0 = time.perf_counter()
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            # custom classes that skip post_init() (MiniCPM-o) never set it; transformers 5's loader reads it
            transformers.PreTrainedModel.all_tied_weights_keys = {}
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model_type = config.model_type
        self.quant_note = None
        self.processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = getattr(self.processor, "chat_template", None)
        kw: dict[str, Any] = dict(dtype=self.torch_dtype, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        if max_memory:
            kw["max_memory"] = max_memory
        self.model = self._load(config, kw).eval()
        head = self._output_head()
        if head is not None:   # only the last position is read
            head.register_forward_pre_hook(lambda m, args: (args[0][:, -1:, :],) + tuple(args[1:]))
        self.template_kwargs = dict(self.template_kwargs_default)
        self.slot_ids = self._verify_slots()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.assistant_suffix = ""
        probe = self._render_text(build_messages({"x": 1}, "Q", self.style), [])
        self.prefix_policy, self.assistant_suffix = assistant_prefix_policy(probe)
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype=dtype, device="auto",
                               fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"adapter": f"jevarena_mm.models.servers.readout_remote.{type(self).__name__}",
                                      "multimodal": True, "assistant_prefix": self.prefix_policy, "quantization": None,
                                      "max_slots": self.max_slots, "one_branch_per_forward": True})
        log.info("loaded %s (%s) in %.1fs; %s; %d slots", model_path, self.model_type, time.perf_counter() - t0,
                 self.prefix_policy, self.max_slots)

    # -- hooks per checkpoint
    def _load(self, config, kw):
        import transformers
        return transformers.AutoModelForCausalLM.from_pretrained(self.model_path, attn_implementation="sdpa", **kw)

    def _output_head(self):
        return self.model.get_output_embeddings()

    def _render_text(self, messages: list[dict], media: list) -> str:
        """Chat-rendered prompt; the default is the chat template with image content parts (HF style)."""
        return mmedia.render_prompt(self.processor, self.template_kwargs, messages, media)

    def _inline(self, messages: list[dict], tag) -> str:
        """Checkpoints whose chat template takes text only: each <image:N> becomes the checkpoint's image tag."""
        user = messages[-1]["content"]
        text = PLACEHOLDER.sub(lambda m: tag(int(m.group(1))), user)
        msgs = messages[:-1] + [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.template_kwargs)

    def _forward_one(self, text: str):
        """Logits at the last position for one rendered prompt with the request's images."""
        import torch
        images = self._loaded[0]
        self.tokenizer.padding_side = "left"
        kw = mmedia.processor_kwargs(images, [], []) if images else {}
        inputs = self.processor(text=[text], return_tensors="pt", **kw)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs, **self.forward_extra)
        return out.logits[0, -1]

    # -- the engine contract
    def _render_prompt(self, state: Any, suffix: str) -> str:
        return self._render_text(build_messages(state, suffix, self.style), self._media) + self.assistant_suffix

    def _forward_batch(self, texts: list[str]):
        """Last-position logits for several prompts of one request; one forward each unless an engine batches."""
        import torch
        return torch.stack([self._forward_one(t) for t in texts])

    def _forward_last(self, texts: list[str]):
        import gc
        import torch
        attempts = [lambda: self._forward_batch(texts)]
        if len(texts) > 1:   # a batch that does not fit runs one prompt per forward
            attempts.append(lambda: torch.stack([self._forward_one(t) for t in texts]))
        for attempt in attempts:
            try:
                return attempt()
            except torch.OutOfMemoryError as e:
                message = str(e).split(". ")[0]
            # outside the except block the traceback (and the activations its frames hold) is released; free the cache
            # so the GPU does not stay full and fail every later request too
            gc.collect()
            torch.cuda.empty_cache()
        raise RuntimeError(message)


class MiniCPMOReadoutEngine(RemoteReadoutEngine):
    IMAGE_TAG = "(<image>./</image>)"
    template_kwargs_default = {"enable_thinking": False}

    def _load(self, config, kw):
        import transformers
        for k in ("init_audio", "init_tts"):
            if hasattr(config, k):
                setattr(config, k, False)
        config.init_vision = True
        # the custom PreTrainedModel predates transformers 5's device-map inference (no all_tied_weights_keys): load on
        # the CPU and move the whole model to one GPU, as the card does
        kw = {k: v for k, v in kw.items() if k not in ("device_map", "max_memory")}
        import torch
        import transformers.models.whisper.modeling_whisper as whisper
        if not hasattr(whisper, "WHISPER_ATTENTION_CLASSES"):   # MiniCPM-o 2.6 imports this pre-5.0 table
            whisper.WHISPER_ATTENTION_CLASSES = {k: whisper.WhisperAttention for k in ("eager", "sdpa", "flash_attention_2")}
        return transformers.AutoModel.from_pretrained(self.model_path, config=config, attn_implementation="sdpa", **kw).to(self.device)

    def _render_text(self, messages, media):
        try:
            return self._inline(messages, lambda n: self.IMAGE_TAG)
        except TypeError:   # the 2.6 template takes no enable_thinking
            self.template_kwargs = {}
            return self._inline(messages, lambda n: self.IMAGE_TAG)

    def _forward_one(self, text):
        import torch
        images = list(self._loaded[0])
        inputs = self.processor([text], [images], None, None, max_slice_nums=None, use_image_id=True, return_tensors="pt")
        def dev(x):   # the processor returns nested lists of CPU tensors (per sample, per slice)
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            return [dev(v) for v in x] if isinstance(x, (list, tuple)) else x
        data = {"input_ids": inputs["input_ids"].to(self.device), "image_bound": dev(inputs["image_bound"]),
                "pixel_values": dev(inputs["pixel_values"]), "tgt_sizes": dev(inputs["tgt_sizes"]),
                "audio_features": None, "audio_feature_lens": None, "audio_bounds": None, "spk_bounds": None}
        with torch.inference_mode():
            embeds, _ = self.model.get_vllm_embedding(data)
            out = self.model.llm(inputs_embeds=embeds, attention_mask=inputs["attention_mask"].to(self.device),
                                 use_cache=False, return_dict=True, logits_to_keep=1)
        return out.logits[0, -1]


class Phi4MMReadoutEngine(RemoteReadoutEngine):
    forward_extra = {"use_cache": False, "return_dict": True, "num_logits_to_keep": 1}

    def _load(self, config, kw):
        import glob
        import torch
        from safetensors.torch import load_file
        import transformers
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        # transformers 5 builds models on the meta device, where this checkpoint's audio front end cannot compute its
        # layer sizes (Tensor.item()); build it for real on the CPU and load the shards (LoRA weights included, as the
        # checkpoint stores them under the module names the modeling code creates)
        config._attn_implementation = "sdpa"
        cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], self.model_path)
        if isinstance(getattr(cls, "_tied_weights_keys", None), list):   # pre-5.0 list form; 5.x wants {target: source}
            cls._tied_weights_keys = ({"lm_head.weight": "model.embed_tokens.weight"}
                                      if getattr(config, "tie_word_embeddings", False) else {})
        import sys
        inner = getattr(sys.modules[cls.__module__], "Phi4MMModel", None)
        if inner is not None and not hasattr(inner, "prepare_inputs_for_generation"):
            inner.prepare_inputs_for_generation = lambda self, *a, **k: {}   # peft 0.21 reads it when wrapping; never generated here
        torch.set_default_dtype(self.torch_dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(torch.float32)
        sd = {}
        for f in sorted(glob.glob(f"{self.model_path}/*.safetensors")):
            sd.update(load_file(f))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "inv_freq" not in k]
        if missing == ["lm_head.weight"]:   # tied to the input embeddings (not stored in the shards)
            model.lm_head.weight = model.get_input_embeddings().weight
            missing = []
        log.info("phi4mm weights: %d loaded, %d missing, %d unexpected", len(sd), len(missing), len(unexpected))
        if missing:
            raise RuntimeError(f"phi4mm: missing weights, e.g. {missing[:5]}")
        del sd
        return model.to(self.torch_dtype).to(self.device)

    def _render_text(self, messages, media):
        return self._inline(messages, lambda n: f"<|image_{n}|>")

    def _forward_one(self, text):
        import torch
        images = list(self._loaded[0])
        inputs = self.processor(text=text, images=images or None, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs, **self.forward_extra)
        return out.logits[0, -1]


def patch_nemotron_mamba(lm_mod, model=None):
    """Run the remote NemotronH Mamba2 mixers through transformers' own NemotronH mixer forward.

    Without the mamba-ssm / causal-conv1d kernels the remote code falls back to a "naive" SSD whose inter-chunk state
    recurrence sums over the wrong chunk index: outputs change with the length of the sequence after them (not causal)
    once a sequence spans more than one 128-token chunk, which every image prompt does; it also materialises 6-D
    float32 tensors that run out of memory on long states. transformers' mixer has the same parameters and computes the
    chunked scan with einsums; it differs only in reading `time_step_limit` as (time_step_min, inf), the clamp the remote
    fallback applies."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer as Native
    cls = lm_mod.NemotronHMamba2Mixer
    if not getattr(cls, "_jev_native", False):
        cls.forward = Native.forward
        cls._jev_native = True
    n = 0
    for m in (model.modules() if model is not None else ()):
        if isinstance(m, cls):
            m.time_step_limit = (m.time_step_min, float("inf"))
            # a device map's accelerate hook keeps the forward bound at load time on the instance
            setattr(m, "_old_forward" if hasattr(m, "_old_forward") else "forward", Native.forward.__get__(m, cls))
            n += 1
    if model is not None and not n:
        raise RuntimeError("no NemotronH Mamba2 mixer found to patch")


class NemotronOmniReadoutEngine(RemoteReadoutEngine):
    template_kwargs_default = {"enable_thinking": False}

    def _load(self, config, kw):
        # the wrapper class does not declare SDPA support although its language model (NemotronH) does and its vision
        # tower only branches on flash-attn; the config default is flash_attention_2, which is not installed here
        import transformers
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], self.model_path)
        cls._supports_sdpa = True
        config._attn_implementation = "sdpa"
        config.llm_config._attn_implementation = "sdpa"
        mm = kw.get("max_memory")
        if mm and len(mm) > 1:
            # the stock "auto" map puts the vision tower and projector after the language model (last GPU) while the
            # forward splices their output into the embeddings on GPU 0 with GPU-0 indices: keep everything but the
            # decoder blocks / final norm / lm_head on GPU 0 (budget lowered by 4 GiB for them)
            import torch
            from accelerate import infer_auto_device_map, init_empty_weights
            with init_empty_weights():
                meta = cls(config)
            gib = lambda v: float(str(v).rstrip("GiB")) if isinstance(v, str) else v / 2**30
            budget = {k: (f"{gib(v) - 4:.0f}GiB" if str(k) == "0" else v) for k, v in mm.items()}
            dm = infer_auto_device_map(meta, max_memory=budget, no_split_module_classes=list(meta._no_split_modules),
                                       dtype=self.torch_dtype)
            del meta
            first = min(dm.values(), key=lambda d: str(d))
            for k in list(dm):
                if not k.startswith("language_model"):
                    dm[k] = first
            kw = {**kw, "device_map": dm}
            kw.pop("max_memory", None)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, config=config, attn_implementation="sdpa", **kw)
        # its NemotronH code calls the pre-5.x masking helper (input_embeds=, cache_position=); translate the call
        import sys
        lm_mod = sys.modules[type(model.language_model).__module__]
        if getattr(lm_mod, "create_causal_mask", None) is not None and not getattr(lm_mod.create_causal_mask, "_jev_shim", False):
            from transformers.masking_utils import create_causal_mask as new_mask

            def create_causal_mask(*a, input_embeds=None, cache_position=None, **k):
                if input_embeds is not None:
                    k["inputs_embeds"] = input_embeds
                return new_mask(*a, **k)
            create_causal_mask._jev_shim = True
            lm_mod.create_causal_mask = create_causal_mask
        patch_nemotron_mamba(lm_mod, model)
        return model

    def _output_head(self):
        return self.model.language_model.lm_head

    def _forward_one(self, text):
        return self._forward_batch([text])[0]

    def _forward_batch(self, texts):
        # The steps of the checkpoint's forward (embed, ViT per image, replace the <image> context tokens, run the
        # language model), done here because images resized to different dynamic resolutions come back as a list that
        # its forward cannot take (it reads pixel_values.shape, and extract_feature concatenates the per-image outputs
        # along the batch dimension, which fails when their token counts differ). Each prompt is processed with exactly
        # its request's images (the processor sizes every image from the number of images in the call); the prompts of
        # one request share them, so the ViT runs once and its tokens go into every row. Rows are left-padded: NemotronH
        # attention has no positional encoding and its Mamba2 mixer masks padded positions, so a row's last-position
        # logits do not depend on the padding. One forward per batch matters here: each of the 23 MoE layers loops over
        # its experts in Python.
        import torch
        images = self._loaded[0]
        if not images:
            raise ValueError("Nemotron-3-Nano-Omni forward needs pixel_values; this benchmark always sends images")
        rows = [self.processor(text=[t], images=list(images), return_tensors=None)["input_ids"][0] for t in texts]
        pv = self.processor(text=[texts[0]], images=list(images), return_tensors=None)["pixel_values"]
        pv = [t.unsqueeze(0) for t in pv] if isinstance(pv, (list, tuple)) else list(pv.split(1))
        with torch.inference_mode():
            vit = torch.cat([self.model._extract_feature_single(t.to(self.device)).reshape(-1, self.model.config.llm_config.hidden_size)
                             for t in pv])
        # the engine sizes batches with an estimate of 1,500 tokens per image; Nemotron's dynamic resolution gives up
        # to 3,328, so split by the real lengths
        groups, i = [], 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and max(map(len, rows[i:j + 1])) * (j + 1 - i) <= self.batch_budget:
                j += 1
            groups.append(rows[i:j])
            i = j
        return torch.cat([self._lm_rows(g, vit) for g in groups])

    batch_budget = 16_384   # padded tokens per language-model forward

    def _lm_rows(self, rows, vit):
        """Last-position logits for token rows sharing `vit`; a batch that runs out of memory is split in halves."""
        import gc
        import torch
        try:
            return self._lm_forward(rows, vit)
        except torch.OutOfMemoryError:
            if len(rows) == 1:
                raise
        gc.collect()
        torch.cuda.empty_cache()
        h = len(rows) // 2
        return torch.cat([self._lm_rows(rows[:h], vit), self._lm_rows(rows[h:], vit)])

    def _lm_forward(self, rows, vit):
        import torch
        width = max(map(len, rows))
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        ids = torch.tensor([[pad] * (width - len(r)) + list(r) for r in rows], device=self.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=self.device)
        with torch.inference_mode():
            emb = self.model.language_model.get_input_embeddings()(ids)
            selected = ids == self.model.img_context_token_id
            counts = selected.sum(1).tolist()
            if any(c != vit.shape[0] for c in counts):
                raise ValueError(f"{vit.shape[0]} vision tokens for {counts} image positions")
            emb[selected] = vit.to(emb.device, emb.dtype).repeat(len(rows), 1)
            out = self.model.language_model(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True,
                                            logits_to_keep=1)
        return out.logits[:, -1]


ENGINES = {"minicpmo": MiniCPMOReadoutEngine, "phi4mm": Phi4MMReadoutEngine,
           "NemotronH_Nano_Omni_Reasoning_V3": NemotronOmniReadoutEngine}
