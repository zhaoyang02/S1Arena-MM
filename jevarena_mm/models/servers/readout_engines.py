"""Model adapters: any vision-language checkpoint behind the same qev System One API.

The benchmark request (state / media / questions) and qev's rendering of it (qev.prompt: system prompt, <state> block,
Question / lettered Options / answer line) are identical for every model. An adapter supplies only what is
model-specific:

  1. load     - checkpoint, processor, quantization (GPTQ / AWQ from the checkpoint's own config; FP8 checkpoints are
                dequantized to BF16 on GPUs without FP8 matmul, e.g. A100 sm_80);
  2. wrap     - the checkpoint's own chat template around qev's system + user text, its own vision tokens, and the
                assistant prefix: a think block is closed empty ("<think>\\n\\n</think>\\n\\n", as OneJev's template does with
                thinking off) when the template opens one, and left out when the template has none;
  3. read out - the next-token logits of the option letters at the answer position (nothing is sampled); qev's own
                DecisionEngine.decide turns them into answers (softmax, debias, >max_slots chunking), so the response
                JSON is the same as from the OneJev server.

Backends (see BACKENDS / backend_for):
  qev-native       qwen3_5, qwen3_5_moe   qev's MMDecisionEngine via serve.py (every OneJev-27B run used this)
  qev-gguf         *.gguf                 qev's LlamaCppEngine (llama-server + mmproj)
  hf               any other VLM          HFReadoutEngine below (AutoModelForImageTextToText, or the Omni thinker)
  none             no vision input        e.g. Qwen2-Audio: cannot take an image
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from qev import media as mmedia
from qev.engine import BranchResult, DecisionEngine, EngineInfo
from qev.prompt import DEFAULT_STYLE, PromptStyle, RenderedQuestion, build_messages
from qev.schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("jevarena_mm.readout")

OMNI = {"qwen2_5_omni": "Qwen2_5OmniForConditionalGeneration", "qwen3_omni_moe": "Qwen3OmniMoeForConditionalGeneration"}
NATIVE = {"qwen3_5", "qwen3_5_moe"}
NO_VISION = {"qwen2_audio"}
THINK_OPEN, THINK_EMPTY = "<think>\n", "<think>\n\n</think>\n\n"


def backend_for(model_type: str, path: str = "") -> str:
    if path.endswith(".gguf") or "GGUF" in path:
        return "qev-gguf"
    if model_type in NATIVE:
        return "qev-native"
    if model_type in NO_VISION:
        return "none"
    if model_type == "diffusion_gemma":
        return "block-diffusion"
    return "hf"


def assistant_prefix_policy(rendered: str) -> tuple[str, str]:
    """How the template ends the prompt, and what the adapter appends so that the answer letter comes next."""
    tail = rendered[-40:]
    if rendered.endswith(THINK_EMPTY) or rendered.rstrip().endswith("</think>"):
        return "template closes the think block (thinking off)", ""
    if rendered.endswith(THINK_OPEN) or rendered.endswith("<think>"):
        return "template opens a think block; adapter closes it empty", ("\n" if rendered.endswith("<think>") else "") + "\n</think>\n\n"
    return f"no think block in the template (prompt ends {tail[-24:]!r})", ""


class HFReadoutEngine(DecisionEngine):
    """qev DecisionEngine for any transformers VLM: one full forward per question branch (exact; no cache forking),
    branches of a request batched with left padding when `batched` is set and verified against sequential at load."""

    def __init__(self, model_path: str, dtype: str = "bfloat16", max_memory: dict | None = None,
                 calibration=None, style: PromptStyle = DEFAULT_STYLE, max_branch_tokens: int = 32768,
                 max_request_tokens: int = 1 << 20, batch_tokens: int = 24_000, template_kwargs: dict | None = None,
                 model_type_override: str | None = None, head_dtype: str = "float32") -> None:
        import torch
        import transformers
        from qev.calibrate import Calibration

        self.model_path = model_path
        self.device = torch.device("cuda:0")
        self.torch_dtype = getattr(torch, dtype)
        self.calibration = calibration or Calibration()
        self.max_branch_tokens, self.max_request_tokens = max_branch_tokens, max_request_tokens
        self.fork_token_budget, self.fork_tolerance = batch_tokens, 5e-2
        self.batch_tokens = batch_tokens
        self.style = style
        self.head_dtype = head_dtype
        self._lock = threading.Lock()
        self._want_mass = False
        self._media, self._loaded = [], ([], [], [])
        t0 = time.perf_counter()
        config = transformers.AutoConfig.from_pretrained(model_path)
        self.model_type = model_type_override or config.model_type
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = self.processor.chat_template
        kw: dict[str, Any] = dict(dtype=self.torch_dtype, device_map="auto", low_cpu_mem_usage=True)
        if max_memory:
            kw["max_memory"] = max_memory
        self.quant_note = None
        qc = getattr(config, "quantization_config", None)
        qc = qc if isinstance(qc, dict) else (qc.to_dict() if qc is not None else None)
        if qc and qc.get("quant_method") == "fp8" and torch.cuda.get_device_capability() < (8, 9):
            kw["quantization_config"] = transformers.FineGrainedFP8Config(
                dequantize=True, weight_block_size=tuple(qc.get("weight_block_size") or (128, 128)))
            self.quant_note = "FP8 weights dequantized to BF16 (no FP8 matmul on this GPU)"
        elif qc:
            self.quant_note = f"{qc.get('quant_method')} {qc.get('bits', '')}-bit weights (checkpoint's own config)"
            skip = qc.get("modules_to_not_convert") or []
            if qc.get("quant_method") == "awq" and self.model_type in OMNI and skip:
                # Omni checkpoints list unquantized modules without the "thinker." prefix of the loaded module names
                # (a quantization_config passed to from_pretrained only overrides loading attributes, so edit the config)
                # the audio tower is stored unquantized as well (not listed); it is unused here but should load as stored
                config.quantization_config = {**qc, "modules_to_not_convert": skip + [f"thinker.{m}" for m in skip] + ["thinker.audio_tower"]}
                kw["config"] = config
            if qc.get("quant_method") == "gptq" and not qc.get("block_name_to_quantize"):
                # optimum cannot find the decoder blocks of multimodal checkpoints by itself
                blocks = {"qwen2_5_omni": "thinker.model.layers", "qwen3_omni_moe": "thinker.model.layers",
                          "qwen3_5_moe": "model.language_model.layers"}.get(self.model_type)
                if blocks:
                    config.quantization_config = {**qc, "block_name_to_quantize": blocks}
                    kw["config"] = config
        if qc and qc.get("quant_method") == "gptq" and self.model_type not in OMNI:
            # MoE GPTQ checkpoints load through gptqmodel itself: it defuses the experts and applies the checkpoint's
            # `dynamic` include/exclude rules, which the generic transformers/optimum path does not (Omni GPTQ checkpoints
            # load through transformers + optimum below, with the decoder block name supplied)
            from gptqmodel import GPTQModel
            from gptqmodel.models.auto import MODEL_MAP
            qcls = MODEL_MAP.get(self.model_type)
            if qcls is not None and qcls.require_pkgs:  # Omni definitions demand audio packages; images only here
                qcls.require_pkgs = [r for r in qcls.require_pkgs if not r.startswith(("audioread", "librosa", "av"))]
            gm = GPTQModel.load(model_path, device_map="auto", **({"max_memory": max_memory} if max_memory else {}))
            full = gm.model
            if hasattr(full, "disable_talker"):
                full.disable_talker()
            self.model = getattr(full, "thinker", full)
            self.quant_note += "; loaded with gptqmodel"
        elif self.model_type in OMNI:
            # text output only: the talker / code2wav are never built (their weights are skipped on load)
            config.enable_audio_output = False
            kw["config"] = config
            full = getattr(transformers, OMNI[self.model_type]).from_pretrained(model_path, **kw)
            if hasattr(full, "disable_talker"):
                full.disable_talker()
            self.model = full.thinker
        else:
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(model_path, **kw)
        self.model.eval()
        # only the last position is read: slice before the output head (not every model honours logits_to_keep, and
        # full-sequence logits over a 150k+ vocabulary are several GB for a batch of image prompts)
        if self.model_type != "diffusion_gemma":  # the diffusion readout needs the canvas logits (DiffusionReadoutEngine)
            self.model.get_output_embeddings().register_forward_pre_hook(lambda m, args: (args[0][:, -1:, :],) + tuple(args[1:]))
        self.template_kwargs = dict(template_kwargs if template_kwargs is not None else
                                    ({"enable_thinking": False} if "qwen" in self.model_type else {}))
        self.slot_ids = self._verify_slots()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        probe = mmedia.render_prompt(self.processor, self.template_kwargs, build_messages({"x": 1}, "Q"), [])
        self.prefix_policy, self.assistant_suffix = assistant_prefix_policy(probe)
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype=dtype, device="auto",
                               fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"adapter": "jevarena_mm.models.servers.readout_engines.HFReadoutEngine", "multimodal": True,
                                      "assistant_prefix": self.prefix_policy, "quantization": self.quant_note,
                                      "max_slots": self.max_slots})
        log.info("loaded %s (%s) in %.1fs; %s; %d slots", model_path, self.model_type, time.perf_counter() - t0,
                 self.prefix_policy, self.max_slots)

    # -- rendering: qev's messages, the checkpoint's chat template, the checkpoint's vision tokens
    def _render_prompt(self, state: Any, suffix: str) -> str:
        text = mmedia.render_prompt(self.processor, self.template_kwargs, build_messages(state, suffix, self.style),
                                    self._media)
        return text + self.assistant_suffix

    def _inputs(self, texts: list[str]):
        images, videos, metas = self._loaded
        kw = mmedia.processor_kwargs(images * len(texts), videos * len(texts), metas * len(texts))
        if images and "gemma" in self.model_type:
            # the Gemma processors read a flat image list as ONE sample with several images (make_nested_list_of_images),
            # so a batch of branches needs one image list per text
            kw["images"] = [list(images) for _ in texts]
        self.tokenizer.padding_side = "left"
        out = self.processor(text=texts, return_tensors="pt", padding=True, **kw)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in out.items()}

    def _forward_last(self, texts: list[str]):
        import torch
        with torch.inference_mode():
            out = self.model(**self._inputs(texts), use_cache=False, return_dict=True, logits_to_keep=1)
        return out.logits[:, -1]

    def _run_sequential(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        """Branches in batches of at most `batch_tokens` tokens (left-padded, last position read)."""
        texts = [self._render_prompt(state, r.suffix) for r in rendered]
        lens = [len(self._encode(t)) + self._vision_tokens() for t in texts]
        results, start = [], 0
        while start < len(texts):
            end = start + 1
            while end < len(texts) and max(lens[start:end + 1]) * (end + 1 - start) <= self.batch_tokens:
                end += 1
            logits = self._forward_last(texts[start:end])
            for row, r in enumerate(rendered[start:end]):
                sel, mass = self._readout(logits[row], r.n_slots)
                results.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, sel, mass, lens[start + row]))
            start = end
        return results

    def _vision_tokens(self) -> int:
        return 1500 * len(self._loaded[0])  # budget estimate only (batch sizing); exact counts come from the processor

    def _readout(self, vocab_logits, n_slots: int):
        import torch
        v = vocab_logits.float()
        sel = v[torch.tensor(self.slot_ids[:n_slots], device=v.device)]
        mass = float((torch.logsumexp(sel, -1) - torch.logsumexp(v, -1)).exp())
        return sel.tolist(), mass

    def decide(self, request: SystemOneRequest, debias: int = 1, debug: bool = False,
               media: list | None = None) -> tuple[SystemOneResponse, dict[str, Any]]:
        items = mmedia.check_items(media if media is not None else getattr(request, "media", None))
        with self._lock:
            t0 = time.perf_counter()
            self._media = items
            self._loaded = mmedia.load(items) if items else ([], [], [])
            t_media = time.perf_counter() - t0
            try:
                response, meta = super().decide(request, debias=debias, debug=debug)
            finally:
                self._media, self._loaded = [], ([], [], [])
        meta["media"] = {"images": sum(m["type"] == "image" for m in items), "load_s": t_media}
        meta["adapter"] = {"assistant_prefix": self.prefix_policy, "quantization": self.quant_note}
        return response, meta


class DiffusionReadoutEngine(HFReadoutEngine):
    """Block-diffusion LMs (DiffusionGemma): there is no next-token step. The prompt (same qev text, the checkpoint's
    chat template) is encoded, and the model's own sampler denoises the first canvas (random-token start, entropy-bound
    acceptance, adaptive stopping; fixed seed). The answer is the option-letter part of the denoiser's raw logits at
    canvas position 0 on the last denoising step, i.e. the first answer token once the canvas has converged."""

    seed = 20260928
    # DiffusionGemma opens every answer with an empty thought channel (probability ~1.0 on all four positions in the
    # format check, e.g. "<|channel>thought\n<channel|>A<turn|>"); it goes into the assistant prefix, as an empty
    # <think></think> does for the Qwen models, so that canvas position 0 is the answer letter
    EMPTY_THOUGHT = "<|channel>thought\n<channel|>"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.assistant_suffix = self.EMPTY_THOUGHT
        self.prefix_policy = "empty thought channel closed by the adapter (the model emits it before every answer)"
        self.info.extra["assistant_prefix"] = self.prefix_policy

    def _forward_last(self, texts: list[str]):
        import torch
        rows = []
        for t in texts:
            inputs = self._inputs([t])
            seen = []
            orig = self.model.forward

            def recording_forward(*a, **k):
                out = orig(*a, **k)
                if getattr(out, "logits", None) is not None and k.get("decoder_input_ids") is not None:
                    seen.append(out.logits[:, 0].float())
                return out

            self.model.forward = recording_forward
            try:
                torch.manual_seed(self.seed)
                with torch.inference_mode():
                    self.model.generate(**inputs, max_new_tokens=int(self.model.config.canvas_length))
            finally:
                del self.model.forward
            rows.append(seen[-1][0])
        return torch.stack(rows)
