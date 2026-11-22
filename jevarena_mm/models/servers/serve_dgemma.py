"""DiffusionGemma through vLLM with the read of the dgemma: adapter, behind the same qev System One API (the qev_dgemma
specs of models/qev.py).

The cyankiwi AWQ-INT4 checkpoint loads only in vLLM, so serve_any.py's DiffusionReadoutEngine cannot serve it. This
server serves it, and the BF16 checkpoint, with the read of dgemma: (models/diffusion_gemma.py, which follows vLLM's
structured reads for DiffusionGemma, vllm-project/vllm#57250; vLLM 0.31 or later) in the image pipeline
(docs/models_img.md):

  steps 1-2 (as for every image model): qev's request, qev's prompt text (system prompt, <state> block,
           question, lettered options, answer line) in the checkpoint's chat template with each image part at its
           placeholder, and qev's labels (A-Z, then the two-letter labels: 256 slots for the Gemma tokenizer). qev's
           DecisionEngine.decide turns the label scores into answers (softmax, debias, chunking), as for the others.
  step 3   dgemma:'s read (layout=prompt, steps=1, reads=1, offset=-1): the prompt ends with the thought
           channel's open tag; the 16-token canvas holds its close tag, the answer slot, the end-of-turn token and
           padding; the slot starts as a random token (random.Random(f"{sample}-{read}"), sample 0, read 0), every other
           position is the template; one read-only denoising pass returns the log-probabilities of the label tokens; in
           vLLM's output the row that predicts canvas position p is row p - 1 (the model keeps the autoregressive
           shift), so the read takes the row before the slot. Label sets above 128 ids (vLLM's limit per request) are
           read in chunks of the same canvas, which give the same distribution. A label missing from a returned row
           scores 5 below the row's lowest value, as in dgemma:.

The qev_diffusion spec (serve_any.py, DiffusionReadoutEngine) reads the BF16 checkpoint differently: its prompt ends
with the whole empty thought block, and it runs the model's own sampler for several steps and reads canvas position 0
on the last one. vLLM's output has no row before canvas position 0, so that read cannot be reproduced here.

    python -m jevarena_mm.models.servers.serve_dgemma --model <path> --port 8012 --name diffusiongemma-26B-A4B-it-AWQ-INT4
           [--gpu 0.85] [--tp 1] [--maxlen N] [--max-images 32] [--meta run_metadata.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from qev import media as mmedia
from qev.calibrate import Calibration
from qev.engine import BranchResult, EngineInfo
from qev.prompt import DEFAULT_STYLE, PromptStyle, RenderedQuestion, build_messages

from jevarena_mm.models.servers.readout_engines import HFReadoutEngine

log = logging.getLogger("jevarena_mm.dgemma")

# the read of dgemma: (models/diffusion_gemma.py, with MAX_TOKEN_IDS from models/local_lm.py) at its defaults
CANVAS = 16  # served canvas length: the answer template of one question, the turn close and padding
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"
TURN_CLOSE_TEXT = "<turn|>"  # DiffusionGemma's end-of-turn token
MAX_TOKEN_IDS = 128  # vLLM's limit on logprob_token_ids per request
STEPS, READS, OFFSET, LAYOUT = 1, 1, -1, "prompt"
SAMPLE = 0  # k = 1: every image run is sample 0


class DGemmaVLLMEngine(HFReadoutEngine):
    """qev DecisionEngine whose branches are read by vLLM's DiffusionGemma (dgemma: layout=prompt, one read-only step,
    the row before the slot). Rendering, media loading and decide() are HFReadoutEngine's; only loading and the
    forward differ."""

    def __init__(self, model_path: str, gpu: float | None = 0.85, tp: int = 1, maxlen: int | None = None,
                 max_images: int = 32, calibration=None, style: PromptStyle = DEFAULT_STYLE,
                 max_branch_tokens: int = 32768, max_request_tokens: int = 1 << 20) -> None:
        import torch
        import transformers
        import vllm
        from vllm import LLM

        t0 = time.perf_counter()
        self.model_path = model_path
        self.calibration = calibration or Calibration()
        self.max_branch_tokens, self.max_request_tokens = max_branch_tokens, max_request_tokens
        self.style = style
        self._lock = threading.Lock()
        self._media, self._loaded = [], ([], [], [])
        config = json.loads((Path(model_path) / "config.json").read_text())
        self.model_type = config["model_type"]
        if self.model_type != "diffusion_gemma":
            raise ValueError(f"{model_path} is a {self.model_type}, not a DiffusionGemma checkpoint")
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = self.processor.chat_template
        self.template_kwargs = {}  # as DiffusionReadoutEngine (no enable_thinking for non-Qwen model types)
        self.slot_ids = self._verify_slots()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.quant_note = None
        qc = config.get("quantization_config")
        if qc:
            self.quant_note = f"{qc.get('quant_method')} {qc.get('format', '')} (checkpoint's own config; vLLM)"
        # vLLM engine as dgemma: starts it (diffusion_gemma.py _load), with room for the image set's largest item
        self.llm = LLM(model=model_path, tensor_parallel_size=tp, max_model_len=maxlen, enable_prefix_caching=True,
                       gpu_memory_utilization=gpu, max_logprobs=64, seed=0, diffusion_config={"canvas_length": CANVAS},
                       limit_mm_per_prompt={"image": max_images})
        tok = self.llm.get_tokenizer()
        self._vocab = len(tok)
        self._pad = tok.pad_token_id or 0
        self._head = self._encode(THOUGHT_CLOSE)  # layout=prompt: the canvas starts with the close tag
        self._turn_close = self._encode(TURN_CLOSE_TEXT)
        if len(self._turn_close) != 1:
            raise ValueError(f"{TURN_CLOSE_TEXT!r} is not one token in {model_path}")
        self._slot = len(self._head)
        self._noise = random.Random(f"{SAMPLE}-0").randrange(self._vocab)  # the slot's start token (read 0)
        self.assistant_suffix = THOUGHT_OPEN  # (used through _render_prompt below)
        plain = mmedia.render_prompt(self.processor, self.template_kwargs, build_messages({"x": 1}, "Q", self.style), [])
        self.template_closes_thought = plain.endswith(THOUGHT_OPEN + THOUGHT_CLOSE)
        self.prefix_policy = ("thought channel opened in the prompt, closed on the canvas (dgemma: layout=prompt); "
                              "one read-only step, the row before the slot"
                              + ("; the template's own empty thought block taken off" if self.template_closes_thought else ""))
        probe = self._render_prompt({"x": 1}, "Q")
        if not probe.endswith("<|turn>model\n" + THOUGHT_OPEN) or probe.count(THOUGHT_CLOSE):
            raise ValueError(f"unexpected prompt end {probe[-80:]!r}")
        if self._encode(probe) != self._encode(probe[:-len(THOUGHT_OPEN)]) + self._encode(THOUGHT_OPEN):
            raise ValueError("the thought open tag does not tokenize apart from the prompt (dgemma: encodes them apart)")
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype=str(config.get("dtype")),
                               device=f"vllm tp={tp}", fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"adapter": "jevarena_mm.models.servers.serve_dgemma.DGemmaVLLMEngine",
                                      "multimodal": True, "vllm": vllm.__version__, "assistant_prefix": self.prefix_policy,
                                      "quantization": self.quant_note, "max_slots": self.max_slots,
                                      "read": {"canvas": CANVAS, "layout": LAYOUT, "steps": STEPS, "reads": READS,
                                               "offset": OFFSET, "slot": self._slot, "noise_token": self._noise,
                                               "template_closes_thought": self.template_closes_thought}})
        log.info("loaded %s (%s) in %.1fs; %s; %d slots", model_path, self.model_type, time.perf_counter() - t0,
                 self.prefix_policy, self.max_slots)

    def _render_prompt(self, state: Any, suffix: str) -> str:
        """qev's text in the checkpoint's chat template, ending with the thought channel's open tag (layout=prompt).
        Some DiffusionGemma templates close an empty thought block themselves when thinking is off (the cyankiwi
        AWQ-INT4 checkpoint's does; google's BF16 snapshot f7f5b7f5 ends at `<|turn>model\\n`): that block is taken off
        first, so every checkpoint gets the same prompt and the canvas holds the only close tag."""
        text = mmedia.render_prompt(self.processor, self.template_kwargs, build_messages(state, suffix, self.style),
                                    self._media)
        if text.endswith(THOUGHT_OPEN + THOUGHT_CLOSE):
            text = text[:-len(THOUGHT_OPEN + THOUGHT_CLOSE)]
        return text + THOUGHT_OPEN

    def _canvas(self, first_label: int) -> list[int]:
        template = self._head + [first_label] + self._turn_close
        canvas = template + [self._pad] * (CANVAS - len(template))
        canvas[self._slot] = self._noise
        return canvas

    def _run_sequential(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        """Every branch of the request in one vLLM call (prefix caching shares the state block)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        images = list(self._loaded[0])
        if self._loaded[1]:
            raise ValueError("videos are not served by this engine")
        prompts, params, owners, lens = [], [], [], []
        for b, r in enumerate(rendered):
            ids = self._encode(self._render_prompt(state, r.suffix))
            lens.append(len(ids))
            labels = self.slot_ids[:r.n_slots]
            extra = {"diffusion_seed_canvas": self._canvas(labels[0]), "diffusion_canvas_length": CANVAS,
                     "diffusion_max_steps": STEPS, "diffusion_read_only": True}
            for k in range(0, len(labels), MAX_TOKEN_IDS):
                prompts.append(TokensPrompt(prompt_token_ids=ids, multi_modal_data={"image": images}) if images
                               else TokensPrompt(prompt_token_ids=ids))
                params.append(SamplingParams(max_tokens=CANVAS, logprob_token_ids=labels[k:k + MAX_TOKEN_IDS],
                                             extra_args=extra))
                owners.append(b)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        got: list[dict[int, float]] = [{} for _ in rendered]
        self.last_prompt_ids = {}  # the prompt as vLLM ran it (image tokens expanded), for checks
        for b, out in zip(owners, outputs):
            row = out.outputs[0].logprobs[self._slot + OFFSET]
            got[b].update({t: lp.logprob for t, lp in row.items()})
            self.last_prompt_ids[b] = list(out.prompt_token_ids)
        results = []
        for b, r in enumerate(rendered):
            values = got[b]
            floor = min(values.values()) - 5.0
            logps = [values.get(i, floor) for i in self.slot_ids[:r.n_slots]]
            results.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logps,
                                        sum(math.exp(x) for x in logps), len(self.last_prompt_ids.get(b, ())) or lens[b]))
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--name", required=True)
    p.add_argument("--gpu", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--maxlen", type=int, default=None)
    p.add_argument("--max-images", type=int, default=32)
    p.add_argument("--meta", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    import transformers
    import uvicorn
    import vllm
    from qev.server import create_app

    engine = DGemmaVLLMEngine(args.model, gpu=args.gpu, tp=args.tp, maxlen=args.maxlen, max_images=args.max_images)
    if args.meta:
        meta = {"server_command": " ".join(sys.argv), "model_name": args.name, "model_path": args.model,
                "dtype": engine.info.dtype, "quantize": engine.quant_note, "backend": f"vllm {vllm.__version__} (dgemma: read)",
                "assistant_prefix": engine.prefix_policy, "max_slots": engine.max_slots, "read": engine.info.extra["read"],
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__,
                "calibration": "none (temperature 1.0)", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta).write_text(json.dumps(meta, indent=1))
    app = create_app(engine, model_name=args.name)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
