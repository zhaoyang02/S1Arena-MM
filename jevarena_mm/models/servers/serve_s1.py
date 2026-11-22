"""Serve a System-One release (Jev-Omni, Clef, Xor, Raya, d3) behind the qev /v1/systemone API, in the model's own format.

Each engine reproduces the release's own calling code (its model card / bundled wrapper), so a run through this server
sends the model what its authors send it; only the transport is qev's. The response is qev's SystemOneResponse, so the
`s1_*` adapters in models/qev.py read it like every other model.

  jevomni  akhilaaa3/Jev-Omni (+ a replacement decision head, e.g. ferdinandl007/jev-omni-general-head-v3)
           jev_omni.py: one forward per question; prompt "<state>\n\n---\n\nQUESTION: ..\n\nOPTIONS:\n1. ..\n2. ..";
           images go first in the user turn; a 256-slot head on the last hidden state, softmax over the options.
           noul -> options ["Yes", "No"] (criteria texts appended), p_true = P(Yes); choice -> "key" or "key: text";
           score -> the level texts in order.
  clef     Cloudflare/clef, clef-flash, clef-omni: joint_schema_model.systemone(request) verbatim (images before the
           state, state as compact sorted JSON, choice keys sorted, all questions in one forward, probabilities
           rounded to 4 decimals). Loaded sharded over the given GPUs.
  xor      juspay/Xor-26B-A4B: serving/xn_base_server.py re-implemented on transformers instead of SGLang: rows
           "Context:\n<state>\n\nQuestion: ..\nOptions:\n(A) ..\n(B) .." in the chat template, label logprobs after
           "Answer: (", temperatures choice 2.0 / noul 1.7 / score 1.0 and two-order averaging (RELEASE_MANIFEST).
  raya     yuyu199741/raya-decision-v1: the card's masked projection: "<images>\n\n<state>\n\n<question>\n\nA. ..\n
           B. ..\n\nAnswer with exactly one label from the options above."; logits of " A".." Z" at the last position.
           Choices above 26 options are asked in chunks of 25 plus "none of the options listed here", as qev does.

  d3       vllm-sr/d3 (Decision 3.0, Qwen3.8-27B): its d3_runtime.D3.system_one(state, questions, images) verbatim
           (the AutoModel(trust_remote_code) path of its card): per question one prompt "State: / Question: / Options:
           <code>: <text>", 255 answer codes read from the last hidden state with the checkpoint's readout matrix,
           noncausal full attention over the prompt, 8 questions per forward, images first at up to 1.6 MP.
           The release loader takes one device; here the backbone is sharded over the given GPUs.

    python -m jevarena_mm.models.servers.serve_s1 --family clef --model <snapshot> --name clef --port 8300 \
        [--max-memory 0=36GiB,1=36GiB] [--head <decision_head.pt>] [--processor <path>] [--meta run_metadata.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("jevarena_mm.s1")
NONE_OPTION = "none of the options listed here"


def render_entry(v: Any) -> str:
    from qev.prompt import render_entry as r
    return r(v)


def compact_json(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def question_dicts(request) -> dict[str, dict]:
    return {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}


def max_memory_of(spec: str | None) -> dict | None:
    return {int(k): v for k, v in (kv.split("=") for kv in spec.split(","))} if spec else None


class S1Engine:
    """The part of qev.engine.DecisionEngine that qev.server.create_app uses."""

    family = ""

    def __init__(self, model_path: str, name: str, **kw):
        from qev.engine import EngineInfo
        import torch
        import transformers
        self.model_path, self.name = model_path, name
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        self.load(**kw)
        self.info = EngineInfo(model_path=model_path, model_type=self.family, dtype="bfloat16", device="auto",
                               fork_mode="s1", fork_max_abs_diff=None, prompt_version=f"s1-{self.family}",
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"adapter": f"jevarena_mm.models.servers.serve_s1.{type(self).__name__}", **self.extra()})
        log.info("loaded %s (%s) in %.0fs", model_path, self.family, time.perf_counter() - t0)

    def load(self, **kw):
        raise NotImplementedError

    def extra(self) -> dict:
        return {}

    def answer(self, state: Any, questions: dict[str, dict], images: list) -> tuple[dict, int, dict]:
        """-> (answers {qid: Answer}, input tokens, diagnostics)."""
        raise NotImplementedError

    def decide(self, request, debias: int = 1, debug: bool = False):
        from qev import media as mmedia
        from qev.schema import SystemOneResponse, Usage
        items = mmedia.check_items(getattr(request, "media", None))
        if any(m["type"] != "image" for m in items):
            raise ValueError("this server takes images only")
        images = [mmedia.open_image(m) for m in items]
        questions = question_dicts(request)
        with self._lock:
            t0 = time.perf_counter()
            answers, n_in, diag = self.answer(request.state, questions, images)
            dt = time.perf_counter() - t0
        resp = SystemOneResponse(model="", answers=answers, usage=Usage(input_tokens=n_in, output_tokens=len(answers)))
        meta = {"timing": {"total_s": dt}, "media": {"images": len(images)}, "adapter": self.info.extra, "diag": diag}
        return resp, meta


# ----------------------------------------------------------------------------------------------- Jev-Omni


class JevOmniEngine(S1Engine):
    family = "jevomni"

    def load(self, head: str | None = None, **_):
        import torch
        sys.path.insert(0, self.model_path)
        import jev_omni
        jev_omni.snapshot_download = lambda *a, **k: self.model_path
        self.clf = jev_omni.load_jev_omni(self.model_path)
        self.prompt = jev_omni._prompt
        self.head_path = head
        if head:
            self.clf.head.load_state_dict(torch.load(head, map_location=self.clf.device, weights_only=True))
            self.clf.head.eval()

    def extra(self):
        return {"head": self.head_path or "release head.pt", "one_forward_per_question": True}

    def options_of(self, q: dict) -> tuple[list[str], list[str]]:
        """(option texts shown to the model, answer keys in the same order)."""
        t = q["type"]
        if t == "noul":
            c = q.get("criteria") or {}
            yes = "Yes" + (f": {render_entry(c['true'])}" if c.get("true") else "")
            no = "No" + (f": {render_entry(c['false'])}" if c.get("false") else "")
            return [yes, no], ["true", "false"]
        if t == "choice":
            keys = list(q["criteria"])
            return [f"{k}: {render_entry(v)}" if render_entry(v) else k for k, v in q["criteria"].items()], keys
        levels = [render_entry(v) for v in q["criteria"]]
        return levels, [str(i) for i in range(len(levels))]

    def answer(self, state, questions, images):
        import numpy as np
        import torch
        from qev.answers import make_choice, make_noul, make_score
        clf = self.clf
        state_text = compact_json(state)
        answers, n_in = {}, 0
        for qid, q in questions.items():
            opts, keys = self.options_of(q)
            if not 2 <= len(opts) <= 256:
                raise ValueError(f"{qid}: Jev-Omni needs 2-256 options, got {len(opts)}")
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": self.prompt(state_text, render_entry(q.get("instructions")), opts)})
            inputs = clf.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                       tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
            n_in += int(inputs["input_ids"].shape[1])
            inputs = {k: v.to(clf.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(clf.device)
                      for k, v in inputs.items()}
            clf._capture.clear()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                clf.model(**inputs, use_cache=False, **clf._extra)
                probs = clf.head(clf._capture["hidden"], torch.tensor([len(opts)], device=clf.device))[0, :len(opts)].softmax(-1)
            p = probs.float().cpu().tolist()
            if q["type"] == "noul":
                answers[qid] = make_noul(p[0])
            elif q["type"] == "choice":
                answers[qid] = make_choice(keys, p)
            else:
                answers[qid] = make_score([render_entry(v) for v in q["criteria"]], p)
        return answers, n_in, {}


# ----------------------------------------------------------------------------------------------- Clef


class ClefEngine(S1Engine):
    family = "clef"

    def load(self, max_memory: dict | None = None, **_):
        import torch
        from safetensors.torch import load_file
        import transformers
        sys.path.insert(0, self.model_path)
        import joint_schema_model as jsm
        self.jsm = jsm
        path = Path(self.model_path)
        kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        if max_memory:
            kw["max_memory"] = max_memory
        cfg = transformers.AutoConfig.from_pretrained(path)
        self.omni = cfg.model_type == "qwen3_omni_moe"
        if self.omni:
            cfg.enable_audio_output = False
            full = transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained(path, config=cfg, **kw)
            if hasattr(full, "disable_talker"):
                full.disable_talker()
            backbone = full.thinker
        else:
            backbone = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(path, **kw)
        backbone.config.use_cache = False
        head = jsm.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
        head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
        out_dev = backbone.get_output_embeddings().weight.device if not self.omni else backbone.lm_head.weight.device
        head = head.to(device=out_dev, dtype=torch.bfloat16)
        orig = head.forward

        def forward_on_head_device(hidden, input_ids, attention_mask, records, weight):  # sharded backbone
            d = weight.device
            return orig(hidden.to(d), input_ids.to(d), attention_mask.to(d), records, weight)
        head.forward = forward_on_head_device
        self.model = jsm.ClefModel(backbone, head).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(path)
        self.max_length = getattr(jsm, "DEFAULT_MAX_LENGTH", 16384)

    def extra(self):
        return {"joint_forward": True, "max_length": self.max_length, "omni": self.omni}

    def answer(self, state, questions, images):
        from qev.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer
        req = {"model": self.name, "state": state, "questions": questions}
        if images:
            req["images"] = images
        out = self.jsm.systemone(self.model, self.processor, req, max_length=self.max_length)
        answers = {}
        for qid, a in out["answers"].items():
            t = a.pop("type")
            answers[qid] = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}[t](**a)
        return answers, int(out["usage"]["input_tokens"]), {}


# ----------------------------------------------------------------------------------------------- Xor


class XorEngine(S1Engine):
    family = "xor"
    TEMP = {"choice": 2.0, "noul": 1.7, "score": 1.0}   # RELEASE_MANIFEST.json of v1.1
    TWO_ORDER = True
    MAX_OPTIONS = 255
    _S = "@@XN_USER@@"

    def load(self, max_memory: dict | None = None, **_):
        import string
        import torch
        import transformers
        kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        if max_memory:
            kw["max_memory"] = max_memory
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(self.model_path, **kw).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        self.tok = self.processor.tokenizer
        E = self.E
        head, tail = self.tok.apply_chat_template([{"role": "user", "content": self._S}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False).split(self._S)
        self.HEAD, self.TAIL, self.ANSWER = E(head), E(tail), E("Answer: (")
        self.head_text, self.tail_text = head, tail
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        out, seen = [], set()
        for n in names:   # the label = the actual next token after "Answer: (" + name (xn_base_server._labels)
            t = E("Answer: (" + n)
            if len(t) <= len(self.ANSWER) or t[-1] in seen:
                continue
            seen.add(t[-1])
            out.append((n, t[-1]))
            if len(out) == self.MAX_OPTIONS:
                break
        assert len(out) == self.MAX_OPTIONS
        self.LABEL_NAMES, self.LABEL_IDS = [n for n, _ in out], [i for _, i in out]
        self.device0 = self.model.get_input_embeddings().weight.device
        self._media_head: dict[int, Any] = {}
        import inspect
        self._ltk = "logits_to_keep" in inspect.signature(self.model.forward).parameters

    def E(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def extra(self):
        return {"temperatures": self.TEMP, "two_order": self.TWO_ORDER, "layout": "chat/state-first/independent-rows",
                "labels": len(self.LABEL_IDS)}

    @staticmethod
    def _txt(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    def render_question(self, qid: str, q: dict) -> tuple[str, list[str]]:
        t, crit, ins = q["type"], q.get("criteria"), q.get("instructions")
        if t == "choice":
            if len(crit) < 2 or len(crit) > self.MAX_OPTIONS:
                raise ValueError(f"{qid}: {len(crit)} options; Xor takes 2-{self.MAX_OPTIONS}")
            opts = [k if crit[k] in (None, "") else f"{k}: {self._txt(crit[k])}" for k in crit]
            if ins in (None, ""):
                raise ValueError(f"{qid}: choice question without instructions")
            return self._txt(ins), opts
        if t == "noul":
            c = crit if isinstance(crit, dict) else {}
            f, tr = c.get("false"), c.get("true")
            opts = ["no" if f in (None, "") else f"no: {self._txt(f)}", "yes" if tr in (None, "") else f"yes: {self._txt(tr)}"]
            if ins in (None, ""):
                ins = "Which answer fits the context?"
            return self._txt(ins), opts
        if not 2 <= len(crit) <= 10:
            raise ValueError(f"{qid}: a score takes 2 to 10 levels")
        return self._txt(ins), [f"{i}: {self._txt(c)}" for i, c in enumerate(crit)]

    def build_row(self, state_ids: list[int], ins: str, opts: list[str]) -> list[int]:
        ids = self.HEAD + self.E("Context:\n") + state_ids + self.E(f"\n\nQuestion: {ins}\nOptions:")
        for j, o in enumerate(opts):
            ids += self.E("\n(" + self.LABEL_NAMES[j] + ") " + o)
        return ids + self.TAIL + self.ANSWER

    def media_inputs(self, images: list) -> dict:
        """The chat-template head with the images (xn_base_server.media_template_head), run through the processor."""
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": self._S}]
        rendered = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        head, tail = rendered.split(self._S)
        if self.E(tail) != self.TAIL:
            raise ValueError("media template suffix differs from the scoring template")
        enc = self.processor(text=[head], images=images, return_tensors="pt")
        return {k: v for k, v in enc.items()}

    def logprobs(self, rows: list[list[int]], images: list) -> list[list[float]]:
        """log-softmax over the vocabulary at the answer position, for every row (one forward per row with images)."""
        import torch
        out = []
        media = self.media_inputs(images) if images else None
        for row in rows:
            if media is None:
                ids = torch.tensor([row], device=self.device0)
                inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            else:
                n_head = media["input_ids"].shape[1]
                ids = torch.cat([media["input_ids"], torch.tensor([row[len(self.HEAD):]])], dim=1).to(self.device0)
                inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                for k, v in media.items():
                    if k in ("input_ids", "attention_mask"):
                        continue
                    if k.endswith("token_type_ids") and v.ndim == 2 and v.shape[1] == n_head:   # per-token: pad for the text
                        v = torch.cat([v, torch.zeros((1, ids.shape[1] - n_head), dtype=v.dtype)], dim=1)
                    inputs[k] = v.to(self.device0)
            with torch.inference_mode():
                o = self.model(**inputs, use_cache=False, **({"logits_to_keep": 1} if self._ltk else {}))
            out.append(torch.log_softmax(o.logits[0, -1].float(), -1).cpu())
        return out

    @staticmethod
    def softmax_T(lp, T):
        m = max(lp)
        e = [math.exp((x - m) / T) for x in lp]
        s = sum(e)
        return [x / s for x in e]

    def answer(self, state, questions, images):
        from qev.answers import make_choice, make_noul, make_score
        state_ids = self.tok.encode(compact_json(state), add_special_tokens=False) if state is not None else []
        order, rows, sets = [], [], []
        for qid, q in questions.items():
            ins, opts = self.render_question(qid, q)
            rows.append(self.build_row(state_ids, ins, opts))
            sets.append(self.LABEL_IDS[:len(opts)])
            order.append((qid, q["type"], opts))
        n_fwd = len(rows)
        rev = []
        if self.TWO_ORDER:
            for qi, (qid, t, opts) in enumerate(order):
                if t == "choice" and len(opts) >= 2:
                    ins, _ = self.render_question(qid, questions[qid])
                    rows.append(self.build_row(state_ids, ins, opts[::-1]))
                    sets.append(self.LABEL_IDS[:len(opts)])
                    rev.append(qi)
        full = self.logprobs(rows, images)
        lps = [[float(lp[i]) for i in ids] for lp, ids in zip(full, sets)]
        for j, ridx in enumerate(range(n_fwd, len(rows))):
            qi = rev[j]
            K = len(lps[qi])
            p_f, p_r = self.softmax_T(lps[qi], 1.0), self.softmax_T(lps[ridx], 1.0)
            pa = [(p_f[k] + p_r[K - 1 - k]) / 2 for k in range(K)]
            lps[qi] = [math.log(max(p, 1e-12)) for p in pa]
        answers, diag = {}, {}
        for (qid, t, opts), lp in zip(order, lps):
            p = self.softmax_T(lp, self.TEMP[t])
            q = questions[qid]
            if t == "noul":
                answers[qid] = make_noul(p[1])
            elif t == "choice":
                answers[qid] = make_choice(list(q["criteria"]), p)
            else:
                answers[qid] = make_score([self._txt(c) for c in q["criteria"]], p)
            diag[qid] = {"x_lp": [round(x, 4) for x in lp]}
        n_in = sum(len(r) for r in rows[:n_fwd])
        return answers, n_in, diag


# ----------------------------------------------------------------------------------------------- Raya


class RayaEngine(S1Engine):
    family = "raya"
    IMG = "<|vision_start|><|image_pad|><|vision_end|>"
    ANSWER_LINE = "Answer with exactly one label from the options above."

    def load(self, processor: str | None = None, **_):
        import string
        import torch
        import transformers
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
        self.processor = transformers.AutoProcessor.from_pretrained(processor or "Qwen/Qwen3.5-2B")  # card: base repo
        self.tok = self.processor.tokenizer
        self.letters = list(string.ascii_uppercase)
        ids = [self.tok.encode(" " + L, add_special_tokens=False) for L in self.letters]   # card: tokenizer.encode(" A")[0]
        if any(len(i) != 1 for i in ids):
            raise ValueError("a label ' X' is not one token")
        self.label_ids = [i[0] for i in ids]
        self.max_slots = len(self.letters)
        import inspect
        self._ltk = "logits_to_keep" in inspect.signature(self.model.forward).parameters

    def extra(self):
        return {"labels": "' A'..' Z' (masked projection)", "chunking": f"choices above {self.max_slots} options: chunks of {self.max_slots - 1} + '{NONE_OPTION}'"}

    def prompt(self, state, instructions: str, opts: list[str], n_images: int) -> str:
        parts = [self.IMG * n_images] if n_images else []
        parts += [compact_json(state), instructions, "\n".join(f"{self.letters[i]}. {o}" for i, o in enumerate(opts)), self.ANSWER_LINE]
        return "\n\n".join(p for p in parts if p != "")

    def probs(self, text: str, images: list, k: int) -> list[float]:
        import torch
        enc = self.processor(text=[text], images=images or None, return_tensors="pt")
        enc = {kk: v.to("cuda:0") for kk, v in enc.items()}
        with torch.inference_mode():
            o = self.model(**enc, use_cache=False, **({"logits_to_keep": 1} if self._ltk else {}))
        sel = o.logits[0, -1].float()[torch.tensor(self.label_ids[:k], device=o.logits.device)]
        return sel.softmax(-1).cpu().tolist(), int(enc["input_ids"].shape[1])

    def answer(self, state, questions, images):
        from qev.answers import make_choice, make_noul, make_score
        answers, n_in = {}, 0
        for qid, q in questions.items():
            ins = render_entry(q.get("instructions"))
            if q["type"] == "noul":
                c = q.get("criteria") or {}
                opts = ["Yes" + (f": {render_entry(c['true'])}" if c.get("true") else ""),
                        "No" + (f": {render_entry(c['false'])}" if c.get("false") else "")]
                p, n = self.probs(self.prompt(state, ins, opts, len(images)), images, 2)
                n_in += n
                answers[qid] = make_noul(p[0])
            elif q["type"] == "score":
                levels = [render_entry(v) for v in q["criteria"]]
                p, n = self.probs(self.prompt(state, ins, levels, len(images)), images, len(levels))
                n_in += n
                answers[qid] = make_score(levels, p)
            else:
                keys = list(q["criteria"])
                texts = [f"{k}: {render_entry(v)}" if render_entry(v) and render_entry(v) != k else k for k, v in q["criteria"].items()]
                if len(keys) <= self.max_slots:
                    p, n = self.probs(self.prompt(state, ins, texts, len(images)), images, len(keys))
                    n_in += n
                else:   # qev's chunked path: each chunk adds NONE_OPTION; p(opt) = p_in_chunk/keep * keep/sum(keep)
                    size = self.max_slots - 1
                    n_chunks = -(-len(keys) // size)
                    size = -(-len(keys) // n_chunks)
                    scores, keeps = {}, []
                    for s in range(0, len(keys), size):
                        ck, ct = keys[s:s + size], texts[s:s + size] + [NONE_OPTION]
                        pc, n = self.probs(self.prompt(state, ins, ct, len(images)), images, len(ct))
                        n_in += n
                        keep = max(1.0 - pc[-1], 1e-9)
                        keeps.append(keep)
                        for kk, pp in zip(ck, pc):
                            scores[kk] = (pp / keep, keep)
                    tot = sum(keeps) or 1.0
                    p = [scores[kk][0] * (scores[kk][1] / tot) for kk in keys]
                answers[qid] = make_choice(keys, p)
        return answers, n_in, {}



# ----------------------------------------------------------------------------------------------- d3


class D3Engine(S1Engine):
    """vllm-sr/d3 through its own runtime (d3_runtime.D3), the code path of AutoModel(..., trust_remote_code=True)."""

    family = "d3"

    def load(self, max_memory: dict | None = None, **_):
        import torch
        import transformers
        sys.path.insert(0, self.model_path)
        import d3_runtime
        cfg = transformers.AutoConfig.from_pretrained(self.model_path)
        cls = d3_runtime.backbone_class(cfg.model_type)
        orig = cls.from_pretrained.__func__
        if max_memory and len(max_memory) > 1:   # the release loader pins one device; shard over the visible GPUs
            def sharded(klass, path, **kw):
                kw.pop("device_map", None)
                return orig(klass, path, device_map="auto", max_memory=max_memory, **kw)
            cls.from_pretrained = classmethod(sharded)
        try:
            self.rt = d3_runtime.D3.from_pretrained(self.model_path, device="cuda:0")
        finally:
            cls.from_pretrained = classmethod(orig)
        self.sharded = {k: str(v) for k, v in (getattr(self.rt.backbone, "hf_device_map", {}) or {}).items()}
        if len(set(self.sharded.values())) > 1:
            # the runtime keeps its readout matrix and masks on cuda:0; a sharded backbone hands the final hidden
            # states back on the last GPU, so bring them home (its PromptEncoder and the stock forward)
            dev0 = self.rt.device
            enc = self.rt.encoder
            if enc is not None:
                self.rt.encoder = lambda *a, **k: enc(*a, **k).to(dev0)
            bb, orig_fwd = self.rt.backbone, self.rt.backbone.forward

            def forward_home(*a, **k):
                out = orig_fwd(*a, **k)
                if getattr(out, "last_hidden_state", None) is not None and out.last_hidden_state.device != dev0:
                    out.last_hidden_state = out.last_hidden_state.to(dev0)
                return out
            bb.forward = forward_home
        if self.rt.image_unavailable:
            raise RuntimeError(f"d3 images unavailable: {self.rt.image_unavailable}")

    def extra(self):
        return {"runtime": "d3_runtime.D3.system_one", "attention_mode": self.rt.attention_mode,
                "batch_size": self.rt.batch_size, "permutation_average": self.rt.permutation_average,
                "fast_path": self.rt.fast_skipped or "on", "kernels": self.rt.kernels, "max_length": self.rt.max_length,
                "devices": sorted(set(self.sharded.values()))}

    def answer(self, state, questions, images):
        import gc
        import torch
        from qev.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer
        try:
            out = self.rt.system_one(state=state, questions=questions, images=images or None)
        except torch.OutOfMemoryError:
            # long states (tables, documents, GUI trees) at the runtime's 8 questions per forward overflow the GPU that
            # holds the embeddings and readout; drop the references the exception keeps, free the cache and run the
            # questions one per forward (the GPU otherwise stays full and every later request fails too)
            gc.collect()
            torch.cuda.empty_cache()
            batch = self.rt.batch_size
            self.rt.batch_size = 1
            try:
                out = self.rt.system_one(state=state, questions=questions, images=images or None)
            finally:
                self.rt.batch_size = batch
                gc.collect()
                torch.cuda.empty_cache()
        answers = {}
        for qid, a in out["answers"].items():
            if "error" in a:
                raise ValueError(f"{qid}: {a['error']}: {a.get('message', '')}")
            t = a.pop("type")
            answers[qid] = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}[t](**a)
        return answers, int(out["usage"]["input_tokens"]), {}


ENGINES = {"jevomni": JevOmniEngine, "clef": ClefEngine, "xor": XorEngine, "raya": RayaEngine, "d3": D3Engine}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=sorted(ENGINES))
    p.add_argument("--model", required=True, help="local snapshot directory")
    p.add_argument("--name", required=True)
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--head", default=None, help="jevomni: replacement decision_head.pt")
    p.add_argument("--processor", default=None, help="raya: processor path (default Qwen/Qwen3.5-2B)")
    p.add_argument("--meta", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    import torch
    import transformers
    import uvicorn
    from qev.server import create_app
    engine = ENGINES[args.family](args.model, args.name, max_memory=max_memory_of(args.max_memory), head=args.head,
                                  processor=args.processor)
    if args.meta:
        meta = {"server_command": " ".join(sys.argv), "model_name": args.name, "model_path": args.model, "dtype": "bfloat16",
                "backend": f"s1:{args.family} ({engine.info.extra['adapter']})", "adapter": engine.info.extra,
                "max_memory": args.max_memory, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__, "transformers": transformers.__version__,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta).write_text(json.dumps(meta, indent=1))
    uvicorn.run(create_app(engine, model_name=args.name), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
