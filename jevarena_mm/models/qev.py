"""Adapters for models served behind a qev HTTP endpoint (POST /v1/systemone), i.e. OneJev and the other VLMs of the
image comparison, through the servers in models/servers/. The server reads each answer as a softmax over the
option-letter logits; nothing is generated.

The image results of the paper are reproduced with these adapters. An adapter sends the state with <image:N>
placeholders, the questions, the images under the img policy and debias; decide() takes a DecisionRequest and returns a
ModelResponse. The in-process adapters of two of these models already use the plain kinds onejev (models/onejev.py)
and jev_omni (models/jev_omni.py), so the served ones are qev_onejev and qev_jev_omni.

Spec: <kind>@model=<served name>,img=auto,debias=1   (server URL: $JEVARENA_MM_QEV_URL, or url= in the spec)
  img=auto   the image comparison's policy: long edge 2560 px for a single screen (info.content == "screen": documents,
             charts, screenshots, renders; without info.content, a PNG/TIFF/BMP/GIF), 1536 px for a single photo; 1024 for
             2-4 images, 768 for 5-8, 512 above; a resized screen is saved as PNG, a photo as JPEG q95; an RGB/greyscale
             JPEG or PNG already within the cap is sent as is
  img=<px>   one long-edge cap for every image

docs/models_img.md lists the specs of the image comparison and the server each needs.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
import urllib.error

from .. import media
from ..registry import register_model
from ..schema import Answer, DecisionRequest, MediaRef, ModelResponse
from .base import ModelAdapter, split_options

LOSSLESS = {"PNG", "TIFF", "BMP", "GIF"}
KINDS = ("image",)  # the media kinds these adapters read from a state


def is_ref(v) -> bool:
    """A media reference in a state dumped to JSON: a dict with one of KINDS, a uri and a sha256."""
    return isinstance(v, dict) and v.get("kind") in KINDS and "uri" in v and "sha256" in v


def raw(ref: dict) -> bytes:
    return media.path_of(MediaRef.model_validate(ref)).read_bytes()


def image(ref: dict):
    """PIL image of the file as fetched (no resizing, no colour conversion)."""
    from PIL import Image
    im = Image.open(io.BytesIO(raw(ref)))
    im.load()
    return im


def long_edge(policy: str, n: int, fmt: str, content: str | None = None) -> int:
    """content (info.content) is "screen" (documents, charts, screenshots, renders) or "photo"; without it the file
    format decides (PNG/TIFF/BMP/GIF read as screens)."""
    if policy != "auto":
        return int(policy)
    if n > 8:
        return 512
    if n >= 5:
        return 768
    if n >= 2:
        return 1024
    return 2560 if is_screen(fmt, content) else 1536


def is_screen(fmt: str, content: str | None) -> bool:
    return content == "screen" if content in ("screen", "photo") else fmt in LOSSLESS


def encode(ref: dict, n: int, policy: str) -> tuple[str, dict]:
    from PIL import Image
    im = image(ref)
    fmt, (w0, h0) = (im.format or "PNG").upper(), im.size
    content = (ref.get("info") or {}).get("content")
    le = long_edge(policy, n, fmt, content)
    if fmt in ("JPEG", "PNG") and im.mode in ("RGB", "L") and max(im.size) <= le:   # already within the cap: as is
        mime = f"image/{fmt.lower()}"
        uri = f"data:{mime};base64,{base64.b64encode(raw(ref)).decode()}"
        return uri, {"sha256": ref["sha256"], "w_h": [w0, h0], "w_h_sent": [w0, h0], "format": mime}
    im = im.convert("RGB")
    if max(im.size) > le:
        s = le / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    if is_screen(fmt, content):   # screens stay lossless when resized
        im.save(buf, format="PNG")
        mime = "image/png"
    else:
        im.save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"
    uri = f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
    return uri, {"sha256": ref["sha256"], "w_h": [w0, h0], "w_h_sent": list(im.size), "format": mime}


INLINE = re.compile(r"<image (\d+)>")


def to_qev(request: dict) -> tuple[dict, list[dict]]:
    """mm state -> qev state with <image:N> placeholders in presentation order, plus the ordered refs.

    An image named inside text (`<image N>` for state key `image_N`, the interleaved convention) is placed there, at its
    first mention, and its own key is dropped; every other image takes the position of its state key."""
    st = request["state"]
    inline = {f"image_{n}" for v in st.values() if isinstance(v, str) for n in INLINE.findall(v)
              if is_ref(st.get(f"image_{n}"))}
    refs, state = [], {}

    def place(ref: dict) -> str:
        refs.append(ref)
        return f"<image:{len(refs)}>"

    def sub(m: re.Match) -> str:
        key = f"image_{m.group(1)}"
        if key not in inline or key in placed:
            return m.group(0)
        placed.add(key)
        return place(st[key])

    placed: set[str] = set()
    for k, v in st.items():
        if k in inline:
            continue
        if isinstance(v, str):
            state[k] = INLINE.sub(sub, v)
        elif is_ref(v):
            state[k] = place(v)
        elif isinstance(v, list) and v and all(is_ref(x) for x in v):
            state[k] = [place(x) for x in v]
        else:
            state[k] = v
    return state, refs


def to_qev_questions(questions: dict) -> dict:
    """mm questions -> qev client questions (boolean -> Noul, choice -> Choice, score -> Score)."""
    from qev.client import Choice, Noul, Score
    out = {}
    for name, q in questions.items():
        if q["type"] == "boolean":
            out[name] = Noul(q["instructions"], q.get("criteria"))
        elif q["type"] == "choice":
            out[name] = Choice(q["instructions"], q["criteria"])
        else:
            out[name] = Score(q["instructions"], q["criteria"])
    return out


def build(request: dict, policy: str = "auto") -> tuple[dict, dict, list[tuple[str, dict]]]:
    """Everything sent to the server for one item: (state, qev questions, [(image data URI, what was sent)])."""
    state, refs = to_qev(request)
    return state, to_qev_questions(request["questions"]), [encode(r, len(refs), policy) for r in refs]


def from_qev(q: dict, a: dict) -> dict:
    if a["type"] == "noul":
        p = float(a["noul"])
        return {"true": p, "false": 1 - p}
    return {k: float(v) for k, v in a["probabilities"].items()}


def as_request(request: DecisionRequest) -> dict:
    """The request as these adapters read it: a dict with the state's media references as dicts."""
    return request.model_dump(mode="json")


@register_model
class Qev(ModelAdapter):
    kind = "qev"
    modalities = frozenset({"image"})
    native_probs = True
    probability_source = "softmax over option-letter logits (qev System-One readout)"

    def __init__(self, arg: str = ""):
        base, self.params = split_options(arg)
        if base:
            raise ValueError(f"{self.kind}: give the options after '@', e.g. {self.kind}@model=<served name>,img=auto")
        self.options = arg.partition("@")[2]

    @property
    def model_id(self) -> str:
        """The spec as written, e.g. "qev@model=Qwen3.5-9B,img=auto,debias=1"."""
        return f"{self.kind}@{self.options}" if self.options else self.kind

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        from qev.client import Client
        req = as_request(request)
        p = self.params
        policy = p.get("img", "auto")
        state, qs, enc = build(req, policy)
        t0 = time.perf_counter()
        try:
            r = Client(p.get("url") or os.environ.get("JEVARENA_MM_QEV_URL", "http://localhost:8000"), timeout=600).system_one(
                state=state, questions=qs, media=[{"type": "image", "data": u} for u, _ in enc],
                debias=int(p.get("debias", 1)))
        except urllib.error.HTTPError as e:
            return ModelResponse(status="failed", latency_ms=(time.perf_counter() - t0) * 1000,
                                 error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        if p.get("model") and r.model != p["model"]:
            return ModelResponse(status="failed", error=f"server answers as {r.model}, spec names {p['model']}")
        answers = {k: Answer(probs=from_qev(req["questions"][k], v.raw)) for k, v in r.answers.items()}
        return ModelResponse(answers=answers, latency_ms=(time.perf_counter() - t0) * 1000,
                             usage={**(r.usage or {}), "input": {"images": [m for _, m in enc], "policy": f"img={policy}"},
                                    "model": r.model})


@register_model
class OneJev(Qev):
    kind = "qev_onejev"  # "onejev" is the in-process adapter (models/onejev.py)
    probability_source = "native: restricted softmax over the answer letters (trained objective)"


@register_model
class QwenReadout(Qev):
    kind = "qev_readout"
    native_probs = False
    probability_source = "letter-logit readout of a chat VLM, one forward per question (servers/serve_any.py)"


@register_model
class LlamaCpp(Qev):
    kind = "qev_gguf"
    native_probs = False
    probability_source = "letter-logit readout from llama.cpp top logprobs (servers/serve_gguf.py)"


@register_model
class DiffusionReadout(Qev):
    kind = "qev_diffusion"
    native_probs = False
    probability_source = "letter logits at the first canvas position on the last denoising step (servers/serve_any.py)"


@register_model
class DGemmaReadout(Qev):
    kind = "qev_dgemma"
    native_probs = False
    probability_source = ("letter log-probabilities at the answer slot of one read-only denoising pass, the row before "
                          "the slot (vLLM; the read of dgemma:, layout=prompt; servers/serve_dgemma.py)")


# System-One releases served by models/servers/serve_s1.py, each in its own format (see that module's docstring)

@register_model
class JevOmni(Qev):
    kind = "qev_jev_omni"  # "jev_omni" is the in-process adapter (models/jev_omni.py)
    native_probs = True
    probability_source = "Jev-Omni: 256-slot decision head on the last hidden state, one forward per question (jev_omni.py)"


@register_model
class S1Clef(Qev):
    kind = "s1_clef"
    native_probs = True
    probability_source = "Clef: joint schema head over the backbone's hidden states, all questions in one forward (joint_schema_model.systemone)"


@register_model
class S1Xor(Qev):
    kind = "s1_xor"
    native_probs = False
    probability_source = "Xor: label-token logprobs after 'Answer: (' with the release's temperatures and two-order averaging (xn_base_server.py)"


@register_model
class S1Raya(Qev):
    kind = "s1_raya"
    native_probs = False
    probability_source = "Raya: masked projection onto the ' A'..' Z' label logits at the last position (model card)"


@register_model
class S1D3(Qev):
    kind = "s1_d3"
    native_probs = True
    probability_source = "d3: 255 answer codes read from the last hidden state (checkpoint readout matrix), noncausal full attention, 8 questions per forward (d3_runtime.py)"
