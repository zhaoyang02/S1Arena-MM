"""Xor 26B-A4B v1.1 (juspay/Xor-26B-A4B@v1.1): a System-One decision model (post-trained Gemma 4 26B-A4B) served as
its card serves it: the pinned SGLang image (run here through Apptainer) and the bundled wrapper xn_base_server.py with
the settings of the release's docker-compose.yml (context 131072, chunked prefill 16384, mem fraction 0.80, label
temperatures choice 2.0 / noul 1.70 / score 1.0, two-order scoring of choices), answering POST /v1/systemone.

- tp (default 2): SGLang tensor-parallel size. The card's setting is one ~96 GB GPU; on 40 GB A100s the BF16 weights
  need two. Not part of the model id.
- concurrency (default 1): requests in flight to the server at once (media are still prepared in parallel). On two
  40 GB A100s the weights and the release's static memory fraction leave too little room for several 32-frame
  prefills at once: SGLang runs out of memory and dies. Not part of the model id. A server that has died ends the run
  (the next run starts it again), instead of failing every remaining call.
- The request is the card's Jev/SystemOne body: "state" is the request's state with each medium replaced by its
  placeholder ("<video 1>", render.media_slots), "questions" the questions (boolean -> noul).
- Media, as the wrapper takes them: 1-8 images as data URLs, or one video as a data URL, never both. The video is the
  window as Gemma4's video processor reads a clip by default: the 32 frames it samples (num_frames=32, arange from the
  window start, all frames of a shorter window), written as an MP4 (MPEG-4, highest quality) at 32 / duration frames
  per second, so the processor samples those same frames and labels them with their mm:ss timestamps. A request with
  several videos, or with images and a video, is rejected (the wrapper takes one video).
- Images go where the wrapper puts them, before the state text (its media_template_head): a text that places images
  with <image:k> (the closed-loop suites' packets) keeps those markers as text, the images first in the same order.
- The wrapper refuses bodies over 8 MB. Frames and images are sent at most twice per side the size Gemma4's processor
  resizes them to (its patch budget: 70 soft tokens per frame, 280 per image; aspect kept, never enlarged), so the
  processor still makes the final resize; if the body is still too large, the MP4 quality steps down (qscale 1 -> 8)
  and images go from PNG to JPEG.

Spec: xor[@tp=N,concurrency=N]; needs Apptainer ($APPTAINER_BIN, else apptainer on the PATH) and the image at $XOR_SIF
(apptainer pull of IMAGE).
XOR_CACHE (default /tmp/xor-jit-<user>) holds the kernels compiled on the node, XOR_LOG_DIR (default
.cache/xor_logs) the server logs.
"""

from __future__ import annotations

import atexit
import base64
import getpass
import io
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path

from .. import media, render
from ..env import CACHE
from ..registry import register_model
from ..schema import Answer, DecisionRequest, ModelResponse
from .base import ModelAdapter, split_options
from .local_lm import source_fps, window

REPO, REVISION, MODEL_NAME = "juspay/Xor-26B-A4B", "v1.1", "xor-26b-a4b-xl5050"
COMMIT = "28ca09c7b76648b24d3561c940d1a7460d12fb8e"  # the v1.1 tag, pinned in case the tag moves (the model id keeps v1.1)
IMAGE = "prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f"  # serving/docker-compose.yml
SIF = os.environ.get("XOR_SIF", "")  # the pinned SGLang image as an Apptainer file (apptainer pull of IMAGE)
APPTAINER = os.environ.get("APPTAINER_BIN") or shutil.which("apptainer") or "apptainer"
TEMPERATURES = '{"choice":2.0,"noul":1.70,"score":1.0}'  # docker-compose.yml, XN_TEMP_JSON
TYPES = {"boolean": "noul", "choice": "choice", "score": "score"}
NUM_FRAMES = 32  # Gemma4's video processor default
MAX_BODY = 8 * 1024 * 1024 - 64 * 1024  # the wrapper's MAX_BODY_BYTES, less room for the rest of the body
PATCH, POOL, VIDEO_TOKENS, IMAGE_TOKENS = 16, 3, 70, 280  # processor_config.json: patch_size, pooling, max_soft_tokens
QSCALES = (1, 2, 3, 5, 8)  # MPEG-4 quality steps (1 = best) tried until the body fits
COMPAT = "/usr/local/cuda-13.0/compat"  # the image's CUDA forward-compatibility driver
JIT = os.environ.get("XOR_CACHE", f"/tmp/xor-jit-{getpass.getuser()}")  # on the node's disk: file locks hang on NFS


def _sent_size(height: int, width: int, soft_tokens: int) -> tuple[int, int]:
    """At most twice per side the size Gemma4's processor resizes to (its patch budget), aspect kept, never enlarged."""
    target = soft_tokens * POOL ** 2 * PATCH ** 2
    scale = min(1.0, 2 * math.sqrt(target / (height * width)))
    return max(2, int(round(height * scale / 2)) * 2), max(2, int(round(width * scale / 2)) * 2)


def _encode_mp4(pictures: list, fps: Fraction, qscale: int) -> bytes:
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.height, stream.width = pictures[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"qscale": str(qscale)}
        for picture in pictures:
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _compat_libraries() -> str:
    """The image's CUDA 13 forward-compatibility driver, put ahead of the host's when the host driver predates CUDA 13
    (R580): data-center GPUs run newer CUDA that way. Empty on a host whose driver is new enough."""
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=120)
        return f"{COMPAT}:" if int(query.stdout.split()[0].split(".")[0]) < 580 else ""  # a prefix of LD_LIBRARY_PATH
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return ""


def _bound_folders(snapshot: Path) -> list[str]:
    """Folders to bind read-only at their own paths so that the snapshot's links resolve in the container: the hub
    cache (a repo's files link into its blobs, and some caches keep the blobs in a store at the cache root) and the
    real folder of any file that lies outside it."""
    roots = [snapshot.parents[2]]
    for f in sorted(snapshot.rglob("*")):
        real = f.resolve().parent
        if f.is_file() and not any(real.is_relative_to(r.resolve()) for r in roots):
            roots.append(real)
    return [str(r) for r in roots]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, process: subprocess.Popen, timeout: float, what: str) -> None:
    started = time.time()
    while time.time() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"{what} exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(5)
    raise TimeoutError(f"{what} not ready after {timeout:.0f}s")


@register_model
class Xor(ModelAdapter):
    kind = "xor"
    modalities = frozenset({"image", "video"})
    native_probs = True
    probability_source = "candidate label-token probabilities (wrapper, two-order)"
    _one_at_a_time = threading.Lock()  # starts the servers once

    def __init__(self, arg: str = "") -> None:
        _, options = split_options(arg)
        self.tp = int(options.get("tp", 2))
        self.slots = threading.BoundedSemaphore(int(options.get("concurrency", 1)))
        self.url = None
        self.load_error = None
        self.processes: list[subprocess.Popen] = []

    @property
    def model_id(self) -> str:
        return f"{REPO}@{REVISION}"

    def _raise_if_dead(self) -> None:
        """End the run when SGLang or the wrapper has exited: no later call can be answered (the next run restarts them)."""
        dead = [p for p in self.processes if p.poll() is not None]
        if dead:
            raise SystemExit(f"{REPO}@{REVISION}: the server exited with {dead[0].returncode}; see XOR_LOG_DIR")

    def stop(self) -> None:
        """Ends SGLang and the wrapper (they would otherwise outlive the run and keep the GPUs)."""
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes = []

    def load(self) -> None:
        if self.url is not None:
            return
        atexit.register(self.stop)
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(REPO, revision=COMMIT))
        model_path = str(snapshot)  # bound at the same path, with wherever its links lead (_bound_folders)
        serving = (snapshot / "serving" / "xn_base_server.py").resolve()
        sglang_port, api_port = _free_port(), _free_port()
        if not SIF or not Path(SIF).is_file():
            raise FileNotFoundError(f"set XOR_SIF to the pinned SGLang image (apptainer pull docker://{IMAGE}); got {SIF!r}")
        logs = Path(os.environ.get("XOR_LOG_DIR") or CACHE / "xor_logs")
        logs.mkdir(parents=True, exist_ok=True)
        compat = _compat_libraries()
        shell = ["bash", "-c", f'export LD_LIBRARY_PATH="{compat}${{LD_LIBRARY_PATH}}"; exec "$@"', "xor"]
        base = [APPTAINER, "exec", "--nv", *[a for d in _bound_folders(snapshot) for a in ("--bind", f"{d}:{d}:ro")]]
        env = {**os.environ, "APPTAINERENV_HF_HUB_OFFLINE": "1", "APPTAINERENV_TOKENIZERS_PARALLELISM": "false",
               "APPTAINERENV_PYTHONNOUSERSITE": "1", "APPTAINERENV_CUDA_HOME": "/usr/local/cuda",
               "APPTAINERENV_XDG_CACHE_HOME": f"{JIT}/xdg", "APPTAINERENV_FLASHINFER_WORKSPACE_BASE": JIT,
               "APPTAINERENV_TRITON_CACHE_DIR": f"{JIT}/xor_triton", "APPTAINERENV_TORCHINDUCTOR_CACHE_DIR": f"{JIT}/xor_inductor"}
        sglang = base + [SIF, *shell, "python3", "-m", "sglang.launch_server", "--model-path", model_path, "--trust-remote-code",
                         "--tp-size", str(self.tp), "--host", "127.0.0.1", "--port", str(sglang_port),
                         "--context-length", "131072", "--max-prefill-tokens", "131072", "--chunked-prefill-size", "16384",
                         "--mem-fraction-static", "0.80"]
        with open(logs / f"sglang-{sglang_port}.log", "w") as out:
            self.processes.append(subprocess.Popen(sglang, stdout=out, stderr=subprocess.STDOUT, env=env))
        _wait(f"http://127.0.0.1:{sglang_port}/health", self.processes[-1], 3600, "SGLang")
        api_env = {**env, "APPTAINERENV_XN_SGLANG_URL": f"http://127.0.0.1:{sglang_port}", "APPTAINERENV_XN_MODEL_PATH": model_path,
                   "APPTAINERENV_XN_MODEL_ID": MODEL_NAME, "APPTAINERENV_XN_MAX_CONTEXT": "131072",
                   "APPTAINERENV_XN_TEMP_JSON": TEMPERATURES, "APPTAINERENV_XN_TWO_ORDER": "1",
                   "APPTAINERENV_XN_HOST": "127.0.0.1", "APPTAINERENV_XN_PORT": str(api_port)}
        api = base + ["--bind", f"{serving}:/app/xn_base_server.py:ro", SIF, *shell, "python3", "-u", "/app/xn_base_server.py"]
        with open(logs / f"api-{api_port}.log", "w") as out:
            self.processes.append(subprocess.Popen(api, stdout=out, stderr=subprocess.STDOUT, env=api_env))
        _wait(f"http://127.0.0.1:{api_port}/health", self.processes[-1], 600, "Xor wrapper")
        self.url = f"http://127.0.0.1:{api_port}/v1/systemone"

    def _video(self, ref, budget: int) -> tuple[str, dict]:
        """The 32 frames Gemma4's processor samples from the window, as an MP4 data URL at 32 / duration fps."""
        import numpy as np
        from PIL import Image

        start, end = window(ref)
        rate = source_fps(ref)
        total = max(1, int(round((end - start) * rate)))
        count = min(NUM_FRAMES, total)
        wanted = [int(i) for i in np.arange(0, total, total / count)][:count]
        frames = media.frames(ref, timestamps=[start + i / rate for i in wanted])
        height, width = _sent_size(*frames[0].array.shape[:2], VIDEO_TOKENS)
        pictures = [np.asarray(Image.fromarray(f.array).resize((width, height), Image.BICUBIC)) for f in frames]
        duration = max(end - start, 1e-3)
        fps = Fraction(len(frames) / duration).limit_denominator(1000) if len(frames) > 1 else Fraction(1)  # MPEG-4 time bases stop at 65535
        for qscale in QSCALES:
            url = "data:video/mp4;base64," + base64.b64encode(_encode_mp4(pictures, fps, qscale)).decode()
            if len(url) <= budget:
                break
        info = {"frames": len(frames), "timestamps_s": [round(f.t, 3) for f in frames], "clip_fps": float(fps),
                "size": [width, height], "qscale": qscale}
        return url, info

    @staticmethod
    def _images(refs, budget: int) -> tuple[list[str], list[dict]]:
        from PIL import Image

        pictures = []
        for ref in refs:
            picture = media.image(ref).convert("RGB")
            height, width = _sent_size(picture.height, picture.width, IMAGE_TOKENS)
            pictures.append(picture.resize((width, height), Image.BICUBIC) if (height, width) != (picture.height, picture.width) else picture)
        for form, options in (("png", {}), ("jpeg", {"quality": 95}), ("jpeg", {"quality": 85})):
            urls = []
            for picture in pictures:
                buffer = io.BytesIO()
                picture.save(buffer, format=form.upper(), **options)
                urls.append(f"data:image/{form};base64," + base64.b64encode(buffer.getvalue()).decode())
            if sum(map(len, urls)) <= budget:
                break
        return urls, [{"image": True, "size": list(p.size), "format": form, **options} for p in pictures]

    def _body(self, request: DecisionRequest) -> tuple[dict | None, list[dict]]:
        """The wrapper's request body and what was sent of each medium; None if the wrapper cannot take the media."""
        state, slots = render.media_slots(request)
        kinds = [ref.kind for _, ref in slots]
        if kinds.count("video") > 1 or ("video" in kinds and "image" in kinds) or kinds.count("image") > 8:
            return None, []
        body = {"model": MODEL_NAME, "state": state, "questions": {}}
        for name, question in request.questions.items():
            entry = {"type": TYPES[question.type], "instructions": question.instructions}
            if question.type != "boolean":
                entry["criteria"] = question.criteria
            body["questions"][name] = entry
        budget = MAX_BODY - len(json.dumps(body).encode())
        used = []
        if "video" in kinds:
            name, ref = slots[0]
            body["video"], info = self._video(ref, budget)
            used.append({"placeholder": name, **info})
        elif kinds:
            body["images"], infos = self._images([ref for _, ref in slots], budget)
            used += [{"placeholder": name, **info} for (name, _), info in zip(slots, infos)]
        return body, used

    def decide(self, request: DecisionRequest, sample: int = 0) -> ModelResponse:
        with self._one_at_a_time:
            if self.load_error:
                raise SystemExit(self.load_error)
            try:
                self.load()
            except Exception as error:  # noqa: BLE001 - no request can be answered: end the run, not each call
                self.stop()
                self.load_error = f"{REPO}@{REVISION}: serving failed to start: {type(error).__name__}: {error}"[:2000]
                raise SystemExit(self.load_error) from error
        # media and the call run in the runner's threads in parallel: SGLang batches concurrent requests
        body, used = self._body(request)
        if body is None:
            return ModelResponse(error="rejected: the Xor wrapper takes 1-8 images or one video per request")
        started = time.perf_counter()
        post = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with self.slots, urllib.request.urlopen(post, timeout=1800) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as error:
            text = error.read().decode(errors="replace")[:300]
            self._raise_if_dead()
            final = error.code in (400, 413, 422)  # refused input (context, options): final
            return ModelResponse(error=("rejected: " if final else "") + f"HTTP {error.code}: {text}", usage={"input": used})
        latency = (time.perf_counter() - started) * 1000
        answers = {}
        for name, question in request.questions.items():
            answer = reply["answers"][name]
            if question.type == "boolean":
                probs = {"true": float(answer["noul"]), "false": 1.0 - float(answer["noul"])}
            else:
                raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
                probs = {option: raw[str(option)] for option in question.option_ids()} if question.type == "choice" \
                    else {option: raw.get(option, raw.get(str(int(option)), 0.0)) for option in question.option_ids()}
            answers[name] = Answer(probs=probs)
        return ModelResponse(answers=answers, latency_ms=round(latency, 1), usage={"input": used, **reply.get("usage", {})},
                             cost_usd=0.0)
