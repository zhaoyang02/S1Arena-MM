"""Run every model spec of the image comparison on one GPU node: start each model's server
(jevarena_mm/models/servers/), run the items it has no cached response for, stop the server. GPUs are handed out
greedily: the slowest multi-GPU models first, single-GPU models on whatever GPU is free. Resumable: a spec with nothing
left to run is skipped without starting a server. Before a spec runs, one item of each question type goes through its
adapter (smoke test); a spec that fails it is not run. Items whose image files are missing are left for later, not run.

    python scripts/img/sweep_gpu.py [--gpus 0,1,2] [--leaf img] [--only SLUG ...] [--dry-run]

Run it from the repository root. Logs: logs/sweep/<slug>.server.txt and <slug>.run.txt; progress: logs/sweep/status.txt.

Settings, read from the environment:
  JEVARENA_MM_IMG_PYTHON            Python of the model servers and of the per-spec runs, with qev and the checkpoints'
                                    own dependencies (docs/models_img.md; default: the Python running this script).
  JEVARENA_MM_IMG_VLLM_PYTHON       Python of the DiffusionGemma servers on vLLM (the qev_dgemma specs,
                                    servers/serve_dgemma.py), with vLLM 0.31 or later and qev (default: the Python of
                                    the other servers, above).
  JEVARENA_MM_HF_HOME               Hugging Face cache holding the checkpoints, read as $JEVARENA_MM_HF_HOME/hub
                                    (default: the standard cache, $HF_HUB_CACHE, else $HF_HOME/hub, else
                                    ~/.cache/huggingface/hub).
  JEVARENA_MM_IMG_QUANT_PYTHONPATH  PYTHONPATH of the GPTQ and AWQ servers only, e.g. a `pip install --target` folder
                                    holding gptqmodel (default: the environment's own PYTHONPATH).
  JEVARENA_MM_LLAMA_SERVER          llama.cpp's llama-server binary, for the GGUF spec (default: llama-server on PATH).
Compilers and CUDA come from your environment: the GPTQ and AWQ servers build gptqmodel's Marlin kernels on first use
(with CC, CXX, CUDA_HOME, nvcc and ninja on PATH; TORCH_CUDA_ARCH_LIST limits the build to your GPUs' architecture,
XDG_CACHE_HOME sets where the built kernels are kept), and llama-server finds its libraries through LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from huggingface_hub import constants as hf_constants

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PY = os.environ.get("JEVARENA_MM_IMG_PYTHON") or sys.executable
VLLM_PY = os.environ.get("JEVARENA_MM_IMG_VLLM_PYTHON") or PY
HUB = (Path(os.environ["JEVARENA_MM_HF_HOME"]) / "hub" if os.environ.get("JEVARENA_MM_HF_HOME")
       else Path(hf_constants.HF_HUB_CACHE))
LLAMA_SERVER = os.environ.get("JEVARENA_MM_LLAMA_SERVER") or "llama-server"
# GPTQ / AWQ checkpoints: gptqmodel side packages; its Marlin kernels are JIT-built on first use
QUANT_PYTHONPATH = os.environ.get("JEVARENA_MM_IMG_QUANT_PYTHONPATH")
QUANT_ENV = {"PYTHONPATH": QUANT_PYTHONPATH} if QUANT_PYTHONPATH else {}

# slug, model spec, HF repo, backend, GPUs, leaves left out, and for some the server's own settings
JOBS = [   # priority order: the core comparison first, the slowest servers (GPTQ MoE, GGUF) last
    ("onejev-27b", "qev_onejev@model=OneJev-27B,img=auto,debias=1", "OmniJev/OneJev-27B", "qev", 2, ()),
    ("qwen3.8-27b-fp8", "qev@model=Qwen3.8-27B-FP8,img=auto,debias=1", "Qwen/Qwen3.8-27B-FP8", "qev", 2, ()),
    ("onejev-9b", "qev_onejev@model=OneJev-9B,img=auto,debias=1", "OmniJev/OneJev-9B", "qev", 1, ()),
    ("qwen3.5-9b", "qev@model=Qwen3.5-9B,img=auto,debias=1", "Qwen/Qwen3.5-9B", "qev", 1, ()),
    ("onejev-4b", "qev_onejev@model=OneJev-4B,img=auto,debias=1", "OmniJev/OneJev-4B", "qev", 1, ()),
    ("qwen3.5-4b", "qev@model=Qwen3.5-4B,img=auto,debias=1", "Qwen/Qwen3.5-4B", "qev", 1, ()),
    ("onejev-0.8b", "qev_onejev@model=OneJev-0.8B,img=auto,debias=1", "OmniJev/OneJev-0.8B", "qev", 1, ()),
    ("qwen3.5-0.8b", "qev@model=Qwen3.5-0.8B,img=auto,debias=1", "Qwen/Qwen3.5-0.8B", "qev", 1, ()),
    ("qwen3-omni-30b-a3b-instruct", "qev_readout@model=Qwen3-Omni-30B-A3B-Instruct,img=auto,debias=1", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "hf", 2, ()),
    ("qwen3-omni-30b-a3b-thinking", "qev_readout@model=Qwen3-Omni-30B-A3B-Thinking,img=auto,debias=1", "Qwen/Qwen3-Omni-30B-A3B-Thinking", "hf", 2, ()),
    ("diffusiongemma-26b-a4b", "qev_diffusion@model=diffusiongemma-26B-A4B-it,img=auto,debias=1", "google/diffusiongemma-26B-A4B-it", "hf", 2, ("img.safety/privacy",)),
    # the same spec for the privacy leaf the job above leaves out (~34 s per item, ~5 h); run it on its own with --only
    ("diffusiongemma-26b-a4b-privacy", "qev_diffusion@model=diffusiongemma-26B-A4B-it,img=auto,debias=1", "google/diffusiongemma-26B-A4B-it", "hf", 2, ()),
    ("qwen2.5-omni-3b", "qev_readout@model=Qwen2.5-Omni-3B,img=auto,debias=1", "Qwen/Qwen2.5-Omni-3B", "hf", 1, ()),
    ("qwen2.5-omni-7b", "qev_readout@model=Qwen2.5-Omni-7B,img=auto,debias=1", "Qwen/Qwen2.5-Omni-7B", "hf", 1, ()),
    ("qwen2.5-omni-7b-awq", "qev_readout@model=Qwen2.5-Omni-7B-AWQ,img=auto,debias=1", "Qwen/Qwen2.5-Omni-7B-AWQ", "hfq", 1, ()),
    ("qwen2.5-omni-7b-gptq", "qev_readout@model=Qwen2.5-Omni-7B-GPTQ-Int4,img=auto,debias=1", "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4", "hfq", 1, ()),
    ("qwen3.5-35b-a3b-gptq", "qev_readout@model=Qwen3.5-35B-A3B-GPTQ-Int4,img=auto,debias=1", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "hfq", 1, ()),
    ("onejev-27b-gguf-q8", "qev_gguf@model=OneJev-27B-GGUF-Q8_0,img=auto,debias=1", "mradermacher/OneJev-27B-GGUF", "gguf", 1, ()),
    # System-1 decision models (models/servers/serve_s1.py), each served in its own format
    ("raya-decision-v1", "s1_raya@model=raya-decision-v1,img=auto,debias=1", "yuyu199741/raya-decision-v1", "s1:raya", 1, (),
     {"processor": "Qwen/Qwen3.5-2B"}),
    ("clef-flash", "s1_clef@model=clef-flash,img=auto,debias=1", "Cloudflare/clef-flash", "s1:clef", 1, ()),
    ("jev-omni", "qev_jev_omni@model=Jev-Omni,img=auto,debias=1", "akhilaaa3/Jev-Omni", "s1:jevomni", 1, ()),
    ("jev-omni-general-head-v3", "qev_jev_omni@model=Jev-Omni-general-head-v3,img=auto,debias=1", "akhilaaa3/Jev-Omni", "s1:jevomni", 1, (),
     {"head": "ferdinandl007/jev-omni-general-head-v3:decision_head.pt"}),
    ("clef-omni", "s1_clef@model=clef-omni,img=auto,debias=1", "Cloudflare/clef-omni", "s1:clef", 2, ()),
    ("xor-26b-a4b", "s1_xor@model=Xor-26B-A4B-v1.1,img=auto,debias=1", "juspay/Xor-26B-A4B", "s1:xor", 2, ()),
    ("clef", "s1_clef@model=clef,img=auto,debias=1", "Cloudflare/clef", "s1:clef", 2, ()),
    # the System-1 model d3, then open MLLMs: the bases of Xor (Gemma 4 26B-A4B) and raya (Qwen3.5-2B), Qwen3-VL-8B,
    # Gemma 4 12B
    # d3: the GPU holding the embeddings, vision encoder and readout needs headroom for long states (the struct and
    # GUI tasks OOMed with an even 36 GiB split: 1,813 items), so the first GPU takes fewer backbone layers
    ("d3", "s1_d3@model=d3,img=auto,debias=1", "vllm-sr/d3", "s1:d3", 2, (), {"max_memory": "0=24GiB,1=36GiB"}),
    # same spec over 3 GPUs (about 18 GB of the 52 GB backbone each): only for the items whose long states still OOM on
    # 2 GPUs (Mind2Web); run after removing those failed lines from the cache, never in the default sweep
    ("d3-3gpu", "s1_d3@model=d3,img=auto,debias=1", "vllm-sr/d3", "s1:d3", 3, (), {"max_memory": "0=16GiB,1=21GiB,2=21GiB"}),
    ("gemma-4-26b-a4b", "qev_readout@model=gemma-4-26B-A4B-it,img=auto,debias=1", "google/gemma-4-26B-A4B-it", "hf", 2, ()),
    ("qwen3-vl-8b", "qev_readout@model=Qwen3-VL-8B-Instruct,img=auto,debias=1", "Qwen/Qwen3-VL-8B-Instruct", "hf", 1, ()),
    ("gemma-4-12b", "qev_readout@model=gemma-4-12B-it,img=auto,debias=1", "google/gemma-4-12B-it", "hf", 1, ()),
    ("qwen3.5-2b", "qev@model=Qwen3.5-2B,img=auto,debias=1", "Qwen/Qwen3.5-2B", "qev", 1, ()),
    # cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 is not loadable by transformers: a vLLM-only compressed-tensors checkpoint
    # whose experts are stored as packed per-expert int4 tensors (experts.N.*.weight_packed); transformers 5.18 has no
    # conversion into its fused DiffusionGemma experts, and the encoder/decoder weight tying expects unpacked weights
    # (load fails with "Linear has no attribute `weight`"). It runs on vLLM 0.31 or later with the read of dgemma:
    # (servers/serve_dgemma.py, the "dgemma" backend), in the qev_dgemma jobs at the end.
    # checkpoints with their own modeling code (servers/readout_remote.py)
    ("phi-4-multimodal", "qev_readout@model=Phi-4-multimodal-instruct,img=auto,debias=1", "microsoft/Phi-4-multimodal-instruct", "hf", 1, ()),
    ("minicpm-o-4.5", "qev_readout@model=MiniCPM-o-4_5,img=auto,debias=1", "openbmb/MiniCPM-o-4_5", "hf", 1, ()),
    ("minicpm-o-2.6", "qev_readout@model=MiniCPM-o-2_6,img=auto,debias=1", "openbmb/MiniCPM-o-2_6", "hf", 1, ()),
    ("nemotron-3-nano-omni-30b-a3b", "qev_readout@model=Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16,img=auto,debias=1", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", "hf", 2, ()),
    # Qwen3-VL-30B-A3B on Qwen3-VL-8B's line (2 GPUs for its 61 GB of BF16 weights)
    ("qwen3-vl-30b-a3b", "qev_readout@model=Qwen3-VL-30B-A3B-Instruct,img=auto,debias=1", "Qwen/Qwen3-VL-30B-A3B-Instruct", "hf", 2, ()),
    # DiffusionGemma on vLLM with the read of dgemma: (servers/serve_dgemma.py, $JEVARENA_MM_IMG_VLLM_PYTHON), tensor
    # parallel over the job's GPUs: the AWQ-INT4 checkpoint on one GPU, the BF16 checkpoint on two. The BF16 checkpoint
    # also has the qev_diffusion jobs above, a different read with its own model id.
    ("diffusiongemma-26b-a4b-awq", "qev_dgemma@model=diffusiongemma-26B-A4B-it-AWQ-INT4,img=auto,debias=1", "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4", "dgemma", 1, ()),
    ("diffusiongemma-26b-a4b-dgemma", "qev_dgemma@model=diffusiongemma-26B-A4B-it,img=auto,debias=1", "google/diffusiongemma-26B-A4B-it", "dgemma", 2, ()),
]
LOGS = Path("logs/sweep")
lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with lock, open(LOGS / "status.txt", "a") as f:
        f.write(line + "\n")


def snapshot(repo: str) -> str:
    return str(sorted((HUB / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())[-1])


def task_items(leaf: str, skip: tuple = ()):
    """(task, its items) for the image tasks under leaf, without the leaves in skip."""
    from jevarena_mm import dataio
    from jevarena_mm.registry import tasks_under
    return [(task, dataio.load_task(task)) for task in tasks_under(leaf)
            if task.name.startswith("img.") and not (skip and task.leaf.startswith(skip))]


def todo(spec: str, leaf: str, skip: tuple) -> int:
    from jevarena_mm import runner
    from jevarena_mm.registry import make_model
    m = make_model(spec)
    n = 0
    for task, items in task_items(leaf, skip):
        done = {(r["item_id"], r["sample"]) for r in runner.read_runs(m.slug, task.name) if runner.is_final(r)}
        n += sum((it.id, 0) not in done for it in items)
    return n


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start(slug, repo, backend, gpus, port, name=None, extras=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    extras = extras or {}
    mem = extras.get("max_memory") or ",".join(f"{i}=36GiB" for i in range(len(gpus)))
    name = name or repo.split("/", 1)[1]
    out = open(LOGS / f"{slug}.server.txt", "a")
    procs = []
    if backend.startswith("s1:"):
        cmd = [PY, "-m", "jevarena_mm.models.servers.serve_s1", "--family", backend[3:], "--model", snapshot(repo),
               "--port", str(port), "--name", name, "--max-memory", mem]
        if extras.get("head"):
            hrepo, hfile = extras["head"].split(":")
            cmd += ["--head", str(Path(snapshot(hrepo)) / hfile)]
        if extras.get("processor"):
            cmd += ["--processor", snapshot(extras["processor"])]
    elif backend == "dgemma":   # vLLM DiffusionGemma with dgemma:'s read (servers/serve_dgemma.py); tensor parallel over the GPUs
        cmd = [VLLM_PY, "-m", "jevarena_mm.models.servers.serve_dgemma", "--model", snapshot(repo), "--port", str(port),
               "--name", name, "--tp", str(len(gpus)), "--max-images", "13"]
    elif backend == "qev":
        cmd = [PY, "-m", "jevarena_mm.models.servers.serve_qev", "--model", snapshot(repo), "--port", str(port),
               "--name", name, "--max-memory", mem]
    elif backend in ("hf", "hfq"):
        if backend == "hfq":
            env.update(QUANT_ENV)
        cmd = [PY, "-m", "jevarena_mm.models.servers.serve_any", "--model", snapshot(repo), "--port", str(port),
               "--name", name, "--max-memory", mem]
    else:   # gguf: llama-server with the vision projector, qev's llama.cpp engine on top
        snap = Path(snapshot(repo))
        gg = next(p for p in sorted(snap.glob("*.gguf")) if "mmproj" not in p.name and "Q8_0" in p.name)
        mp = next(p for p in sorted(snap.glob("*mmproj*f16*.gguf")))
        lport = port + 100
        procs.append(subprocess.Popen([LLAMA_SERVER, "--host", "127.0.0.1", "--port",
                                       str(lport), "-m", str(gg), "--mmproj", str(mp), "-ngl", "999", "-c", "32768", "-np", "1",
                                       "--image-min-tokens", "64", "--image-max-tokens", "16384"],
                                      env=env, stdout=open(LOGS / f"{slug}.llama.txt", "a"), stderr=subprocess.STDOUT))
        for _ in range(180):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{lport}/health", timeout=5) as r:
                    if b"ok" in r.read():
                        break
            except Exception:
                pass
            if procs[0].poll() is not None:
                raise RuntimeError("llama-server died")
            time.sleep(10)
        cmd = [PY, "-m", "jevarena_mm.models.servers.serve_gguf", "--llama-url", f"http://127.0.0.1:{lport}",
               "--tokenizer", snapshot("OmniJev/OneJev-27B"), "--name", "OneJev-27B-GGUF-Q8_0", "--port", str(port),
               "--host", "0.0.0.0"]
    procs.append(subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT))
    while not healthy(port):
        if procs[-1].poll() is not None:
            raise RuntimeError(f"server exited ({procs[-1].returncode}); see {LOGS}/{slug}.server.txt")
        time.sleep(15)
    return procs


def spec_model(spec: str) -> str:
    return dict(kv.split("=", 1) for kv in spec.partition("@")[2].split(",") if "=" in kv)["model"]


def run_job(job, gpus, port, leaf, free, cond):
    slug, spec, repo, backend, _, skip = job[:6]
    extras = job[6] if len(job) > 6 else {}
    procs = []
    try:
        log(f"{slug}: starting on GPU {gpus}, port {port}")
        procs = start(slug, repo, backend, gpus, port, name=spec_model(spec), extras=extras)
        bad = smoke(spec, port, leaf)
        if bad:
            log(f"{slug}: FAILED smoke test: {bad}")
            return
        log(f"{slug}: server up, smoke test passed; running")
        env = {**os.environ, "JEVARENA_MM_QEV_URL": f"http://localhost:{port}"}
        leaves = [leaf] if not skip else None
        prefix_args = ["--leaf", leaf]
        r = subprocess.run([PY, "scripts/img/sweep_gpu.py", "--_run", spec, "--_skip", ",".join(skip), *prefix_args],
                           env=env, stdout=open(LOGS / f"{slug}.run.txt", "a"), stderr=subprocess.STDOUT)
        log(f"{slug}: run exited {r.returncode}; {todo(spec, leaf, skip)} items left")
    except Exception as e:
        log(f"{slug}: FAILED {type(e).__name__}: {e}")
    finally:
        for p in reversed(procs):
            p.terminate()
            try:
                p.wait(120)
            except subprocess.TimeoutExpired:
                p.kill()
        with cond:
            free.update(gpus)
            cond.notify_all()


def smoke(spec: str, port: int, leaf: str) -> str:
    """One item of each question type through the adapter; the error text if any call fails, else ''."""
    os.environ["JEVARENA_MM_QEV_URL"] = f"http://localhost:{port}"
    from jevarena_mm.registry import make_model
    m = make_model(spec)
    seen, picked = set(), []
    for it in (it for _, items in task_items(leaf) for it in items):
        kinds = frozenset((q.type, len(q.criteria or []) > 26) for q in it.request.questions.values())
        if kinds - seen:
            seen |= kinds
            picked.append(it)
        if len(picked) >= 5:
            break
    for it in picked:
        r = m.decide(it.request)
        if r.status != "ok":
            return f"{it.id}: {r.error}"
        if set(r.answers) != set(it.request.questions):
            return f"{it.id}: answers for {sorted(r.answers)} instead of {sorted(it.request.questions)}"
    return ""


def run_items(spec: str, skip: str, leaf: str, shard: str = "0/1") -> None:
    """shard k/n: only the items whose id hashes to k mod n, so n servers of one model can split its items."""
    import hashlib
    from jevarena_mm import media, runner
    from jevarena_mm.registry import make_model
    m = make_model(spec)
    sk = tuple(s for s in skip.split(",") if s)
    k, n = map(int, shard.split("/"))
    for task, items in task_items(leaf, sk):
        items = [it for it in items if int(hashlib.sha256(it.id.encode()).hexdigest(), 16) % n == k]
        ready = [it for it in items if all(media.check(ref) is None for _, ref in it.request.media())]
        calls = runner.run_task(m, task.name, ready, k=1, workers=1, shard=(k, n) if n > 1 else None)
        print(f"{task.name}: {calls} calls, {len(items) - len(ready)} items with missing media left", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--leaf", default="img")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port-base", type=int, default=8200)
    ap.add_argument("--_run")
    ap.add_argument("--_skip", default="")
    ap.add_argument("--_shard", default="0/1")
    ap.add_argument("--extra", nargs=3, metavar=("SLUG", "GPUS", "SHARD"),
                    help="one more server of SLUG on GPUS (e.g. 2) running shard k/n of its items, beside the sweep")
    a = ap.parse_args()
    if a._run:
        return run_items(a._run, a._skip, a.leaf, a._shard)
    if a.extra:
        LOGS.mkdir(parents=True, exist_ok=True)
        slug, g, shard = a.extra
        j = next(j for j in JOBS if j[0] == slug)
        gpus = [int(x) for x in g.split(",")]
        port = 8300 + gpus[0]
        procs = start(f"{slug}.extra{gpus[0]}", j[2], j[3], gpus, port, name=spec_model(j[1]), extras=j[6] if len(j) > 6 else {})
        log(f"{slug}: extra server on GPU {gpus}, shard {shard}")
        env = {**os.environ, "JEVARENA_MM_QEV_URL": f"http://localhost:{port}"}
        # "auto/n": claim chunks k/n one at a time (mkdir is atomic), so any number of extra servers share the work
        shards = [shard] if not shard.startswith("auto/") else None
        n_auto = int(shard.split("/")[1]) if shards is None else 0
        claims = LOGS / "claims" / slug
        claims.mkdir(parents=True, exist_ok=True)
        try:
            for k in range(n_auto) if shards is None else [None]:
                sh = shard if k is None else f"{k}/{n_auto}"
                if k is not None:
                    try:
                        (claims / str(k)).mkdir()
                    except FileExistsError:
                        continue
                log(f"{slug}: GPU {gpus} runs chunk {sh}")
                subprocess.run([PY, "scripts/img/sweep_gpu.py", "--_run", j[1], "--_skip", ",".join(j[5]), "--_shard", sh,
                                "--leaf", a.leaf], env=env, stdout=open(LOGS / f"{slug}.extra{gpus[0]}.run.txt", "a"),
                               stderr=subprocess.STDOUT)
        finally:
            for p in reversed(procs):
                p.terminate()
        log(f"{slug}: extra shard {shard} exited; {todo(j[1], a.leaf, j[5])} items left")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in JOBS if (j[0] in a.only if a.only else not j[0].endswith("-3gpu"))]
    pending = []
    for j in jobs:
        n = todo(j[1], a.leaf, j[5])
        log(f"{j[0]}: {n} items to run")
        if n:
            pending.append(j)
    if a.dry_run:
        return
    free, cond, threads, port = set(int(g) for g in a.gpus.split(",")), threading.Condition(), [], a.port_base
    while pending:
        with cond:
            j = next((j for j in pending if j[4] <= len(free)), None)
            if j is None:
                cond.wait()
                continue
            gpus = sorted(free)[:j[4]]
            free.difference_update(gpus)
        pending.remove(j)
        port += 1
        t = threading.Thread(target=run_job, args=(j, gpus, port, a.leaf, free, cond))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log("SWEEP DONE")


if __name__ == "__main__":
    main()
