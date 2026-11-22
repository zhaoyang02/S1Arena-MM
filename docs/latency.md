# Latency

How long one decision takes, measured the same way for every model: one request at a time (batch size 1), on one
fixed set of requests, on an NVIDIA H100 80 GB PCIe per model (`sweeps/latency_h100.toml`), with the model configured
as in its accuracy run. The accuracy runs cannot give this number: the vLLM models answer whole batches
(`decide_many`), so their time per request is amortized, and the other runs shared nodes, split models over 40 GB
GPUs and decoded with several threads at once.

## What is timed

- **One request**: the wall time of `model.decide(request)`, from the request to its probabilities. This includes
  decoding and sampling the media (the reading policy's cost) and every question of the request, as a deployment
  pays it and as a realtime episode charges it. Loading the weights is not included; it is reported as `load_s`.
- **Batch size 1**: the next request starts when the last one has returned. No concurrency, no batching.
- **Warm-up**: 12 requests (10 without audio, 2 with audio, so an audio tower warms too) answered after loading and
  not counted.
- **Noise**: the first 30 answered requests are timed a second time; the summary gives the ratio of the two times.
- **Output tokens**: the tokens a model writes before its answer is read (`output_tokens`). No model in the accuracy
  runs writes any: the System-1 models (d3, Clef, OneJev, Jev-Omni, Xor, Raya) read their answer from a decision head
  or the label logits at the last prompt position, and the open MLLMs run with thinking off (`think=0`, in every model
  id): `lm:` reads the label log-probabilities at the answer position of the prompt (`max_tokens=1` with
  `logprob_token_ids`, the token it samples is dropped; labels of several tokens take a few more short reads that hit
  the prefix cache), a model that always thinks answers after an empty thinking block written into the prompt, and
  DiffusionGemma reads its answer slot after one denoising step. A request's latency is then media decoding, the
  vision and audio encoders and the prefill, not decoding. With `think=B > 0`, `lm:` and `dgemma:` report the thinking
  tokens as output tokens.

## The latency set

`data/latency/v1.json` (`jevarena-mm latency-set`) names the requests every model gets:

| Input kind | Leaves | Items per leaf | Requests |
|---|---:|---:|---:|
| short (vid, without audio) | 20 | 10 | 200 |
| long (lvid, without audio) | 7 | 10 | 70 |
| av (leaves whose tasks need audio) | 3 | 20 | 60 |
| bench (public subsets) | 5 | 10 | 50 |
| loop (closed-loop decisions) | GameWorld 50, BALROG 30, VS-Bench 20 | | 100 |

- Items are drawn with a fixed seed; a leaf's sources take turns (`ret/moment`: 4 + 3 + 3 of its three sources), and
  no two requests, warm-up included, come from one source recording (`meta["group"]`), so no request reuses a medium
  that an earlier one decoded (vLLM's processed-media cache, a decoder's file cache).
- Items of recipe tasks are drawn from their recipe rows, so the set is the same before and after Ego4D or Video-MME
  items are rebuilt; the measurement needs them built (`jevarena-mm build`), and it checks every request before the
  weights load.
- **Closed-loop decisions** are decision packets copied from the attempt folders of recorded episodes
  ([envs.md](envs.md#playing)): answered steps with at least two candidates, at most one per episode, the environments
  of a suite in turn. The set lists them; the packets and their screenshots are under
  `$JEVARENA_MM_MEDIA/latency/v1/loop/`; `jevarena-mm fetch` downloads them with the rest of the media (only them:
  `jevarena-mm fetch --leaf latency`). For a set of your own, this command draws them from a folder of attempts, copies
  them there and writes their entries into the set, keeping its items:

  ```bash
  jevarena-mm latency-set --loop-only --loop-attempts "$JEVARENA_MM_ATTEMPTS" [--loop-model <slug>]
  ```

  Every model then gets the same packets. Their median is the summary's `median_ms`, the latency that realtime
  episodes charge (`envs.runner.realtime_latency`, for `play --realtime` and the sweeps). GameWorld deletes its
  temporary screenshots after an episode; a packet takes the copies the attempt kept under `native/` (found by hash).

## Hardware and configuration

- **One NVIDIA H100 80 GB PCIe per model**, on a node of its own (`#SBATCH --exclusive`): no other job shares the CPUs
  that decode the media, the disk or the PCIe bus. Four models ran out of memory on one GPU with their accuracy-run
  options and run tensor parallel on two (`gpus = 2` in the plan): Qwen3-VL-30B-A3B-Instruct,
  Qwen3-Omni-30B-A3B-Instruct and -Thinking, and Nemotron-3-Nano-Omni-30B-A3B. Every summary records the GPU, driver,
  CPU, package versions and commit.
- **Kind of GPU.** `latency` names it after the model number of the visible GPUs (`h100`, `a100`, ...; `--gpu-label`,
  or a plan's `gpu_label`, sets another) and writes under that name. The repository holds two: `h100`, the H100 PCIe
  measurement above, which the results page shows and realtime episodes charge, and `h100-sxm`, a replication on an
  H100 80 GB SXM machine shared with other users (`sweeps/latency_h100_sxm.toml`, with `gpu_label = "h100-sxm"` since
  both read as `h100`; `results/summary/latency/h100-sxm/`). `play --realtime` charges the `h100` measurement's
  `median_ms` (`--latency-gpu` names another kind of GPU's) and records the kind of GPU in each episode row
  (`latency_gpu`); without a measurement, the median latency of the model's static runs, fixed in
  `results/latency.json` the first time. A sweep plan's model takes
  `latency_gpu` likewise.
- **CUDA on an older driver.** The vLLM builds are CUDA 12.9; a node whose driver supports only 12.8 (driver 570, as on
  the H100 PCIe nodes measured here) runs them with NVIDIA's CUDA 12.9 forward-compatibility library first on
  `LD_LIBRARY_PATH` (conda-forge `cuda-compat`; vLLM JIT-compiles some sm_90 PTX, e.g. its Marlin MoE repack, which the
  570 driver refuses), `DG_JIT_NVCC_COMPILER` set to an nvcc >= 12.9 (conda-forge `cuda-nvcc`; DeepGEMM compiles the
  FP8 kernels with it) with a C++20 host compiler in `NVCC_CCBIN`, and `CUDA_HOME` set to a full CUDA toolkit
  (FlashInfer JIT-builds its sm_90 FP8 GEMM against the cuBLASLt, cuRAND and NVRTC headers). The summary's
  `machine.env` records these variables.
- **Flash attention first on the H100** (`TORCH_CUDNN_SDPA_DEPRIORITIZED=1`). On Hopper, torch 2.11 picks cuDNN's
  kernel for `scaled_dot_product_attention`, which builds a plan for every new sequence length (~0.7 s on the H100
  PCIe, cuDNN 9.19); the A100 picks flash attention. Without the variable, the Transformers models (Clef, OneJev,
  Jev-Omni, raya) pay this on nearly every request, a closed-loop decision 0.8-1.4 s slower on the H100 than on the
  A100, while a repeated request is fast (its plan cached). With it, torch takes flash attention as on the A100; the
  vLLM models' attention is vLLM's own and does not change. `machine.env` records it.
- **Specs**: `sweeps/latency_h100.toml`. Each is the spec of the model's accuracy run with only engine options changed
  (`tp`, `pp`, `pplayers`, `gpus`, `gpu_layers`, `gpu`; none is part of the model id). Options that are part of the
  id stay, also where 80 GB would allow more: Nemotron keeps 64 frames and 600 s of audio, the Omni models their
  65,536-token context. The timed model is then the scored model.
- **OneJev** keeps every decoder layer on the GPU (`gpu_layers=1000`); its default keeps 32 and offloads the rest to
  the CPU, which is fine for scoring but not for timing.
- **Environments.** A plan names the software each model needs (`env`); `latency-plan` passes it to the job as
  `JV_ENV`, for the site file to activate (`scripts/slurm/site.env.example`):
  - `jev`: the `jev` extra (transformers 5.17) and, for OneJev, its engine (`qev`, see `pyproject.toml`); for Xor also
    Apptainer and the pinned SGLang image as `$XOR_SIF` (`jevarena_mm/models/xor.py`). The System-One models.
  - `vllm`: vLLM 0.30 or later with `nvcc` (its kernels compile at engine start) and a libstdc++ of gcc 9 or later;
    Nemotron also a C++20 compiler as `$CXX` (its processor compiles a CPU kernel). Qwen2.5-Omni-7B-GPTQ-Int4 runs
    from a dequantized copy: `python scripts/gptq_dequant.py <snapshot> $JV_DEQUANT_DIR/Qwen2.5-Omni-7B-GPTQ-Int4-bf16`.
  - `vllm031`: vLLM 0.31 or later (vllm-project/vllm#57250), for DiffusionGemma.
  - `d3`: the `jev` extra with flash-linear-attention, for d3's own runtime.
- **Media on the node's own disk**: the job copies the set's media there first (`jevarena-mm latency-stage`), so the
  shared file system does not time in. Decoding uses `JEVARENA_MM_DECODE_THREADS=4`, as the accuracy runs.

## Running it

Copy `scripts/slurm/site.env.example` to `scripts/slurm/site.env` and fill it in, then:

```bash
hf download <repo> --revision <sha>                              # weights first: jobs run with HF_HUB_OFFLINE=1
jevarena-mm latency-plan sweeps/latency_h100.toml -p <h100 partition> -A <account> --site-env scripts/slurm/site.env
jevarena-mm latency-plan sweeps/latency_h100.toml -p <h100 partition> -A <account> --site-env scripts/slurm/site.env --go
```

The first prints one job per model; `--go` submits them. On a machine without Slurm, `scripts/slurm/latency_local.py`
runs a plan's models one after another through the same job script (`--dry-run` prints what would run):

```bash
python scripts/slurm/latency_local.py sweeps/latency_h100.toml --site-env scripts/slurm/site.env --dry-run
```

Each job (`scripts/slurm/latency.slurm`) stages the media, runs a **fit test** and then the set:

1. **Fit test** (`jevarena-mm latency --fit 10`): the 10 requests with the most media seconds of each input kind,
   which set the peak memory. It fails (exit 3) when one of them fails and its accuracy run did not fail on it too
   (requests that were over a model's context there, such as whole AMI meetings for Jev-Omni with head v3, stay
   failures by design). Then use the two-GPU spec in the comment beside the model (tensor parallel; `gpus = 2`), as
   for Clef-Omni (66 GiB of weights); `latency_local.py` runs a plan entry's `fallback` spec on two GPUs by itself.
2. **The set** (`jevarena-mm latency`): about 480 requests; 1 to 2 hours for most models.

The whole set is the measurement, also for realtime episodes, which charge its closed-loop median. `--kinds loop`
(`JV_KINDS` for the job, `kinds` in a plan or `latency-plan --kinds`) times some input kinds only, as a quick check:
it stages only their media and writes `<slug>.<kinds>.json` and `.jsonl` beside the measurement, never in its place.

`JEVARENA_MM_RUNS` must point to the accuracy runs' cache, so the probabilities can be compared with them and the fit
test knows which requests the accuracy run rejected. If the caches are on another machine, copy just the rows the set
needs there (a few MB a model) and point `JEVARENA_MM_RUNS` at that copy where you measure:

```bash
jevarena-mm latency-runs --plan sweeps/latency_h100.toml --out <folder>   # where the caches are; --models for some
```

## Outputs

- `results/latency/v1/<gpu>/<slug>.jsonl`: one row per request: kind, leaf, task and item (or environment and packet),
  media seconds, questions, `latency_ms`, the adapter's own time, status, what the adapter fed the model
  (`usage["input"]`: frames, audio seconds), `output_tokens` and the probabilities. These rows stay where they are
  measured (gitignored); the repository keeps the summaries below.
- `results/summary/latency/<gpu>/<slug>.json`, per input kind: requests, failed, unsupported, **median**, p90, mean, a
  bootstrap 95% interval of the median, and **GPU seconds per 1,000 requests** (mean seconds x GPUs x 1,000; the mean,
  because cost adds up), and the output tokens (mean and max per request). Also `load_s`, the warm-up times, the peak
  GPU memory (nvidia-smi, every process on the visible GPUs), the noise of the repeated requests, and the machine.
- `consistency` in the summary: the top option of each question against the model's cached accuracy run, and the
  largest probability difference. Below 98% agreement (`verdict: differs from the accuracy run`) the timed model is not
  the scored one (another spec, revision or engine); fix that before using the numbers.

## What to report

The results page ([results/README.md](../results/README.md#latency-and-realtime)) shows each model's medians per
input kind from `results/summary/latency/h100/` (`results-page --latency`) and the closed-loop median on the
leaderboard.

Per model and input kind: the median latency per request with its interval, and GPU seconds per 1,000 requests, with
the GPU and the number of GPUs. Do not average across input kinds. Throughput at large batches is a different number
(requests per GPU-hour with batching); it favors models whose serving stack batches and is reported apart, if at all.
