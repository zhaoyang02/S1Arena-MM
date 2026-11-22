# Models: image

The image tasks run through model servers. Each adapter in [`jevarena_mm/models/qev.py`](../jevarena_mm/models/qev.py)
sends an item to a server's `POST /v1/systemone`, the System-One API of qev (OneJev's engine), and the server reads
each answer as a distribution over the options from the model's own scores (the logits of the option labels, or a
release's decision head), not from generated text. The servers are in
[`jevarena_mm/models/servers/`](../jevarena_mm/models/servers/) and run as their own processes. Image runs are scored
with `jevarena-mm img-score` ([scoring](#scoring)). The video and audio adapters are in [models.md](models.md).

`qev_onejev` and `qev_jev_omni` are OneJev and Jev-Omni behind a server; `onejev` and `jev_omni` are their in-process
adapters for video and audio.

## Run and score

```bash
jevarena-mm build --leaf img      # the image tasks shipped as recipes (Winoground, Hateful Memes, ArtEmis)
# start the model's server (below), then
export JEVARENA_MM_QEV_URL=http://localhost:8000
jevarena-mm run --model "qev_onejev@model=OneJev-27B,img=auto,debias=1" --leaf img --workers 1
jevarena-mm img-score --models "qev_onejev@model=OneJev-27B,img=auto,debias=1"
jevarena-mm img-compare           # results/summary/img/README.md: overall, per domain, per leaf
```

The recipes need their sources first ([Get the media](../README.md#get-the-media)). A task that is not built is left
out of the scores: its leaf is missing, and the overall is a mean over fewer leaves.

A call the server refuses with HTTP 400, 413 or 422 is final and scores as a failure. A call that found no server,
got another error, or reached a server answering as another model than the spec's `model=` is not cached as final and
is made again on the next run.

Runs recorded earlier with the specs below sent the same requests these adapters build to the same servers, so a new
run gives each model the same input.

### Every spec on one GPU node

```bash
python scripts/img/sweep_gpu.py --gpus 0,1,2 --leaf img
```

Run it from the repository root. For every spec with items left, it starts the server on the free GPUs the spec needs,
sends one item of each question type through the adapter (a spec that fails this smoke test is not run), runs the rest
and stops the server. Items whose image files are missing are left for later. `--only <slug> ...` picks specs by the
slugs in its `JOBS` table (`d3-3gpu`, d3 over three GPUs for items that run out of memory on two, runs only when named),
`--dry-run` only counts what is left, and the logs go to `logs/sweep/`. Settings, from the environment:

| Variable | Default | |
|---|---|---|
| `JEVARENA_MM_IMG_PYTHON` | the Python running the script | the Python of the servers and of the runs; it needs qev and the checkpoints' dependencies ([servers](#servers)) |
| `JEVARENA_MM_IMG_VLLM_PYTHON` | the Python of the other servers | the Python of `serve_dgemma` (the `qev_dgemma` specs); it needs vLLM and qev ([servers](#servers)) |
| `JEVARENA_MM_HF_HOME` | the standard Hugging Face cache | the cache holding the checkpoints, read as `$JEVARENA_MM_HF_HOME/hub`; each server loads the snapshot found there (the last by name if there are several) |
| `JEVARENA_MM_IMG_QUANT_PYTHONPATH` | the environment's `PYTHONPATH` | `PYTHONPATH` of the GPTQ and AWQ servers only, e.g. a folder holding gptqmodel |
| `JEVARENA_MM_LLAMA_SERVER` | `llama-server` on the `PATH` | llama.cpp's server, for the GGUF spec |

The GPTQ and AWQ servers build gptqmodel's Marlin kernels on first use, with `CC`, `CXX`, `CUDA_HOME`, `nvcc` and
`ninja` from your environment (`TORCH_CUDA_ARCH_LIST` limits the build to your GPUs).

## Specs

A spec is `<kind>@model=<served name>,img=auto,debias=1`, and the spec as written is the model id. These kinds take
images only. The image comparison's specs (`JOBS` in `scripts/img/sweep_gpu.py`) and the GPUs that script gives each
server (36 GiB each unless noted):

| Checkpoint | `kind` | `model=` | Server | GPUs |
|---|---|---|---|---|
| OmniJev/OneJev-27B | `qev_onejev` | `OneJev-27B` | `serve_qev` | 2 |
| OmniJev/OneJev-9B, -4B, -0.8B | `qev_onejev` | `OneJev-9B`, `OneJev-4B`, `OneJev-0.8B` | `serve_qev` | 1 |
| Qwen/Qwen3.8-27B-FP8 | `qev` | `Qwen3.8-27B-FP8` | `serve_qev` | 2 |
| Qwen/Qwen3.5-9B, -4B, -2B, -0.8B | `qev` | `Qwen3.5-9B`, `Qwen3.5-4B`, `Qwen3.5-2B`, `Qwen3.5-0.8B` | `serve_qev` | 1 |
| Qwen/Qwen3-Omni-30B-A3B-Instruct, -Thinking | `qev_readout` | `Qwen3-Omni-30B-A3B-Instruct`, `Qwen3-Omni-30B-A3B-Thinking` | `serve_any` | 2 |
| Qwen/Qwen2.5-Omni-7B, -3B | `qev_readout` | `Qwen2.5-Omni-7B`, `Qwen2.5-Omni-3B` | `serve_any` | 1 |
| Qwen/Qwen2.5-Omni-7B-AWQ, -GPTQ-Int4 | `qev_readout` | `Qwen2.5-Omni-7B-AWQ`, `Qwen2.5-Omni-7B-GPTQ-Int4` | `serve_any` with gptqmodel | 1 |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | `qev_readout` | `Qwen3.5-35B-A3B-GPTQ-Int4` | `serve_any` with gptqmodel | 1 |
| google/gemma-4-26B-A4B-it | `qev_readout` | `gemma-4-26B-A4B-it` | `serve_any` | 2 |
| google/gemma-4-12B-it | `qev_readout` | `gemma-4-12B-it` | `serve_any` | 1 |
| Qwen/Qwen3-VL-8B-Instruct | `qev_readout` | `Qwen3-VL-8B-Instruct` | `serve_any` | 1 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | `qev_readout` | `Qwen3-VL-30B-A3B-Instruct` | `serve_any` | 2 |
| openbmb/MiniCPM-o-4_5, MiniCPM-o-2_6 | `qev_readout` | `MiniCPM-o-4_5`, `MiniCPM-o-2_6` | `serve_any` (own modeling code) | 1 |
| microsoft/Phi-4-multimodal-instruct | `qev_readout` | `Phi-4-multimodal-instruct` | `serve_any` (own modeling code) | 1 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | `qev_readout` | `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | `serve_any` (own modeling code) | 2 |
| google/diffusiongemma-26B-A4B-it | `qev_diffusion` | `diffusiongemma-26B-A4B-it` | `serve_any` | 2 |
| google/diffusiongemma-26B-A4B-it | `qev_dgemma` | `diffusiongemma-26B-A4B-it` | `serve_dgemma` | 2 (vLLM, 0.85 of each GPU's memory) |
| cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 | `qev_dgemma` | `diffusiongemma-26B-A4B-it-AWQ-INT4` | `serve_dgemma` | 1 (vLLM, 0.85 of its memory) |
| mradermacher/OneJev-27B-GGUF (Q8_0) | `qev_gguf` | `OneJev-27B-GGUF-Q8_0` | `llama-server` and `serve_gguf` | 1 |
| akhilaaa3/Jev-Omni | `qev_jev_omni` | `Jev-Omni` | `serve_s1 --family jevomni` | 1 |
| akhilaaa3/Jev-Omni with ferdinandl007/jev-omni-general-head-v3 | `qev_jev_omni` | `Jev-Omni-general-head-v3` | `serve_s1 --family jevomni --head <decision_head.pt>` | 1 |
| Cloudflare/clef-flash | `s1_clef` | `clef-flash` | `serve_s1 --family clef` | 1 |
| Cloudflare/clef, clef-omni | `s1_clef` | `clef`, `clef-omni` | `serve_s1 --family clef` | 2 |
| juspay/Xor-26B-A4B (v1.1) | `s1_xor` | `Xor-26B-A4B-v1.1` | `serve_s1 --family xor` | 2 |
| yuyu199741/raya-decision-v1 | `s1_raya` | `raya-decision-v1` | `serve_s1 --family raya --processor <Qwen/Qwen3.5-2B snapshot>` | 1 |
| vllm-sr/d3 | `s1_d3` | `d3` | `serve_s1 --family d3` | 2 (`--max-memory 0=24GiB,1=36GiB`) |

`cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` does not load in transformers (its experts are stored as packed int4
tensors that transformers cannot convert), so `serve_any` cannot serve it; `serve_dgemma` runs it in vLLM with the read
of `dgemma:` ([DiffusionGemma on vLLM](#diffusiongemma-on-vllm-qev_dgemma)). The BF16 checkpoint has two specs that
read it differently, `qev_diffusion` and `qev_dgemma`, with their own model ids. Both of its servers answer as
`diffusiongemma-26B-A4B-it`, so the `model=` check below cannot tell them apart: send each spec to its own server.
`img-score` scores the image runs of any spec, so the adapters of [models.md](models.md) that take images are scored
the same way.

Spec options:

- `model`: the served name, which is the server's `--name`. A call fails if the server answers as another model.
- `img`: the image policy. `auto` caps the long edge at 2560 px for a single screen-like image (`info.content` is
  `screen`: documents, charts, screenshots, renders; without `info.content`, a PNG, TIFF, BMP or GIF file) and at
  1536 px for a single photo, at 1024 px for 2 to 4 images, 768 for 5 to 8 and 512 above. A number sets one cap for
  every image. An RGB or greyscale JPEG or PNG within the cap is sent as it is; a resized screen is sent as PNG, a
  resized photo as JPEG (quality 95).
- `debias`: how many cyclic shifts of the option order the server averages over; `1`, as in every spec above, is one
  pass in criteria order.
- `url`: the server, by default `$JEVARENA_MM_QEV_URL`, else `http://localhost:8000`. Prefer the variable: `url=` in the
  spec changes the model id.

Every adapter records what it sent of each image (size, format) in the response's `usage["input"]`.

## Servers

Each server serves one model at `/v1/systemone` (and `/health`). They need qev, OneJev's engine, at the commit the
OneJev adapter pins, with its `torch` extra (torch, torchvision, accelerate, safetensors; qev itself brings
transformers 5 or later, FastAPI and uvicorn):

```bash
pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev@81ce62f1597c91e46767d4d02d6ac2e18534fe94"
```

The adapters import qev's client, so the environment that runs `jevarena-mm run` needs qev too. GPTQ and AWQ
checkpoints also need gptqmodel (transformers loads the Omni GPTQ checkpoint through optimum), and the GGUF model
llama.cpp's `llama-server`. `serve_dgemma` runs DiffusionGemma in vLLM: its Python needs vLLM 0.31 or later (with
vllm-project/vllm#57250, DiffusionGemma's structured reads) and qev as above; `scripts/img/sweep_gpu.py` starts it with
`$JEVARENA_MM_IMG_VLLM_PYTHON`, so vLLM can have an environment of its own. `<snapshot>` below is a local Hugging Face
snapshot folder of the checkpoint; `CUDA_VISIBLE_DEVICES` picks the GPUs and `--max-memory` caps each of them; `--name`
must be the spec's `model=`.

```bash
# OneJev, Qwen3.5 and Qwen3.8 (qev's own engine), sharded over the visible GPUs
python -m jevarena_mm.models.servers.serve_qev --model <snapshot> --name OneJev-27B --port 8000 --max-memory 0=36GiB,1=36GiB

# other chat models, DiffusionGemma, and checkpoints with their own modeling code (MiniCPM-o, Phi-4, Nemotron)
python -m jevarena_mm.models.servers.serve_any --model <snapshot> --name Qwen2.5-Omni-7B --port 8000 --max-memory 0=36GiB

# DiffusionGemma in vLLM with the read of dgemma: (the qev_dgemma specs), tensor parallel over --tp visible GPUs
python -m jevarena_mm.models.servers.serve_dgemma --model <snapshot> --name diffusiongemma-26B-A4B-it-AWQ-INT4 --port 8000 --tp 1 --max-images 13

# System-One releases, each in its own format: --family jevomni, clef, xor, raya or d3
python -m jevarena_mm.models.servers.serve_s1 --family clef --model <snapshot> --name clef --port 8000 --max-memory 0=36GiB,1=36GiB

# the GGUF model: llama-server with the vision projector, then qev's llama.cpp engine on top
llama-server --host 127.0.0.1 --port 8100 -m <OneJev-27B Q8_0 .gguf> --mmproj <mmproj f16 .gguf> -ngl 999 -c 32768 -np 1 \
  --image-min-tokens 64 --image-max-tokens 16384
python -m jevarena_mm.models.servers.serve_gguf --llama-url http://127.0.0.1:8100 --tokenizer <OneJev-27B snapshot> \
  --name OneJev-27B-GGUF-Q8_0 --host 0.0.0.0 --port 8000
```

`serve_s1 --family jevomni --head <file>` loads a replacement decision head (head-v3: its `decision_head.pt`), and
`--family raya --processor <path>` the processor (default `Qwen/Qwen3.5-2B`, as raya's card does). `serve_dgemma` takes
no `--max-memory`: `--gpu` is the fraction of each GPU's memory vLLM takes (default 0.85), `--tp` the number of GPUs
the model is split over (tensor parallel; the sweep gives the AWQ-INT4 checkpoint one and the BF16 checkpoint two),
`--maxlen` the served context (default the checkpoint's) and `--max-images` the most images per prompt vLLM makes room
for (default 32; the sweep passes 13, the most an image item has). `serve_qev`, `serve_any`, `serve_s1` and
`serve_dgemma` write their settings (command, device map or GPUs, versions) to a file with `--meta <file>`.

## How an item becomes the model's input

Steps 1 and 2 are the same for every server except `serve_s1`, whose releases take step 1's request in their own
format ([System-One releases](#system-one-releases)).

### 1. Item to request (`build()` in `qev.py`, the same for every spec)

| In the item | Sent to the server |
|---|---|
| `state` text values | unchanged, in key order (the order they are shown in) |
| an image reference in `state` | the placeholder `<image:N>` in its key, and the image as media item N |
| a list of image references | a list of placeholders |
| `<image N>` inside a text value (MMMU and MMMU-Pro items) | `<image:k>` at its first mention; the `image_N` key itself is not sent |
| boolean question | qev's `Noul(instructions, criteria)`; the answer is P(yes) |
| choice question | `Choice(instructions, criteria)`, option keys and texts in criteria order |
| score question | `Score(instructions, criteria)`; level i is the i-th criteria text, low to high |
| images | under the `img` policy, as base64 data URIs |

`gold` and `meta` are never sent. Placeholders are numbered in order of appearance.

### 2. Request to prompt (`qev.prompt`, in the server)

Each question is one branch: the system prompt, then a user turn with the state block and that question. The branches
of an item share everything up to the end of the state block. The state is step 1's JSON (indent 2), and each
`<image:N>` is where image N's vision tokens go. Choice options are lettered in criteria order as
`<letter>. <key>: <text>`. A boolean is the two options `yes` and `no`, in that order. Score levels are `level 0` to
`level n-1`, after "Rate the state:" instead of "Question:". Past 26 options the labels go on with two letters (`AA`,
`AB`, ...) and the last line reads "Answer with one label: ...". An item of `img.basic/custom_categories/cub`:

```text
[system] Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase
         letter, with no explanation or reasoning.
[user]   <state>
         {
           "task": "Which value of has breast pattern does this bird have?",
           "image": "<image:1>"
         }
         </state>

         Question: Which value of has breast pattern does this bird have?

         Options:
         A. spotted: spotted
         B. striped: striped
         C. multi-colored: multi-colored
         D. solid: solid

         Answer with one letter: A, B, C, D.
```

The item's boolean question becomes `Question: This bird has bill length: about the same as head` with
`A. yes: the statement is true / the answer is yes` and `B. no: the statement is false / the answer is no`. A KonIQ
score question lists `A. level 0: bad (1)` to `E. level 4: excellent (5)`.

### 3. Prompt to model

| Server | Models | Chat template and answer prefix | Images | Answer read from |
|---|---|---|---|---|
| `serve_qev` (qev's `MMDecisionEngine`) | OneJev, Qwen3.5, Qwen3.8-27B-FP8 | the checkpoint's template, thinking off | resized as in OneJev's training; vision tokens at each `<image:N>` | the output-head rows of the option labels at the answer position, softmax over the labels |
| `serve_gguf` (qev's `LlamaCppEngine` on `llama-server`) | OneJev-27B Q8_0 | OneJev's template, thinking off | resized to the engine's pixel bounds, sent as PNG | `llama-server`'s top log-probabilities of the labels; labels missing from them come from a second call with a logit bias on the labels; questions are sent 8 per call |
| `serve_any` (`HFReadoutEngine`) | Qwen2.5-Omni, Qwen3-Omni, Qwen3.5-35B-A3B-GPTQ, Gemma 4, Qwen3-VL | the checkpoint's template (`enable_thinking=False` for Qwen models); a think block the template opens is closed empty; Omni models run the thinker only | the checkpoint's processor | the logits at the last position; each question is a whole prompt (no shared cache), and a request's questions are batched with left padding up to `--batch-tokens` (default 24,000) |
| `serve_any` with the checkpoint's own code (`readout_remote.py`) | MiniCPM-o, Phi-4-multimodal, Nemotron 3 Nano Omni | as above, with each card's image format: MiniCPM-o `(<image>./</image>)` tags (vision only); Phi-4 `<\|image_N\|>` tags, its text-only template and its vision LoRA; Nemotron image parts in its template (`enable_thinking=False`) | the checkpoint's processor | as above, one question per forward pass; Nemotron batches a request's questions over one pass of its vision encoder |
| `serve_any` (`DiffusionReadoutEngine`) | DiffusionGemma-26B-A4B-it | Gemma's template plus the empty thought channel the model writes before every answer | the checkpoint's processor | the label logits at canvas position 0 on the last denoising step of the model's own sampler (fixed seed) |
| `serve_dgemma` (`DGemmaVLLMEngine`, vLLM) | DiffusionGemma-26B-A4B-it and its AWQ-INT4 | Gemma's template, ending with the thought channel's open tag | the checkpoint's processor, in vLLM | the label log-probabilities at the answer slot of one read-only denoising pass, read from the row before the slot ([DiffusionGemma on vLLM](#diffusiongemma-on-vllm-qev_dgemma)) |

- A choice with more options than the server has labels is asked in chunks of at most one option fewer than the labels,
  each with the added option "none of the options listed here"; an option's probability is its probability in its
  chunk divided by the sum over chunks of 1 − P(none). A server has 26 labels unless its tokenizer keeps every
  two-letter label as one token (at most 256); `serve_any` and `serve_dgemma` log the count when they load. Eight
  tasks have choices with more than 26 options (the largest count in brackets): `img.basic/fine_grained/cub` (200),
  `img.game/board_twin/lichess_puzzles` (59), `img.know/clock_calendar/rendered` (144), `img.pro/damage/resisc45` (45),
  `img.pro/medical/slake` (46), `img.pro/phishing_page/phishpedia` (255), `img.step/grounding/androidcontrol` (218)
  and `img.step/gui_action/mind2web` (255).
- On GPUs without FP8 matrix multiplication (compute capability below 8.9), `serve_any` dequantizes FP8 checkpoints to
  BF16.
- Nemotron's Mamba2 layers run transformers' own NemotronH mixer on the checkpoint's weights
  (`patch_nemotron_mamba`): without the mamba-ssm kernels, the checkpoint's code falls back to a scan whose output at a
  position depends on the tokens after it once a sequence spans more than one 128-token chunk.

### DiffusionGemma on vLLM (`qev_dgemma`)

`serve_dgemma` reads DiffusionGemma as [`dgemma:`](models.md#dgemma-diffusiongemma) does with its defaults
(`layout=prompt`, one denoising step, one read, the row before the slot), on the prompt of steps 1 and 2:

- The prompt ends with the thought channel's open tag. A template that closes an empty thought block by itself (the
  AWQ-INT4 checkpoint's does) has that block taken off first, so both checkpoints get the same prompt.
- vLLM then denoises a 16-token canvas: the thought channel's close tag, the answer slot, the end-of-turn token and
  padding. The slot starts as a fixed random token, the same for every request; every other position holds the
  template. One read-only pass gives the log-probability of every label at the slot. In vLLM's output the row that
  predicts canvas position p is row p − 1 (the model keeps the autoregressive shift), so the read takes the row before
  the slot.
- A label missing from the returned row scores 5 below the row's lowest value. Label sets larger than 128 (vLLM's
  limit per request) are read in chunks of the same canvas, which give the same distribution. qev's engine turns the
  label scores into the answer, as for the other servers.

`qev_diffusion` reads the BF16 checkpoint another way: its prompt ends with the whole empty thought block, the model's
own sampler runs for several steps, and the answer is read at canvas position 0 on the last one. vLLM's output has no
row before canvas position 0, so `serve_dgemma` cannot give that read.

### System-One releases

`serve_s1` serves each release in its own format, as its model card or bundled code calls it; only the transport is
qev's.

| Kind | Release | Its format | Booleans and other details |
|---|---|---|---|
| `qev_jev_omni` | akhilaaa3/Jev-Omni, with its own head or a replacement (`--head`) | the repository's `jev_omni.py`: one forward pass per question; the user turn is the images, then its prompt over the state as compact JSON, the question and numbered options; its chat template, thinking off; softmax over the decision head's first K slots, read from the last hidden state | `Yes` / `No` (criteria texts appended), P(true) = P(Yes); choice options as `key` or `key: text`; score as the level texts; 2 to 256 options |
| `s1_clef` | Cloudflare/clef-flash, clef, clef-omni | `joint_schema_model.systemone(request)`: all questions in one forward pass; images before the state; the state as compact JSON with sorted keys; probabilities rounded to 4 decimals | choice keys in sorted order; `<image:N>` stays text; the release's `max_length` (16,384 tokens for clef and clef-flash, 64,000 for clef-omni): the state is cut to what fits after the images and the schema, and an item whose images and schema alone exceed it fails |
| `s1_xor` | juspay/Xor-26B-A4B v1.1 | `serving/xn_base_server.py` on transformers (the release serves it with SGLang): per question one row `Context:` (the state), `Question:`, `Options:` with `(A) ...` lines, in the chat template, then `Answer: (`; label log-probabilities, softmax with the release's temperatures (choice 2.0, noul 1.7, score 1.0); a choice is also scored with its options reversed, and the two are averaged | `no` / `yes` in that order (criteria texts appended), P(true) = P(yes); choices take 2 to 255 options, scores 2 to 10 levels |
| `s1_raya` | yuyu199741/raya-decision-v1 | the card's masked projection, no chat template: image placeholders, the state, the question, `A. ...` lines and "Answer with exactly one label from the options above."; logits of ` A` to ` Z` at the last position, softmax | `A. Yes` / `B. No`; score as the level texts; a choice over 26 options is asked in chunks of at most 25, as above |
| `s1_d3` | vllm-sr/d3 | `d3_runtime.D3.system_one(state, questions, images)`, the code behind `AutoModel(..., trust_remote_code=True)` on its card: per question one prompt; 255 answer codes read from the last hidden state by the checkpoint's readout matrix; 8 questions per forward pass; images first, at up to 1.6 MP | the backbone is sharded over the visible GPUs (the release loader takes one device); a request that runs out of memory is answered again with one question per forward pass |

## Scoring

`jevarena-mm img-score` (`jevarena_mm/metrics_img.py`) writes `results/summary/img/<model>.json`; `img-compare`
collects every scored model into `results/summary/img/README.md`. It differs from the scoring of the video and audio
tasks (`jevarena_mm/metrics.py`):

- An item's questions are averaged first, so every item weighs the same in its leaf, and a leaf pools the items of all
  its tasks. A domain is the mean of its leaves; the overall score is the mean of every leaf outside `img.std`, whose 7
  standard benchmarks have their own tables.
- A failed call or a missing answer scores accuracy 0 and the Brier score of an all-zero distribution (1 for a one-hot
  target). Items not run and items the model cannot take are left out (coverage).
- A reply's distribution is renormalized before scoring; a boolean answer is `true` when P(true) ≥ 0.5.
- Brier: the sum over options of (p − t)², with the annotators' distribution as the target where the item has one
  (KonIQ, ArtEmis, New Yorker), else one-hot; options in `gold.accept` count as one outcome. `brier_hard` uses one-hot
  targets everywhere. ECE: ten equal-mass bins of top-1 confidence. `-k` gives avg@k and maj@k over k samples.
