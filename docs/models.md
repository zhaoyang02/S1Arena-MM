# Models: video and audio

The adapters in [`jevarena_mm/models/`](../jevarena_mm/models/) that answer the video and audio tasks and play the
closed-loop games. Image tasks run through model servers instead: [models_img.md](models_img.md). Spoken
conversations, with audio in and out at the same time, have their own clients: [audio_streaming.md](audio_streaming.md).

A model is named by a spec, `kind[:argument][@option=value,...]`, e.g. `onejev:OmniJev/OneJev-9B` or
`lm:Qwen/Qwen3.5-4B@maxlen=131072`. Options that change what the model sees (frame rate and count, context length,
decision head) are part of the model id, and runs are cached under that id, so two input policies of one model are two
entries. Engine options (GPUs, memory fraction, parallelism) are not part of it: change them to fit your hardware and
the cached runs still count.

## Run and score

```bash
jevarena-mm run --model "onejev:OmniJev/OneJev-9B" --leaf vid      # vid: short video, lvid: long video, aud: audio
jevarena-mm score --models "onejev:OmniJev/OneJev-9B" --leaf vid
jevarena-mm play --model "onejev:OmniJev/OneJev-9B" --env loop.game/vsbench/kuhn_poker --episodes 1
```

- The adapters run inside the `jevarena-mm` process and load their model on the first call; `xor` starts and stops its
  own SGLang server. Only the image adapters need a server started by hand ([models_img.md](models_img.md)).
- Items in a modality the model does not take are recorded as unsupported, without a call, and left out of its scores.
- A rejected request is final and scores 0: an input over the model's context, media the model cannot take (e.g. two
  videos for Xor), or an out-of-memory error while the process has the GPU to itself. `run --retry-rejected` calls
  them again, e.g. on a larger GPU. Calls that failed for a transport or server error are retried on the next run.
- `--shard I/N` runs one of N disjoint parts of each task, so several processes or nodes can share a model's items.
- Every adapter reads only the item's time window, decodes frames with `jevarena_mm/media.py` (which draws the box some
  videos mark on their frames) and records what it gave the model (frames, timestamps, audio seconds) in each
  response's `usage["input"]`.

## Leaderboard specs

The specs of the 33 models of the [leaderboard](../results/README.md), as in the plans in [`sweeps/`](../sweeps/)
(`video_baselines.toml`, `system1_fulleval.toml` and `video_dev.toml` for video and the closed loop;
`audio_baselines.toml` runs the audio models with other engine options under the same model ids). Engine options
(`gpus`, `tp`, `pp`, `pplayers`, `gpu`, `batched`, `seqs`) can be changed for other GPUs; `maxlen`, `max_frames`,
`dequant` and `head` are part of the id.

| Model | Spec |
|---|---|
| d3 | `d3@gpus=2` |
| clef | `clef:Cloudflare/clef@gpus=2` |
| clef-flash | `clef` |
| clef-omni | `clef:Cloudflare/clef-omni@gpus=3` |
| OneJev-27B | `onejev` |
| OneJev-9B, 4B, 0.8B | `onejev:OmniJev/OneJev-9B`, `onejev:OmniJev/OneJev-4B`, `onejev:OmniJev/OneJev-0.8B` |
| Jev-Omni | `jev_omni` |
| Jev-Omni + head-v3 | `jev_omni@head=ferdinandl007/jev-omni-general-head-v3` |
| Xor-26B-A4B | `xor` |
| raya-decision-v1 | `raya` |
| Qwen3.5-35B-A3B (GPTQ-Int4) | `lm:Qwen/Qwen3.5-35B-A3B-GPTQ-Int4@gpu=0.8,maxlen=131072` |
| Qwen3.8-27B (FP8) | `lm:Qwen/Qwen3.8-27B-FP8@tp=2,gpu=0.7,maxlen=131072` |
| Qwen3.5-9B | `lm:Qwen/Qwen3.5-9B@gpu=0.7,maxlen=131072` |
| Qwen3.5-4B | `lm:Qwen/Qwen3.5-4B@maxlen=131072` |
| Qwen3.5-2B | `lm:Qwen/Qwen3.5-2B@maxlen=131072` |
| Qwen3.5-0.8B | `lm:Qwen/Qwen3.5-0.8B` |
| Qwen3-VL-30B-A3B-Instruct | `lm:Qwen/Qwen3-VL-30B-A3B-Instruct@pp=3,pplayers=10+19+19,maxlen=131072,gpu=0.88` |
| Qwen3-Omni-30B-A3B-Instruct | `lm:Qwen/Qwen3-Omni-30B-A3B-Instruct@pp=3,pplayers=10+19+19,maxlen=65536,gpu=0.88,batched=32768` |
| Qwen3-Omni-30B-A3B-Thinking | `lm:Qwen/Qwen3-Omni-30B-A3B-Thinking@pp=3,pplayers=10+19+19,maxlen=65536,gpu=0.88,batched=32768` |
| Qwen2.5-Omni-7B | `lm:Qwen/Qwen2.5-Omni-7B` |
| Qwen2.5-Omni-7B (AWQ) | `lm:Qwen/Qwen2.5-Omni-7B-AWQ` |
| Qwen2.5-Omni-7B (GPTQ-Int4) | `lm:Qwen/Qwen2.5-Omni-7B-GPTQ-Int4@dequant=<dir>` (see `dequant` below) |
| Qwen2.5-Omni-3B | `lm:Qwen/Qwen2.5-Omni-3B` |
| Gemma 4 26B-A4B | `lm:google/gemma-4-26B-A4B-it@tp=2,gpu=0.85,maxlen=65536` |
| Gemma 4 12B | `lm:google/gemma-4-12B-it@gpu=0.85,maxlen=65536` |
| DiffusionGemma-26B-A4B | `dgemma:google/diffusiongemma-26B-A4B-it@tp=2,gpu=0.9` |
| DiffusionGemma-26B-A4B (AWQ-INT4) | `dgemma:cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4@gpu=0.85` |
| MiniCPM-o 4.5 | `lm:openbmb/MiniCPM-o-4_5@gpu=0.85` |
| MiniCPM-o 2.6 | `lm:openbmb/MiniCPM-o-2_6@gpu=0.85` |
| Phi-4-multimodal | `lm:microsoft/Phi-4-multimodal-instruct@max_frames=16,gpu=0.85,maxlen=65536` |
| Nemotron-3-Nano-Omni-30B-A3B | `lm:nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@tp=2,maxlen=65536,gpu=0.9,seqs=64,batched=4096` |

The OneJev-27B and Jev-Omni video rows come from runs recorded earlier with the same input policy; for those items
both adapters rebuild the recorded inputs ([recorded presentation](#recorded-presentation)).

## Adapters

| Kind | Spec | Takes | Probabilities | Video frames (default) | Needs |
|---|---|---|---|---|---|
| `onejev` | `onejev[:<repo>][@fps=1,min_frames=2]` | image, video | the engine's own distribution | every frame of a 1 fps grid from the window start, at least 2 | one BF16-capable CUDA GPU, `jev` extra, the OneJev engine |
| `jev_omni` | `jev_omni[@fps=1,frames=16,head=<repo>]` | image, video, audio | its 256-slot decision head | 16 bin-centre frames of a 1 fps grid | one CUDA GPU, `jev` extra |
| `lm` | `lm:<Hugging Face id>[@...]` | what the checkpoint's config declares | label-token probabilities at the answer position | the family's default reading | vLLM 0.30 or later |
| `dgemma` | `dgemma:<Hugging Face id>[@...]` | image, video | label probabilities at the answer slot of a read-only denoising pass | 32 frames from the window start | vLLM 0.31 or later |
| `raya` | `raya[@frames=8]` | image, video | label-token logits at the last position | 8 frames spread over the window | one CUDA GPU, `jev` extra |
| `clef` | `clef[:<repo>][@gpus=N]` | image, video; clef-omni also audio | its joint-schema head | clef-flash, clef: the Qwen3-VL processor's default; clef-omni: a 1 fps grid fitted to its context | 1 to 3 CUDA GPUs, `jev` extra |
| `xor` | `xor[@tp=2,concurrency=1]` | 1 to 8 images or one video per request | its wrapper's label probabilities | 32 frames from the window start, as an MP4 | Apptainer, its SGLang image, two 40 GB GPUs (`tp=2`) |
| `d3` | `d3[@gpus=2]` | image, video | its answer-code readout | 2 fps, 4 to 32 frames over the window | one 80 GB GPU, or two 40 GB GPUs with `gpus=2` |
| `audio_gemini` | `audio_gemini:<Gemini model id>[@sample_rate=16000]` | one audio clip (and text) | verbalized | – | `audio-gemini` extra, `GEMINI_API_KEY` |
| `audio_qwen` | `audio_qwen` | one audio clip (and text) | verbalized | – | `audio-qwen` extra in its own environment, CUDA |
| `random` | `random[@modalities=image+video]` | every kind, or the listed ones | random | – | nothing (pipeline tests only) |

### OneJev (`onejev`)

- `onejev` is OneJev-27B; `onejev:OmniJev/OneJev-9B`, `-4B` and `-0.8B` are the other sizes. Each loads at a pinned
  commit ([pinned revisions](#pinned-revisions)); another repository needs `revision=<commit>` in the spec. The model id
  of a size other than 27B carries its revision, e.g. `OmniJev/OneJev-9B@fps=1,min_frames=2,revision=59429e8`.
- It runs through OneJev's official engine, `qev`: install it with
  `pip install "qev @ git+https://github.com/OmniJev/OneJev@81ce62f1597c91e46767d4d02d6ac2e18534fe94"`, or set
  `ONEJEV_ENGINE_DIR` to a checkout of that commit. Weights come from the Hub; `ONEJEV_MODEL_DIR` points OneJev-27B
  (only) at a local copy.
- BF16, with 32 decoder layers on the GPU and the rest offloaded to the CPU. `gpu_layers=N` changes the split
  (`gpu_layers=1000` keeps every layer on the GPU), `gpus=N` spreads the layers over N GPUs without offloading. Neither
  is part of the model id.
- Prompt: the state text, a media-timing note (each video's window and sampling grid), images as `<image:k>`
  placeholders (unless the text already places them), and per video a block `BEGIN VIDEO VIEW k; independent timeline`,
  `<video:k>`, `END VIDEO VIEW k`. Each video is every frame of the `fps` grid from the window start, passed as video at
  that rate.
- Short windows: the official video processor rejects a one-frame video, so a window that gives fewer than
  `min_frames` frames is read at `min_frames` frames per window length (for 2: the window start and middle).
- One question per request (every video item has one). Booleans use the engine's yes/no answer, with yes listed first.
  For choices with more than 20 options, the engine's probabilities are read before its 6-decimal rounding when the
  rounded ones do not sum to 1.

### Jev-Omni (`jev_omni`)

- Loaded with the model repository's own helper (`jev_omni.py`) and decision head at the pinned revision;
  `JEV_OMNI_MODEL_DIR` points to a local copy. `gpus=N` spreads the weights over N GPUs, e.g. for long audio (not part
  of the model id).
- `head=<repo>` loads another 256-slot decision head trained on the same backbone revision, as its card loads it (e.g.
  `ferdinandl007/jev-omni-general-head-v3`); it is part of the model id. A head whose card names another backbone
  revision is refused.
- Video: a grid of `fps` from the window start, of which the model's sampler keeps `frames` frames at the centres of
  equal bins (every grid frame when the grid is shorter). Several videos get one note per view.
- Audio: the whole window at 16 kHz mono, never cut, with a note on its source clock. A request longer than the model's
  context is rejected.
- Prompt: audio, images and the sampled frames, then the repository's prompt (numbered options) over the state text,
  the media-timing note and the sampled timestamps. Images that the text places with `<image:k>`, with no video
  (closed-loop screenshots), are interleaved with the text.
- Booleans are shown as Yes / No in a per-item order (the recorded order where there is one, else one fixed by the
  request's hash, about half No first), so a position bias does not push every item the same way.
- One question per request; the questions of a multi-question audio item are asked one at a time. Probabilities are
  the softmax over the head's first K slots, read from the last hidden state.

### lm: (vLLM)

Every question is one prompt: the system prompt "You are a classifier. Read the input and answer the question. Reply
with the label of your answer only.", then a user turn with the media, a line naming them, the state as JSON under
`Input:` (each medium as a placeholder such as `<video 1>`) and the question with labelled options. For the example item
of the [README](../README.md#items):

```text
Media, in the order shown: "<video 1>"

Input:
{
 "text": "Observe the main pushing action in the clip. Use image coordinates.",
 "video": "<video 1>"
}

Question: During the main pushing action, which direction does the object move?
Options:
A. From left to right.
B. From right to left.
Reply with the label of one option.
```

- Labels: A to Z, zero-padded numbers past 26 options (`001`, `002`, ...), `0` to `n-1` for score levels ("Reply with
  the number of one level."), `yes` / `no` for booleans ("Reply yes or no."; `Yes` and `YES` are summed in).
- Read-out: one step at the answer position gives each label's log-probability, and the softmax over the labels is the
  answer. Labels of several tokens are scored on a prefix tree of their tokens.
- Thinking is off (`think=0`): templates with a switch get it off, templates that always think get an empty thinking
  block. `think=<tokens>` lets the model think first. The read-out options (`mode`, default `label`; `think`; `effort`;
  `norm` with `mode=text`) are part of the model id.
- The media go before the text, in placeholder order. A state text that places each of its images once, in order, as
  `<image:k>` (the closed-loop packets) gets them there.
- Checkpoints that ship their own code (an `auto_map` in their config) run with `trust_remote_code`. `lm:` and
  `dgemma:` load the checkpoint's current Hub revision; they pin none.

Video and audio, as each family's own preprocessing reads a whole clip by default:

| Family | Video | Audio (16 kHz mono) |
|---|---|---|
| Qwen3.5, Qwen3.8, Qwen3-VL | the frames the checkpoint's video processor samples (2 fps, 4 to 768 frames, spread over the window), passed with their timestamps so the processor sizes them | – |
| Qwen2.5-Omni, Qwen3-Omni | qwen-omni-utils 0.0.9 defaults: 2 fps, 4 to 768 frames (an even count), within a pixel budget of 0.9 × the served context; frames that would overflow it are reduced to the most that fit, spread the same way; with audio or several videos, each video gets an equal share of what the audio leaves | Qwen2.5-Omni: the first 300 s (its processor's cut, made before vLLM, which would skip it); Qwen3-Omni: the whole track |
| Gemma 4, DiffusionGemma | 32 frames from the window start (every frame of a shorter window), each with its mm:ss timestamp | Gemma 4 12B: at most 30 s (its processor's 750 audio tokens of 40 ms); Gemma 4 26B-A4B and DiffusionGemma take none |
| MiniCPM-o 4.5, 2.6 | 1 fps, at most 64 frames spread over the window, as frames | the first 800 s (4.5) or 320 s (2.6): 8,000 audio tokens, which fit vLLM's encoder cache at the default batch size |
| Nemotron 3 Nano Omni | 1 fps, at most 64 timestamped frames (its card allows 128) | the first 600 s (its card allows an hour) |
| Phi-4-multimodal | as images (`video=frames`): `max_frames=16` frames spread over the window; each image cut into at most 4 crops (default 36) | the first 640 s |
| any other image model | as images: `max_frames` frames spread over the window, else one per second | – |

Phi-4-multimodal uses the checkpoint's vision LoRA when a prompt holds an image (a video's frames are images) and its
speech LoRA for audio alone, as its own code picks them. Nemotron needs a C++20 compiler as `CXX` (vLLM's processor for
it compiles a CPU kernel).

Options of `lm:`:

- `maxlen=<tokens>`: the served context (`-ctx<N>` in the model id); without it vLLM serves the checkpoint's own. It
  sets the Omni video budget and which requests fit.
- `dequant=<dir>`: load a dequantized copy of a GPTQ checkpoint quantized with `desc_act=True`, which vLLM 0.30 and
  later refuse (Qwen2.5-Omni-7B-GPTQ-Int4); `-dequant` in the model id. Make it with
  `python scripts/gptq_dequant.py <snapshot dir> <output dir>`.
- `quant=<method>` (vLLM's online weight quantization), `video=native` or `video=frames`, and `max_frames=N` (for
  `video=frames`) are part of the model id. `media=image+video+audio` overrides the kinds read from the config.
- Engine, not part of the id: `tp`, `pp`, `pplayers` (layers per pipeline stage, e.g. `10+19+19`; the first stage also
  runs the media encoders), `gpu` (memory fraction; default 0.6 with media), `batched` (vLLM's max_num_batched_tokens,
  which also sizes the encoder cache), `seqs` (max_num_seqs; Mamba hybrids need one state block per sequence), `chunk`
  (requests per vLLM call), `decoders` (media decoding threads).

### dgemma: (DiffusionGemma)

- The prompt and media of `lm:`. After the prompt, the model denoises a 16-token canvas that holds the answer template
  (an empty thought block, the answer label, the end of turn). The label position starts as a random token, every other
  position is pinned, and one read-only pass gives the log-probability of every label at the answer slot; the softmax
  over the labels is the answer.
- Labels must be single tokens: A to Z, then the two-letter labels the tokenizer keeps whole; a question whose options
  cannot all get one is rejected.
- Options: `steps` (denoising steps before the read, default 1), `reads` (reads with independent noise, averaged;
  default 1), `think`, `layout` (`prompt` or `canvas`) and `offset` are part of the model id, e.g.
  `google/diffusiongemma-26B-A4B-it@steps1-reads1-think0-prompt-1-video`, as are `maxlen`, `video` and `max_frames`
  (as for `lm:`). `tp` and `gpu` are not.
- Needs a vLLM with vllm-project/vllm#57250 (0.31 or later).

### raya

- As its card runs it: no chat template. The prompt is the media placeholders, a blank line, the state text, the
  question, the options as `A. text` lines and "Answer with exactly one label from the options above."; the logits of
  ` A`, ` B`, ... at the last position, softmaxed over the labels, are the answer.
- Booleans are the options `A. Yes` / `B. No`, score levels are options in level order. Past 26 options it uses the
  two-letter labels (` AA`, ...) that the tokenizer keeps whole; a question with more options is rejected.
- Images and video go through `Qwen/Qwen3.5-2B`'s processor at a pinned revision (the checkpoint ships none). Video:
  `frames` (8) frames at equal steps from the window start (every frame of a shorter window), passed with their
  metadata so the processor sizes them (`cap_pixels_per_frame`) and writes their timestamps.
- Images that the text places with `<image:k>` get their placeholders there.

### clef

- `clef` is clef-flash; `clef:Cloudflare/clef` and `clef:Cloudflare/clef-omni` are the others. Each runs with its
  repository's own `joint_schema_model.py` at a pinned commit. The request is the card's record (the state with media
  placeholders, the questions, the media); all questions of an item go through one forward pass, and each question's
  probabilities are the softmax of the head's logits.
- The repository's encoder puts every medium before the state, so `<image:k>` markers stay text.
- clef-flash and clef (`<repo>@video-default`): each video as the Qwen3-VL processor reads a whole video by default
  (2 fps, 4 to 768 frames spread over the window, its size budget). A record longer than the release's context is
  rejected.
- clef-omni (`Cloudflare/clef-omni@fps=1,fit=64000`): frames on a 1 fps grid from the window start (`fps=` changes it,
  part of the id), each scaled to at most 256 × 32 × 32 pixels, an even count. When the record exceeds the release's
  64,000-token context, each video gets the most frames that fit, spread over its grid; audio (16 kHz mono) is never
  cut. An item is rejected only when two frames per video still do not fit.
- `gpus=N` spreads the backbone over N GPUs (not part of the id). The video plans run clef with `gpus=2` and clef-omni
  with `gpus=3` on 40 GB GPUs.

### xor

- Served as its card serves it: the release's pinned SGLang image, run through Apptainer, and its wrapper
  `serving/xn_base_server.py` with the release's settings (context 131,072, chunked prefill 16,384, memory fraction
  0.80, label temperatures choice 2.0 / noul 1.70 / score 1.0, choices scored in both option orders). The adapter
  starts both on free local ports and stops them when the run ends; a run whose server dies ends, and the next run
  starts it again.
- Setup: pull the image once and point `XOR_SIF` at it; Apptainer comes from `APPTAINER_BIN` or the `PATH`.

  ```bash
  apptainer pull xor-sglang.sif docker://prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f
  export XOR_SIF=$PWD/xor-sglang.sif
  ```

  Server logs go to `XOR_LOG_DIR` (default `.cache/xor_logs`), compiled kernels to `XOR_CACHE` (default
  `/tmp/xor-jit-<user>`). On hosts whose NVIDIA driver is older than R580, the image's CUDA 13 forward-compatibility
  libraries are used.
- `tp=2` (SGLang tensor parallelism: the BF16 weights need two 40 GB GPUs) and `concurrency=1` (requests in flight: on
  40 GB GPUs concurrent 32-frame prefills run SGLang out of memory) are not part of the model id,
  `juspay/Xor-26B-A4B@v1.1`.
- Media: 1 to 8 images, or one video; other requests (two views, images with a video) are rejected. The video is the 32
  frames Gemma 4's processor samples from the window (every frame of a shorter window), written as an MP4 at 32 frames
  per window length, so the processor samples the same frames and labels them with their mm:ss timestamps. Frames and
  images are sent at most twice per side the size the processor resizes them to, so a request stays under the
  wrapper's 8 MB (if not, the MP4 quality steps down and images go from PNG to JPEG).
- Images go before the state text, where the wrapper puts them; `<image:k>` markers stay text.

### d3

- Loaded as its card loads it (AutoModel with `trust_remote_code` at the pinned revision) and asked through its own
  `system_one(state, questions, images, videos)`: the state with media placeholders, choices with their criteria,
  booleans as `noul`, scores with their levels. Each question is its own prompt; d3 reads its 255 answer codes at the
  last prompt position (8 questions per forward pass).
- Images go before the text, each at up to 1.6 MP. Video: as its runtime reads a video file, the window being the whole
  file: 2 frames per second, at least 4 and at most 32, spread over the window, at most 0.2 MP each, timestamped by d3.
  A request whose videos exceed 16,384 video tokens is rejected, as is a question d3 refuses. It takes no audio.
- Its runtime runs on one device and peaks at about 62-65 GiB: one 80 GB GPU per process (`d3`). `d3@gpus=2` loads
  the backbone over two 40 GB GPUs instead, with two fewer decoder layers on the first, which also holds the vision
  tower, the embeddings, the final norm and the readout; every other step is the runtime's own. `d3@device=cpu` runs
  it on the CPU, as a functional check.
- flash-linear-attention, when installed, runs its Gated DeltaNet layers faster (else its PyTorch reference
  implementation runs).

### Audio-only adapters

Both take one audio reference with optional state text, send the shared prompt that asks for a probability per option
as one JSON object (no transcript), and record the input's duration and rate. Details: [audio.md](audio.md).

- `audio_gemini:<model id>[@sample_rate=16000]`: Gemini generateContent with mono PCM16 WAV at 8 to 48 kHz (the Files
  API for requests over 20 MB), temperature 1, at most 4,000 output tokens. Needs `GEMINI_API_KEY`.
- `audio_qwen`: Qwen3-Omni-30B-A3B-Instruct at a pinned commit, BF16 with SDPA, sampled at temperature 1 with at most
  4,000 output tokens. It needs transformers 4.57.1 (the `audio-qwen` extra), so install it apart from the `jev` extra,
  and `CUDA_VISIBLE_DEVICES` set. A request over 16,384 tokens (with the output) is rejected, not truncated.

## Recorded presentation

`jevarena_mm/models/recorded_presentation.json.gz` holds, for 17,000 video items, how the recorded OneJev-27B and
Jev-Omni runs presented them: the media-timing note (in one of several wordings, and where it sits in the state), the
frame timestamps printed for Jev-Omni and the Yes/No order shown to Jev-Omni. Entries are keyed by a hash of the
request (`jev_common.request_key`). Both adapters use them, so they rebuild those runs' inputs; any other item,
including one whose request has changed, gets the generated note and a Yes/No order drawn from the request's hash.
Two frame rules come with it: where a recorded note names the last permitted observation, the last frame before the
window end is added after the grid, and a window the recorded runs treated as a prediction prefix gives Jev-Omni at
least two frames.

## Pinned revisions

The adapters load the checkpoints at these commits (`hf download <repo> --revision <commit>` fetches one before an
offline run); the OneJev engine is installed at its commit.

| Repository | Commit |
|---|---|
| OmniJev/OneJev-27B | `aec969d68e7e8e8b0df051e065c04304c0cd7163` |
| OmniJev/OneJev-9B | `59429e80aa6cbb82c49d4ff8802359044a084eee` |
| OmniJev/OneJev-4B | `c88e18653ceb7a8770716287f55fdefc79d6b588` |
| OmniJev/OneJev-0.8B | `c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8` |
| OneJev engine `qev` ([github.com/OmniJev/OneJev](https://github.com/OmniJev/OneJev)) | `81ce62f1597c91e46767d4d02d6ac2e18534fe94` |
| akhilaaa3/Jev-Omni | `5addda86ddee081a68fb067477ea100c221b8917` |
| ferdinandl007/jev-omni-general-head-v3 | `10e061b811f4add247dc0cbf06a67cb5c0233ad8` |
| Cloudflare/clef-flash | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| Cloudflare/clef | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| Cloudflare/clef-omni | `0db1cd2607d76a7bdb2a382f659e7b313079f84b` |
| juspay/Xor-26B-A4B (tag v1.1) | `28ca09c7b76648b24d3561c940d1a7460d12fb8e` |
| yuyu199741/raya-decision-v1 | `036f2fb7fb775e95246327200a532a24394b07bd` |
| Qwen/Qwen3.5-2B (raya's processor) | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| vllm-sr/d3 | `fc2d74bd48ec29ae42d4330a543913f7858e3f40` |
| Qwen/Qwen3-Omni-30B-A3B-Instruct (`audio_qwen`) | `26291f793822fb6be9555850f06dfe95f2d7e695` |

The software environments the models need (vLLM, transformers, the OneJev engine, Apptainer for Xor) are listed
under Environments in [latency.md](latency.md).
