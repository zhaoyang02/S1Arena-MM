# S1Arena-MM

### Which multimodal decisions can be left to System 1?

Many applications ask a model for a discrete decision about an image, a video or a sound: is this meme hateful, does
this clip show a foul, is this voice spoofed, should this request go to a slower model. S1Arena-MM measures how well
models make such decisions and how good their probabilities are. Every item holds an input (images, video or audio)
and one or more typed questions, and every model answers each question with a probability distribution over its
options. Fast System-1 decision models and multimodal LLMs read at the answer token answer the same items and get the
same scores. S1Arena-MM is the multimodal companion of [S1-Arena](https://github.com/TianyuCodings/S1-Arena).

[🤗 Media dataset](https://huggingface.co/datasets/Swindl/S1Arena-MM) · [Results](results/README.md) · [Models](docs/models.md)

## What is in it

| Part | Size |
|---|---|
| Image | 47 sources in 40 kinds of decision ("leaves"), 23,000 items; and 7 standard benchmarks (3,454 items), reported apart |
| Video | 36 sources in 30 leaves, short and long video, 15,000 items; and 5 subsets of public benchmarks (2,500 items), reported apart |
| Audio | 21 sources in 19 leaves, 10,560 items |
| Closed loop | 102 games (GameWorld, BALROG, VS-Bench) and 4 spoken-conversation environments |
| Leaderboard | 33 models: 12 System-1 decision models and 21 multimodal LLMs |

| Modality | Domains (task cards) | Leaves | Tasks | Items |
|---|---|---:|---:|---:|
| Image | [basic](docs/tasks_img_basic.md), [batch](docs/tasks_img_batch.md), [code](docs/tasks_img_code.md), [embodied](docs/tasks_img_embodied.md), [game](docs/tasks_img_game.md), [judge](docs/tasks_img_judge.md), [know](docs/tasks_img_know.md), [pro](docs/tasks_img_pro.md), [ret](docs/tasks_img_ret.md), [route](docs/tasks_img_route.md), [safety](docs/tasks_img_safety.md), [social](docs/tasks_img_social.md), [std](docs/tasks_img_std.md), [step](docs/tasks_img_step.md), [struct](docs/tasks_img_struct.md) | 47 | 54 | 26,454 |
| Short video | [basic](docs/tasks_vid_basic.md), [batch](docs/tasks_vid_batch.md), [bench](docs/tasks_vid_bench.md), [embodied](docs/tasks_vid_embodied.md), [fore](docs/tasks_vid_fore.md), [game](docs/tasks_vid_game.md), [judge](docs/tasks_vid_judge.md), [know](docs/tasks_vid_know.md), [pro](docs/tasks_vid_pro.md), [ret](docs/tasks_vid_ret.md), [safety](docs/tasks_vid_safety.md) | 23 | 27 | 11,500 |
| Long video | [bench](docs/tasks_lvid_bench.md), [dialog](docs/tasks_lvid_dialog.md), [embodied](docs/tasks_lvid_embodied.md), [judge](docs/tasks_lvid_judge.md), [know](docs/tasks_lvid_know.md), [pro](docs/tasks_lvid_pro.md), [ret](docs/tasks_lvid_ret.md), [route](docs/tasks_lvid_route.md) | 12 | 14 | 6,000 |
| Audio | [basic](docs/tasks_aud_basic.md), [dialog](docs/tasks_aud_dialog.md), [judge](docs/tasks_aud_judge.md), [know](docs/tasks_aud_know.md), [pro](docs/tasks_aud_pro.md), [ret](docs/tasks_aud_ret.md), [safety](docs/tasks_aud_safety.md) | 19 | 21 | 10,560 |

<!-- results:start -->
## Leaderboard

Video: 33 models on the 41 video tasks (17,500 items) and the 102 closed-loop games, one sample per item. Per-leaf and per-environment tables: [results/README.md](results/README.md).

| # | Model | Type | Static | Brier ↓ | Video | Long video | Benchmarks | Audio-visual | GameWorld | BALROG | VS-Bench | GameWorld realtime | Latency ↓ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | [d3](https://huggingface.co/vllm-sr/d3) | System-One | **66.9** | 0.419 | **66.9** | 60.3 | 76.2 | – | **40.9** | 29.2 | **45.3** | 34.2 | 780 |
| 2 | [clef](https://huggingface.co/Cloudflare/clef) | System-One | 66.5 | 0.425 | 65.9 | 60.5 | **77.5** | – | 30.9 | **33.3** | 25.3 | 28.1 | 1,140 |
| 3 | [Qwen3.5-35B-A3B (GPTQ-Int4)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) | MLLM | 66.0 | 0.427 | 65.5 | 59.4 | 77.3 | – | 31.2 | 7.0 | 25.0 | 33.3 | 237 |
| 4 | [OneJev-27B](https://huggingface.co/OmniJev/OneJev-27B) | System-One | 65.8 | **0.417** | 64.4 | **61.7** | 76.8 | – | 39.8 | 22.0 | 32.4 | **38.2** | 1,093 |
| 5 | [OneJev-9B](https://huggingface.co/OmniJev/OneJev-9B) | System-One | 64.6 | 0.428 | 64.6 | 57.6 | 74.5 | – | 32.2 | 4.1 | 13.0 | 35.7 | 446 |
| 6 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | MLLM | 63.9 | 0.457 | 63.9 | 55.7 | 75.2 | – | 29.5 | 3.7 | 24.8 | 27.5 | 240 |
| 7 | [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | System-One | 63.3 | 0.464 | 63.3 | 55.3 | 74.7 | – | 30.1 | 4.8 | 20.8 | 29.9 | 472 |
| 8 | [OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B) | System-One | 62.9 | 0.448 | 64.2 | 52.9 | 72.0 | – | 32.0 | 3.3 | 20.8 | 28.0 | 374 |
| 9 | [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | MLLM | 62.9 | 0.512 | 62.7 | 55.4 | 74.3 | – | 23.4 | 2.3 | 24.0 | 27.7 | 473 |
| 10 | [Qwen3.8-27B (FP8)](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | MLLM | 62.5 | 0.509 | 62.9 | 53.9 | 73.2 | – | 34.6 | 8.1 | 19.9 | 33.2 | 429 |
| 11 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | MLLM | 62.0 | 0.478 | 62.8 | 53.4 | 70.5 | – | 31.8 | 3.4 | 27.6 | 25.9 | 167 |
| 12 | [clef-omni](https://huggingface.co/Cloudflare/clef-omni) | System-One | 60.2 | 0.508 | 60.1 | 53.2 | 70.3 | **69.5** | 24.6 | 1.8 | 23.9 | 23.1 | 368 |
| 13 | [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | MLLM | 60.0 | 0.543 | 60.8 | 53.8 | 65.6 | 55.2 | 21.6 | 2.4 | 24.6 | 21.0 | 386 |
| 14 | [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) | MLLM | 59.6 | 0.617 | 62.0 | 46.8 | 67.7 | 59.8 | 20.5 | 2.5 | 17.0 | 26.6 | 537 |
| 15 | [Nemotron-3-Nano-Omni-30B-A3B](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) | MLLM | 58.7 | 0.546 | 59.7 | 48.9 | 68.0 | 65.5 | 18.8 | 5.3 | 22.0 | 22.6 | 458 |
| 16 | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | MLLM | 58.2 | 0.532 | 58.5 | 50.8 | 67.7 | 55.3 | 24.6 | 2.3 | 35.2 | 27.3 | 528 |
| 17 | [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | MLLM | 56.8 | 0.777 | 57.9 | 50.0 | 61.8 | – | 32.5 | 9.7 | 22.2 | 31.5 | 228 |
| 18 | [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | MLLM | 55.5 | 0.555 | 56.7 | 44.9 | 65.4 | 57.6 | 23.8 | 3.3 | 25.1 | 22.2 | 332 |
| 19 | [Qwen2.5-Omni-7B (AWQ)](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ) | MLLM | 55.4 | 0.550 | 56.9 | 44.6 | 64.6 | 57.5 | 21.2 | 2.6 | 28.6 | 22.0 | 343 |
| 20 | [Qwen2.5-Omni-7B (GPTQ-Int4)](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4) | MLLM | 55.2 | 0.551 | 57.3 | 42.5 | 64.6 | 56.4 | 23.4 | 2.4 | 28.8 | 23.4 | 335 |
| 21 | [Xor-26B-A4B](https://huggingface.co/juspay/Xor-26B-A4B) | System-One | 54.7 | 0.534 | 55.3 | 47.4 | 62.5 | – | 25.1 | 6.4 | 30.3 | 30.0 | 492 |
| 22 | [DiffusionGemma-26B-A4B (AWQ-INT4)](https://huggingface.co/cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4) | MLLM | 54.2 | 0.663 | 56.1 | 45.5 | 58.9 | – | 25.6 | 4.1 | 9.2 | 28.2 | 237 |
| 23 | [DiffusionGemma-26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | MLLM | 54.1 | 0.651 | 55.8 | 45.8 | 59.2 | – | 25.9 | 6.2 | 11.7 | 26.2 | 228 |
| 24 | [OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B) | System-One | 53.6 | 0.542 | 56.7 | 42.8 | 56.1 | – | 20.7 | 3.9 | 12.1 | 22.1 | 183 |
| 25 | [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | MLLM | 53.5 | 0.558 | 55.1 | 43.1 | 61.3 | 55.1 | 17.2 | 4.3 | 21.5 | 20.0 | 271 |
| 26 | [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6) | MLLM | 53.4 | 0.669 | 55.3 | 43.1 | 60.5 | 52.1 | 18.3 | 0.5 | 17.5 | 16.8 | 431 |
| 27 | [Jev-Omni + head-v3](https://huggingface.co/ferdinandl007/jev-omni-general-head-v3) | System-One | 53.2 | 0.546 | 54.6 | 45.8 | 58.0 | 40.1 | 24.5 | 4.8 | 15.6 | 27.0 | 187 |
| 28 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | MLLM | 53.0 | 0.555 | 54.8 | 41.7 | 62.0 | – | 13.6 | 1.3 | 15.9 | 13.9 | 136 |
| 29 | [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | System-One | 52.9 | 0.575 | 54.8 | 44.8 | 56.2 | 45.5 | 25.8 | 5.6 | 19.7 | 24.7 | 188 |
| 30 | [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it) | MLLM | 51.8 | 0.790 | 53.1 | 45.4 | 55.3 | 46.4 | 25.1 | 7.5 | 40.4 | 27.5 | 175 |
| 31 | [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | MLLM | 49.7 | 0.655 | 52.6 | 38.7 | 53.3 | 58.8 | 18.0 | 2.2 | 19.8 | 19.5 | 206 |
| 32 | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | MLLM | 47.9 | 0.632 | 49.3 | 38.8 | 54.9 | – | 18.5 | 0.0 | 9.6 | 20.5 | **94** |
| 33 | [raya-decision-v1](https://huggingface.co/yuyu199741/raya-decision-v1) | System-One | 46.1 | 0.706 | 50.4 | 30.7 | 50.5 | – | 15.2 | 0.0 | 24.3 | 19.9 | 234 |

- **Static**: accuracy on the 32 leaves every model takes (no audio); **Video** (20 leaves), **Long video** (7) and **Benchmarks** (5 public subsets) split them. **Audio-visual**: the 3 leaves that need audio, for the models that take audio.
- **Brier ↓**: Brier score of the answer distributions on the same leaves as Static, sum over options of (p − y)² (0–2, lower is better; a question without an answer counts as uniform).
- **Closed loop** (paused mode): GameWorld progress and BALROG progression, mean over environments; VS-Bench return, min-max normalized per game over the models above.
- **GameWorld realtime**: the same games with the game running for the model's latency before each action lands; **Latency ↓**: median milliseconds per closed-loop decision at batch size 1 on H100 80 GB PCIe ([latency and realtime](results/README.md#latency-and-realtime)).
- System-One: decision models with their own read-out (a decision head, a scoring server). MLLM: multimodal LLMs used as classifiers (label-token probabilities).

<!-- results:end -->

Image and audio: each score averages the tasks with equal weight; ECE uses ten equal-mass bins; Public is the mean
accuracy on the 7 standard image benchmarks; – means the model does not take audio.

| Model | Image acc. | Brier ↓ | ECE ↓ | Public | Audio acc. | Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|---|---|
| d3 | **69.7** | **0.377** | 0.095 | **87.2** | – | – | – |
| Clef | 67.3 | 0.416 | 0.115 | 85.1 | – | – | – |
| OneJev-27B | 67.8 | 0.388 | **0.077** | 85.9 | – | – | – |
| OneJev-9B | 64.5 | 0.426 | 0.084 | 84.0 | – | – | – |
| Clef-Flash | 62.8 | 0.465 | 0.113 | 83.7 | – | – | – |
| OneJev-4B | 62.4 | 0.444 | 0.082 | 83.6 | – | – | – |
| Clef-Omni | 62.0 | 0.488 | 0.139 | 83.9 | **62.9** | 0.495 | 0.241 |
| Xor-26B-A4B | 64.7 | 0.439 | 0.134 | 83.8 | – | – | – |
| OneJev-0.8B | 51.8 | 0.548 | 0.101 | 72.3 | – | – | – |
| Jev-Omni (head v3) | 55.1 | 0.543 | 0.148 | 78.1 | 48.4 | 0.541 | **0.132** |
| Jev-Omni | 57.0 | 0.546 | 0.166 | 77.3 | 47.7 | 0.568 | 0.179 |
| Raya-Decision-v1 | 41.2 | 0.739 | 0.232 | 65.8 | – | – | – |
| Qwen3.5-35B-A3B (Int4) | 64.5 | 0.461 | 0.148 | 84.8 | – | – | – |
| Qwen3.5-9B | 60.7 | 0.492 | 0.150 | 82.0 | – | – | – |
| Qwen3-VL-30B-A3B | 58.8 | 0.599 | 0.236 | 83.2 | – | – | – |
| Qwen3.8-27B (FP8) | 64.8 | 0.439 | 0.112 | 83.5 | – | – | – |
| Qwen3.5-4B | 57.3 | 0.521 | 0.136 | 81.0 | – | – | – |
| MiniCPM-o 4.5 | 59.3 | 0.563 | 0.207 | 81.4 | 57.9 | 0.548 | 0.237 |
| Qwen3-Omni-30B-A3B (Thinking) | 57.5 | 0.666 | 0.289 | 82.1 | 59.8 | 0.570 | 0.274 |
| Nemotron-3-Nano-Omni | 54.6 | 0.629 | 0.236 | 83.3 | 54.4 | 0.573 | 0.231 |
| Qwen3-Omni-30B-A3B | 54.3 | 0.672 | 0.278 | 81.0 | 62.5 | 0.520 | 0.235 |
| Gemma 4 26B-A4B | 64.5 | 0.629 | 0.318 | 82.6 | – | – | – |
| Qwen2.5-Omni-7B | 55.4 | 0.568 | 0.170 | 79.8 | 57.0 | 0.479 | 0.154 |
| Qwen2.5-Omni-7B (AWQ) | 54.5 | 0.570 | 0.168 | 78.9 | 56.7 | 0.480 | 0.151 |
| Qwen2.5-Omni-7B (GPTQ) | 54.7 | 0.560 | 0.153 | 79.1 | 55.5 | 0.499 | 0.163 |
| DiffusionGemma-26B-A4B (AWQ) | 60.7 | 0.579 | 0.244 | 81.8 | – | – | – |
| DiffusionGemma-26B-A4B | 61.9 | 0.543 | 0.217 | 82.1 | – | – | – |
| Qwen2.5-Omni-3B | 49.6 | 0.583 | 0.135 | 76.4 | 58.1 | **0.476** | 0.145 |
| MiniCPM-o 2.6 | 53.8 | 0.615 | 0.220 | 79.5 | 54.4 | 0.583 | 0.244 |
| Qwen3.5-2B | 46.6 | 0.647 | 0.199 | 75.6 | – | – | – |
| Gemma 4 12B | 61.6 | 0.647 | 0.316 | 78.9 | 53.0 | 0.730 | 0.359 |
| Phi-4-multimodal | 37.0 | 0.872 | 0.384 | 55.8 | 43.7 | 0.716 | 0.273 |
| Qwen3.5-0.8B | 38.8 | 0.695 | 0.210 | 64.5 | – | – | – |

## Install

Python 3.11 or later. In a clone of this repository:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[image,video,audio]'
```

Other extras: `ego4d` (Ego4D's clips), `audio-stream` (the spoken-conversation environments), `jev` (OneJev and
Jev-Omni on a GPU), `dev` (tests).

## Get the media

```bash
hf auth login                              # once your access request on the dataset page is granted
export JEVARENA_MM_MEDIA=/path/to/media    # or put it in .env
jevarena-mm fetch                          # everything, or --leaf img, vid, lvid, aud
jevarena-mm build                          # rebuild the items shipped as recipes
jevarena-mm verify                         # every file present, sha256 as the items record
```

`fetch` downloads the media the dataset hosts and gets the others from their publishers, under each source's own
terms. Everything takes about 600 GB, nearly all of it video; the image and audio media take about 9 and 5 GB. A few
sources need your own access first; without it their items are skipped and the rest runs:

| Step | Where | Needed for |
|---|---|---|
| Request access to the media dataset (name, affiliation, country; you agree to follow each file's terms), then `hf auth login` | [Swindl/S1Arena-MM](https://huggingface.co/datasets/Swindl/S1Arena-MM) | every task |
| Accept Winoground's license | [facebook/winoground](https://huggingface.co/datasets/facebook/winoground) (granted at once) | `img.ret/compositional` |
| Fill in BetterUp's form and accept the CANDOR Corpus License | [betterup/candor-full-duplex-bench](https://huggingface.co/datasets/betterup/candor-full-duplex-bench) (granted at once) | `aud.dialog/turn_timing` |
| Get an Ego4D license and its AWS keys (`aws configure` or `AWS_PROFILE`), and `pip install -e ".[ego4d]"` | [ego4d-data.org](https://ego4d-data.org/) (reviewed by Ego4D; takes days) | `lvid.ret/moment`, `vid.basic/state_change`, `lvid.route/escalation` |
| Register, download the InfographicVQA images (one archive holds every split), unpack them and set `INFOGRAPHICVQA_DIR` to that folder | [RRC portal](https://rrc.cvc.uab.es/?ch=17) | `img.struct/document_qa` |
| Request ArtEmis 2.0 through its form and set `ARTEMIS_CSV` to the CSV you receive | [artemisdataset-v2.org](https://www.artemisdataset-v2.org/) ("Dataset") | `img.social/art_emotion` |

The dataset card lists where each source comes from and its terms.

## Run the full evaluation

A model is named by a spec, `<kind>[:<checkpoint>][@<options>]`. The specs of the 33 models above are in
[docs/models.md](docs/models.md) (video, audio, closed loop) and [docs/models_img.md](docs/models_img.md) (image); a
model of your own needs an adapter ([Add a model or a task](#add-a-model-or-a-task)). With the media in place:

1. Install the closed-loop suites once (GameWorld, BALROG and VS-Bench at their pinned revisions) and set the `.env`
   lines the script prints:

   ```bash
   bash scripts/loop/install.sh /path/to/suites
   ```

2. Put your model's spec into [sweeps/full.toml](sweeps/full.toml) and run every task (image, video, long video,
   audio) and every closed-loop game, one sample per item:

   ```bash
   jevarena-mm sweep sweeps/full.toml
   ```

   Every answer and episode is cached, so a stopped sweep continues where it stopped. `--only static` or
   `--only envs` runs one half, `--shard I/N` splits the work over N processes or nodes, `jevarena-mm sweep-status
   sweeps/full.toml` shows how far it is, and `jevarena-mm sweep-submit sweeps/full.toml -p <partition>` submits it
   to Slurm ([scripts/slurm/README.md](scripts/slurm/README.md)).

3. Score it (no model calls):

   ```bash
   jevarena-mm sweep-report sweeps/full.toml   # per leaf and environment: results/summary/sweep_full_report.md
   jevarena-mm img-score --models <spec>       # image scores as in the table above
   ```

4. Latency and GameWorld in realtime ([docs/latency.md](docs/latency.md)), and the four spoken-conversation
   environments, which stream audio in real time ([docs/audio_streaming.md](docs/audio_streaming.md)), have their
   own commands.

To try it on a few items first: `jevarena-mm run --model random --leaf img.basic --limit 20`, then `jevarena-mm score
--models random --leaf img.basic --limit 20`; one game: `jevarena-mm play --model random --env
loop.game/vsbench/kuhn_poker --episodes 1`.

## Items

Every task produces the same item format (`jevarena_mm/schema.py`):

```json
{
  "id": "vid.basic.temporal.ssv2:c7d77c0ecbdc",
  "leaf": "vid.basic/temporal",
  "task": "vid.basic/temporal/ssv2",
  "request": {
    "state": {
      "text": "Observe the main pushing action in the clip. Use image coordinates.",
      "video": {"kind": "video", "uri": "ssv2/videos/163230.webm", "sha256": "bfdda5ea...", "window": [0.0, 3.666]}
    },
    "questions": {
      "answer": {"type": "choice",
                 "instructions": "During the main pushing action, which direction does the object move?",
                 "criteria": {"left_to_right": "From left to right.", "right_to_left": "From right to left."}}
    }
  },
  "gold": {"answer": {"label": "left_to_right"}},
  "meta": {"source": "Something-Something v2, validation split (Goyal et al., 2017)", "license": "..."}
}
```

- `request` is everything the model sees; `gold` and `meta` are never sent to a model.
- `state` holds text and media references (a file under `$JEVARENA_MM_MEDIA`, its sha256 and, for video and audio, a
  time window). How to sample frames or resample audio is the model adapter's choice, recorded with its results.
- Question types: `choice` (2 to 255 options), `boolean` and `score` (2 to 10 levels).
- An item's id is a hash of its request and gold, so any change to a question or a file changes the id.

## Metrics

All models are scored by the same code (`jevarena_mm/metrics.py`; image tasks: `jevarena_mm/metrics_img.py`), from
their distributions only:

- **Accuracy** (avg@k, and maj@k over k samples). A leaf's accuracy is the mean over its tasks.
- **Brier score** against the one-hot or soft target, and the **expected calibration error** (top option, ten
  equal-mass bins).
- **Latency** (median per decision at batch size 1) and **cost**; **NDCG** and **pair accuracy** where a task declares
  them; in closed loops, each suite's success or progress score.
- **A question without an answer scores 0**: a failed call, an unusable reply or a rejected input. Items in a modality
  the model does not take are left out and reported as coverage.

## Add a model or a task

A model adapter subclasses `ModelAdapter` (`jevarena_mm/models/base.py`), declares the media kinds it takes, and
implements `decide(request, sample)`, which returns one distribution per question; `@register_model` makes it
available as `--model <kind>@<options>`. A task subclasses `TaskAdapter` (`jevarena_mm/tasks/base.py`) and turns the
source's records into items; the modules under `jevarena_mm/tasks/` are examples. `jevarena-mm check`, `ruff` and the
tests run on every push.

## Data and licenses

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)). Each source keeps its own license; check it before you
use a source (the task cards, each item's `meta.license`, and the dataset card with its `LICENSES/`). Items and media
that S1Arena-MM generates are licensed CC BY 4.0. Some tasks contain offensive or disturbing content by design
(safety sources).

## Citation

```bibtex
@misc{s1arenamm2026,
  title  = {S1Arena-MM: Which Multimodal Decisions Can Be Left to System 1?},
  author = {Lin, Lang and Chen, Tianyu and Gu, Yi and Zhao, Yang and Song, Xinran and Liu, Xinyang and Cheng, Jianpeng
            and Zhou, Mingyuan},
  year   = {2026}
}
```
