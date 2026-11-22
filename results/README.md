# Results

Video results of S1Arena-MM: every model below on the 41 video tasks (`vid`, `lvid`; 17,500 items) and the 102
closed-loop game environments (`loop.game`: GameWorld, BALROG, VS-Bench), one sample per item, each model with the
input policy of its model card or official preprocessing ([docs/models.md](../docs/models.md)). The runs follow the
plans in [`sweeps/`](../sweeps/). Latency is each model's batch-1 measurement on one H100 80 GB
([docs/latency.md](../docs/latency.md)), and GameWorld realtime replays the GameWorld games at that latency.

Nemotron-3-Nano-Omni-30B-A3B hears the first 600 s of audio and sees at most 64 frames (its card allows up to an hour
and 128 frames), as longer inputs did not fit on 40 GB GPUs.

## Leaderboard

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
- **GameWorld realtime**: the same games with the game running for the model's latency before each action lands; **Latency ↓**: median milliseconds per closed-loop decision at batch size 1 on H100 80 GB PCIe ([latency and realtime](#latency-and-realtime)).
- System-One: decision models with their own read-out (a decision head, a scoring server). MLLM: multimodal LLMs used as classifiers (label-token probabilities).

## Static accuracy per leaf

Accuracy per leaf (mean over its released tasks). 20 video and 7 long-video taxonomy leaves, 3 audio-visual leaves (models that take audio) and 5 public benchmark subsets; the best score of each table's row in bold.

### System-One decision models

| Leaf | d3 | clef | OneJev-27B | OneJev-9B | clef-flash | OneJev-4B | clef-omni | Xor-26B-A4B | OneJev-0.8B | Jev-Omni + head-v3 | Jev-Omni | raya-decision-v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vid.basic/action_candidates` | 95.4 | **96.6** | 95.0 | 96.0 | **96.6** | 94.8 | 95.8 | 94.4 | 88.0 | 92.2 | 91.2 | 87.8 |
| `vid.basic/action_category` | **88.6** | 86.2 | 82.8 | 86.2 | 87.4 | 87.2 | 86.8 | 81.2 | 68.2 | 30.4 | 31.0 | 10.6 |
| `vid.basic/repetition` | **25.8** | 25.6 | 16.4 | 14.8 | 19.6 | 13.8 | 14.0 | 18.0 | 10.4 | 14.4 | 18.2 | 13.8 |
| `vid.basic/state_change` | 73.0 | 70.9 | 75.9 | **79.5** | 72.4 | 77.3 | 71.7 | 66.3 | 59.7 | 63.5 | 67.2 | 54.9 |
| `vid.basic/temporal` | **96.6** | 95.2 | 92.6 | 90.4 | 82.4 | 85.4 | 68.4 | 59.4 | 63.4 | 78.4 | 74.2 | 55.6 |
| `vid.batch/quality` | 60.0 | 60.0 | 56.0 | 53.4 | 57.6 | 62.8 | 57.8 | 38.2 | **66.6** | 39.4 | 50.8 | 58.6 |
| `vid.embodied/feasibility` | 62.8 | 58.8 | 61.2 | **63.8** | 63.4 | 60.6 | 60.8 | 49.2 | 56.4 | 52.2 | 48.2 | 48.4 |
| `vid.embodied/procedure` | 51.0 | 51.4 | **53.2** | 51.8 | 51.0 | 50.8 | 50.0 | 52.8 | 50.0 | 50.8 | 50.8 | 49.6 |
| `vid.embodied/robot_success` | 60.0 | **63.6** | 56.2 | 59.8 | 56.4 | 57.8 | 50.0 | 52.0 | 49.2 | 54.2 | 48.6 | 49.6 |
| `vid.fore/next_action` | **70.6** | 60.4 | 51.8 | 46.2 | 55.6 | 46.6 | 54.0 | 63.2 | 36.2 | 56.2 | 51.2 | 31.8 |
| `vid.fore/pedestrian` | 55.4 | 52.9 | 55.0 | 59.8 | 50.4 | **61.0** | 49.1 | 48.9 | 55.5 | 49.2 | 50.5 | 55.5 |
| `vid.game/physics_bug` | 62.4 | **63.8** | 62.8 | 56.2 | 62.6 | 57.8 | 52.4 | 52.0 | 43.6 | 45.8 | 46.2 | 37.8 |
| `vid.judge/hallucination` | **85.2** | 81.8 | 83.8 | 82.0 | 82.6 | 79.6 | 80.2 | 79.6 | 64.2 | 73.6 | 76.4 | 69.6 |
| `vid.judge/t2v` | 39.8 | 41.0 | 39.0 | 40.4 | 38.8 | 41.6 | 32.0 | 30.6 | **45.8** | 39.4 | 37.2 | 35.8 |
| `vid.know/causal_qa` | **69.0** | 66.0 | 64.2 | 68.2 | 61.4 | 59.2 | 56.4 | 45.8 | 45.2 | 46.0 | 44.0 | 40.8 |
| `vid.know/physics` | **55.6** | 52.2 | 50.8 | 53.4 | 52.8 | 55.2 | 51.2 | 51.0 | 53.0 | 51.0 | 50.6 | 51.8 |
| `vid.pro/driving_risk` | 51.8 | 54.4 | 53.2 | 54.0 | 49.6 | 54.4 | 49.8 | 50.2 | 50.0 | 50.6 | **54.6** | 50.0 |
| `vid.pro/foul` | **48.6** | 47.0 | 47.4 | 46.0 | 40.0 | 48.0 | 41.4 | 0.0 | 43.4 | 27.2 | 26.8 | 41.2 |
| `vid.ret/text_match` | **98.6** | **98.6** | 97.8 | 97.6 | 98.4 | 97.4 | 96.8 | 95.6 | 92.2 | 95.6 | 94.6 | 86.0 |
| `vid.safety/violence` | 88.0 | 92.0 | **93.1** | 92.6 | 86.3 | 91.9 | 83.1 | 77.8 | 92.8 | 81.3 | 84.5 | 78.6 |
| `lvid.embodied/ego_plan` | 44.0 | **49.0** | 48.8 | 44.6 | 38.2 | 34.6 | 37.4 | 32.4 | 32.2 | 30.8 | 29.8 | 22.6 |
| `lvid.know/long_qa` | 60.8 | **63.0** | 57.0 | 54.2 | 61.6 | 48.8 | 59.0 | 53.2 | 35.0 | 50.4 | 50.8 | 32.8 |
| `lvid.know/needle` | 79.0 | 83.0 | **94.2** | 82.2 | 74.4 | 77.6 | 64.2 | 52.6 | 53.2 | 43.4 | 42.2 | 30.2 |
| `lvid.pro/surveillance` | 71.8 | 76.2 | **77.0** | 71.2 | 75.6 | 71.8 | 71.0 | 61.0 | 62.2 | 67.8 | 66.6 | 38.6 |
| `lvid.ret/moment` | **61.4** | 61.3 | 57.8 | 57.1 | 58.0 | 44.2 | 60.7 | 43.8 | 29.0 | 39.7 | 40.2 | 29.3 |
| `lvid.route/escalation` | **64.6** | 51.6 | 56.4 | 50.4 | 50.8 | 51.6 | 49.8 | 50.2 | 50.0 | 51.8 | 50.2 | 47.8 |
| `lvid.route/segment` | 40.2 | 39.2 | 40.8 | **43.6** | 28.6 | 41.4 | 30.4 | 38.4 | 38.0 | 36.8 | 33.6 | 13.6 |
| `lvid.dialog/meeting_decision` | – | – | – | – | – | – | **71.0** | – | – | 31.8 | 52.2 | – |
| `lvid.judge/meeting_summary` | – | – | – | – | – | – | **67.8** | – | – | 60.0 | 58.0 | – |
| `lvid.know/lecture` | – | – | – | – | – | – | **69.8** | – | – | 28.4 | 26.4 | – |
| `lvid.bench/egoschema` | 77.4 | 77.8 | **79.0** | 75.0 | 70.8 | 73.4 | 71.0 | 66.4 | 48.4 | 54.4 | 53.4 | 42.6 |
| `lvid.bench/vsi_bench` | 59.4 | 67.8 | 69.6 | 67.4 | **70.8** | 64.2 | 60.0 | 45.2 | 39.8 | 36.4 | 31.6 | 29.8 |
| `vid.bench/mvbench` | **75.2** | 73.8 | 68.6 | 67.0 | 65.6 | 63.8 | 65.2 | 55.6 | 52.6 | 57.8 | 56.6 | 53.8 |
| `vid.bench/nextqa` | 85.4 | 84.4 | 84.0 | 84.2 | **86.0** | 82.4 | 82.4 | 73.8 | 72.6 | 71.0 | 70.0 | 69.6 |
| `vid.bench/tempcompass` | 83.6 | **83.8** | 82.6 | 79.0 | 80.4 | 76.4 | 72.8 | 71.4 | 67.0 | 70.6 | 69.6 | 56.6 |

### Multimodal LLMs as classifiers

| Leaf | Qwen3.5-35B-A3B (GPTQ-Int4) | Qwen3.5-9B | Qwen3-VL-30B-A3B-Instruct | Qwen3.8-27B (FP8) | Qwen3.5-4B | MiniCPM-o 4.5 | Qwen3-Omni-30B-A3B-Thinking | Nemotron-3-Nano-Omni-30B-A3B | Qwen3-Omni-30B-A3B-Instruct | Gemma 4 26B-A4B | Qwen2.5-Omni-7B | Qwen2.5-Omni-7B (AWQ) | Qwen2.5-Omni-7B (GPTQ-Int4) | DiffusionGemma-26B-A4B (AWQ-INT4) | DiffusionGemma-26B-A4B | Qwen2.5-Omni-3B | MiniCPM-o 2.6 | Qwen3.5-2B | Gemma 4 12B | Phi-4-multimodal | Qwen3.5-0.8B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vid.basic/action_candidates` | 96.8 | 96.6 | **97.4** | 95.6 | 96.8 | 94.4 | 97.0 | 97.0 | 97.2 | 95.0 | 95.4 | 95.8 | 96.2 | 94.0 | 93.4 | 95.6 | 93.4 | 95.8 | 92.4 | 88.2 | 92.4 |
| `vid.basic/action_category` | **89.8** | 85.8 | 85.8 | 84.6 | 86.2 | 76.6 | 83.6 | 73.4 | 85.2 | 82.8 | 43.6 | 45.4 | 41.0 | 81.0 | 81.2 | 30.4 | 37.0 | 79.0 | 66.4 | 48.4 | 49.6 |
| `vid.basic/repetition` | 26.8 | 26.6 | 22.6 | **28.6** | 22.6 | 15.2 | 14.8 | 20.4 | 13.2 | 11.0 | 16.2 | 15.4 | 10.4 | 20.4 | 18.4 | 4.8 | 20.8 | 11.0 | 15.4 | 15.8 | 11.4 |
| `vid.basic/state_change` | 73.6 | 74.5 | 74.7 | 74.3 | 76.2 | **76.4** | 76.0 | 69.6 | 70.6 | 71.1 | 69.2 | 69.8 | 66.9 | 67.3 | 66.0 | 63.4 | 70.2 | 56.2 | 63.7 | 57.8 | 57.8 |
| `vid.basic/temporal` | 87.4 | **88.6** | 79.0 | 86.6 | 81.4 | 79.0 | 69.0 | 76.0 | 64.6 | 66.0 | 70.8 | 72.2 | 71.8 | 59.0 | 61.4 | 52.8 | 64.4 | 60.0 | 61.2 | 53.0 | 49.2 |
| `vid.batch/quality` | 68.0 | **68.8** | 49.8 | 48.6 | 56.2 | 42.2 | 66.8 | 35.4 | 21.8 | 41.6 | 26.4 | 25.6 | 46.8 | 51.6 | 46.6 | 54.4 | 38.0 | 27.4 | 33.0 | 59.6 | 0.2 |
| `vid.embodied/feasibility` | 65.6 | 61.2 | **66.4** | 60.4 | 62.0 | 57.8 | 61.8 | 53.0 | 63.6 | 52.6 | 64.4 | 65.0 | 64.0 | 50.6 | 48.6 | 62.6 | 52.2 | 53.4 | 42.8 | 44.6 | 54.2 |
| `vid.embodied/procedure` | 50.4 | 50.2 | 50.4 | 50.0 | 50.0 | **52.2** | 49.2 | 50.0 | 50.2 | 49.6 | 50.0 | 50.0 | 50.0 | 49.8 | 50.0 | 50.6 | 50.8 | 49.8 | 50.0 | 50.8 | 50.0 |
| `vid.embodied/robot_success` | **60.6** | 54.0 | 54.0 | 50.6 | 60.4 | 52.6 | 56.8 | 51.6 | 54.8 | 54.4 | 53.8 | 53.6 | 51.6 | 50.6 | 51.4 | 51.2 | 49.6 | 50.2 | 51.0 | 50.0 | 50.0 |
| `vid.fore/next_action` | **67.0** | 53.8 | 64.4 | 65.4 | 59.2 | 59.0 | 59.6 | 64.0 | 49.2 | 59.0 | 49.2 | 49.8 | 48.2 | 45.4 | 48.2 | 48.8 | 47.6 | 43.2 | 59.6 | 57.0 | 47.8 |
| `vid.fore/pedestrian` | 55.4 | 49.4 | 51.3 | 49.4 | 54.5 | 50.1 | 47.9 | 54.1 | 48.4 | 48.5 | 55.1 | 56.2 | 54.4 | 49.2 | 48.4 | 51.5 | **56.5** | 51.8 | 48.4 | 51.6 | 47.5 |
| `vid.game/physics_bug` | 60.0 | 57.8 | 57.8 | **60.8** | 57.2 | 51.0 | 54.8 | 49.8 | 55.2 | 53.4 | 50.0 | 50.4 | 49.6 | 47.6 | 48.0 | 46.2 | 51.2 | 49.8 | 43.6 | 40.6 | 42.0 |
| `vid.judge/hallucination` | **84.2** | 82.2 | 83.0 | 77.4 | 82.2 | 82.4 | 75.4 | 82.0 | 82.0 | 78.2 | 82.0 | 81.0 | 81.2 | 80.6 | 78.0 | 76.4 | 78.2 | 63.8 | 76.4 | 76.8 | 67.8 |
| `vid.judge/t2v` | 15.4 | 23.4 | 17.0 | 21.4 | 11.6 | 27.6 | **32.4** | 9.8 | 29.8 | 20.6 | 17.0 | 18.0 | 20.0 | 24.8 | 23.2 | 16.2 | 18.8 | 32.2 | 17.0 | 8.6 | 18.4 |
| `vid.know/causal_qa` | 65.6 | **67.0** | 65.8 | 65.0 | 65.4 | 56.4 | 64.0 | 66.0 | 57.2 | 49.4 | 55.8 | 51.8 | 52.6 | 44.8 | 45.6 | 50.6 | 50.2 | 46.2 | 44.6 | 40.4 | 39.8 |
| `vid.know/physics` | 53.4 | 51.8 | 49.4 | 52.2 | 50.8 | 53.2 | 52.8 | 50.4 | **55.2** | 49.8 | 50.6 | 52.8 | 51.8 | 46.2 | 50.4 | 53.0 | 53.6 | 51.4 | 51.4 | 51.0 | 52.4 |
| `vid.pro/driving_risk` | 53.2 | 53.6 | 57.4 | 51.4 | 50.0 | 57.4 | 50.2 | 51.2 | 50.0 | 51.0 | 50.6 | 52.6 | 52.8 | 50.0 | 50.0 | **60.8** | 49.2 | 51.0 | 50.0 | 50.0 | 50.0 |
| `vid.pro/foul` | 43.6 | 45.2 | 40.6 | **48.2** | 44.6 | 47.0 | 42.4 | 48.0 | 38.2 | 41.6 | 42.4 | 42.2 | 45.2 | 29.2 | 28.4 | 44.8 | 38.8 | 37.4 | 29.2 | 24.4 | 36.4 |
| `vid.ret/text_match` | **97.8** | 97.6 | 96.8 | 97.0 | 97.6 | 96.4 | 97.0 | 97.0 | 97.6 | 96.2 | **97.8** | 97.6 | **97.8** | 96.8 | 96.6 | 97.0 | 96.4 | 95.0 | 93.0 | 94.0 | 92.6 |
| `vid.safety/violence` | 94.5 | 90.0 | 90.1 | 90.7 | 92.1 | 89.7 | 88.1 | **95.9** | 85.1 | 85.5 | 94.0 | 93.3 | 92.8 | 83.3 | 82.7 | 91.3 | 89.4 | 91.0 | 73.3 | 89.7 | 77.1 |
| `lvid.embodied/ego_plan` | 45.4 | 38.0 | 45.6 | **48.0** | 39.4 | 29.4 | 33.8 | 28.2 | 36.8 | 35.2 | 27.8 | 27.8 | 26.4 | 26.8 | 29.4 | 29.0 | 27.6 | 33.0 | 32.6 | 29.4 | 27.2 |
| `lvid.know/long_qa` | **61.8** | 57.6 | 58.4 | 60.0 | 51.2 | 56.6 | 53.0 | 54.8 | 53.2 | 50.4 | 48.6 | 46.4 | 42.6 | 46.6 | 45.2 | 46.4 | 49.2 | 47.0 | 47.2 | 41.8 | 37.0 |
| `lvid.know/needle` | **84.6** | 72.0 | 74.4 | 64.4 | 73.6 | 71.6 | 50.2 | 66.4 | 64.2 | 60.0 | 45.8 | 45.2 | 42.8 | 48.6 | 47.8 | 35.2 | 38.8 | 45.6 | 50.6 | 40.2 | 43.0 |
| `lvid.pro/surveillance` | **75.6** | 72.8 | 72.6 | 54.8 | 71.4 | 73.8 | 53.0 | 56.4 | 69.8 | 69.0 | 70.6 | 69.6 | 65.2 | 69.4 | 69.4 | 48.6 | 51.8 | 40.2 | 63.4 | 38.8 | 41.4 |
| `lvid.ret/moment` | 58.7 | 59.8 | 52.3 | **60.0** | 54.3 | 52.9 | 51.9 | 50.7 | 45.1 | 49.3 | 47.9 | 47.5 | 45.7 | 41.2 | 41.5 | 49.3 | 49.6 | 41.2 | 37.3 | 36.3 | 41.6 |
| `lvid.route/escalation` | 51.4 | 50.6 | 47.2 | 51.0 | 49.0 | **54.6** | 48.0 | 52.0 | 48.0 | 49.4 | 48.2 | 48.8 | 50.2 | 48.6 | 48.6 | 53.8 | 47.8 | 49.6 | 51.0 | 47.8 | 49.6 |
| `lvid.route/segment` | 38.4 | **39.4** | 37.2 | 38.8 | 35.2 | 37.6 | 37.8 | 34.0 | 38.6 | 36.8 | 25.2 | 27.0 | 24.6 | 37.6 | 38.4 | 39.2 | 36.6 | 35.2 | 36.0 | 36.8 | 32.0 |
| `lvid.dialog/meeting_decision` | – | – | – | – | – | 51.4 | 50.8 | 61.8 | 53.2 | – | 55.6 | 56.8 | 55.4 | – | – | 56.0 | 47.8 | – | 30.6 | **68.4** | – |
| `lvid.judge/meeting_summary` | – | – | – | – | – | 67.8 | 68.4 | **71.6** | 64.4 | – | 65.0 | 66.0 | 66.0 | – | – | 64.4 | 65.8 | – | 52.4 | 58.8 | – |
| `lvid.know/lecture` | – | – | – | – | – | 46.4 | 60.2 | **63.2** | 48.4 | – | 52.2 | 49.8 | 47.8 | – | – | 45.0 | 42.6 | – | 56.2 | 49.2 | – |
| `lvid.bench/egoschema` | **77.0** | 74.2 | 73.0 | 76.0 | 72.2 | 65.0 | 68.2 | 68.4 | 65.4 | 64.6 | 70.2 | 68.4 | 68.0 | 61.4 | 61.8 | 63.6 | 56.6 | 56.2 | 53.4 | 51.4 | 46.0 |
| `lvid.bench/vsi_bench` | 68.8 | **70.4** | 64.4 | 53.2 | 54.0 | 49.2 | 47.0 | 44.6 | 54.4 | 43.4 | 34.4 | 33.4 | 33.2 | 39.6 | 41.2 | 31.0 | 37.8 | 48.2 | 36.0 | 32.6 | 41.6 |
| `vid.bench/mvbench` | **73.0** | 72.0 | 72.4 | 71.8 | 68.2 | 63.2 | 66.8 | 70.8 | 65.6 | 56.0 | 71.8 | 71.6 | 71.0 | 52.4 | 51.6 | 66.4 | 61.6 | 62.6 | 51.4 | 51.0 | 56.0 |
| `vid.bench/nextqa` | **87.0** | 84.0 | 84.8 | 83.0 | 81.4 | 81.8 | 83.8 | 83.2 | 80.8 | 75.0 | 81.0 | 80.2 | 80.4 | 72.4 | 71.6 | 79.2 | 78.4 | 77.2 | 68.2 | 68.8 | 70.0 |
| `vid.bench/tempcompass` | 80.6 | 75.4 | 76.8 | **82.2** | 76.8 | 69.0 | 72.6 | 72.8 | 72.4 | 70.2 | 69.6 | 69.2 | 70.6 | 68.6 | 69.6 | 66.4 | 68.2 | 65.8 | 67.4 | 62.6 | 61.0 |

## Calibration

Brier score of the answer distributions (lower is better): sum over options of (p − y)², 0–2, a question without an answer counted as uniform; mean over the 32 leaves of Static and over its Video, Long video and Benchmarks groups.

| # | Model | Type | Static | Video | Long video | Benchmarks |
|---:|---|---|---:|---:|---:|---:|
| 1 | OneJev-27B | System-One | **0.417** | 0.438 | **0.428** | 0.321 |
| 2 | d3 | System-One | 0.419 | **0.430** | 0.455 | 0.323 |
| 3 | clef | System-One | 0.425 | 0.442 | 0.457 | **0.310** |
| 4 | Qwen3.5-35B-A3B (GPTQ-Int4) | MLLM | 0.427 | 0.442 | 0.461 | 0.318 |
| 5 | OneJev-9B | System-One | 0.428 | 0.434 | 0.479 | 0.336 |
| 6 | OneJev-4B | System-One | 0.448 | 0.440 | 0.527 | 0.366 |
| 7 | Qwen3.5-9B | MLLM | 0.457 | 0.463 | 0.521 | 0.344 |
| 8 | clef-flash | System-One | 0.464 | 0.476 | 0.514 | 0.349 |
| 9 | Qwen3.5-4B | MLLM | 0.478 | 0.480 | 0.537 | 0.386 |
| 10 | clef-omni | System-One | 0.508 | 0.528 | 0.536 | 0.392 |
| 11 | Qwen3.8-27B (FP8) | MLLM | 0.509 | 0.518 | 0.590 | 0.362 |
| 12 | Qwen3-VL-30B-A3B-Instruct | MLLM | 0.512 | 0.521 | 0.568 | 0.399 |
| 13 | Qwen3-Omni-30B-A3B-Instruct | MLLM | 0.532 | 0.526 | 0.609 | 0.450 |
| 14 | Xor-26B-A4B | System-One | 0.534 | 0.530 | 0.579 | 0.486 |
| 15 | OneJev-0.8B | System-One | 0.542 | 0.516 | 0.627 | 0.529 |
| 16 | MiniCPM-o 4.5 | MLLM | 0.543 | 0.544 | 0.582 | 0.486 |
| 17 | Nemotron-3-Nano-Omni-30B-A3B | MLLM | 0.546 | 0.542 | 0.626 | 0.447 |
| 18 | Jev-Omni + head-v3 | System-One | 0.546 | 0.537 | 0.587 | 0.526 |
| 19 | Qwen2.5-Omni-7B (AWQ) | MLLM | 0.550 | 0.540 | 0.631 | 0.479 |
| 20 | Qwen2.5-Omni-7B (GPTQ-Int4) | MLLM | 0.551 | 0.529 | 0.661 | 0.484 |
| 21 | Qwen3.5-2B | MLLM | 0.555 | 0.542 | 0.635 | 0.491 |
| 22 | Qwen2.5-Omni-7B | MLLM | 0.555 | 0.548 | 0.638 | 0.467 |
| 23 | Qwen2.5-Omni-3B | MLLM | 0.558 | 0.546 | 0.644 | 0.490 |
| 24 | Jev-Omni | System-One | 0.575 | 0.556 | 0.635 | 0.566 |
| 25 | Qwen3-Omni-30B-A3B-Thinking | MLLM | 0.617 | 0.582 | 0.781 | 0.528 |
| 26 | Qwen3.5-0.8B | MLLM | 0.632 | 0.619 | 0.709 | 0.576 |
| 27 | DiffusionGemma-26B-A4B | MLLM | 0.651 | 0.643 | 0.732 | 0.570 |
| 28 | Phi-4-multimodal | MLLM | 0.655 | 0.627 | 0.738 | 0.651 |
| 29 | DiffusionGemma-26B-A4B (AWQ-INT4) | MLLM | 0.663 | 0.646 | 0.758 | 0.596 |
| 30 | MiniCPM-o 2.6 | MLLM | 0.669 | 0.656 | 0.758 | 0.600 |
| 31 | raya-decision-v1 | System-One | 0.706 | 0.682 | 0.814 | 0.652 |
| 32 | Gemma 4 26B-A4B | MLLM | 0.777 | 0.771 | 0.864 | 0.678 |
| 33 | Gemma 4 12B | MLLM | 0.790 | 0.794 | 0.833 | 0.716 |

## Closed loop

Per suite (paused mode). Per-environment scores: [closed_loop.md](closed_loop.md).

| Model | GameWorld progress | success | BALROG progression | success | VS-Bench normalized | Environments played |
|---|---:|---:|---:|---:|---:|---:|
| d3 | 40.9 | 15.8 | 29.2 | 29.4 | 45.3 | 97 / 97 |
| clef | 30.9 | 11.2 | 33.3 | 33.5 | 25.3 | 96 / 96 |
| Qwen3.5-35B-A3B (GPTQ-Int4) | 31.2 | 10.9 | 7.0 | 6.9 | 25.0 | 97 / 97 |
| OneJev-27B | 39.8 | 18.8 | 22.0 | 21.8 | 32.4 | 97 / 97 |
| OneJev-9B | 32.2 | 11.5 | 4.1 | 3.8 | 13.0 | 97 / 97 |
| Qwen3.5-9B | 29.5 | 10.9 | 3.7 | 3.5 | 24.8 | 97 / 97 |
| clef-flash | 30.1 | 11.9 | 4.8 | 4.6 | 20.8 | 96 / 96 |
| OneJev-4B | 32.0 | 11.5 | 3.3 | 3.1 | 20.8 | 97 / 97 |
| Qwen3-VL-30B-A3B-Instruct | 23.4 | 6.1 | 2.3 | 2.1 | 24.0 | 97 / 97 |
| Qwen3.8-27B (FP8) | 34.6 | 13.9 | 8.1 | 7.9 | 19.9 | 97 / 97 |
| Qwen3.5-4B | 31.8 | 13.9 | 3.4 | 3.2 | 27.6 | 97 / 97 |
| clef-omni | 24.6 | 7.9 | 1.8 | 1.7 | 23.9 | 97 / 97 |
| MiniCPM-o 4.5 | 21.6 | 6.7 | 2.4 | 2.3 | 24.6 | 97 / 97 |
| Qwen3-Omni-30B-A3B-Thinking | 20.5 | 5.5 | 2.5 | 2.5 | 17.0 | 97 / 97 |
| Nemotron-3-Nano-Omni-30B-A3B | 18.8 | 6.1 | 5.3 | 5.3 | 22.0 | 97 / 97 |
| Qwen3-Omni-30B-A3B-Instruct | 24.6 | 9.1 | 2.3 | 2.1 | 35.2 | 97 / 97 |
| Gemma 4 26B-A4B | 32.5 | 12.7 | 9.7 | 9.7 | 22.2 | 97 / 97 |
| Qwen2.5-Omni-7B | 23.8 | 7.9 | 3.3 | 3.3 | 25.1 | 97 / 97 |
| Qwen2.5-Omni-7B (AWQ) | 21.2 | 6.1 | 2.6 | 2.6 | 28.6 | 97 / 97 |
| Qwen2.5-Omni-7B (GPTQ-Int4) | 23.4 | 8.5 | 2.4 | 2.5 | 28.8 | 97 / 97 |
| Xor-26B-A4B | 25.1 | 8.5 | 6.4 | 6.3 | 30.3 | 97 / 97 |
| DiffusionGemma-26B-A4B (AWQ-INT4) | 25.6 | 8.5 | 4.1 | 4.0 | 9.2 | 97 / 97 |
| DiffusionGemma-26B-A4B | 25.9 | 7.9 | 6.2 | 6.1 | 11.7 | 97 / 97 |
| OneJev-0.8B | 20.7 | 4.2 | 3.9 | 4.0 | 12.1 | 97 / 97 |
| Qwen2.5-Omni-3B | 17.2 | 6.7 | 4.3 | 4.4 | 21.5 | 97 / 97 |
| MiniCPM-o 2.6 | 18.3 | 7.3 | 0.5 | 0.4 | 17.5 | 97 / 97 |
| Jev-Omni + head-v3 | 24.5 | 9.1 | 4.8 | 4.7 | 15.6 | 97 / 97 |
| Qwen3.5-2B | 13.6 | 4.2 | 1.3 | 1.3 | 15.9 | 97 / 97 |
| Jev-Omni | 25.8 | 8.5 | 5.6 | 5.4 | 19.7 | 97 / 97 |
| Gemma 4 12B | 25.1 | 9.1 | 7.5 | 7.2 | 40.4 | 97 / 97 |
| Phi-4-multimodal | 18.0 | 3.6 | 2.2 | 2.3 | 19.8 | 97 / 97 |
| Qwen3.5-0.8B | 18.5 | 5.5 | 0.0 | 0.0 | 9.6 | 97 / 97 |
| raya-decision-v1 | 15.2 | 3.0 | 0.0 | 0.0 | 24.3 | 97 / 97 |

Not counted for a model: environments whose decision prompt is over the model's input limit (every call rejected; final, as rejected static items are): `19_minesweeper` (clef, clef-flash).

## Latency and realtime

How long one decision takes at batch size 1 ([docs/latency.md](../docs/latency.md)): the median over the latency set (`data/latency/v1.json`) per input kind, on an H100 80 GB PCIe with the node to itself (two GPUs, tensor parallel, for the models that do not fit on one; flash attention first), with the model as in its accuracy run; *closed loop* is the median over its 100 decisions of recorded episodes (GameWorld, BALROG, VS-Bench). The same set on an H100 80 GB SXM machine: [summary/latency/h100-sxm/](summary/latency/h100-sxm/). GameWorld *realtime* plays the same games with the game running for the model's closed-loop latency (rounded to 10 ms) before each chosen action lands; *Δ* is realtime minus paused on the games the model played in both modes (percentage points).

| # | Model | Type | Closed loop (ms) | Short video (s) | Long video (s) | Benchmarks (s) | Audio-visual (s) | GameWorld paused | realtime | Δ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Qwen3.5-0.8B | MLLM | **94** | 0.27 | 0.61 | 0.75 | – | 18.5 | 20.5 | +2.0 |
| 2 | Qwen3.5-2B | MLLM | 136 | 0.34 | 0.77 | 0.93 | – | 13.6 | 13.9 | +0.3 |
| 3 | Qwen3.5-4B | MLLM | 167 | 0.40 | 0.84 | 1.01 | – | 31.8 | 25.9 | -5.9 |
| 4 | Gemma 4 12B | MLLM | 175 | 0.66 | 0.65 | 0.81 | 2.70 | 25.1 | 27.5 | +2.4 |
| 5 | OneJev-0.8B | System-One | 183 | 0.35 | 0.86 | 0.93 | – | 20.7 | 22.1 | +1.4 |
| 6 | Jev-Omni + head-v3 | System-One | 187 | 0.73 | 1.34 | 1.26 | 14.95 | 24.5 | 27.0 | +2.5 |
| 7 | Jev-Omni | System-One | 188 | 0.72 | 1.47 | 1.17 | 15.37 | 25.8 | 24.7 | -1.1 |
| 8 | Phi-4-multimodal | MLLM | 206 | 1.40 | 0.97 | 1.50 | 3.24 | 18.0 | 19.5 | +1.5 |
| 9 | Gemma 4 26B-A4B | MLLM | 228 | 0.70 | 0.77 | 0.87 | – | 32.5 | 31.5 | -1.0 |
| 10 | DiffusionGemma-26B-A4B | MLLM | 228 | 0.75 | 0.80 | 0.87 | – | 25.9 | 26.2 | +0.3 |
| 11 | raya-decision-v1 | System-One | 234 | 0.29 | 0.21 | 0.29 | – | 15.2 | 19.9 | +4.8 |
| 12 | DiffusionGemma-26B-A4B (AWQ-INT4) | MLLM | 237 | 0.68 | 0.78 | 0.98 | – | 25.6 | 28.2 | +2.6 |
| 13 | Qwen3.5-35B-A3B (GPTQ-Int4) | MLLM | 237 | 0.48 | 1.04 | 1.21 | – | 31.2 | 33.3 | +2.1 |
| 14 | Qwen3.5-9B | MLLM | 240 | 0.51 | 1.07 | 1.24 | – | 29.5 | 27.5 | -2.0 |
| 15 | Qwen2.5-Omni-3B | MLLM | 271 | 0.59 | 1.84 | 1.67 | 4.75 | 17.2 | 20.0 | +2.7 |
| 16 | Qwen2.5-Omni-7B | MLLM | 332 | 0.68 | 2.21 | 1.95 | 5.42 | 23.8 | 22.2 | -1.5 |
| 17 | Qwen2.5-Omni-7B (GPTQ-Int4) | MLLM | 335 | 0.67 | 2.20 | 2.08 | 5.43 | 23.4 | 23.4 | 0.0 |
| 18 | Qwen2.5-Omni-7B (AWQ) | MLLM | 343 | 0.68 | 2.25 | 2.00 | 5.47 | 21.2 | 22.0 | +0.8 |
| 19 | clef-omni | System-One | 368 | 0.40 | 1.39 | 1.12 | 22.54 | 24.6 | 23.1 | -1.5 |
| 20 | OneJev-4B | System-One | 374 | 0.48 | 1.28 | 1.42 | – | 32.0 | 28.0 | -4.0 |
| 21 | MiniCPM-o 4.5 | MLLM | 386 | 0.49 | 1.58 | 1.43 | 4.22 | 21.6 | 21.0 | -0.6 |
| 22 | Qwen3.8-27B (FP8) | MLLM | 429 | 0.79 | 1.70 | 1.94 | – | 34.6 | 33.2 | -1.3 |
| 23 | MiniCPM-o 2.6 | MLLM | 431 | 0.48 | 1.82 | 1.41 | 3.71 | 18.3 | 16.8 | -1.4 |
| 24 | OneJev-9B | System-One | 446 | 0.53 | 1.39 | 1.53 | – | 32.2 | 35.7 | +3.4 |
| 25 | Nemotron-3-Nano-Omni-30B-A3B | MLLM | 458 | 0.46 | 1.09 | 1.00 | 3.16 | 18.8 | 22.6 | +3.8 |
| 26 | clef-flash | System-One | 472 | 0.80 | 2.07 | 2.13 | – | 30.1 | 29.9 | -0.3 |
| 27 | Qwen3-VL-30B-A3B-Instruct | MLLM | 473 | 0.64 | 1.39 | 1.61 | – | 23.4 | 27.7 | +4.3 |
| 28 | Xor-26B-A4B | System-One | 492 | 0.90 | 0.94 | 1.32 | – | 25.1 | 30.0 | +5.0 |
| 29 | Qwen3-Omni-30B-A3B-Instruct | MLLM | 528 | 0.65 | 2.21 | 1.80 | 8.89 | 24.6 | 27.3 | +2.7 |
| 30 | Qwen3-Omni-30B-A3B-Thinking | MLLM | 537 | 0.65 | 2.29 | 1.76 | 8.94 | 20.5 | 26.6 | +6.1 |
| 31 | d3 | System-One | 780 | 0.61 | 0.76 | 1.25 | – | 40.9 | 34.2 | -6.7 |
| 32 | OneJev-27B | System-One | 1,093 | 0.99 | 2.86 | 2.90 | – | 39.8 | 38.2 | -1.6 |
| 33 | clef | System-One | 1,140 | 1.72 | 4.41 | 4.57 | – | 30.9 | 28.1 | -2.8 |

## Items without an answer

Static items a model was given but did not answer: inputs over the model's context or memory (rejected, final), or requests its server refuses (Xor: two-view videos). They count as wrong; items needing a modality the model does not take are not given to it and do not count.

| Model | Answered |
|---|---:|
| clef | 15,975 / 16,000 |
| clef-flash | 15,975 / 16,000 |
| clef-omni | 17,492 / 17,500 |
| Xor-26B-A4B | 15,500 / 16,000 |
| Jev-Omni + head-v3 | 17,328 / 17,500 |
| Jev-Omni | 17,498 / 17,500 |

