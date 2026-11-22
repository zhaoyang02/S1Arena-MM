# Spoken conversations: results

Sixteen configurations on the 631 episodes of the four spoken-conversation environments
([audio_streaming.md](audio_streaming.md)): 117 turn-taking, 216 pause handling with 277 marked pauses, 200
interruption and 98 backchannel episodes. Every configuration completed all 631 episodes without an execution failure,
one episode at a time, on RTX A6000 GPUs. The episodes are every eligible recording of the four sets
(`stream-prepare --per-task 0`). The 80 on which the protocol was tested are among them, so this is not a held-out
test set.

## Columns

- **Turn (ms, n)**: turn-taking. The median time from the moment the other speaker took the turn in the recording to
  the model's first audible speech after it (0 if it was already speaking), over the n episodes, of 117, in which the
  model spoke after that moment.
- **Early**: turn-taking episodes, of 117, in which the model spoke before that moment.
- **Pauses**: marked pauses, of 277, during which the model's speech was audible.
- **Interruption**, **Backchannel**: episodes in which the response was cancelled / episodes in which the interruption
  (of 200) or the backchannel (of 98) was played. Stopping is expected after an interruption and unwanted after a
  backchannel.
- **Censored**: episodes, of 631, whose window ended with speech still buffered or being generated, or with a decision
  pending. They are not failures.
- **Report**: the configuration's summary, and its JSON with per-episode metrics and the sha256 of every episode's input
  (`input_fingerprints`), against which a newly prepared pool can be checked. Neither holds audio or transcripts.

The counts are not combined into a score, and none of them is an official Full-Duplex-Bench score. A model that seldom
speaks also seldom speaks early, into pauses, or long enough for the user's utterance to be played, so read each count
with n and with the number of episodes played.

## Shared voice

Each model decides at the pauses the client's voice activity detection finds, and Qwen3-Omni-30B-A3B-Instruct speaks
for all of them with its own turn detection off (`--backend controller`). The models run with their specs in
`sweeps/audio_baselines.toml`, Qwen2.5-Omni-7B (GPTQ-Int4) from its dequantized copy. A cancellation here always comes
from the model's `stop`. The time to speak includes the 500 ms of silence the voice activity detection waits for, the
decision and the speech generation, so it is not comparable with the decision latencies of the
[leaderboard](../README.md).

| Model | Type | Turn (ms, n) | Early | Pauses | Interruption | Backchannel | Censored | Report |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Jev-Omni + head-v3 | System-One | 1,322.4 (65) | 18 | 39 | 99 / 114 | 35 / 49 | 353 | [JSON](../results/summary/audio_streaming/controller-full-00-full.json) |
| Jev-Omni | System-One | 1,326.9 (74) | 17 | 32 | 112 / 124 | 48 / 53 | 352 | [JSON](../results/summary/audio_streaming/controller-full-01-full.json) |
| clef-omni | System-One | 1,463.0 (35) | 1 | 3 | 148 / 151 | 67 / 87 | 281 | [JSON](../results/summary/audio_streaming/controller-full-02-full.json) |
| Qwen3-Omni-30B-A3B-Instruct | MLLM | 1,327.4 (72) | 15 | 17 | 122 / 125 | 10 / 59 | 323 | [JSON](../results/summary/audio_streaming/controller-full-03-full.json) |
| Qwen3-Omni-30B-A3B-Thinking | MLLM | 1,299.9 (25) | 5 | 5 | 85 / 86 | 1 / 1 | 115 | [JSON](../results/summary/audio_streaming/controller-full-04-full.json) |
| Qwen2.5-Omni-7B | MLLM | 1,319.1 (10) | 2 | 4 | 98 / 101 | 2 / 11 | 120 | [JSON](../results/summary/audio_streaming/controller-full-05-full.json) |
| Qwen2.5-Omni-7B (AWQ) | MLLM | 1,294.7 (22) | 5 | 11 | 131 / 142 | 19 / 48 | 214 | [JSON](../results/summary/audio_streaming/controller-full-06-full.json) |
| Qwen2.5-Omni-7B (GPTQ-Int4) | MLLM | 1,347.2 (31) | 5 | 11 | 155 / 157 | 23 / 81 | 289 | [JSON](../results/summary/audio_streaming/controller-full-07-full.json) |
| Qwen2.5-Omni-3B | MLLM | 1,386.7 (3) | 0 | 2 | 9 / 9 | 0 / 0 | 19 | [JSON](../results/summary/audio_streaming/controller-full-08-full.json) |
| Gemma 4 12B | MLLM | 1,298.5 (100) | 35 | 51 | 188 / 192 | 47 / 97 | 519 | [JSON](../results/summary/audio_streaming/controller-full-09-full.json) |
| MiniCPM-o 4.5 | MLLM | – (0) | 0 | 0 | 0 / 0 | 0 / 0 | 1 | [JSON](../results/summary/audio_streaming/controller-full-10-full.json) |
| MiniCPM-o 2.6 | MLLM | 1,264.4 (74) | 14 | 22 | 173 / 179 | 29 / 58 | 429 | [JSON](../results/summary/audio_streaming/controller-full-11-full.json) |
| Phi-4-multimodal | MLLM | – (0) | 0 | 0 | 4 / 4 | 0 / 0 | 4 | [JSON](../results/summary/audio_streaming/controller-full-12-full.json) |
| Nemotron-3-Nano-Omni-30B-A3B | MLLM | 1,302.8 (116) | 32 | 69 | 198 / 200 | 98 / 98 | 573 | [JSON](../results/summary/audio_streaming/controller-full-13-full.json) |

With the shared voice, MiniCPM-o 4.5 chose `wait` in 980 of its 981 decisions (it spoke once) and Phi-4-multimodal in
973 of 981 (it spoke four times and stopped four times), so neither spoke in turn-taking.

## Own speech

Two systems also speak with their own voice and turn policy: Qwen3-Omni-30B-A3B-Instruct with the voice activity
detection and cancellation of its serving runtime (`--backend qwen`), and MiniCPM-o 4.5 in its own full-duplex mode
(`--backend minicpm`). These rows compare turn policies, not decisions, so they are listed apart from the shared-voice
table. The other twelve configurations have no validated speech interface in this runtime (vLLM-Omni 0.30.0).

| System | Turn (ms, n) | Early | Pauses | Interruption | Backchannel | Censored | Report |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-Omni-30B-A3B-Instruct, runtime turn policy | 1,145.4 (117) | 19 | 65 | 199 / 200 | 98 / 98 | 594 | [JSON](../results/summary/audio_streaming/native-qwen-full-full.json) |
| MiniCPM-o 4.5, own full-duplex mode | 1,920.7 (109) | 11 | 21 | – / 199 | – / 98 | 513 | [JSON](../results/summary/audio_streaming/native-minicpm-r5-full-full.json) |

–: MiniCPM-o 4.5 reported no cancellation event in its own mode, so its stops cannot be counted. The played
interruptions and backchannels (199 and 98) are still shown.
