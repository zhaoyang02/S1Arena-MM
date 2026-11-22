# Spoken conversations

Four environments put a model in a live spoken conversation. The user's recorded audio is streamed to the model in
real time while the model's speech plays, and the model has to choose when to speak: whether it takes the turn when the
user is done, waits through the user's pauses, and stops when the user interrupts but not when the user only says
"uh-huh". Results: [audio_streaming_results.md](audio_streaming_results.md). Environment cards:
[envs_st_audio.md](envs_st_audio.md).

| Environment | Source | Episode | Measured |
|---|---|---|---|
| `st.audio/turn_taking/candor` | CANDOR conversations selected by Full-Duplex-Bench v1 | the user's audio, then 5 s of silence | when the model starts to speak, against the moment the other speaker took the turn in the recording |
| `st.audio/pause_handling/candor` | CANDOR conversations selected by Full-Duplex-Bench v1 | the user's audio, then 5 s of silence | whether the model speaks during the marked pauses in the user's turn |
| `st.audio/user_interruption/full_duplex_bench_v15` | Full-Duplex-Bench v1.5, synthetic speech | the user's request, then an interruption once the model has spoken for 1 s | whether the model's response is cancelled, which is expected |
| `st.audio/backchannel/full_duplex_bench_v15` | Full-Duplex-Bench v1.5, synthetic speech | the same, with a short acknowledgment instead of an interruption | whether the response is cancelled, which is unwanted |

The model hears and speaks at the same time, which a single decision request cannot express, so these environments
have their own commands: `stream-prepare`, `stream-run` and `stream-score` (`play` and `sweep` skip them). Their
measures are counts and times, reported apart from every accuracy table.

## Install

```bash
python -m venv .venv-audio-stream                  # client for Gemini Live
.venv-audio-stream/bin/pip install -e '.[audio-stream,audio-stream-gemini]'

python -m venv .venv-audio-stream-qwen             # client for the other backends
.venv-audio-stream-qwen/bin/pip install -e '.[audio-stream,audio-stream-qwen]'
.venv-audio-stream-qwen/bin/pip install silero-vad onnxruntime torch   # controller and jev: the client runs Silero VAD
```

The two clients need separate environments (Python 3.11 or later): the Gemini SDK requires websockets older than 17,
the other client websockets 17.2. The model servers run in environments of their own (below).

## Prepare the episodes

`stream-prepare` reads four sets from one folder, each as a folder `<set>/` or a ZIP `<set>.zip`:

- `candor_turn_taking` and `candor_pause_handling`, from BetterUp's gated
  [betterup/candor-full-duplex-bench](https://huggingface.co/datasets/betterup/candor-full-duplex-bench). Fill in
  BetterUp's form and accept the CANDOR Corpus License on its page, then:

  ```bash
  hf download betterup/candor-full-duplex-bench --repo-type dataset --local-dir /path/to/fdb-sets
  ```

- `user_interruption.zip` and `user_backchannel.zip`, the synthetic v1.5 sets of
  [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench), copied into the same folder.

```bash
.venv-audio-stream-qwen/bin/jevarena-mm stream-prepare --source /path/to/fdb-sets \
  --out data_local/audio-stream/full --per-task 0
```

- `--per-task N` takes the first N eligible recordings of each set, by numeric id (default 20). `0` takes all of them:
  the results use these 631 episodes, 117 turn-taking, 216 pause handling (277 marked pauses), 200 interruption and 98
  backchannel.
- A recording is eligible when its initial audio lasts at most 60 s and its held-out event windows are not empty and
  lie inside it. The manifest lists the others under `selection.skipped`. For the results, these were two turn-taking
  recordings, one of 94 s and one whose turn has zero length.
- The audio is converted to 16-bit PCM, mono, 16 kHz. `manifest.json` records each set's hash (a ZIP's sha256, or for a
  folder one sha256 over its files' names and hashes), every converted file's sha256 and the held-out event times.
  `--out` must be a new folder.
- `stream-run` checks every file's sha256, duration and format before it connects to a model. A model receives only
  the audio, as it arrives: no transcripts, event times or future audio.
- The audio prepared from CANDOR stays under the CANDOR Corpus License v2.0: do not redistribute it, and state in work
  that uses it that it uses the CANDOR Corpus from BetterUp Labs, BetterUp, Inc.
  ([betterup/candor-tabular](https://huggingface.co/datasets/betterup/candor-tabular)). The v1.5 sets are MIT.

## Backends

| `--backend` | Speaks | Decides when to speak | Needs |
|---|---|---|---|
| `gemini` | a Gemini Live model | Gemini's automatic activity detection | `--model <Live model id>`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `qwen` | Qwen3-Omni-30B-A3B-Instruct | the serving runtime's voice activity detection and cancellation | the Qwen3-Omni voice server |
| `minicpm` | MiniCPM-o 4.5 | the model, in its own full-duplex mode | its vLLM-Omni duplex server (`scripts/audio/minicpm-duplex.yaml`) and `MINICPM_REF_AUDIO` |
| `controller` | Qwen3-Omni-30B-A3B-Instruct, for any model | the model under test, at each pause the client detects | the Qwen3-Omni voice server and `scripts/audio/controller_server.py` |

`--backend jev` expects a Jev-Omni decision service that this repository does not include; Jev-Omni runs with
`controller` and a `jev_omni` spec, as in the results.

Gemini runs up to four episodes at once (`--workers 1` to `4`), the other backends one (`--workers 1`). In every
backend, the model that speaks gets the same system prompt (`SYSTEM` in `jevarena_mm/streaming/gemini.py`).

### Gemini Live

```bash
.venv-audio-stream/bin/jevarena-mm stream-run --backend gemini --model "$GEMINI_LIVE_MODEL" --workers 4 \
  --manifest data_local/audio-stream/full/manifest.json --out results/episodes/audio-stream/gemini
```

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, in the environment or in `.env`. `--model` is required: a Live model id
available to your account. The model answers with audio, with Gemini's automatic activity detection on, and the
transcription of its speech is recorded. Google bills the calls, so try one episode first with
`--env st.audio/turn_taking --limit 1`.

### Qwen3-Omni voice server

The `qwen`, `controller` and `jev` backends talk to Qwen3-Omni-30B-A3B-Instruct, served in duplex mode by vLLM-Omni at
`ws://127.0.0.1:8099/v1/realtime?duplex=1` (the default `--url`). The results used vLLM 0.30.0+cu129, vLLM-Omni
0.30.0, Torch 2.13.0+cu129, TorchCodec 0.17.0+cpu and Silero VAD 6.2.3 on four RTX A6000 GPUs. No extra of this
package installs them: the server runs in an environment of its own.

1. Download the checkpoint at the pinned revision:

   ```bash
   hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
     --local-dir /path/to/qwen3-omni
   ```

2. vLLM-Omni 0.30.0 drops a response that has audio but no transcript before its audio has been played, and the
   episode then fails with `playback_item_not_found`. Write a patched copy of the installed package (the script
   refuses any other version) and check it:

   ```bash
   python scripts/audio/patch_duplex_runtime.py --source /path/to/site-packages/vllm_omni --out /path/to/overlay
   PYTHONPATH=/path/to/overlay python scripts/audio/verify_duplex_runtime.py
   ```

3. Start the server with the patched copy first on `PYTHONPATH`:

   ```bash
   PYTHONPATH=/path/to/overlay CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_FLASHINFER_SAMPLER=0 \
     VLLM_WORKER_MULTIPROC_METHOD=spawn \
     vllm serve /path/to/qwen3-omni --omni \
       --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
       --host 127.0.0.1 --port 8099 --deploy-config scripts/audio/qwen-duplex.yaml
   ```

`scripts/audio/qwen-duplex.yaml` puts the Thinker on two GPUs (tensor parallel) and the Talker and Code2Wav on one GPU
each, serves one sequence at a time with a 4,096-token context, and sets the Thinker's default sampling to greedy with
at most 128 tokens. Before it sends any audio, the client checks that the server reports `turn_based_duplex`,
separate input and output streams, and playback acknowledgments.

With `--backend qwen`, the server's voice activity detection decides (threshold 0.5, 500 ms of silence, 300 ms of
padding): it starts a response when the user stops and cancels it when the user speaks. This is the runtime's turn
policy, not one the model learned.

```bash
.venv-audio-stream-qwen/bin/jevarena-mm stream-run --backend qwen \
  --manifest data_local/audio-stream/full/manifest.json --out results/episodes/audio-stream/qwen
```

### Shared-voice controller

With `--backend controller`, the model under test only decides, and Qwen3-Omni-30B-A3B-Instruct speaks for it with the
server's turn detection off:

- The client runs Silero VAD (threshold 0.5, 500 ms of silence, 300 ms of padding) on the user audio as it arrives.
- At each end of user speech it detects, the decision service receives the user audio observed so far and whether the
  assistant is speaking or still has speech buffered. It asks the model "Which action should the assistant take now?"
  with two options: `wait` or `speak` while the assistant listens, `continue` or `stop` while it speaks. A hash of the
  episode and the decision number sets the order of the two options. The option with the highest probability is
  taken.
- A decision is dropped if the user starts speaking again before it returns.
- After `speak`, the voice server answers the user audio since its last response. After `stop`, the buffered speech is
  cleared, the current response is cancelled, and a new response answers the new user audio. `continue` keeps the
  current response, and `wait` does nothing.

Start the voice server, then the decision service with a spec of `sweeps/audio_baselines.toml`, on GPUs the voice
server does not use and in the environment in which that model runs offline. Then start the client:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/audio/controller_server.py --spec "lm:Qwen/Qwen2.5-Omni-3B"
.venv-audio-stream-qwen/bin/jevarena-mm stream-run --backend controller \
  --manifest data_local/audio-stream/full/manifest.json --out results/episodes/audio-stream/qwen2.5-omni-3b
```

The decision service listens on `127.0.0.1:8767` and accepts only models that take audio. Its state and option texts
are in `scripts/audio/controller_server.py`. It sets `VLLM_PLUGINS=` for Qwen2.5-Omni-7B-AWQ and -GPTQ-Int4 itself
([audio.md](audio.md)).

### MiniCPM-o 4.5

`--backend minicpm` talks to openbmb/MiniCPM-o-4_5 in its own full-duplex mode, served by vLLM-Omni at
`ws://127.0.0.1:8100/v1/realtime?duplex=1`. The model itself decides when to listen and when to speak (no server turn
detection). The results ran the server in the environment of the Qwen3-Omni voice server, with Transformers 5.14.1,
s3tokenizer 0.3.0 and onnxruntime 1.31.0 in it and only the folder of step 2 on `PYTHONPATH`, on three RTX A6000 GPUs.

1. Download the checkpoint at the pinned revision. Its `assets/HT_ref_audio.wav` (sha256
   `cb8f06ba5080cdf548969138881fb8ad8b04e2516108f4e08ba0363b68b613ea`) is the model's voice:

   ```bash
   hf download openbmb/MiniCPM-o-4_5 --revision 503e754207c94da6bb26850b4469f367c9ea3582 \
     --local-dir /path/to/minicpm-o-4.5
   ```

2. Install the model's audio packages into a folder of their own, at the versions of the results:

   ```bash
   pip install --no-deps --target /path/to/minicpm-deps -r scripts/audio/minicpm-deps.txt
   ```

3. Start the server:

   ```bash
   PYTHONPATH=/path/to/minicpm-deps:/path/to/minicpm-deps/stepaudio2 CUDA_VISIBLE_DEVICES=0,1,2 \
     MINICPM_REF_AUDIO=/path/to/minicpm-o-4.5/assets/HT_ref_audio.wav OMP_NUM_THREADS=1 \
     TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
     vllm serve /path/to/minicpm-o-4.5 --omni --served-model-name openbmb/MiniCPM-o-4_5 \
       --host 127.0.0.1 --port 8100 --deploy-config scripts/audio/minicpm-duplex.yaml --trust-remote-code
   ```

`scripts/audio/minicpm-duplex.yaml` puts the model's three stages on one GPU each, serves one duplex session at a time
and fixes each stage's sampling (seed 42). Then:

```bash
MINICPM_REF_AUDIO=/path/to/minicpm-o-4.5/assets/HT_ref_audio.wav .venv-audio-stream-qwen/bin/jevarena-mm stream-run \
  --backend minicpm --url 'ws://127.0.0.1:8100/v1/realtime?duplex=1' \
  --manifest data_local/audio-stream/full/manifest.json --out results/episodes/audio-stream/minicpm-o-4.5
```

The backend does not start without `MINICPM_REF_AUDIO`.

## Run, stop and continue

- `--env` selects environments by prefix (default `st.audio`, all four), and `--limit N` runs the first N episodes of
  each (0, the default, runs all).
- `--out` must be a new folder: `stream-run` never overwrites a run and never retries an episode by itself. It stops at
  the first failed episode and cancels the ones in flight, and the episodes not run stay pending in `progress.json`.
- To continue, start a new run with `--resume-from <earlier --out>` (it may be given several times). It copies the
  completed episodes of the earlier runs, which must have used the same manifest, backend, model and configuration,
  and runs the others.
- Each episode folder holds its event log (`events.jsonl`), `result.json` and the model's audio (`received.wav`,
  and for Gemini also `playback_timeline.wav`). `results/episodes/` is not committed.

## Timing

All backends use the same clock:

- The user audio is sent in 100 ms packets, each when its last sample is due, and how late each packet went out is
  recorded.
- The model's speech plays on a virtual sink in 20 ms blocks at 24 kHz, and a block is audible when its RMS exceeds
  0.005 of full scale. There is no physical speaker or microphone.
- Turn-taking and pause handling: the user audio, then 5 s of silence.
- Interruption and backchannel: the user's request, then silence until the model has played 1 s of audible speech and
  is still speaking, then the interruption or backchannel, then 8 s of silence. If that does not happen within 20 s
  after the request, the episode ends without it.
- A session that runs longer than 110 s fails.
- An episode is scored up to the end of this window. Anything later, such as audio that arrives while the session
  closes, is not counted.

## Scoring

```bash
.venv-audio-stream-qwen/bin/jevarena-mm stream-score --manifest data_local/audio-stream/full/manifest.json \
  --run results/episodes/audio-stream/qwen --out results/summary/audio-stream-qwen
```

`stream-score` reads a run's traces, also those of a stopped run, without calling a model, and writes `report.json` and
`report.md` into a new folder. The reports hold counts, times and the sha256 of each episode's input, but no audio,
transcripts or event logs.

| Environment | Measures |
|---|---|
| Turn-taking | the time from the start of the held-out turn, taken when that audio was sent, to the model's first audible speech after it (0 if it is already speaking then), as a median over the episodes in which it speaks after that moment, with their number. Also the episodes in which it speaks before that moment. |
| Pause handling | marked pauses during which the model's speech is audible |
| Interruption, backchannel | the episodes in which the user utterance was played. Among them, those in which the server cancelled (Gemini: interrupted) the response, and the time from the start of the utterance to that event, which is not the moment the speech stops. |

- **Right-censored**: when the window ends, speech is still buffered, a response is still being generated (reported by
  the servers of every backend but Gemini), or a controller decision is pending. A censored episode is not a failed
  one.
- **Completed** means the episode ran to the end of its window, not that the model behaved well. Stopping is expected
  after an interruption and unwanted after a backchannel.
- No combined score is defined. The content of the replies is not scored, and these are not official
  Full-Duplex-Bench scores.
