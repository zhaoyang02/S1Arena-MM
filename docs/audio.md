# Audio tasks

The audio part of S1Arena-MM has 21 tasks, one per source, in 19 leaves: 10,560 items. An item is one audio clip and
one to eight questions about it, with no other text, and a model answers each question with a probability distribution
over its options. The task cards list each task's source, license and labels: [basic](tasks_aud_basic.md),
[dialog](tasks_aud_dialog.md), [judge](tasks_aud_judge.md), [know](tasks_aud_know.md), [pro](tasks_aud_pro.md),
[ret](tasks_aud_ret.md), [safety](tasks_aud_safety.md). Live spoken conversations, in which a model listens and speaks
at the same time, are separate environments: [audio_streaming.md](audio_streaming.md).

| Leaf | Source (task) | Items | Question |
|---|---|---:|---|
| `aud.basic/acoustic_scene` | TAU Urban Acoustic Scenes 2019 (`tau2019`) | 500 | which of 10 scenes the recording comes from |
| `aud.basic/emotion` | CREMA-D (`cremad`) | 1,470 | the emotions listeners perceive in the voice (6 options) |
| `aud.basic/intent` | SLURP (`slurp`) | 500 | the intent of a spoken request (77 options) |
| `aud.basic/language_id` | VoxLingua107, dev set (`voxlingua107_dev`) | 1,609 | the language spoken (33 options) |
| `aud.basic/sound_event` | SONYC-UST (`sonyc_ust`) | 500 | for each of 8 sound categories, whether it is audible |
| `aud.basic/speaker_change` | AMI (`ami`) | 500 | whether more than one speaker is heard, one after the other |
| `aud.dialog/dialogue_state` | SpokenWOZ (`spokenwoz`) | 500 | the user's constraints so far for 4 hotel and restaurant slots |
| `aud.dialog/interrupt_backchannel` | Full-Duplex-Bench v1.5 (`full_duplex_bench_v15`) | 298 | whether a given user utterance is an interruption or a backchannel |
| `aud.dialog/turn_timing` | Full-Duplex-Bench v1, CANDOR conversations (`full_duplex_bench_v1`) | 335 | whether, at the end of the audio so far, the user has yielded the floor |
| `aud.judge/data_cleaning` | NISQA, TEST_LIVETALK (`nisqa_livetalk`) | 232 | the listeners' quality rating (1 to 5), and whether the recording meets a Good-or-better bar |
| `aud.judge/data_cleaning` | signal defects added to SLURP recordings (`signal_defects_synthetic`) | 436 | whether to reject the recording as silent or clipped, by thresholds the question states |
| `aud.judge/naturalness` | BVCC (`bvcc`) | 741 | the listeners' naturalness rating (1 to 5) of synthesized or voice-converted speech |
| `aud.know/music_understanding` | MMAU test-mini, music (`mmau_mini`) | 105 | a multiple-choice question (4 options) |
| `aud.know/reasoning` | MMAU test-mini, speech (`mmau_mini`) | 330 | a multiple-choice question (2 to 5 options) |
| `aud.know/sound_qa` | MMAU test-mini, sound (`mmau_mini`) | 36 | a multiple-choice question (4 options) |
| `aud.pro/heart_sound` | PhysioNet/CinC Challenge 2016 (`physionet2016`) | 301 | whether the heart sound is abnormal |
| `aud.pro/heart_sound` | ICBHI 2017 respiratory sounds (`respiratory_icbhi2017`) | 500 | crackles, wheezes, both or neither in a respiratory cycle |
| `aud.pro/machine_anomaly` | MIMII pump, DCASE 2020 Task 2 (`mimii_pump_dcase2020`) | 500 | whether the pump sounds abnormal |
| `aud.ret/audio_text_matching` | SONYC-UST tag descriptions (`sonyc_tag_descriptions`) | 500 | whether the audio supports a one-sentence description |
| `aud.safety/spoof_detection` | ASVspoof 2019, logical access (`asvspoof2019_la`) | 500 | whether the utterance is synthesized or voice-converted |
| `aud.safety/toxicity` | MuTox (`mutox`) | 167 | whether the spoken segment contains toxic language |

- The clips are WAV files (16-bit PCM) at each source's own sample rate and channel count, 0.2 s to 406 s long (the
  SpokenWOZ dialogues are the longest), and an item uses its whole clip. How a model hears a clip (resampling, and for
  some model families a cut of long audio) is part of its adapter's input policy ([models.md](models.md)).
- Emotion, naturalness and both NISQA questions are scored against the listeners' votes: accuracy takes the most
  frequent rating as the answer, and the Brier score compares with the whole distribution. Every other question has
  one label.
- An item's `meta`, which no model sees, holds its source, its license and, in `meta.audio_provenance`, where the clip
  comes from.

## Get the audio

```bash
pip install -e '.[audio]'
hf auth login                              # once your access request on the dataset page is granted
export JEVARENA_MM_MEDIA=/path/to/media    # or put it in .env
jevarena-mm fetch --leaf aud
jevarena-mm verify --leaf aud
```

The items are in the repository (`data/aud/`), so the audio tasks need no `build`. `fetch` puts about 5 GB of audio
under `audio_delivery/` in the media folder. The [media dataset](https://huggingface.co/datasets/Swindl/S1Arena-MM)
hosts all of it except the files below, which `fetch` gets from their publishers with
`scripts/fetch/audio_delivery.py` (log: `$JEVARENA_MM_MEDIA/.fetch-logs/audio_delivery.log`). It keeps a file only if
its sha256 is the one its item records.

| Tasks | Fetched from | Your access |
|---|---|---|
| `aud.know/reasoning/mmau_mini`: its 80 clips from IEMOCAP (whose license does not allow redistribution) and VoxCeleb (whose authors no longer distribute its audio) | MMAU's release, [gamma-lab-umd/MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) at a pinned revision, decoded to WAV | none |
| `aud.dialog/turn_timing/full_duplex_bench_v1` | the start of each recording, up to the moment its question is about, from BetterUp's [betterup/candor-full-duplex-bench](https://huggingface.co/datasets/betterup/candor-full-duplex-bench) at a pinned revision | fill in BetterUp's form and accept the CANDOR Corpus License on its page before you run `fetch` |

- The media dataset hosts MMAU's other clips. Its Song Describer and FMA clips are MP3s that libsndfile decodes
  differently on x86 and ARM CPUs, so they could not be rebuilt byte for byte everywhere. The `audio` extra pins
  soundfile 0.14.0 (libsndfile 1.2.2) for the clips `fetch` decodes.
- Each task's license is on its card and in every item's `meta.license`. MMAU test-mini is CC BY-NC 4.0, and its audio
  stays under the terms of the datasets MMAU took it from. The CANDOR Corpus License lets a benchmark refer to its
  audio by id and timestamp, with labels, but not include it, so the repository holds only those references. The media
  dataset keeps TAU 2019's license file next to its audio (`audio_delivery/LICENSE-TAU-urban-acoustic-scenes-2019.txt`)
  and BVCC's license files under `LICENSES/BVCC/` ([licenses/README.md](../licenses/README.md)). ICBHI 2017 asks for
  research use and a citation of Rocha et al. (2019).

## Run a model

```bash
jevarena-mm run --model random --leaf aud --limit 2
jevarena-mm score --models random --leaf aud --limit 2
```

`--limit N` runs the first N items of every task. Items whose audio is missing are not run. A model that does not take
audio gets every item recorded as unsupported, without a call, and has no audio score. The adapters that take audio,
and their input policies, are in [models.md](models.md).

The 14 models of the results, with their settings, form the plan `sweeps/audio_baselines.toml`:

```bash
jevarena-mm sweep sweeps/audio_baselines.toml --models "lm:Qwen/Qwen2.5-Omni-3B"   # one model of the plan
jevarena-mm sweep sweeps/audio_baselines.toml --parallel-models 1                  # all 14, one after another
jevarena-mm sweep-report sweeps/audio_baselines.toml
```

- Qwen2.5-Omni-7B-GPTQ-Int4: vLLM 0.30 and later refuse its GPTQ format (desc_act=True). Write a dequantized copy with
  `python scripts/gptq_dequant.py <snapshot dir> <output dir>` and put `<output dir>` in place of `DEQUANT_PATH` in the
  plan.
- Qwen2.5-Omni-7B-AWQ and -GPTQ-Int4: if vLLM-Omni is installed in the same environment, set `VLLM_PLUGINS=` (empty).
  Its plugin replaces these checkpoints' architecture with a speech-generating model, while plain vLLM's model gives
  the label probabilities.
- `python scripts/audio/run_unified.py --spec SPEC --status STATUS.json` runs one spec of the plan the way the results
  were run. It loads the model first, runs two items and stops if either fails, then runs the rest in batches of 32
  and stops after three technical errors in a batch (rejected inputs count as unanswered questions, not as errors). It
  sets `VLLM_PLUGINS` for the two checkpoints above, records its progress in `STATUS.json` and writes
  `results/summary/sweep_audio_<model>_report.md` and `.json`.

## Scoring

Audio is scored by the same code as video (`jevarena_mm/metrics.py`), from each answer's distribution:

- **Accuracy**: the top option is the label. A task's accuracy is the mean over its questions (SONYC-UST asks 8 per
  item, SpokenWOZ 4, NISQA 2), a leaf's is the mean over its tasks, and the audio score is the mean over the 19 leaves.
- **Brier score**: sum over options of (p − t)², where t is one-hot or the listeners' distribution (0 to 2, lower is
  better). **ECE**: top option, ten equal-mass bins. Both are averaged like accuracy.
- A question without an answer (a failed call, an unusable reply, a rejected input) counts as wrong, and as a uniform
  distribution for Brier and ECE.
- `sweep-report` also gives, per task, the p50 and p95 latency, the unanswered questions and the items not yet run.

The [README](../README.md) lists audio accuracy, Brier score and ECE for the 14 models, in its table of image and
audio results.

## Gemini and Qwen3-Omni with written probabilities

Two more adapters take one audio clip (and optional text) and ask the model to write its probabilities as JSON,
instead of reading them from label tokens. They are not used for the results. Both send the clip's requested window
and never a transcript, keep their preprocessing and prompt version in the model id, and record the duration and
sample rate they sent in `usage.input`.

`audio_gemini:<model id>[@sample_rate=16000]` calls a Gemini model that takes audio through the Gemini API:

```bash
pip install -e '.[audio,audio-gemini]'
export GEMINI_API_KEY=...                  # or put it in .env
jevarena-mm run --model "audio_gemini:<model id>@sample_rate=16000" --leaf aud.basic/emotion --limit 2
```

- `<model id>` is the Gemini model's id without the `models/` prefix.
- The clip goes as mono 16-bit WAV at `sample_rate` (8,000 to 48,000 Hz, default 16,000), with temperature 1.0 and at
  most 4,000 output tokens. A request over 20 MB goes through the Files API, and the uploaded file is deleted
  afterwards, also when the call fails.
- Token usage is recorded but the cost is not, so no dollar budget limits these calls. Google bills them.

`audio_qwen` runs Qwen3-Omni-30B-A3B-Instruct at revision `26291f7` locally with Transformers:

```bash
pip install -e '.[audio,audio-qwen]'       # in an environment without the jev extra
CUDA_VISIBLE_DEVICES="$GPUS" jevarena-mm run --model audio_qwen --leaf aud.basic/emotion --limit 2
```

- It needs Transformers 4.57.1 (the `jev` extra pins 5.17.0), CUDA, and `CUDA_VISIBLE_DEVICES` set to the GPUs it may
  use. The model runs in BF16 with SDPA attention, entirely on those GPUs (at most 42 GB on each), with no CPU offload.
- Audio is mono 16 kHz, and nothing is cut: a request over 16,384 tokens, counting 4,000 output tokens, is rejected,
  and so is audio the processor would shorten.
- Sampling uses temperature 1.0 and top-p 1.0, seeded by the request and the sample number.
