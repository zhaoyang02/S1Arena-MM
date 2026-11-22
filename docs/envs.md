# Closed-loop environments

The 102 game environments come from three suites, each run by its own upstream harness at a pinned revision. The
harness owns the game loop and the score; the model answers each step from the native screenshots and the suite's
text, with the suite's legal actions as the options. The environments are registered from `data/loop/suites.json`
(as `Environment` classes, `jevarena_mm/envs/base.py`); their cards are in [envs_loop_game.md](envs_loop_game.md).

The four spoken-conversation environments (`st.audio`) need persistent audio input and output, so they run through
`stream-run` and `stream-score` instead, with timing and coverage reported apart from the environments below; see
[audio streaming](audio_streaming.md) and [their cards](envs_st_audio.md).

| Suite | Leaf | Environments | Episodes per model | Recorded episodes (OneJev-27B / Jev-Omni) | Not playable |
|---|---|---|---|---|---|
| [GameWorld](https://github.com/gameworld-project/gameworld) @ `3c26bda` | `loop.game/gameworld` | 34 games | 5 tasks each, 100 steps | 165 / 165 completed | Wordle: any 5-letter word is an action (5 episodes) |
| [BALROG](https://github.com/balrog-ai/BALROG) @ `b7afe79` | `loop.game/balrog_pixels` | 58 tasks over BabyAI, BabaIsAI, MiniHack, Crafter, NetHack, TextWorld | the seeds of the suite record: 10 per task (BabaIsAI 3, MiniHack and NetHack 5) | 220 / 220 completed, 5 / 5 technical failures (NetHack) | TextWorld: free-text commands (30 episodes); NetHack: blocked upstream, 100 of its 248 actions cannot be executed (5 episodes) |
| [VS-Bench](https://github.com/zelaix/VS-Bench) @ `2c3531c` | `loop.game/vsbench` | 10 multi-agent environments | 10 | 100 / 100 completed (version 2; version 1 also recorded) | none |

- **Recorded and live.** OneJev-27B and Jev-Omni are scored from episodes recorded with the same input policy
  (`sweeps/video_dev.toml`). Any model can play the environments live: `jevarena-mm play` (or `sweep` for a plan)
  serves the model and runs each episode through a bridge in the suite's own Python (see [Playing](#playing)); the
  other models of the results page were played this way. Episodes are cached per model, environment and seed in
  `results/episodes/`, so seeds already recorded for a model are not replayed.
- **Scores are raw**: GameWorld's normalized task progress, BALROG's progression (0 to 1), VS-Bench's mean return of
  the model's roles. Success is reported where the suite defines it.
- **Unsupported and blocked.** `unsupported` environments need actions a classifier cannot give (free text) and are
  recorded as such for every model; `blocked` ones cannot run on the pinned upstream and are not played until it is
  fixed (remove `blocked` from `data/loop/suites.json` then).
- **Failures are not losses.** Episodes that ended in a technical failure (the NetHack upstream cannot execute 100 of
  its 248 advertised actions) are counted as crashed; unsupported episodes are counted apart.
- **VS-Bench versions.** Version 2 is current: each episode's environment and fixed opponents draw from their own
  seeded random streams (the bridge's `--rng isolated`). Version 1 (`--rng legacy`) is the original seeding, under
  which some environments repeat their initial state across seeds; its recorded episodes are kept, and `env-score`
  and `sweep-report` count only the current version.

## Playing

Each suite keeps its own checkout and Python environment, separate from S1Arena-MM's. A bridge
(`jevarena_mm/bridges/<suite>.py`, standard library only) runs inside the suite's Python, lets the upstream harness
play one episode, and replaces only its model client: each step becomes one packet to a local decision service
(`jevarena_mm/serve.py`), which builds a `DecisionRequest` and calls the model's `decide()`.

| Suite | State text and images | Options | Episode |
|---|---|---|---|
| GameWorld | native system and user prompt; current and up to two previous screenshots in place | every complete semantic control, each enumerable parameter expanded (at most 255) | game task `i+1`, seed 42, `main` protocol, 100 steps |
| BALROG | naive agent's messages (16 text turns, 1 image) | the environment's language action space | task seed from the suite record, BALROG's step limit |
| VS-Bench | official system prompt, the role's images and observation | all legal actions | episode `i` at seed `i`; the model plays the prompt-agent roles |

- **Packet** (`jevarena-mm-decision-v1`): `state` text with `<image:k>` where the k-th image sits, `question`,
  `candidates` `[{id, text}]`, `images` `[{path, sha256}]`. The service answers the argmax option (ties to the earlier
  one); a single option is taken without a call.
- **No defaults.** A failed call ends the episode as a technical failure, never with a default action. A model that does
  not take images, or an action space over 255 options (Wordle), is `unsupported`.
- **Files.** Each attempt is kept under `results/episodes/<model>/<env>/seed-NNN/attempt-*/`: `outcome.json`, every
  request and reply (`decisions/`), the upstream logs (`native/`) and `bridge.log`. The episode row records the
  attempt. These are about 200 files an episode, some 600,000 for a full sweep: `JEVARENA_MM_ATTEMPTS` puts the
  folders elsewhere, such as a file system without a file-count quota.
- **Realtime (GameWorld).** `play --realtime` charges each decision a fixed latency in game time: the game still pauses
  while the model decides (GameWorld's `pause_during_inference`), then runs for the latency before the chosen action
  lands, so an episode depends neither on the serving hardware nor on the episodes sharing the model. The latency is the
  model's measured one ([latency.md](latency.md)): the median over the closed-loop decisions of its batch-1
  measurement on an H100 PCIe (`--latency-gpu` names another kind of GPU; without a measurement, the fixed table
  `results/latency.json`); `--latency-ms` sets another. BALROG and VS-Bench wait for every move and have no realtime
  protocol: `--realtime` skips them. Rows record `mode`, `assumed_latency_ms` and `latency_gpu`; `env-score` reports
  realtime rows apart. `sweep-report` with `realtime = true` in the plan reports them beside the paused ones
  (`sweeps/gameworld_realtime.toml`); the results page shows each model's realtime progress and its change from
  paused.
- **Same inputs as the recorded episodes.** The bridges rebuild the recorded requests exactly (state text, options
  and images, checked on BabyAI goto, Kuhn poker and 2048), and OneJev-27B and Jev-Omni give the recorded
  probabilities. GameWorld screenshots depend on browser timing, so a long episode can drift from its recording.

### Installing the suites

```bash
scripts/loop/install.sh /path/to/suites       # about 10 minutes and 3 GB; prints the .env lines below
```

The script needs git, network access and uv (or a `python3` to install uv into the folder). It checks out each suite
at the revision above, with GameWorld's game library
([GameWorld-Games](https://github.com/gameworld-project/GameWorld-Games), which GameWorld does not pin) at `5532292`,
and installs each in its own Python as its README asks (GameWorld 3.12 with Playwright's Chromium; BALROG and
VS-Bench 3.10, through uv), then runs BALROG's post-install (MiniHack's Boxoban levels, the TextWorld games). BALROG
pins neither Gymnasium nor NumPy; `scripts/loop/balrog-constraints.txt` holds them at the versions VS-Bench pins
(Gymnasium 0.29.1: from 1.0 on, the wrappers BALROG reads its action space through no longer forward attributes).
Chromium needs the system's usual browser libraries (`playwright install-deps chromium`, as root). Running the script
again skips what is installed.

Set the checkout and Python of each suite in `.env` (see `.env.example`); a suite without them is not played, and
`play` says which variables to set. With the folder above, the script prints:

```bash
JEVARENA_MM_GAMEWORLD=/path/to/suites/GameWorld
JEVARENA_MM_GAMEWORLD_PYTHON=/path/to/suites/venv-gameworld/bin/python
JEVARENA_MM_GAMEWORLD_BROWSERS=/path/to/suites/browsers
JEVARENA_MM_BALROG=/path/to/suites/BALROG
JEVARENA_MM_BALROG_PYTHON=/path/to/suites/venv-balrog/bin/python
JEVARENA_MM_VSBENCH=/path/to/suites/VS-Bench
JEVARENA_MM_VSBENCH_PYTHON=/path/to/suites/venv-vsbench/bin/python
```

```bash
jevarena-mm envs --env loop.game
jevarena-mm play --model onejev@fps=1,min_frames=2 --env loop.game/vsbench --workers 4
jevarena-mm env-score --models onejev jev_omni --env loop.game/gameworld
jevarena-mm play --model onejev@fps=1,min_frames=2 --env loop.game/gameworld --realtime --latency-gpu h100 --workers 4

# or serve once and run one bridge by hand (in the suite's Python, with jevarena_mm on PYTHONPATH;
# --rng isolated is VS-Bench version 2, as play runs it)
jevarena-mm serve --model onejev@fps=1,min_frames=2 --port 8765
python -m jevarena_mm.bridges.vsbench --upstream $JEVARENA_MM_VSBENCH --env kuhn_poker --episode 0 --rng isolated \
    --endpoint http://127.0.0.1:8765/ --out /tmp/kuhn-0
```

`play` loads the model once and serves all episodes of the run; `--workers` episodes run in parallel and share it,
one call at a time. `--max-decisions` does not apply: the upstream harness sets the episode length.
