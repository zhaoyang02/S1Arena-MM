# Slurm

`run_shards.slurm` runs one model over the selected leaves as a job array. Each array task takes one node and starts
one shard per GPU group (`JV_GPUS_PER_SHARD`, default 1), so a cluster that bills whole nodes bills no idle GPU. Shard
`g` of array task `i` is `i * groups + g` of `array size * groups`; each shard writes its own run files
(`<task>.shard-IofN.jsonl`), so the tasks may run at once on a shared file system.

```bash
cp scripts/slurm/site.env.example scripts/slurm/site.env      # modules, venv, media, caches; not committed
hf download <repo> --revision <sha>                           # weights once, at the revision in docs/models.md
JV_MODEL='onejev@fps=1,min_frames=2' JV_SITE_ENV=$PWD/scripts/slurm/site.env \
  sbatch -p <gpu partition> -A <account> -t 48:00:00 --array=0-3 scripts/slurm/run_shards.slurm
jevarena-mm score --models 'onejev@fps=1,min_frames=2' --leaf vid    # reads every shard's files
```

- `JV_ENVS=loop` also plays the closed-loop environments after the static tasks (`JV_LEAVES=` for those alone): each
  shard plays every N-th environment (`play --shard I/N`), so no two shards write one environment's episodes. The
  suites must be installed (`scripts/loop/install.sh`) and set in the site file.
- `JV_FILL_MODEL=<spec>` runs a second model on the GPUs left over when a shard takes several (the third GPU of a
  three-GPU node when `JV_GPUS_PER_SHARD=2`), one shard per GPU, over the same leaves and environments.
- `jevarena-mm sweep-submit <plan> -p <partition> -A <account> --site-env <file>` prints one such array per model of
  a sweep plan and submits them with `--go`. Each model's `slurm` table in the plan sets its GPUs per shard, its fill
  model, its node count (`array`; 0 for a model that runs only as a fill) and its time limit
  (`sweeps/video_baselines.toml`); `jevarena-mm sweep-status <plan>` shows what every array has done.
- Jobs run with `HF_HUB_OFFLINE=1`: fetch the weights before submitting, so the nodes do not download them at once.
- A job that hits its time limit loses at most the row being written; submit the array again (any size, e.g. with
  `--dependency=afterany:<job id>`) and it calls only what is missing.
- Each shard logs to `$JV_LOG_DIR/<job name>.<job id>.<model>.shard-IofN.log`; the first line of the array task's
  output names the node, the GPUs, the shard count and the repository commit.
- `OMP_NUM_THREADS` is set to the node's cores per shard (sites often default it to 1).
- Set the glibc allocator lines of `site.env.example`: without them, shards that decode long videos keep their freed
  frames and run a node out of memory.

## Latency

`latency.slurm` times one model at batch size 1 on an exclusive node: it copies the latency set's media to the node's
own disk, runs a fit test on the longest requests of each input kind and then the whole set
([docs/latency.md](../../docs/latency.md)). `jevarena-mm latency-plan sweeps/latency_h100.toml -p <partition> -A
<account> --site-env <file>` prints one job per model of the plan (`--go` submits); `JV_GPUS` sets the GPUs a spec uses.
