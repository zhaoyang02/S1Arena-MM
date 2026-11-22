#!/bin/bash
#SBATCH -J jevarena-run
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o slurm-%x.%A_%a.out
# One array task = one model on one node, one shard per GPU group (clusters that bill whole nodes waste nothing).
# Shard of GPU group g on array task i: i * groups + g of (array size) * groups. Each shard writes its own run files
# (static tasks) or plays its own environments (closed loop), so every task of the array (and later resubmissions
# with any array size) can run at once and resume.
#
#   JV_MODEL=onejev@fps=1,min_frames=2 JV_SITE_ENV=$PWD/scripts/slurm/site.env \
#     sbatch -p <partition> -A <account> -t 48:00:00 --array=0-3 scripts/slurm/run_shards.slurm
#
# JV_MODEL           model spec (required)
# JV_SITE_ENV        file sourced first: modules, venv, JEVARENA_MM_MEDIA, HF_HOME, ... (see site.env.example)
# JV_LEAVES          static leaf prefixes to run (default: "vid lvid"; empty: none)
# JV_ENVS            environment prefixes to play after them (default: none; e.g. "loop"; the suites installed with
#                    scripts/loop/install.sh and set in the site file or .env)
# JV_GPUS_PER_SHARD  GPUs one shard process uses (default 1; 2 for a model split over two GPUs)
# JV_FILL_MODEL      a model run on the GPUs a node has beyond its whole groups (e.g. the third of three GPUs when a
#                    shard takes two), one GPU per shard, over the same leaves and environments: no GPU idles
# JV_RUN_ARGS        extra `jevarena-mm run` arguments, e.g. "--limit 3"
# JV_PLAY_ARGS       extra `jevarena-mm play` arguments (default: "--workers 4")
# JV_TASK, JV_TASKS  this node's task index and the array's task count, to redo one task of an array whose node
#                    failed while the others still run (default: from the Slurm array)
# JV_REPO            repository to run in (default: the submit directory)
# JV_LOG_DIR         where each shard writes its log (default: the submit directory)
set -u
: "${JV_MODEL:?set JV_MODEL to a model spec}"
if [ -n "${JV_SITE_ENV:-}" ]; then  # without set -u: module systems (Lmod) and site files read unset variables
  set +u; source "$JV_SITE_ENV" || { echo "sourcing $JV_SITE_ENV failed"; exit 2; }; set -u
fi
cd "${JV_REPO:-$SLURM_SUBMIT_DIR}" && [ -f pyproject.toml ] || { echo "not an S1Arena-MM checkout: $PWD"; exit 2; }
logs=${JV_LOG_DIR:-$SLURM_SUBMIT_DIR}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # weights are fetched once before the array, not by every node at once

gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
per=${JV_GPUS_PER_SHARD:-1}
groups=$(( gpus / per )); [ "$groups" -ge 1 ] || groups=1
fill=0; [ -n "${JV_FILL_MODEL:-}" ] && fill=$(( gpus - groups * per ))
tasks=${JV_TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}  # JV_TASK / JV_TASKS: redo one task of an earlier array (--array=0)
task=${JV_TASK:-$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))}
cores=${SLURM_CPUS_ON_NODE:-$(nproc --all)}  # not nproc: it follows OMP_NUM_THREADS, which sites often set to 1
export OMP_NUM_THREADS=$(( cores / (groups + fill) ))

echo "job ${SLURM_JOB_ID:-} task $task on $(hostname): $JV_MODEL, $gpus GPUs, $groups shard(s) of $(( tasks * groups ))$( [ "$fill" -gt 0 ] && echo " + $fill of $(( tasks * fill )) of $JV_FILL_MODEL"), repo $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- jevarena_mm || echo '+local changes')"
pids=()
shard() {  # shard <model> <index> <count> <devices>
  (
    export CUDA_VISIBLE_DEVICES=$4
    for leaf in ${JV_LEAVES-vid lvid}; do
      jevarena-mm run --model "$1" --leaf "$leaf" --shard "$2/$3" ${JV_RUN_ARGS:-} || exit $?
    done
    for prefix in ${JV_ENVS:-}; do
      jevarena-mm play --model "$1" --env "$prefix" --shard "$2/$3" ${JV_PLAY_ARGS:---workers 4} || exit $?
    done
  ) > "$logs/${SLURM_JOB_NAME:-run}.${SLURM_JOB_ID:-0}.$(echo "$1" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-60).shard-${2}of${3}.log" 2>&1 &
  pids+=($!)
}
for g in $(seq 0 $(( groups - 1 ))); do
  shard "$JV_MODEL" $(( task * groups + g )) $(( tasks * groups )) "$(seq -s, $(( g * per )) $(( g * per + per - 1 )))"
done
for f in $(seq 0 $(( fill - 1 ))); do
  shard "$JV_FILL_MODEL" $(( task * fill + f )) $(( tasks * fill )) $(( groups * per + f ))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "done $(date -Is), status $status"
exit $status
