#!/bin/bash
#SBATCH -J jevarena-latency
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -o slurm-%x.%j.out
# One model's latency on one whole node (docs/latency.md): the set's media are copied to the node's own disk, a fit test
# answers the longest requests of each input kind, and only if they fit is the whole set timed at batch size 1. The node
# is exclusive, so no other job shares its CPUs, disk or PCIe while requests are timed.
#
#   JV_SPEC='lm:Qwen/Qwen3.5-9B@gpu=0.9,maxlen=131072' JV_SITE_ENV=$PWD/scripts/slurm/site.env \
#     sbatch -p <gpu partition> -A <account> -t 04:00:00 scripts/slurm/latency.slurm
#   jevarena-mm latency-plan sweeps/latency_h100.toml -p <gpu partition> -A <account> --site-env scripts/slurm/site.env
#
# JV_SPEC       model spec (required); $VARIABLES in it are expanded by `jevarena-mm latency`
# JV_GPUS       GPUs the spec uses (default 1); the job uses GPUs 0 .. JV_GPUS-1 and leaves the others idle
# JV_ENV        the software the model needs (a plan's env: jev, vllm, vllm031, d3), for JV_SITE_ENV to activate
# JV_SET        latency set (default data/latency/v1.json)
# JV_SITE_ENV   file sourced first: modules, venv, JEVARENA_MM_MEDIA, JEVARENA_MM_RUNS, HF_HOME (site.env.example)
# JV_LOCAL      folder on the node's own disk for the media (default: $TMPDIR, else /tmp, then jevarena-latency-media)
# JV_FIT        requests per input kind in the fit test (default 10; 0 skips it)
# JV_KINDS      only these input kinds, e.g. loop (the closed-loop decisions that realtime episodes charge); default all
# JV_GPU_LABEL  the kind of GPU in the output paths (default: from the GPUs' names, e.g. h100, a100)
# JV_REPO       repository to run in (default: the submit directory)
# JV_CUDA_DEVICES  the GPUs to use (default 0 .. JV_GPUS-1), e.g. on a shared machine where some are taken
# OMP_NUM_THREADS_LATENCY  OMP_NUM_THREADS for the job (default: all cores); "unset" leaves it unset
set -u
: "${JV_SPEC:?set JV_SPEC to a model spec}"
if [ -n "${JV_SITE_ENV:-}" ]; then  # without set -u: module systems (Lmod) and site files read unset variables
  set +u; source "$JV_SITE_ENV" || { echo "sourcing $JV_SITE_ENV failed"; exit 2; }; set -u
fi
cd "${JV_REPO:-$SLURM_SUBMIT_DIR}" && [ -f pyproject.toml ] || { echo "not an S1Arena-MM checkout: $PWD"; exit 2; }
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # weights are fetched before the job

gpus=${JV_GPUS:-1}
set=${JV_SET:-data/latency/v1.json}
export CUDA_VISIBLE_DEVICES=${JV_CUDA_DEVICES:-$(seq -s, 0 $(( gpus - 1 )))}
export JEVARENA_MM_DECODE_THREADS=${JEVARENA_MM_DECODE_THREADS:-4}  # FFmpeg threads per file, as in the accuracy runs
cores=${SLURM_CPUS_ON_NODE:-$(nproc --all)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS_LATENCY:-$cores}
# flash attention before cuDNN's: on the H100, cuDNN's builds a plan per new sequence length (docs/latency.md)
export TORCH_CUDNN_SDPA_DEPRIORITIZED=${TORCH_CUDNN_SDPA_DEPRIORITIZED:-1}
local_media=${JV_LOCAL:-${TMPDIR:-/tmp}/jevarena-latency-media}
only=(); [ -n "${JV_KINDS:-}" ] && only=(--kinds "$JV_KINDS")
label=(); [ -n "${JV_GPU_LABEL:-}" ] && label=(--gpu-label "$JV_GPU_LABEL")
[ "${OMP_NUM_THREADS_LATENCY:-}" = unset ] && unset OMP_NUM_THREADS  # Phi-4-multimodal's vLLM engine hung at start with it set

echo "job ${SLURM_JOB_ID:-} on $(hostname): $JV_SPEC on $gpus of $(nvidia-smi -L 2>/dev/null | wc -l) GPUs, set $set, repo $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- jevarena_mm || echo '+local changes')"
jevarena-mm latency-stage --set "$set" --to "$local_media" "${only[@]}" || exit $?
export JEVARENA_MM_MEDIA=$local_media
if [ "${JV_FIT:-10}" -gt 0 ]; then
  jevarena-mm latency --model "$JV_SPEC" --set "$set" --gpus "$gpus" --fit "${JV_FIT:-10}" "${only[@]}" "${label[@]}" \
    || { echo "fit test failed: see results/latency/*/*/*.fit.json; try more GPUs (docs/latency.md)"; exit 3; }
fi
jevarena-mm latency --model "$JV_SPEC" --set "$set" --gpus "$gpus" "${only[@]}" "${label[@]}"
