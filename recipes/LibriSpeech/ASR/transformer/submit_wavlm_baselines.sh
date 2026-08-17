#!/bin/bash
# Submit the WavLM-large versions of the fixed-rate frontier and the native-rate
# (no-downsampling) three-seed baseline. All jobs are independent and may run in
# parallel.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

FIXED_RATES=(3 4 5 6 8)
SEEDS=(3407 3408 3409)
FIXED_SEED=3407
EPOCHS=10

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
WAVLM_MODEL_ROOT="$WAVLM_CACHE/models--microsoft--wavlm-large"
BASE=results/speechllm_fixed_pooling_wavlm
DRY_RUN=${DRY_RUN:-0}

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --exclude=ga129
  --time=3-00:00:00
)

die() {
  echo "$*" >&2
  exit 1
}

validate_inputs() {
  local revision snapshot k seed path
  [[ -s "$WAVLM_MODEL_ROOT/refs/main" ]] || die "Missing WavLM cache ref"
  revision=$(tr -d '[:space:]' < "$WAVLM_MODEL_ROOT/refs/main")
  snapshot="$WAVLM_MODEL_ROOT/snapshots/$revision"
  [[ -s "$snapshot/config.json" ]] || die "Missing WavLM config: $snapshot/config.json"
  if [[ ! -s "$snapshot/pytorch_model.bin" && ! -s "$snapshot/model.safetensors" ]]; then
    die "Missing WavLM weights under $snapshot"
  fi
  for k in "${FIXED_RATES[@]}"; do
    path="$BASE/fixed_rate_k${k}_tc100/$FIXED_SEED"
    [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
  done
  for seed in "${SEEDS[@]}"; do
    path="$BASE/no_downsampling_tc100/$seed"
    [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
  done
  echo "Validated offline WavLM snapshot: $snapshot"
}

submit_job() {
  local job_name=$1
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "${COMMON[@]}" --job-name="$job_name" --parsable "$@" >&2
    printf '\n' >&2
    echo "DRY_$job_name"
  else
    sbatch "${COMMON[@]}" --job-name="$job_name" --parsable "$@"
  fi
}

mkdir -p slurm_logs
validate_inputs

encoder_args="--experiment_name speechllm_fixed_pooling_wavlm --ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM"

echo "### WavLM fixed-rate frontier (seed $FIXED_SEED) ###"
for k in "${FIXED_RATES[@]}"; do
  out="$BASE/fixed_rate_k${k}_tc100/$FIXED_SEED"
  jid=$(submit_job "wl_fr_k${k}" \
    --export="ALL,FIXED_K=$k,SEED=$FIXED_SEED,EPOCHS=$EPOCHS,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,EXTRA_ARGS=$encoder_args" \
    run_fixed_rate.slurm)
  echo "fixed-rate k=$k seed=$FIXED_SEED -> $jid ($out)"
done

echo "### WavLM no-downsampling baseline (3 seeds) ###"
for seed in "${SEEDS[@]}"; do
  out="$BASE/no_downsampling_tc100/$seed"
  jid=$(submit_job "wl_nods_${seed}" \
    --export="ALL,SEED=$seed,EPOCHS=$EPOCHS,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,EXTRA_ARGS=$encoder_args" \
    run_no_downsampling.slurm)
  echo "no-downsampling seed=$seed -> $jid ($out)"
done

echo "### submitted WavLM baselines; monitor with: squeue -u \$USER ###"
