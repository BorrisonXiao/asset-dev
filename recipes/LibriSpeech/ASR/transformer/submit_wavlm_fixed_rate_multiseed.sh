#!/bin/bash
# Complete the WavLM fixed-rate frontier to three seeds. Seed 3407 already
# exists; this submits the missing seeds 3408 and 3409 for k=3,4,5,6,8.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

FIXED_RATES=(3 4 5 6 8)
SEEDS=(3408 3409)
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
    for seed in "${SEEDS[@]}"; do
      path="$BASE/fixed_rate_k${k}_tc100/$seed"
      [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
    done
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

for k in "${FIXED_RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$BASE/fixed_rate_k${k}_tc100/$seed"
    jid=$(submit_job "wl_fr_k${k}_s${seed}" \
      --export="ALL,FIXED_K=$k,SEED=$seed,EPOCHS=$EPOCHS,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,EXTRA_ARGS=$encoder_args" \
      run_fixed_rate.slurm)
    echo "fixed-rate k=$k seed=$seed -> $jid ($out)"
  done
done

echo "Submitted 10 WavLM fixed-rate jobs; monitor with: squeue -u $USER"
