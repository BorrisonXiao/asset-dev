#!/bin/bash
# Submit the transcript-informed character-CTC alignment oracle with the same
# WavLM-Large, LibriSpeech-100h, 10-epoch, and three-seed protocol as the
# fixed-rate/no-downsampling baselines.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
EPOCHS=10
WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
WAVLM_MODEL_ROOT="$WAVLM_CACHE/models--microsoft--wavlm-large"
BOUNDARY_DIR=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets/wavlm_boundaries/char
BASE=results/speechllm_fixed_pooling_wavlm/char_alignment_tc100
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
  local revision snapshot seed path target_count
  [[ -s "$WAVLM_MODEL_ROOT/refs/main" ]] || die "Missing WavLM cache ref"
  revision=$(tr -d '[:space:]' < "$WAVLM_MODEL_ROOT/refs/main")
  snapshot="$WAVLM_MODEL_ROOT/snapshots/$revision"
  [[ -s "$snapshot/config.json" ]] || die "Missing WavLM config: $snapshot/config.json"
  if [[ ! -s "$snapshot/pytorch_model.bin" && ! -s "$snapshot/model.safetensors" ]]; then
    die "Missing WavLM weights under $snapshot"
  fi
  [[ -d "$BOUNDARY_DIR" ]] || die "Missing char-alignment directory: $BOUNDARY_DIR"
  target_count=$(find "$BOUNDARY_DIR" -maxdepth 1 -name '*.pt' | wc -l)
  (( target_count >= 292367 )) || die "Incomplete char targets: found $target_count"
  for seed in "${SEEDS[@]}"; do
    path="$BASE/$seed"
    [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
  done
  echo "Validated WavLM snapshot and $target_count character-alignment targets"
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
for seed in "${SEEDS[@]}"; do
  out="$BASE/$seed"
  jid=$(submit_job "wl_char_a_${seed}" \
    --export="ALL,SEED=$seed,EPOCHS=$EPOCHS,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,BOUNDARY_TARGET_DIR=$BOUNDARY_DIR,EXTRA_ARGS=$encoder_args" \
    run_char_alignment.slurm)
  echo "char-alignment seed=$seed -> $jid ($out)"
done

echo "Submitted three WavLM char-alignment jobs; monitor with: squeue -u $USER"
