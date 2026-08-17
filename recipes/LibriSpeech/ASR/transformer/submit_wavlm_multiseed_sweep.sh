#!/bin/bash
# Reproduce the corrected shared-warmup learned-segmenter sweep with
# microsoft/wavlm-large as the frozen SSL encoder.
#
# Per backbone, the dependency graph is:
#   cold-start (10 epochs, seed 3407)
#       -> shared decoder warm-up (6 epochs, seed 3407)
#           -> {nll_frozen,nll_mt,cer_mt} x {3407,3408,3409}
#
# CNN jobs are submitted first. The Transformer graph is submitted immediately
# afterwards; its independent stages may run in parallel with the CNN graph.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
WARMUP_SEED=3407
COLDSTART_EPOCHS=10
WARMUP_EPOCHS=6
TOTAL_EPOCHS=10

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
WAVLM_MODEL_ROOT="$WAVLM_CACHE/models--microsoft--wavlm-large"

BASE=results/speechllm_segmenter_wavlm
OUT_ROOT="$BASE/multiseed_shared_warmup_onpolicy"
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

validate_wavlm_cache() {
  local revision snapshot
  [[ -s "$WAVLM_MODEL_ROOT/refs/main" ]] || \
    die "Missing WavLM cache ref: $WAVLM_MODEL_ROOT/refs/main"
  revision=$(tr -d '[:space:]' < "$WAVLM_MODEL_ROOT/refs/main")
  snapshot="$WAVLM_MODEL_ROOT/snapshots/$revision"
  [[ -s "$snapshot/config.json" ]] || \
    die "Missing WavLM config: $snapshot/config.json"
  [[ -s "$snapshot/preprocessor_config.json" ]] || \
    die "Missing WavLM feature-extractor config: $snapshot/preprocessor_config.json"
  if [[ ! -s "$snapshot/pytorch_model.bin" && ! -s "$snapshot/model.safetensors" ]]; then
    die "Missing WavLM weights under $snapshot"
  fi
  echo "Validated offline WavLM snapshot: $snapshot"
}

validate_output_paths() {
  local bb variant seed path
  for bb in cnn transformer; do
    for path in \
      "$BASE/coldstart/$bb/$WARMUP_SEED" \
      "$BASE/shared_warmup/$bb/$WARMUP_SEED"; do
      [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
    done
    for variant in nll_frozen nll_mt cer_mt; do
      for seed in "${SEEDS[@]}"; do
        path="$OUT_ROOT/$variant/$bb/$seed"
        [[ ! -e "$path" ]] || die "Refusing to reuse existing output path: $path"
      done
    done
  done
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

submit_backbone() {
  local bb=$1
  local bb_short=$2
  local cold_out="$BASE/coldstart/$bb/$WARMUP_SEED"
  local cold_dir="$(pwd)/$cold_out/save"
  local warm_out="$BASE/shared_warmup/$bb/$WARMUP_SEED"
  local warm_source="$(pwd)/$warm_out/save"
  local encoder_args="--experiment_name speechllm_segmenter_wavlm --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE --ssl_feat_dims $WAVLM_DIM"
  local cold_extra="$encoder_args --seed $WARMUP_SEED --output_folder $cold_out"
  local cold_job warm_job warm_extra

  echo "### WavLM $bb: cold-start ###"
  cold_job=$(submit_job "wl_${bb_short}_cs_${WARMUP_SEED}" \
    --export="ALL,MODE=coldstart,BACKBONE=$bb,EPOCHS=$COLDSTART_EPOCHS,EXTRA_ARGS=$cold_extra" \
    run_segmenter.slurm)
  echo "cold-start $bb -> $cold_job ($cold_out)"

  echo "### WavLM $bb: shared six-epoch decoder warm-up ###"
  warm_extra="$encoder_args --seed $WARMUP_SEED --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy --lr_segmenter 0.0001 --lr_decoder 0.0005 --pg_weight 1.0 --max_decode_ratio 3.0 --output_folder $warm_out"
  warm_job=$(submit_job "wl_${bb_short}_w6_${WARMUP_SEED}" \
    --dependency="afterok:$cold_job" \
    --export="ALL,MODE=joint,BACKBONE=$bb,EPOCHS=$WARMUP_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$cold_dir,EXTRA_ARGS=$warm_extra" \
    run_segmenter.slurm)
  echo "shared warm-up $bb -> $warm_job (afterok:$cold_job; $warm_out)"

  submit_variant() {
    local variant=$1
    local seed=$2
    local reward=$3
    local freeze_decoder=$4
    local short=$5
    local update_mode=$6
    local lr_segmenter=$7
    local pg_weight=$8
    local max_decode_ratio=$9
    local out="$OUT_ROOT/$variant/$bb/$seed"
    local output_abs="$(pwd)/$out"
    local extra="$encoder_args --seed $seed --segmenter_reward $reward --freeze_decoder_in_joint $freeze_decoder --rl_update_mode $update_mode --lr_segmenter $lr_segmenter --lr_decoder 0.0005 --pg_weight $pg_weight --max_decode_ratio $max_decode_ratio --output_folder $out"
    local jid

    jid=$(submit_job "wl_${bb_short}_${short}_${seed}" \
      --dependency="afterok:$warm_job" \
      --export="ALL,MODE=joint,BACKBONE=$bb,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$cold_dir,WARMUP_SOURCE_DIR=$warm_source,OUTPUT_FOLDER=$output_abs,EXTRA_ARGS=$extra" \
      run_segmenter_from_shared_warmup.slurm)
    echo "$variant $bb seed=$seed -> $jid (afterok:$warm_job; $out)"
  }

  echo "### WavLM $bb: 3 variants x 3 seeds ###"
  local seed
  for seed in "${SEEDS[@]}"; do
    submit_variant nll_frozen "$seed" nll True nf on_policy 0.0001 1.0 3.0
    submit_variant nll_mt "$seed" nll False nm on_policy 0.0001 1.0 3.0
    submit_variant cer_mt "$seed" cer False cm batched_on_policy 0.00005 0.5 2.0
  done
}

mkdir -p slurm_logs
validate_wavlm_cache
validate_output_paths

# Preserve the requested priority in submission order while leaving the two
# dependency graphs independent for maximum scheduler-level parallelism.
submit_backbone cnn c
submit_backbone transformer t

echo "### submitted WavLM sweep; monitor with: squeue -u \$USER ###"
