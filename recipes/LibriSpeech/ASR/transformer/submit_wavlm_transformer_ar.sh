#!/bin/bash
# Three-seed WavLM study for the full-prefix label-autoregressive Transformer
# boundary policy. A fresh supervised policy is required because this architecture
# is not checkpoint-compatible with the independent Transformer segmenter.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
COLDSTART_EPOCHS=10
WARMUP_EPOCHS=6
TOTAL_EPOCHS=10
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
WAVLM_MODEL_ROOT="$WAVLM_CACHE/models--microsoft--wavlm-large"
OUT_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar
COLD_OUT="$OUT_ROOT/coldstart/3407"
WARM_OUT="$OUT_ROOT/shared_warmup/3407"
COLD_DIR="$(pwd)/$COLD_OUT/save"
WARM_SOURCE="$(pwd)/$WARM_OUT/save"

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

validate_inputs() {
  local revision snapshot
  [[ -s "$WAVLM_MODEL_ROOT/refs/main" ]] || \
    die "Missing WavLM cache ref: $WAVLM_MODEL_ROOT/refs/main"
  revision=$(tr -d '[:space:]' < "$WAVLM_MODEL_ROOT/refs/main")
  snapshot="$WAVLM_MODEL_ROOT/snapshots/$revision"
  [[ -s "$snapshot/config.json" ]] || die "Missing WavLM config: $snapshot/config.json"
  if [[ ! -s "$snapshot/pytorch_model.bin" && ! -s "$snapshot/model.safetensors" ]]; then
    die "Missing WavLM weights under $snapshot"
  fi
  [[ ! -e "$COLD_OUT" ]] || die "Refusing to reuse output path: $COLD_OUT"
  [[ ! -e "$WARM_OUT" ]] || die "Refusing to reuse output path: $WARM_OUT"
  local seed
  for seed in "${SEEDS[@]}"; do
    [[ ! -e "$OUT_ROOT/nll_mt/$seed" ]] || \
      die "Refusing to reuse output path: $OUT_ROOT/nll_mt/$seed"
  done
}

mkdir -p slurm_logs
validate_inputs

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_transformer_ar"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"

cold_extra="$ENCODER_ARGS $POLICY_ARGS --seed 3407 --output_folder $COLD_OUT"
cold_job=$(submit_job wl_tar_cs_3407 \
  --export="ALL,MODE=coldstart,BACKBONE=transformer_ar,EPOCHS=$COLDSTART_EPOCHS,EXTRA_ARGS=$cold_extra" \
  run_segmenter.slurm)
echo "cold-start -> $cold_job ($COLD_OUT)"

warm_extra="$ENCODER_ARGS $POLICY_ARGS --seed 3407 --output_folder $WARM_OUT"
warm_extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
warm_extra+=" --rl_update_mode combined_on_policy --lr_segmenter 0.0001 --lr_decoder 0.0005"
warm_extra+=" --pg_weight 1.0 --max_decode_ratio 3.0"
warm_job=$(submit_job wl_tar_w6_3407 \
  --dependency="afterok:$cold_job" \
  --export="ALL,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$WARMUP_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$warm_extra" \
  run_segmenter.slurm)
echo "decoder warm-up -> $warm_job (afterok:$cold_job; $WARM_OUT)"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/nll_mt/$seed"
  output_abs="$(pwd)/$out"
  rl_extra="$ENCODER_ARGS $POLICY_ARGS --seed $seed --output_folder $out"
  rl_extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
  rl_extra+=" --rl_update_mode combined_on_policy --lr_segmenter 0.0001 --lr_decoder 0.0005"
  rl_extra+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  rl_extra+=" --rate_mode band --rho_lo 0.10 --rho_hi 0.30 --lambda_cap 1.0"
  job=$(submit_job "wl_tar_nm_$seed" \
    --dependency="afterok:$warm_job" \
    --export="ALL,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,WARMUP_SOURCE_DIR=$WARM_SOURCE,OUTPUT_FOLDER=$output_abs,EXTRA_ARGS=$rl_extra" \
    run_segmenter_from_shared_warmup.slurm)
  echo "NLL-MT seed=$seed -> $job (afterok:$warm_job; $out)"
done

echo "Schedule: fresh 10-epoch char cold-start -> 6-epoch decoder warm-up -> three 4-epoch joint-RL runs"
echo "Output root: $OUT_ROOT"
