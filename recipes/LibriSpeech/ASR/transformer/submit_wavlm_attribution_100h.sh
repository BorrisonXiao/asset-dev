#!/bin/bash
# Matched 100h attribution study:
#   B1 fixed k=5 + BiGRU, decoder/pooler CE only
#   B2 frozen supervised Transformer-AR segmenter + BiGRU, decoder/pooler CE only
#   B3 the same initialization + BiGRU, decoder CE + joint NLL-GRPO
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
JOINT_RL_EPOCHS=${JOINT_RL_EPOCHS:-5}
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save
OUT_ROOT=${OUT_ROOT:-results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru}

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=12
  --mem=60G
  --time=2-00:00:00
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

for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
compgen -G "$COLD_DIR/CKPT*/segmenter.ckpt" >/dev/null \
  || die "No Transformer-AR cold-start checkpoint under $COLD_DIR"

for arm in fixed_k5_bigru frozen_segmenter_bigru joint_rl_bigru; do
  for seed in "${SEEDS[@]}"; do
    [[ ! -e "$OUT_ROOT/$arm/$seed" ]] \
      || die "Refusing to reuse output: $OUT_ROOT/$arm/$seed"
  done
done
mkdir -p slurm_logs

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_attribution_100h"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
ENCODER_ARGS+=" --train_splits ['train-clean-100']"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
TRAIN_ARGS="--decoder_init_ckpt_dir $DECODER_CKPT"
TRAIN_ARGS+=" --segmenter_reward nll --freeze_decoder_in_joint False"
TRAIN_ARGS+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
TRAIN_ARGS+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
TRAIN_ARGS+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
TRAIN_ARGS+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
TRAIN_ARGS+=" --test_batch_size 8"

submit_arm() {
  local arm=$1
  local short=$2
  local seed=$3
  local control_args=$4
  local out="$OUT_ROOT/$arm/$seed"
  local extra="$ENCODER_ARGS $POLICY_ARGS $POOL_ARGS $TRAIN_ARGS"
  extra+=" --seed $seed --output_folder $out $control_args"
  local job
  job=$(submit_job "tc100_${short}_${seed}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "$arm seed=$seed -> $job ($out)"
}

echo "Dataset: train-clean-100 (100h attribution; not LS960)"
echo "Shared initialization: $COLD_DIR and $DECODER_CKPT"
echo "Shared decoder/pooler training length: $TOTAL_EPOCHS epochs"
for seed in "${SEEDS[@]}"; do
  submit_arm fixed_k5_bigru fxbg "$seed" \
    "--freeze_boundary_policy True --fixed_rate_k 5"
  submit_arm frozen_segmenter_bigru frbg "$seed" \
    "--freeze_boundary_policy True"
  submit_arm joint_rl_bigru jrbg "$seed" \
    "--freeze_boundary_policy False"
done

echo "Output root: $OUT_ROOT"
