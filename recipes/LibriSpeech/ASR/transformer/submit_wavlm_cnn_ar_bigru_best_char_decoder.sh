#!/bin/bash
# Matched three-seed pooling ablation for the WavLM CNN first-order AR policy.
# This changes only mean pooling -> zero-residual BiGRU pooling relative to
# oracleclose_bestinit/autoregressive, keeping initialization and optimization
# fixed.  Together with the existing Transformer-AR runs, it completes the
# {CNN, Transformer} x {mean, BiGRU} comparison.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
WARMUP_EPOCHS=2
JOINT_RL_EPOCHS=10
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
SEGMENTER_CKPT=$(pwd)/results/speechllm_segmenter_wavlm/coldstart/cnn/3407/save/CKPT+2026-08-07+11-19-21+00/segmenter.ckpt
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
OUT_ROOT=results/speechllm_segmenter_wavlm/cnn_first_order_ar_bigru_best_char_decoder/nll_mt

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=8
  --mem=60G
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

[[ -s "$SEGMENTER_CKPT" ]] || die "Missing segmenter checkpoint: $SEGMENTER_CKPT"
for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Seed must be an integer: $seed"
  [[ ! -e "$OUT_ROOT/$seed" ]] \
    || die "Refusing to reuse seed output: $OUT_ROOT/$seed"
done
mkdir -p slurm_logs

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  extra="--experiment_name speechllm_segmenter_wavlm_cnn_ar_bigru_best_char"
  extra+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE --ssl_feat_dims $WAVLM_DIM"
  extra+=" --segmenter_input_dim $WAVLM_DIM --seed $seed --output_folder $out"
  extra+=" --segmenter_init_checkpoint $SEGMENTER_CKPT"
  extra+=" --decoder_init_ckpt_dir $DECODER_CKPT"
  extra+=" --segmenter_sampling autoregressive"
  extra+=" --segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
  extra+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002"
  extra+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0"
  extra+=" --stage_timing_file $out/stage_timing.jsonl"

  job=$(submit_job "wl_car_bg_$seed" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=cnn,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "CNN first-order AR + BiGRU seed=$seed -> $job ($out)"
done

echo "Configuration: CNN first-order AR, BiGRU residual pooling (hidden=128), K=4"
echo "Schedule: $WARMUP_EPOCHS decoder/pooler adaptation + $JOINT_RL_EPOCHS joint-RL epochs"
echo "Output root: $OUT_ROOT"
