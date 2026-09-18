#!/bin/bash
# Matched 100h attribution control: fixed k=5 boundaries, parameter-free mean
# pooling, and the same best character-decoder initialization/training length
# used by the fixed-k=5 + BiGRU arm.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
EPOCHS=${EPOCHS:-7}
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save
OUT_ROOT=${OUT_ROOT:-results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru}
ARM=fixed_k5_mean

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
for seed in "${SEEDS[@]}"; do
  [[ ! -e "$OUT_ROOT/$ARM/$seed" ]] \
    || die "Refusing to reuse output: $OUT_ROOT/$ARM/$seed"
done
mkdir -p slurm_logs

EXTRA="--experiment_name speechllm_segmenter_wavlm_attribution_100h"
EXTRA+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
EXTRA+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
EXTRA+=" --train_splits ['train-clean-100']"
EXTRA+=" --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
EXTRA+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
EXTRA+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
EXTRA+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
EXTRA+=" --segment_pooling mean"
EXTRA+=" --decoder_init_ckpt_dir $DECODER_CKPT"
EXTRA+=" --segmenter_reward nll --freeze_decoder_in_joint False"
EXTRA+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
EXTRA+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
EXTRA+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
EXTRA+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
EXTRA+=" --test_batch_size 8"
EXTRA+=" --freeze_boundary_policy True --fixed_rate_k 5"

echo "Dataset: train-clean-100 (100h attribution; not LS960)"
echo "Boundary/pooling: fixed k=5 / mean"
echo "Decoder initialization: $DECODER_CKPT"
echo "Decoder CE training length: $EPOCHS epochs"
for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$ARM/$seed"
  args="$EXTRA --seed $seed --output_folder $out"
  job=$(submit_job "tc100_fxmn_${seed}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$EPOCHS,WARMUP_EPOCHS=0,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$args" \
    run_segmenter.slurm)
  echo "$ARM seed=$seed -> $job ($out)"
done

echo "Output root: $OUT_ROOT/$ARM"
