#!/bin/bash
# One shared local-attention Transformer-AR char cold start, followed by a
# three-seed best-char-decoder study using the production combined rollout.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
COLDSTART_EPOCHS=10
WARMUP_EPOCHS=2
JOINT_RL_EPOCHS=10
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
OUT_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder
COLD_OUT=$OUT_ROOT/coldstart/3407
COLD_DIR=$(pwd)/$COLD_OUT/save

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --cpus-per-task=12
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

for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
[[ ! -e "$OUT_ROOT" ]] || die "Refusing to reuse output root: $OUT_ROOT"
mkdir -p slurm_logs

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_transformer_ar_local64_best_char"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"

cold_extra="$ENCODER_ARGS $POLICY_ARGS --seed 3407 --output_folder $COLD_OUT"
cold_extra+=" --test_batch_size 8"
cold_job=$(submit_job "wl_tar_l64_cs" \
  --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=coldstart,BACKBONE=transformer_ar,EPOCHS=$COLDSTART_EPOCHS,EXTRA_ARGS=$cold_extra" \
  run_segmenter.slurm)
echo "local-64 char cold start -> $cold_job ($COLD_OUT)"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/nll_mt/$seed"
  extra="$ENCODER_ARGS $POLICY_ARGS --seed $seed --output_folder $out"
  extra+=" --decoder_init_ckpt_dir $DECODER_CKPT"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
  extra+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
  extra+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
  extra+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --test_batch_size 8"
  job=$(submit_job "wl_tar_l64_$seed" \
    --dependency="afterok:$cold_job" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "local-64 best-char seed=$seed -> $job (afterok:$cold_job; $out)"
done

echo "Configuration: local history=64, preallocated KV, combined_on_policy, K=4"
echo "Schedule: $COLDSTART_EPOCHS shared cold-start epochs; then $WARMUP_EPOCHS decoder-adaptation + $JOINT_RL_EPOCHS joint-RL epochs"
echo "Output root: $OUT_ROOT"
