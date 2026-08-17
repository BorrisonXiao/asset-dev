#!/bin/bash
# Three-seed full-prefix Transformer-AR runs initialized with the strongest
# oracle-char WavLM decoder. Matches the best learned WavLM setting, except that
# the joint-RL phase uses the same 10-epoch budget.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
WARMUP_EPOCHS=2
JOINT_RL_EPOCHS=10
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
COLDSTART_JOB=${COLDSTART_JOB:-64597}
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/coldstart/3407/save
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
OUT_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_best_char_decoder/nll_mt

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

for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
for seed in "${SEEDS[@]}"; do
  [[ ! -e "$OUT_ROOT/$seed" ]] || die "Refusing to reuse output path: $OUT_ROOT/$seed"
done

mkdir -p slurm_logs
ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_transformer_ar_best_char"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  extra="$ENCODER_ARGS $POLICY_ARGS --seed $seed --output_folder $out"
  extra+=" --decoder_init_ckpt_dir $DECODER_CKPT"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
  extra+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
  extra+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
  extra+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  job=$(submit_job "wl_tar_bc_$seed" \
    --dependency="afterok:$COLDSTART_JOB" \
    --export="ALL,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "best-char decoder seed=$seed -> $job (afterok:$COLDSTART_JOB; $out)"
done

echo "Schedule: $WARMUP_EPOCHS decoder-adaptation epochs + $JOINT_RL_EPOCHS joint-RL epochs"
echo "Initialization: best full-prefix char segmenter from job $COLDSTART_JOB + best oracle-char decoder"
echo "Output root: $OUT_ROOT"
