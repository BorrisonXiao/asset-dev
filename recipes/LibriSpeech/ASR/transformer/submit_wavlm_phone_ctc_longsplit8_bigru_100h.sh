#!/bin/bash
# Submit the ~11 Hz phone-CTC long-segment refinement downstream baseline.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

BOUNDARY=/export/jsalt26/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_tc100_best_longsplit8
OUT_ROOT=results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/wavlm_phone_ctc_longsplit8_bigru_3ep
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save
SEEDS=(3407 3408 3409)

identity=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
)

die() { echo "$*" >&2; exit 1; }
[[ -s "$BOUNDARY/_stats/long_segment_split_summary.json" ]] \
  || die "Missing refined-boundary summary: $BOUNDARY"
[[ "$(find "$BOUNDARY" -maxdepth 1 -type f -name '*.pt' | wc -l)" -eq 39665 ]] \
  || die "Expected 39,665 refined boundary tensors under $BOUNDARY"
for seed in "${SEEDS[@]}"; do
  [[ ! -e "$OUT_ROOT/$seed" ]] || die "Refusing to overwrite $OUT_ROOT/$seed"
done

echo "Dataset: train-clean-100 (100h baseline; not LS960)"
echo "Boundary rule: preserve phone-CTC boundaries; split segments >=8 frames once at midpoint"
echo "Boundary source: $BOUNDARY"
echo "GPU policy: three one-A100 downstream jobs"
echo "Outputs: $OUT_ROOT/{3407,3408,3409}; estimated under 3 GB total"
echo "Checkpoints: retain/evaluate three per seed"

wavlm_args="--experiment_name speechllm_segmenter_wavlm_attribution_100h"
wavlm_args+=" --ssl_hub microsoft/wavlm-large"
wavlm_args+=" --ssl_folder /export/jsalt26/omnienc/users/cxiao/hf/hub"
wavlm_args+=" --ssl_feat_dims 1024 --segmenter_input_dim 1024"
wavlm_args+=" --train_splits ['train-clean-100']"
policy_args="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
policy_args+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
policy_args+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
policy_args+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
pool_args="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
pool_args+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
train_args="--decoder_init_ckpt_dir $DECODER_CKPT --coldstart_ckpt_dir $COLD_DIR"
train_args+=" --boundary_source alignment --boundary_target_dir $BOUNDARY"
train_args+=" --segmenter_reward nll --freeze_decoder_in_joint False"
train_args+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
train_args+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
train_args+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
train_args+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
train_args+=" --test_batch_size 8 --freeze_boundary_policy True"
common_args="$wavlm_args $policy_args $pool_args $train_args"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  seed_args="$common_args --seed $seed"
  downstream_args="$seed_args --output_folder $out --checkpoints_to_keep 3"
  downstream_args+=" --test_splits [] --test_csv []"
  train_job=$(sbatch --parsable \
    "${identity[@]}" \
    --cpus-per-task=12 --mem=80G --time=2-00:00:00 \
    --job-name="tc100_ph11_$seed" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=3,WARMUP_EPOCHS=0,EXTRA_ARGS=$downstream_args" \
    run_segmenter.slurm)
  eval_job=$(sbatch --parsable \
    "${identity[@]}" \
    --dependency="afterok:$train_job" \
    --job-name="tc100_ph11_e$seed" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NUMBER_OF_EPOCHS=3,OUTPUT_FOLDER=$out,EXTRA_ARGS=$seed_args" \
    eval_retained_phone_ctc_bigru.slurm)
  echo "seed=$seed train=$train_job eval=$eval_job"
done
