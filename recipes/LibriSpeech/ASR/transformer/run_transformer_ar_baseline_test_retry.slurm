#!/bin/bash
# Test-only recovery of the completed baseline checkpoint from job 70131.
# EpochCounter is already at its one-epoch limit, so fit() recovers and performs
# no additional training before evaluating test-clean.
#SBATCH --job-name=tar_base_test
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

ROOT=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm
RECIPE="$ROOT/recipes/LibriSpeech/ASR/transformer"
OUT="$RECIPE/results/speechllm_segmenter_wavlm/transformer_ar_combined_rollout_validation/full_baseline_70131"
SOURCE="$RECIPE/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_best_char_decoder/nll_mt/3407"
COLD_CKPT="$RECIPE/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/coldstart/3407/save/CKPT+2026-08-11+21-43-59+00/segmenter.ckpt"
DECODER_CKPT="$RECIPE/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00"

cd "$RECIPE"
started=$(date +%s)
set +e
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  train_speechllm_with_segmenter.py hparams/speechllm_segmenter.yaml \
  --segmenter_mode joint --segmenter_backbone transformer_ar \
  --number_of_epochs 1 --warmup_epochs 0 --grpo_k 4 \
  --output_folder "$OUT" \
  --stage_timing_file "$OUT/stage_timing.jsonl" \
  --train_csv "$SOURCE/train.csv" --valid_csv "$SOURCE/dev-clean.csv" \
  --test_csv "[$SOURCE/test-clean.csv]" --skip_prep True \
  --test_batch_size 8 \
  --ssl_hub microsoft/wavlm-large \
  --ssl_folder /home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub \
  --ssl_feat_dims 1024 --segmenter_input_dim 1024 \
  --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4 \
  --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024 \
  --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096 \
  --segmenter_ar_cache_mode preallocated --segmenter_ar_history_window 64 \
  --segmenter_init_checkpoint "$COLD_CKPT" \
  --decoder_init_ckpt_dir "$DECODER_CKPT" \
  --segmenter_reward nll --freeze_decoder_in_joint False \
  --rl_update_mode on_policy --grad_accumulation_factor 4 \
  --initial_lr 0.0002 --lr_decoder 0.0002 \
  --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005 \
  --pg_weight 1.0 --max_decode_ratio 3.0 \
  --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0
exit_status=$?
set -e
ended=$(date +%s)
printf '{"job_id":"%s","source_job":"70131","test_batch_size":8,"wall_seconds":%s,"exit_status":%s}\n' \
  "$SLURM_JOB_ID" "$((ended-started))" "$exit_status" \
  > "$OUT/test_retry_${SLURM_JOB_ID}.json"
exit "$exit_status"
