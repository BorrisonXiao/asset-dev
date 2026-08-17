#!/bin/bash
# Fixed-window-64 real-batch timing. Experimental hook is opt-in via the environment.
#SBATCH --job-name=tar_flex_train
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

: "${VARIANT:?set VARIANT=dense64_ref|flex64_ref|dense64_reward|dense64_group|flex64_reward|flex64_group}"

ROOT=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm
RECIPE="$ROOT/recipes/LibriSpeech/ASR/transformer"
SOURCE="$RECIPE/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_best_char_decoder/nll_mt/3407"
COLD_CKPT="$RECIPE/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/coldstart/3407/save/CKPT+2026-08-11+21-43-59+00/segmenter.ckpt"
DECODER_CKPT="$RECIPE/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00"
OUT_ROOT="$RECIPE/results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench"
OUT="$OUT_ROOT/${VARIANT}_${SLURM_JOB_ID}"
HEADER_ROOT="$RECIPE/results/speechllm_segmenter_wavlm/transformer_ar_attention_bench/uv-python/cpython-3.11.15-linux-x86_64-gnu/include/python3.11"

case "$VARIANT" in
  dense64_ref)    RUNTIME=;                    RL_MODE=on_policy ;;
  flex64_ref)     RUNTIME=flex_ref;            RL_MODE=on_policy ;;
  dense64_reward) RUNTIME=dense_reward_batch;  RL_MODE=batched_on_policy ;;
  dense64_group)  RUNTIME=dense_group_batch;   RL_MODE=batched_on_policy ;;
  flex64_reward)  RUNTIME=flex_reward_batch;   RL_MODE=batched_on_policy ;;
  flex64_group)   RUNTIME=flex_group_batch;    RL_MODE=batched_on_policy ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

cd "$RECIPE"
mkdir -p "$OUT" slurm_logs
export SEGMENTER_EXPERIMENTAL_RUNTIME="$RUNTIME"
export CPATH="$HEADER_ROOT${CPATH:+:$CPATH}"

started=$(date +%s)
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  train_speechllm_with_segmenter.py hparams/speechllm_segmenter.yaml \
  --debug --debug_batches 30 --debug_epochs 1 --debug_persistently \
  --segmenter_mode joint --segmenter_backbone transformer_ar \
  --number_of_epochs 1 --warmup_epochs 0 --grpo_k 4 \
  --output_folder "$OUT" \
  --train_csv "$SOURCE/train.csv" --valid_csv "$SOURCE/dev-clean.csv" \
  --test_csv "[$SOURCE/test-clean.csv]" --skip_prep True \
  --ssl_hub microsoft/wavlm-large \
  --ssl_folder /home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub \
  --ssl_feat_dims 1024 --segmenter_input_dim 1024 \
  --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4 \
  --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024 \
  --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096 \
  --segmenter_ar_cache_mode preallocated \
  --segmenter_ar_history_window 64 \
  --segmenter_init_checkpoint "$COLD_CKPT" \
  --decoder_init_ckpt_dir "$DECODER_CKPT" \
  --segmenter_reward nll --freeze_decoder_in_joint False \
  --rl_update_mode "$RL_MODE" --grad_accumulation_factor 4 \
  --initial_lr 0.0002 --lr_decoder 0.0002 \
  --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005 \
  --pg_weight 1.0 --max_decode_ratio 3.0 \
  --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0
ended=$(date +%s)

runtime_json=null
if [[ -n "$RUNTIME" ]]; then runtime_json="\"$RUNTIME\""; fi
printf '{"job_id":"%s","variant":"%s","runtime":%s,"history_window":64,"grpo_k":4,"debug_batches":30,"wall_seconds":%s}\n' \
  "$SLURM_JOB_ID" "$VARIANT" "$runtime_json" "$((ended-started))" \
  > "$OUT/benchmark_summary.json"
