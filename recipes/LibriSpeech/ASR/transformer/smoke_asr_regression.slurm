#!/bin/bash
# Regression guard: the ASR-only segmenter recipe must still train and evaluate
# normally after the multi-task work. The only shared file that changed is
# segmenter.py (active_task_id / _resolve_task), which must be inert here.
#SBATCH --job-name=asr_regress
#SBATCH --time=1:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
WARM=results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00
OUT=${OUT:-results/smoke_asr_regression/$$}
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

echo "=== ASR-only regression smoke (single-task path, modified segmenter.py) ==="
echo "Node: $(hostname)"

"$PYTHON" train_speechllm_with_segmenter.py hparams/speechllm_segmenter.yaml \
  --seed 3408 --output_folder "$OUT" \
  --segmenter_mode joint --segmenter_backbone transformer_ar \
  --ssl_hub microsoft/wavlm-large --ssl_folder "$WAVLM_CACHE" \
  --ssl_feat_dims 1024 --segmenter_input_dim 1024 \
  --segment_pooling bigru_residual --segment_pooling_hidden_dim 128 \
  --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated \
  --segmenter_num_tasks 1 \
  --rl_update_mode combined_on_policy --grpo_k 4 \
  --boundary_source none --warmstart_target none \
  --segmenter_init_checkpoint "$WARM/segmenter.ckpt" \
  --decoder_init_ckpt_dir "$WARM" \
  --skip_prep True --csv_folder "$LS" \
  --train_csv "$LS/dev-clean.csv" --valid_csv "$LS/dev-clean.csv" \
  --test_csv "[$LS/dev-clean.csv]" --test_splits "['dev-clean']" \
  --number_of_epochs 1 --optimizer_step_limit 12 \
  --validation_interval_optimizer_steps 6 --warmup_optimizer_steps 4 \
  --max_batch_length_train 60 --max_batch_length_val 60 \
  --num_workers 2 --test_batch_size 4

echo "=== ASR regression train log ==="
cat "$OUT/train_log.txt"
