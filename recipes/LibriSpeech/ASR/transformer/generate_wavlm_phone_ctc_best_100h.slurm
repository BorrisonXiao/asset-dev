#!/bin/bash
# Audio-only full boundary generation from the best dev-PER CTC checkpoint.
#SBATCH --job-name=tc100_phctcgen
#SBATCH --array=0-3%2
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%A_%a.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
OUTPUT=$ROOT/artifacts/models/wavlm_phone_ctc_tc100_seed3407
BOUNDARY=/export/jsalt26/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_tc100_best
AUDIT=$ROOT/artifacts/segmenter/wavlm_phone_ctc_tc100_checkpoint_audit
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$CACHE

cd "$RECIPE"
echo "Dataset: train-clean-100 plus dev/test splits (100h baseline; not LS960)"
echo "Boundary source: self-trained WavLM phone-CTC checkpoint rank 0"
echo "Inference: audio only; ordinary nonblank CTC runs"
echo "Output: $BOUNDARY (estimated under 1 GB)"
"$PY" wavlm_phone_ctc.py infer \
  --audio_root "$DATA" \
  --output_dir "$OUTPUT" \
  --cache_dir "$CACHE" \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --boundary_dir "$BOUNDARY" \
  --audit_dir "$AUDIT" \
  --checkpoint_rank 0 \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --num_shards 4 \
  --require_max_per 0.40 \
  --require_rate_min 7 \
  --require_rate_max 14 \
  --max_padded_seconds 160 \
  --num_workers 6
