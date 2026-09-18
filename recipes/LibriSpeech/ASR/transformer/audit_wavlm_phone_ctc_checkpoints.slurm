#!/bin/bash
# Audio-only dev-clean inference for each of the three retained CTC checkpoints.
#SBATCH --job-name=tc100_phctcaud
#SBATCH --array=0-2%1
#SBATCH --time=03:00:00
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
AUDIT=$ROOT/artifacts/segmenter/wavlm_phone_ctc_tc100_checkpoint_audit
BOUNDARY=/export/jsalt26/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_tc100_best
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$CACHE

cd "$RECIPE"
echo "Audio-only dev-clean inference for CTC checkpoint rank $SLURM_ARRAY_TASK_ID"
"$PY" wavlm_phone_ctc.py infer \
  --audio_root "$DATA" \
  --output_dir "$OUTPUT" \
  --cache_dir "$CACHE" \
  --splits dev-clean \
  --boundary_dir "$BOUNDARY" \
  --audit_dir "$AUDIT" \
  --checkpoint_rank "$SLURM_ARRAY_TASK_ID" \
  --stats_only \
  --require_rate_min 0 \
  --require_rate_max 50 \
  --max_padded_seconds 160 \
  --num_workers 6
