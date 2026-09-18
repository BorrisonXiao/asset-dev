#!/bin/bash
# Generate the frozen phone-CTC + long-segment-split boundary stream for LS960.
# Submission identity and the array shape are supplied by the launcher.
#SBATCH --job-name=ls960_phctc11_gen
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%A_%a.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
CTC_OUTPUT=$ROOT/artifacts/models/wavlm_phone_ctc_tc100_seed3407
AUDIT=$ROOT/artifacts/segmenter/wavlm_phone_ctc_tc100_checkpoint_audit
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

: "${BOUNDARY_DIR:?set BOUNDARY_DIR}"
: "${NUM_SHARDS:?set NUM_SHARDS}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$CACHE

cd "$RECIPE"
echo "Scale: LS960 = train-clean-100 + train-clean-360 + train-other-500"
echo "Additional inference splits: dev-clean dev-other test-clean test-other"
echo "Boundary rule: phone-CTC nonblank runs plus midpoint for segments >=8 frames"
echo "Transcript used during boundary inference: no"
echo "Output: $BOUNDARY_DIR (estimated 1.2 GB total across the full corpus)"
"$PY" wavlm_phone_ctc.py infer \
  --audio_root "$DATA" \
  --output_dir "$CTC_OUTPUT" \
  --cache_dir "$CACHE" \
  --splits train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other \
  --boundary_dir "$BOUNDARY_DIR" \
  --audit_dir "$AUDIT" \
  --checkpoint_rank 0 \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS" \
  --long_segment_split_min_frames 8 \
  --require_max_per 0.40 \
  --require_rate_min 7 \
  --require_rate_max 14 \
  --max_padded_seconds 180 \
  --num_workers 6
