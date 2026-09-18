#!/bin/bash
# Audio-only CTC posterior-collapse preprocessing for the 100h ASR baseline.
# Account/partition/comment/reservation/exclude are supplied by the submitter.
#SBATCH --job-name=tc100_ctcgen
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%A_%a.out

set -euo pipefail

RECIPE=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
OUT=/export/jsalt26/omnienc/users/cxiao/boundary_targets/ctc_posterior_argmax_nonblank
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export TORCH_HOME=/export/jsalt26/omnienc/users/cxiao/.cache/torch
export PYTHONUNBUFFERED=1

cd "$RECIPE"
echo "Dataset: train-clean-100 plus dev/test evaluation splits (100h baseline; not LS960)"
echo "Boundary source: audio-only wav2vec2 CTC argmax; no reference transcript"
echo "Output: $OUT"

"$PY" ctc_posterior_collapse.py \
  --audio_root "$DATA" \
  --out_root "$OUT" \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --device cuda \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --num_shards 8
