#!/bin/bash
# Resolve the seed-matched phone-oracle decoder checkpoint (selected by dev
# CER), then delegate to the standard segmenter runner for the learned
# 24k-step joint run. Env: DECODER_SAVE_DIR, MODE=joint, plus everything
# run_segmenter.slurm consumes.
#SBATCH --job-name=xling_joint
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${DECODER_SAVE_DIR:?set DECODER_SAVE_DIR to a completed phone-oracle save directory}"
: "${MODE:?set MODE=joint}"
[[ "$MODE" == joint ]] || { echo "This resolver is for MODE=joint only" >&2; exit 1; }

best_ckpt=""
best_metric=""
for meta in "$DECODER_SAVE_DIR"/CKPT*/CKPT.yaml; do
  [[ -f "$meta" ]] || continue
  ckpt=${meta%/CKPT.yaml}
  [[ -s "$ckpt/llm.ckpt" && -s "$ckpt/proj.ckpt" && -s "$ckpt/normalize.ckpt" ]] || continue
  metric=$(awk '/^CER:/ {print $2; exit}' "$meta")
  [[ -n "$metric" ]] || metric=$(awk '/^WER:/ {print $2; exit}' "$meta")
  [[ -n "$metric" ]] || continue
  if [[ -z "$best_metric" ]] || awk -v lhs="$metric" -v rhs="$best_metric" 'BEGIN {exit !(lhs < rhs)}'; then
    best_metric=$metric
    best_ckpt=$ckpt
  fi
done

[[ -n "$best_ckpt" ]] || {
  echo "No complete decoder checkpoint with CER/WER metadata under $DECODER_SAVE_DIR" >&2
  exit 1
}

echo "Resolved phone-oracle decoder init: $best_ckpt (dev CER=$best_metric)"
EXTRA_ARGS="${EXTRA_ARGS:-} --decoder_init_ckpt_dir $best_ckpt"
export EXTRA_ARGS

exec bash run_segmenter.slurm
