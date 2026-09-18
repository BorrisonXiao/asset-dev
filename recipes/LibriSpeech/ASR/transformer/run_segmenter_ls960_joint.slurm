#!/bin/bash
# Resolve the best completed LibriSpeech-960h oracle-char decoder checkpoint,
# then delegate to the standard segmenter runner. Submission identity and
# resources are supplied by submit_ls960_scaleup.sh.
#SBATCH --job-name=ls960_joint
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --requeue
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${DECODER_SAVE_DIR:?set DECODER_SAVE_DIR to a completed oracle-char save directory}"
: "${MODE:?set MODE=joint}"
[[ "$MODE" == joint ]] || { echo "This resolver is for MODE=joint only" >&2; exit 1; }

best_ckpt=""
best_wer=""
for meta in "$DECODER_SAVE_DIR"/CKPT*/CKPT.yaml; do
  [[ -f "$meta" ]] || continue
  ckpt=${meta%/CKPT.yaml}
  [[ -s "$ckpt/llm.ckpt" && -s "$ckpt/proj.ckpt" && -s "$ckpt/normalize.ckpt" ]] || continue
  wer=$(awk '/^WER:/ {print $2; exit}' "$meta")
  [[ -n "$wer" ]] || continue
  if [[ -z "$best_wer" ]] || awk -v lhs="$wer" -v rhs="$best_wer" 'BEGIN {exit !(lhs < rhs)}'; then
    best_wer=$wer
    best_ckpt=$ckpt
  fi
done

[[ -n "$best_ckpt" ]] || {
  echo "No complete decoder checkpoint with WER metadata under $DECODER_SAVE_DIR" >&2
  exit 1
}

echo "Resolved LS960 oracle-char decoder: $best_ckpt (dev-clean WER=$best_wer)"
EXTRA_ARGS="${EXTRA_ARGS:-} --decoder_init_ckpt_dir $best_ckpt"
export EXTRA_ARGS

exec bash run_segmenter.slurm
