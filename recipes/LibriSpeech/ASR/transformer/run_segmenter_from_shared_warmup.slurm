#!/bin/bash
# Job shape for a post-warm-up run. The submitter supplies a dependency on the
# shared warm-up job; this wrapper stages that job's exact final epoch checkpoint
# into the destination before invoking the standard segmenter runner.
#SBATCH --job-name=seg_from_warmup
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${WARMUP_SOURCE_DIR:?set WARMUP_SOURCE_DIR to the shared warm-up save directory}"
: "${OUTPUT_FOLDER:?set OUTPUT_FOLDER to the destination for this run}"
: "${WARMUP_EPOCHS:?set WARMUP_EPOCHS}"

selected=""
for ckpt in "$WARMUP_SOURCE_DIR"/CKPT*; do
  [[ -d "$ckpt" ]] || continue
  [[ -f "$ckpt/CKPT.yaml" && -f "$ckpt/counter.ckpt" ]] || continue
  if grep -q '^end-of-epoch: true$' "$ckpt/CKPT.yaml" \
      && [[ "$(tr -d '[:space:]' < "$ckpt/counter.ckpt")" == "$WARMUP_EPOCHS" ]]; then
    if [[ -z "$selected" || "$ckpt" -nt "$selected" ]]; then
      selected="$ckpt"
    fi
  fi
done

if [[ -z "$selected" ]]; then
  echo "No final epoch-$WARMUP_EPOCHS checkpoint under $WARMUP_SOURCE_DIR" >&2
  exit 1
fi
if [[ -e "$OUTPUT_FOLDER" ]]; then
  echo "Refusing to reuse existing output path: $OUTPUT_FOLDER" >&2
  exit 1
fi

required=(
  CKPT.yaml brain.ckpt counter.ckpt dataloader-TRAIN.ckpt llm.ckpt
  model_optimizer.ckpt normalize.ckpt proj.ckpt segmenter.ckpt
)
for name in "${required[@]}"; do
  if [[ ! -f "$selected/$name" ]]; then
    echo "Shared warm-up checkpoint is incomplete: $selected/$name" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_FOLDER/save"
cp -al "$selected" "$OUTPUT_FOLDER/save/"
echo "Staged exact shared warm-up checkpoint: $selected -> $OUTPUT_FOLDER/save/"

exec bash run_segmenter.slurm
