#!/bin/bash
# Evaluate one fixed RTF operating point while reusing an existing batch-1
# measurement only for the hypothesis comparison. Results always go to a fresh
# directory and never modify the source training or batch-1 artifacts.
#SBATCH --job-name=rtf_fixed_batch
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${SYSTEM_KEY:?set SYSTEM_KEY}"
: "${EVAL_KIND:?set EVAL_KIND=baseline|learned}"
: "${RUN_DIR:?set RUN_DIR to the source seed run}"
: "${BATCH1_DIR:?set BATCH1_DIR to the completed batch-1 artifact}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}"
: "${SEED:?set SEED}"
: "${BATCH_SIZE:?set BATCH_SIZE}"

PREWARM_BATCHES=${PREWARM_BATCHES:-10}
PROTOCOL_TAG=${PROTOCOL_TAG:-fixed_batch_rtf_v1}
SELECTION_REASON=${SELECTION_REASON:-declared_common_batch}

[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid BATCH_SIZE=$BATCH_SIZE" >&2
  exit 1
}
(( (BATCH_SIZE & (BATCH_SIZE - 1)) == 0 )) || {
  echo "BATCH_SIZE must be a power of two, got $BATCH_SIZE" >&2
  exit 1
}
[[ "$EVAL_KIND" == baseline || "$EVAL_KIND" == learned ]] || {
  echo "Invalid EVAL_KIND=$EVAL_KIND" >&2
  exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Refusing to reuse fixed-batch output: $OUTPUT_DIR" >&2
  exit 1
}
for split in test-clean test-other; do
  [[ -s "$BATCH1_DIR/benchmark_${split}_utterances.jsonl" ]] || {
    echo "Missing batch-1 hypotheses for $split: $BATCH1_DIR" >&2
    exit 1
  }
done

prewarm_utterances=$((BATCH_SIZE * PREWARM_BATCHES))
mkdir -p "$OUTPUT_DIR/prewarm"
prewarm_csv="$OUTPUT_DIR/prewarm/longest_${prewarm_utterances}_batch${BATCH_SIZE}.csv"
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  rtf_batch_selection.py prepare \
  --source "$RUN_DIR/dev-other.csv" \
  --output "$prewarm_csv" \
  --num-utterances "$prewarm_utterances"

benchmark_dir="$OUTPUT_DIR/batch${BATCH_SIZE}"
csv_override="[$prewarm_csv,$benchmark_dir/test-clean.csv,$benchmark_dir/test-other.csv]"
echo "=== fixed-batch system=$SYSTEM_KEY seed=$SEED batch=$BATCH_SIZE ==="
echo "source=$RUN_DIR batch1_source=$BATCH1_DIR output=$OUTPUT_DIR"
echo "prewarm_batches=$PREWARM_BATCHES allocator=$PYTORCH_CUDA_ALLOC_CONF"

if [[ "$EVAL_KIND" == baseline ]]; then
  BENCHMARK_DIR="$benchmark_dir" \
  BATCH_SIZE="$BATCH_SIZE" \
  TEST_CSV_OVERRIDE="$csv_override" \
  INFERENCE_WARMUP_BATCHES=0 \
    bash eval_fixed_inference_benchmark.slurm
else
  BENCHMARK_DIR="$benchmark_dir" \
  BATCH_SIZE="$BATCH_SIZE" \
  TEST_CSV_OVERRIDE="$csv_override" \
  INFERENCE_WARMUP_BATCHES=0 \
    bash eval_segmenter_inference_benchmark.slurm
fi

for split in test-clean test-other; do
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    rtf_batch_selection.py compare \
    --batch1 "$BATCH1_DIR/benchmark_${split}_utterances.jsonl" \
    --selected "$benchmark_dir/benchmark_${split}_utterances.jsonl" \
    --output "$OUTPUT_DIR/batch_invariance_${split}.json"
done

{
  echo "protocol=$PROTOCOL_TAG"
  echo "system=$SYSTEM_KEY"
  echo "seed=$SEED"
  echo "batch1_source=$BATCH1_DIR"
  echo "batch_size=$BATCH_SIZE"
  echo "selection_reason=$SELECTION_REASON"
  echo "prewarm_batches=$PREWARM_BATCHES"
  echo "allocator=$PYTORCH_CUDA_ALLOC_CONF"
  echo "test_splits=test-clean,test-other"
} > "$OUTPUT_DIR/protocol.txt"

echo "Completed system=$SYSTEM_KEY seed=$SEED batch=$BATCH_SIZE"
