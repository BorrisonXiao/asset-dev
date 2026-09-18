#!/bin/bash
# Run publication-oriented RTF evaluation at batch 1 and at the batch selected
# by select_inference_batch_size.slurm. A separate long-utterance dataset warms
# the GPU before the complete test-clean and test-other measurements.
#SBATCH --job-name=rtf_selected_eval
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${SYSTEM_KEY:?set SYSTEM_KEY}"
: "${EVAL_KIND:?set EVAL_KIND=baseline|learned}"
: "${RUN_DIR:?set RUN_DIR to the source seed run}"
: "${SELECT_DIR:?set SELECT_DIR to the completed seed-3407 selection}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new per-seed result directory}"
: "${SEED:?set SEED}"

PREWARM_BATCHES=${PREWARM_BATCHES:-10}
selected_file="$SELECT_DIR/selected_batch_size.txt"
[[ -s "$selected_file" ]] || {
  echo "Missing selected batch: $selected_file" >&2
  exit 1
}
selected_batch=$(<"$selected_file")
[[ "$selected_batch" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid selected batch: $selected_batch" >&2
  exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Refusing to reuse final evaluation output: $OUTPUT_DIR" >&2
  exit 1
}
[[ -s "$RUN_DIR/dev-other.csv" ]] || {
  echo "Missing source dev-other manifest: $RUN_DIR/dev-other.csv" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR/prewarm"
ln -s "$SELECT_DIR" "$OUTPUT_DIR/batch_selection"

batch_sizes=(1)
if (( selected_batch != 1 )); then
  batch_sizes+=("$selected_batch")
fi

for batch_size in "${batch_sizes[@]}"; do
  prewarm_utterances=$((batch_size * PREWARM_BATCHES))
  prewarm_csv="$OUTPUT_DIR/prewarm/longest_${prewarm_utterances}_batch${batch_size}.csv"
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    rtf_batch_selection.py prepare \
    --source "$RUN_DIR/dev-other.csv" \
    --output "$prewarm_csv" \
    --num-utterances "$prewarm_utterances"

  benchmark_dir="$OUTPUT_DIR/batch$batch_size"
  csv_override="[$prewarm_csv,$benchmark_dir/test-clean.csv,$benchmark_dir/test-other.csv]"
  echo "=== final system=$SYSTEM_KEY seed=$SEED batch=$batch_size ==="
  echo "prewarm_batches=$PREWARM_BATCHES prewarm_csv=$prewarm_csv"
  if [[ "$EVAL_KIND" == baseline ]]; then
    BENCHMARK_DIR="$benchmark_dir" \
    BATCH_SIZE="$batch_size" \
    TEST_CSV_OVERRIDE="$csv_override" \
    INFERENCE_WARMUP_BATCHES=0 \
      bash eval_fixed_inference_benchmark.slurm
  elif [[ "$EVAL_KIND" == learned ]]; then
    BENCHMARK_DIR="$benchmark_dir" \
    BATCH_SIZE="$batch_size" \
    TEST_CSV_OVERRIDE="$csv_override" \
    INFERENCE_WARMUP_BATCHES=0 \
      bash eval_segmenter_inference_benchmark.slurm
  else
    echo "Invalid EVAL_KIND=$EVAL_KIND" >&2
    exit 1
  fi
done

if (( selected_batch != 1 )); then
  for split in test-clean test-other; do
    /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
      rtf_batch_selection.py compare \
      --batch1 "$OUTPUT_DIR/batch1/benchmark_${split}_utterances.jsonl" \
      --selected "$OUTPUT_DIR/batch${selected_batch}/benchmark_${split}_utterances.jsonl" \
      --output "$OUTPUT_DIR/batch_invariance_${split}.json"
  done
fi

{
  echo "system=$SYSTEM_KEY"
  echo "seed=$SEED"
  echo "batch1=1"
  echo "selected_batch_size=$selected_batch"
  echo "prewarm_batches=$PREWARM_BATCHES"
  echo "test_splits=test-clean,test-other"
  echo "inference_warmup_batches_inside_test=0"
} > "$OUTPUT_DIR/protocol.txt"

echo "Completed system=$SYSTEM_KEY seed=$SEED selected_batch=$selected_batch"
