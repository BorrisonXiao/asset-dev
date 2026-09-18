#!/bin/bash
# Select one throughput batch by GPU memory only. Seed 3407 supplies the model;
# the selected batch is then frozen for all seeds of the same system.
#SBATCH --job-name=rtf_batch_select
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${SYSTEM_KEY:?set SYSTEM_KEY}"
: "${EVAL_KIND:?set EVAL_KIND=baseline|learned}"
: "${RUN_DIR:?set RUN_DIR to the seed-3407 source run}"
: "${SELECT_DIR:?set SELECT_DIR to a new selection output directory}"
: "${SEED:?set SEED=3407}"

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-2}
MAX_MEMORY_FRACTION=${MAX_MEMORY_FRACTION:-0.90}

[[ "$SEED" == 3407 ]] || {
  echo "Batch selection must use seed 3407, got $SEED" >&2
  exit 1
}
[[ "$EVAL_KIND" == baseline || "$EVAL_KIND" == learned ]] || {
  echo "Invalid EVAL_KIND=$EVAL_KIND" >&2
  exit 1
}
[[ -s "$RUN_DIR/dev-other.csv" ]] || {
  echo "Missing source dev-other manifest: $RUN_DIR/dev-other.csv" >&2
  exit 1
}
[[ ! -e "$SELECT_DIR" ]] || {
  echo "Refusing to reuse selection output: $SELECT_DIR" >&2
  exit 1
}

mkdir -p "$SELECT_DIR/calibration" "$SELECT_DIR/candidates"
printf 'batch_size\tstatus\tbenchmark_dir\tsummary\n' > "$SELECT_DIR/candidates.tsv"

selected=0
termination=""
batch_size=1
while (( batch_size <= MAX_BATCH_SIZE )); do
  utterances=$((batch_size * CALIBRATION_BATCHES))
  calibration_csv="$SELECT_DIR/calibration/longest_${utterances}.csv"
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    rtf_batch_selection.py prepare \
    --source "$RUN_DIR/dev-other.csv" \
    --output "$calibration_csv" \
    --num-utterances "$utterances"

  candidate_dir="$SELECT_DIR/candidates/batch$batch_size"
  runner_log="$SELECT_DIR/candidates/batch${batch_size}.runner.log"
  csv_override="[$calibration_csv]"
  echo "=== candidate system=$SYSTEM_KEY batch=$batch_size utterances=$utterances ==="

  set +e
  {
    if [[ "$EVAL_KIND" == baseline ]]; then
      BENCHMARK_DIR="$candidate_dir" \
      BATCH_SIZE="$batch_size" \
      TEST_CSV_OVERRIDE="$csv_override" \
      INFERENCE_WARMUP_BATCHES=0 \
        bash eval_fixed_inference_benchmark.slurm
    else
      BENCHMARK_DIR="$candidate_dir" \
      BATCH_SIZE="$batch_size" \
      TEST_CSV_OVERRIDE="$csv_override" \
      INFERENCE_WARMUP_BATCHES=0 \
        bash eval_segmenter_inference_benchmark.slurm
    fi
  } > "$runner_log" 2>&1
  candidate_status=$?
  set -e

  if (( candidate_status != 0 )); then
    if rg -q 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$runner_log"; then
      printf '%d\toom\t%s\t—\n' "$batch_size" "$candidate_dir" \
        >> "$SELECT_DIR/candidates.tsv"
      termination="oom"
      echo "batch=$batch_size failed with CUDA OOM; selecting batch=$selected"
      break
    fi
    echo "Candidate batch=$batch_size failed for a non-OOM reason" >&2
    tail -n 80 "$runner_log" >&2
    exit "$candidate_status"
  fi

  summary="$SELECT_DIR/candidates/batch${batch_size}.summary.json"
  set +e
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    rtf_batch_selection.py inspect \
    --benchmark "$candidate_dir/benchmark.jsonl" \
    --expected-batch "$batch_size" \
    --max-memory-fraction "$MAX_MEMORY_FRACTION" \
    --output "$summary"
  inspect_status=$?
  set -e
  if (( inspect_status == 0 )); then
    selected=$batch_size
    printf '%d\tpass\t%s\t%s\n' "$batch_size" "$candidate_dir" "$summary" \
      >> "$SELECT_DIR/candidates.tsv"
  elif (( inspect_status == 2 )); then
    printf '%d\tmemory_limit\t%s\t%s\n' \
      "$batch_size" "$candidate_dir" "$summary" \
      >> "$SELECT_DIR/candidates.tsv"
    termination="memory_limit"
    echo "batch=$batch_size exceeded memory ceiling; selecting batch=$selected"
    break
  else
    echo "Could not validate batch=$batch_size benchmark" >&2
    exit "$inspect_status"
  fi
  batch_size=$((batch_size * 2))
done

(( selected >= 1 )) || {
  echo "No batch satisfied the memory rule" >&2
  exit 1
}
if [[ -z "$termination" ]]; then
  echo "Batch $MAX_BATCH_SIZE passed; raise MAX_BATCH_SIZE before claiming max-fit" >&2
  exit 1
fi

printf '%d\n' "$selected" > "$SELECT_DIR/selected_batch_size.txt"
{
  echo "system=$SYSTEM_KEY"
  echo "seed=$SEED"
  echo "selected_batch_size=$selected"
  echo "max_memory_fraction=$MAX_MEMORY_FRACTION"
  echo "calibration_batches=$CALIBRATION_BATCHES"
  echo "termination=$termination"
} > "$SELECT_DIR/selection.txt"

echo "Selected system=$SYSTEM_KEY batch=$selected termination=$termination"
