#!/bin/bash
# Evaluation-only end-to-end benchmark for an existing learned-segmenter run.
# RUN_DIR is read-only: checkpoints and prepared CSVs are referenced in place,
# while all benchmark logs and WER files go to BENCHMARK_DIR.
#SBATCH --job-name=seg_infer_bench
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${RUN_DIR:?set RUN_DIR to an existing learned-segmenter seed directory}"
: "${BENCHMARK_DIR:?set BENCHMARK_DIR to a new benchmark output directory}"
: "${SEED:?set SEED to the source run seed}"
: "${BACKBONE:?set BACKBONE=cnn|transformer_ar}"
: "${BATCH_SIZE:?set BATCH_SIZE=1 or a throughput batch size}"
: "${EXTRA_ARGS:?set EXTRA_ARGS to the source architecture overrides}"

INFERENCE_WARMUP_BATCHES=${INFERENCE_WARMUP_BATCHES:-5}
TEST_CSV_OVERRIDE=${TEST_CSV_OVERRIDE:-}

[[ -d "$RUN_DIR/save" ]] || {
  echo "Missing checkpoint directory: $RUN_DIR/save" >&2
  exit 1
}
[[ ! -e "$BENCHMARK_DIR" ]] || {
  echo "Refusing to reuse benchmark output: $BENCHMARK_DIR" >&2
  exit 1
}

mkdir -p "$BENCHMARK_DIR" "$BENCHMARK_DIR/wer_results"
for csv in train.csv dev-clean.csv test-clean.csv test-other.csv dev-other.csv; do
  [[ -s "$RUN_DIR/$csv" ]] || {
    echo "Missing prepared manifest: $RUN_DIR/$csv" >&2
    exit 1
  }
  ln -s "$RUN_DIR/$csv" "$BENCHMARK_DIR/$csv"
done

args=(
  hparams/speechllm_segmenter.yaml
  --segmenter_mode joint
  --segmenter_backbone "$BACKBONE"
  --seed "$SEED"
  --number_of_epochs 12
  --output_folder "$BENCHMARK_DIR"
  --save_folder "$RUN_DIR/save"
  --output_wer_folder "$BENCHMARK_DIR/wer_results"
  --train_log "$BENCHMARK_DIR/eval.log"
  --stage_timing_file "$BENCHMARK_DIR/stage_timing.jsonl"
  --eval_only True
  --inference_benchmark_file "$BENCHMARK_DIR/benchmark.jsonl"
  --inference_warmup_batches "$INFERENCE_WARMUP_BATCHES"
  --test_batch_size "$BATCH_SIZE"
  --skip_prep True
)

if [[ -n "$TEST_CSV_OVERRIDE" ]]; then
  args+=(--test_csv "$TEST_CSV_OVERRIDE")
fi

# Match the source run's encoder, policy, and pooler construction exactly.
args+=($EXTRA_ARGS)

echo "=== $(date) inference benchmark on $(hostname) ==="
echo "source=$RUN_DIR backbone=$BACKBONE seed=$SEED batch=$BATCH_SIZE"
echo "warmup_batches=$INFERENCE_WARMUP_BATCHES"
echo "test_csv_override=${TEST_CSV_OVERRIDE:-default}"
echo "output=$BENCHMARK_DIR"

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  train_speechllm_with_segmenter.py "${args[@]}"
