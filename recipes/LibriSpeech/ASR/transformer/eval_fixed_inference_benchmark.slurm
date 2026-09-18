#!/bin/bash
# Evaluation-only inference benchmark for fixed/no-downsampling/oracle baselines.
# The source run remains read-only; benchmark artifacts go to BENCHMARK_DIR.
#SBATCH --job-name=fixed_infer_bench
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${RUN_DIR:?set RUN_DIR to an existing baseline seed directory}"
: "${BENCHMARK_DIR:?set BENCHMARK_DIR to a new benchmark output directory}"
: "${SEED:?set SEED to the source run seed}"
: "${BOUNDARY_SOURCE:?set BOUNDARY_SOURCE=none|fixed_rate|alignment}"
: "${BATCH_SIZE:?set BATCH_SIZE to the evaluation batch size}"

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
  hparams/speechllm_fixed_pooling.yaml
  --seed "$SEED"
  --boundary_source "$BOUNDARY_SOURCE"
  --train_splits "['train-clean-100']"
  --number_of_epochs 10
  --ssl_hub microsoft/wavlm-large
  --ssl_folder /export/jsalt26/omnienc/users/cxiao/hf/hub
  --ssl_feat_dims 1024
  --output_folder "$BENCHMARK_DIR"
  --save_folder "$RUN_DIR/save"
  --output_wer_folder "$BENCHMARK_DIR/wer_results"
  --train_log "$BENCHMARK_DIR/eval.log"
  --stage_timing_file "$BENCHMARK_DIR/stage_timing.jsonl"
  --eval_only True
  --inference_benchmark_file "$BENCHMARK_DIR/benchmark.jsonl"
  --inference_warmup_batches "$INFERENCE_WARMUP_BATCHES"
  --test_batch_size "$BATCH_SIZE"
  --max_decode_ratio 3.0
  --skip_prep True
)

if [[ -n "$TEST_CSV_OVERRIDE" ]]; then
  args+=(--test_csv "$TEST_CSV_OVERRIDE")
elif [[ "${TEST_ONLY:-0}" == 1 ]]; then
  args+=(
    --test_csv
    "[$BENCHMARK_DIR/test-clean.csv,$BENCHMARK_DIR/test-other.csv]"
  )
fi

if [[ "$BOUNDARY_SOURCE" == fixed_rate ]]; then
  : "${FIXED_K:?set FIXED_K for fixed-rate pooling}"
  args+=(--fixed_rate_k "$FIXED_K")
elif [[ "$BOUNDARY_SOURCE" == alignment ]]; then
  : "${BOUNDARY_TARGET_DIR:?set BOUNDARY_TARGET_DIR for oracle pooling}"
  [[ -d "$BOUNDARY_TARGET_DIR" ]] || {
    echo "Missing alignment directory: $BOUNDARY_TARGET_DIR" >&2
    exit 1
  }
  args+=(--boundary_target_dir "$BOUNDARY_TARGET_DIR")
fi

echo "=== $(date) baseline inference benchmark on $(hostname) ==="
echo "source=$RUN_DIR boundary_source=$BOUNDARY_SOURCE batch=$BATCH_SIZE"
echo "test_only=${TEST_ONLY:-0} warmup_batches=$INFERENCE_WARMUP_BATCHES"
echo "test_csv_override=${TEST_CSV_OVERRIDE:-default}"
echo "output=$BENCHMARK_DIR"

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  train_speechllm.py "${args[@]}"
