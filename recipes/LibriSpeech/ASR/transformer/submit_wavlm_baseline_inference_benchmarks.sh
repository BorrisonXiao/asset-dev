#!/bin/bash
# Queue matched efficiency evaluations for the complete WavLM baseline family.
# Every system gets batch-1 latency.  The second setting is the largest
# conservative throughput batch for its audio-token rate; this avoids knowingly
# OOMing high-rate systems while retaining batch-1 as the controlled comparison.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEED=${SEED:-3407}
DRY_RUN=${DRY_RUN:-0}
RESULT_ROOT=$(pwd)/results/speechllm_fixed_pooling_wavlm
OUT_ROOT=$(pwd)/../../../../artifacts/segmenter/inference_benchmark_baselines
CHAR_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char
PHONE_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/phone

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=8
  --mem=60G
  --time=1-00:00:00
)

submit_job() {
  local job_name=$1
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "${COMMON[@]}" --job-name="$job_name" --parsable "$@" >&2
    printf '\n' >&2
    echo "DRY_$job_name"
  else
    sbatch "${COMMON[@]}" --job-name="$job_name" --parsable "$@"
  fi
}

submit_system() {
  local key=$1
  local short=$2
  local run_name=$3
  local source=$4
  local throughput_batch=$5
  local fixed_k=${6:-}
  local target_dir=${7:-}
  local run_dir="$RESULT_ROOT/$run_name/$SEED"

  [[ -d "$run_dir/save" ]] || {
    echo "Missing completed source run: $run_dir" >&2
    exit 1
  }
  for batch_size in 1 "$throughput_batch"; do
    benchmark_dir="$OUT_ROOT/$key/seed$SEED/batch$batch_size"
    [[ ! -e "$benchmark_dir" ]] || {
      echo "Refusing to reuse benchmark output: $benchmark_dir" >&2
      exit 1
    }
    export_arg="ALL,RUN_DIR=$run_dir,BENCHMARK_DIR=$benchmark_dir,SEED=$SEED"
    export_arg+=",BOUNDARY_SOURCE=$source,BATCH_SIZE=$batch_size"
    [[ -z "$fixed_k" ]] || export_arg+=",FIXED_K=$fixed_k"
    [[ -z "$target_dir" ]] || export_arg+=",BOUNDARY_TARGET_DIR=$target_dir"
    job=$(submit_job "fb_${short}_b$batch_size" \
      --export="$export_arg" eval_fixed_inference_benchmark.slurm)
    echo "$key batch=$batch_size -> $job"
  done
}

mkdir -p slurm_logs

# Native 50-Hz prefixes need a smaller throughput batch; its historical safe
# decode setting was batch 4 with a shorter generation cap, so batch 2 leaves
# room for the now-matched 3x cap.
submit_system no_downsampling nods no_downsampling_tc100 none 2

# Full fixed-rate frontier.  High-rate k=3/4 use conservative throughput batches.
submit_system fixed_k3 fk3 fixed_rate_k3_tc100 fixed_rate 4 3
submit_system fixed_k4 fk4 fixed_rate_k4_tc100 fixed_rate 6 4
submit_system fixed_k5 fk5 fixed_rate_k5_tc100 fixed_rate 8 5
submit_system fixed_k6 fk6 fixed_rate_k6_tc100 fixed_rate 8 6
submit_system fixed_k8 fk8 fixed_rate_k8_tc100 fixed_rate 8 8

# Oracle labels are precomputed; these timings measure pooling + ASR conditioned
# on those labels and are not deployable segmentation costs.
submit_system oracle_phone orph phone_alignment_tc100 alignment 8 "" "$PHONE_DIR"
submit_system oracle_char orch char_alignment_tc100 alignment 4 "" "$CHAR_DIR"

echo "Benchmark output root: $OUT_ROOT"
echo "Protocol: best dev checkpoint, BF16, batch 1 plus safe throughput batch, five warm-up batches"
