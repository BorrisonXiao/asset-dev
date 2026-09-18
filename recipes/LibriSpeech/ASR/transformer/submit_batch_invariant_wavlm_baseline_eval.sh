#!/bin/bash
# Corrected evaluation for every trained WavLM baseline family.
#
# Set PHASE_DEPENDENCY to the colon-separated learned job IDs to enforce the
# requested learned-first, baselines-second schedule.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=${SEEDS:-"3407 3408 3409"}
DRY_RUN=${DRY_RUN:-0}
PHASE_DEPENDENCY=${PHASE_DEPENDENCY:-}
TEST_ONLY=${TEST_ONLY:-0}
INCLUDE_BATCH1=${INCLUDE_BATCH1:-1}
INCLUDE_ORACLES=${INCLUDE_ORACLES:-1}
NODELIST=${NODELIST:-}
RESULT_ROOT=$(pwd)/results/speechllm_fixed_pooling_wavlm
OUT_ROOT=${OUT_ROOT:-/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/inference_eval_batch_invariant_leftpack_durationcap_v1/baselines_wavlm}
JOB_MANIFEST="$OUT_ROOT/submitted_jobs.tsv"
CHAR_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char
PHONE_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/phone

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=8
  --mem=60G
  --time=04:00:00
)
if [[ -n "$NODELIST" ]]; then
  COMMON+=(--nodelist="$NODELIST")
fi

submit_job() {
  local job_name=$1
  shift
  local dependency=()
  if [[ -n "$PHASE_DEPENDENCY" ]]; then
    dependency=(--dependency="afterok:$PHASE_DEPENDENCY")
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "${COMMON[@]}" "${dependency[@]}" --job-name="$job_name" --parsable "$@" >&2
    printf '\n' >&2
    echo "DRY_$job_name"
  else
    sbatch "${COMMON[@]}" "${dependency[@]}" --job-name="$job_name" --parsable "$@"
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

  for seed in $SEEDS; do
    local run_dir="$RESULT_ROOT/$run_name/$seed"
    [[ -d "$run_dir/save" ]] || {
      echo "Missing completed source run: $run_dir" >&2
      exit 1
    }
    local batch_sizes="$throughput_batch"
    [[ "$seed" != 3407 || "$INCLUDE_BATCH1" != 1 ]] \
      || batch_sizes="1 $throughput_batch"
    for batch_size in $batch_sizes; do
      local benchmark_dir="$OUT_ROOT/$key/seed$seed/batch$batch_size"
      [[ ! -e "$benchmark_dir" ]] || {
        echo "Refusing to reuse corrected output: $benchmark_dir" >&2
        exit 1
      }
      local export_arg="ALL,RUN_DIR=$run_dir,BENCHMARK_DIR=$benchmark_dir,SEED=$seed"
      export_arg+=",BOUNDARY_SOURCE=$source,BATCH_SIZE=$batch_size,TEST_ONLY=$TEST_ONLY"
      [[ -z "$fixed_k" ]] || export_arg+=",FIXED_K=$fixed_k"
      [[ -z "$target_dir" ]] || export_arg+=",BOUNDARY_TARGET_DIR=$target_dir"
      local job
      job=$(submit_job "cie_${short}_s${seed: -1}_b$batch_size" \
        --export="$export_arg" eval_fixed_inference_benchmark.slurm)
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$job" "$key" "$seed" "$batch_size" "$benchmark_dir" >> "$JOB_MANIFEST"
      echo "$key seed=$seed batch=$batch_size -> $job"
    done
  done
}

mkdir -p "$OUT_ROOT" slurm_logs
printf 'job_id\tsystem\tseed\tbatch_size\toutput_dir\n' > "$JOB_MANIFEST"

submit_system no_downsampling nods no_downsampling_tc100 none 2
submit_system fixed_k3 fk3 fixed_rate_k3_tc100 fixed_rate 4 3
submit_system fixed_k4 fk4 fixed_rate_k4_tc100 fixed_rate 6 4
submit_system fixed_k5 fk5 fixed_rate_k5_tc100 fixed_rate 8 5
submit_system fixed_k6 fk6 fixed_rate_k6_tc100 fixed_rate 8 6
submit_system fixed_k8 fk8 fixed_rate_k8_tc100 fixed_rate 8 8
if [[ "$INCLUDE_ORACLES" == 1 ]]; then
  submit_system oracle_phone orph phone_alignment_tc100 alignment 8 "" "$PHONE_DIR"
  submit_system oracle_char orch char_alignment_tc100 alignment 4 "" "$CHAR_DIR"
fi

echo "Corrected baseline output root: $OUT_ROOT"
echo "Phase dependency: ${PHASE_DEPENDENCY:-none}"
echo "Test only: $TEST_ONLY; include batch-1: $INCLUDE_BATCH1; include oracles: $INCLUDE_ORACLES"
echo "Requested node: ${NODELIST:-scheduler choice}"
