#!/bin/bash
# Queue the paper-oriented two-point RTF protocol for all systems in the report:
#   (1) batch-1 latency;
#   (2) largest power-of-two batch below 90% allocated A100 memory.
# Batch selection uses seed 3407 and dev-other duration only. The selected batch
# is frozen across seeds 3407/3408/3409 before test-clean/test-other evaluation.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=${SEEDS:-"3407 3408 3409"}
DRY_RUN=${DRY_RUN:-0}
OUT_ROOT=${OUT_ROOT:-/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/inference_eval_memory_limited_batch_v1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-2}
PREWARM_BATCHES=${PREWARM_BATCHES:-10}
MAX_MEMORY_FRACTION=${MAX_MEMORY_FRACTION:-0.90}

BASELINE_ROOT=$(pwd)/results/speechllm_fixed_pooling_wavlm
LEARNED_ROOT=$(pwd)/results/speechllm_segmenter_wavlm
CHAR_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char
PHONE_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/phone

WAVLM_ARGS="--ssl_hub microsoft/wavlm-large"
WAVLM_ARGS+=" --ssl_folder /export/jsalt26/omnienc/users/cxiao/hf/hub"
WAVLM_ARGS+=" --ssl_feat_dims 1024 --segmenter_input_dim 1024"
CNN_ARGS="$WAVLM_ARGS --segmenter_sampling autoregressive"
TRANSFORMER_ARGS="$WAVLM_ARGS --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
TRANSFORMER_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
TRANSFORMER_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
TRANSFORMER_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
BIGRU_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
BIGRU_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

[[ ! -e "$OUT_ROOT" ]] || {
  echo "Refusing to reuse experiment root: $OUT_ROOT" >&2
  exit 1
}
[[ -d "$CHAR_DIR" && -d "$PHONE_DIR" ]] || {
  echo "Missing oracle boundary directories" >&2
  exit 1
}

# Preflight every source checkpoint before submitting the first job.
baseline_runs=(
  no_downsampling_tc100
  fixed_rate_k3_tc100
  fixed_rate_k4_tc100
  fixed_rate_k5_tc100
  fixed_rate_k6_tc100
  fixed_rate_k8_tc100
  phone_alignment_tc100
  char_alignment_tc100
)
learned_runs=(
  oracleclose_bestinit/autoregressive
  cnn_first_order_ar_bigru_best_char_decoder/nll_mt
  fullprefix_transformer_ar_local64_best_char_decoder/nll_mt
  fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt
)
for run_name in "${baseline_runs[@]}"; do
  for seed in $SEEDS; do
    [[ -d "$BASELINE_ROOT/$run_name/$seed/save" ]] || {
      echo "Missing baseline checkpoint: $BASELINE_ROOT/$run_name/$seed/save" >&2
      exit 1
    }
  done
done
for run_name in "${learned_runs[@]}"; do
  for seed in $SEEDS; do
    [[ -d "$LEARNED_ROOT/$run_name/$seed/save" ]] || {
      echo "Missing learned checkpoint: $LEARNED_ROOT/$run_name/$seed/save" >&2
      exit 1
    }
  done
done

mkdir -p "$OUT_ROOT/slurm_logs" "$OUT_ROOT/selection" "$OUT_ROOT/results"
selection_manifest="$OUT_ROOT/selection_jobs.tsv"
evaluation_manifest="$OUT_ROOT/evaluation_jobs.tsv"
printf 'job_id\tsystem\tseed\tselection_dir\n' > "$selection_manifest"
printf 'job_id\tdependency\tsystem\tseed\toutput_dir\n' > "$evaluation_manifest"

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=8
  --mem=60G
)

submit_job() {
  local job_name=$1
  local walltime=$2
  local dependency=$3
  local export_arg=$4
  local script=$5
  local dependency_arg=()
  [[ -z "$dependency" ]] || dependency_arg=(--dependency="afterok:$dependency")
  local command=(
    sbatch "${COMMON[@]}" "${dependency_arg[@]}"
    --time="$walltime"
    --job-name="$job_name"
    --output="$OUT_ROOT/slurm_logs/%x_%j.out"
    --error="$OUT_ROOT/slurm_logs/%x_%j.err"
    --export="$export_arg"
    --parsable
    "$script"
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "${command[@]}" >&2
    printf '\n' >&2
    echo "DRY_$job_name"
  else
    local submitted
    submitted=$("${command[@]}")
    echo "${submitted%%;*}"
  fi
}

submit_system() {
  local key=$1
  local short=$2
  local kind=$3
  local run_rel=$4
  local boundary_source=$5
  local fixed_k=$6
  local boundary_target_dir=$7
  local backbone=$8
  local extra_args=$9

  local root=$BASELINE_ROOT
  [[ "$kind" == learned ]] && root=$LEARNED_ROOT
  local selection_dir="$OUT_ROOT/selection/$key"
  local seed3407_run="$root/$run_rel/3407"
  local select_export="ALL,SYSTEM_KEY=$key,EVAL_KIND=$kind,RUN_DIR=$seed3407_run"
  select_export+=",SELECT_DIR=$selection_dir,SEED=3407"
  select_export+=",MAX_BATCH_SIZE=$MAX_BATCH_SIZE,CALIBRATION_BATCHES=$CALIBRATION_BATCHES"
  select_export+=",MAX_MEMORY_FRACTION=$MAX_MEMORY_FRACTION"
  if [[ "$kind" == baseline ]]; then
    select_export+=",BOUNDARY_SOURCE=$boundary_source"
    [[ -z "$fixed_k" ]] || select_export+=",FIXED_K=$fixed_k"
    [[ -z "$boundary_target_dir" ]] \
      || select_export+=",BOUNDARY_TARGET_DIR=$boundary_target_dir"
  else
    select_export+=",BACKBONE=$backbone,EXTRA_ARGS=$extra_args"
  fi

  local select_job
  select_job=$(submit_job "rtfs_${short}" 04:00:00 "" "$select_export" \
    select_inference_batch_size.slurm)
  printf '%s\t%s\t3407\t%s\n' \
    "$select_job" "$key" "$selection_dir" >> "$selection_manifest"
  echo "$key selection -> $select_job"

  for seed in $SEEDS; do
    local run_dir="$root/$run_rel/$seed"
    local output_dir="$OUT_ROOT/results/$key/seed$seed"
    local eval_export="ALL,SYSTEM_KEY=$key,EVAL_KIND=$kind,RUN_DIR=$run_dir"
    eval_export+=",SELECT_DIR=$selection_dir,OUTPUT_DIR=$output_dir,SEED=$seed"
    eval_export+=",PREWARM_BATCHES=$PREWARM_BATCHES"
    if [[ "$kind" == baseline ]]; then
      eval_export+=",BOUNDARY_SOURCE=$boundary_source"
      [[ -z "$fixed_k" ]] || eval_export+=",FIXED_K=$fixed_k"
      [[ -z "$boundary_target_dir" ]] \
        || eval_export+=",BOUNDARY_TARGET_DIR=$boundary_target_dir"
    else
      eval_export+=",BACKBONE=$backbone,EXTRA_ARGS=$extra_args"
    fi
    local eval_job
    eval_job=$(submit_job "rtfe_${short}_s${seed: -1}" 08:00:00 \
      "$select_job" "$eval_export" eval_memory_selected_inference.slurm)
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$eval_job" "$select_job" "$key" "$seed" "$output_dir" \
      >> "$evaluation_manifest"
    echo "$key seed=$seed evaluation -> $eval_job afterok:$select_job"
  done
}

submit_system no_downsampling nods baseline no_downsampling_tc100 none "" "" "" ""
submit_system fixed_k3 fk3 baseline fixed_rate_k3_tc100 fixed_rate 3 "" "" ""
submit_system fixed_k4 fk4 baseline fixed_rate_k4_tc100 fixed_rate 4 "" "" ""
submit_system fixed_k5 fk5 baseline fixed_rate_k5_tc100 fixed_rate 5 "" "" ""
submit_system fixed_k6 fk6 baseline fixed_rate_k6_tc100 fixed_rate 6 "" "" ""
submit_system fixed_k8 fk8 baseline fixed_rate_k8_tc100 fixed_rate 8 "" "" ""
submit_system oracle_phone orph baseline phone_alignment_tc100 alignment "" "$PHONE_DIR" "" ""
submit_system oracle_char orch baseline char_alignment_tc100 alignment "" "$CHAR_DIR" "" ""

submit_system cnn_ar_mean carm learned oracleclose_bestinit/autoregressive \
  "" "" "" cnn "$CNN_ARGS --segment_pooling mean"
submit_system cnn_ar_bigru carb learned \
  cnn_first_order_ar_bigru_best_char_decoder/nll_mt \
  "" "" "" cnn "$CNN_ARGS $BIGRU_ARGS"
submit_system transformer_ar_local64_mean tarm learned \
  fullprefix_transformer_ar_local64_best_char_decoder/nll_mt \
  "" "" "" transformer_ar "$TRANSFORMER_ARGS --segment_pooling mean"
submit_system transformer_ar_local64_bigru tarb learned \
  fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt \
  "" "" "" transformer_ar "$TRANSFORMER_ARGS $BIGRU_ARGS"

{
  echo "protocol=memory_limited_batch_v1"
  echo "selection_seed=3407"
  echo "seeds=$SEEDS"
  echo "candidate_batches=powers_of_two_until_failure"
  echo "max_batch_size_guard=$MAX_BATCH_SIZE"
  echo "max_memory_fraction=$MAX_MEMORY_FRACTION"
  echo "calibration_batches=$CALIBRATION_BATCHES"
  echo "prewarm_batches=$PREWARM_BATCHES"
  echo "hardware=A100_80GB"
  echo "precision=bf16"
  echo "test_splits=test-clean,test-other"
} > "$OUT_ROOT/protocol.txt"

echo "Output root: $OUT_ROOT"
echo "Selection jobs: $selection_manifest"
echo "Evaluation jobs: $evaluation_manifest"
