#!/bin/bash
# Corrected evaluation for the matched WavLM learned-system series.
#
# All three seeds run at batch 8 for aggregate WER.  Seed 3407 additionally
# runs at batch 1, providing a direct batch-invariance and latency check without
# tripling the expensive batch-1 benchmark.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=${SEEDS:-"3407 3408 3409"}
DRY_RUN=${DRY_RUN:-0}
START_DEPENDENCY=${START_DEPENDENCY:-}
OUT_ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/inference_eval_batch_invariant_leftpack_durationcap_v1/learned_wavlm
JOB_MANIFEST="$OUT_ROOT/submitted_jobs.tsv"

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

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=8
  --mem=60G
  --time=02:00:00
)

submit_job() {
  local job_name=$1
  shift
  local dependency=()
  if [[ -n "$START_DEPENDENCY" ]]; then
    dependency=(--dependency="afterok:$START_DEPENDENCY")
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
  local backbone=$3
  local run_rel=$4
  local extra_args=$5

  for seed in $SEEDS; do
    local run_dir="$(pwd)/results/speechllm_segmenter_wavlm/$run_rel/$seed"
    [[ -d "$run_dir/save" ]] || {
      echo "Missing completed source run: $run_dir" >&2
      exit 1
    }
    local batch_sizes="8"
    [[ "$seed" != 3407 ]] || batch_sizes="1 8"
    for batch_size in $batch_sizes; do
      local benchmark_dir="$OUT_ROOT/$key/seed$seed/batch$batch_size"
      [[ ! -e "$benchmark_dir" ]] || {
        echo "Refusing to reuse corrected output: $benchmark_dir" >&2
        exit 1
      }
      local job
      job=$(submit_job "cie_${short}_s${seed: -1}_b$batch_size" \
        --export="ALL,RUN_DIR=$run_dir,BENCHMARK_DIR=$benchmark_dir,SEED=$seed,BACKBONE=$backbone,BATCH_SIZE=$batch_size,EXTRA_ARGS=$extra_args" \
        eval_segmenter_inference_benchmark.slurm)
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$job" "$key" "$seed" "$batch_size" "$benchmark_dir" >> "$JOB_MANIFEST"
      echo "$key seed=$seed batch=$batch_size -> $job"
      SUBMITTED_IDS+=("$job")
    done
  done
}

mkdir -p "$OUT_ROOT" slurm_logs
printf 'job_id\tsystem\tseed\tbatch_size\toutput_dir\n' > "$JOB_MANIFEST"
declare -a SUBMITTED_IDS=()

submit_system cnn_ar_mean carm cnn \
  oracleclose_bestinit/autoregressive \
  "$CNN_ARGS --segment_pooling mean"
submit_system cnn_ar_bigru carb cnn \
  cnn_first_order_ar_bigru_best_char_decoder/nll_mt \
  "$CNN_ARGS $BIGRU_ARGS"
submit_system transformer_ar_local64_mean tarm transformer_ar \
  fullprefix_transformer_ar_local64_best_char_decoder/nll_mt \
  "$TRANSFORMER_ARGS --segment_pooling mean"
submit_system transformer_ar_local64_bigru tarb transformer_ar \
  fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt \
  "$TRANSFORMER_ARGS $BIGRU_ARGS"

dependency=$(IFS=:; echo "${SUBMITTED_IDS[*]}")
echo "Corrected learned output root: $OUT_ROOT"
echo "LEARNED_AFTEROK=$dependency"
