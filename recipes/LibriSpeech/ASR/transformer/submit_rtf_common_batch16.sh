#!/bin/bash
# Run the headline throughput comparison at a common batch size of 16: the
# largest stable batch already demonstrated by the no-downsampling reference.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=${SEEDS:-"3407 3408 3409"}
DRY_RUN=${DRY_RUN:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
BASE_ROOT=${BASE_ROOT:-/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/inference_eval_memory_limited_batch_v1}
OUT_ROOT=${OUT_ROOT:-$BASE_ROOT/common_batch16_v1}
PREWARM_BATCHES=${PREWARM_BATCHES:-10}

[[ "$BATCH_SIZE" == 16 ]] || {
  echo "The common-batch launcher requires BATCH_SIZE=16" >&2
  exit 1
}
[[ ! -e "$OUT_ROOT" ]] || {
  echo "Refusing to reuse common-batch root: $OUT_ROOT" >&2
  exit 1
}

BASELINE_ROOT=$(pwd)/results/speechllm_fixed_pooling_wavlm
LEARNED_ROOT=$(pwd)/results/speechllm_segmenter_wavlm
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

mkdir -p "$OUT_ROOT/slurm_logs" "$OUT_ROOT/results"
manifest="$OUT_ROOT/evaluation_jobs.tsv"
printf 'job_id\tsystem\tseed\tbatch_size\toutput_dir\n' > "$manifest"

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=8
  --mem=60G
  --time=08:00:00
)

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

  local source_root=$BASELINE_ROOT
  [[ "$kind" == learned ]] && source_root=$LEARNED_ROOT

  for seed in $SEEDS; do
    local run_dir="$source_root/$run_rel/$seed"
    local batch1_dir="$BASE_ROOT/results/$key/seed$seed/batch1"
    local output_dir="$OUT_ROOT/results/$key/seed$seed"
    [[ -d "$run_dir/save" ]] || {
      echo "Missing source checkpoint: $run_dir/save" >&2
      exit 1
    }
    for split in test-clean test-other; do
      [[ -s "$batch1_dir/benchmark_${split}_utterances.jsonl" ]] || {
        echo "Missing batch-1 source for $key seed=$seed split=$split" >&2
        exit 1
      }
    done

    local export_arg="ALL,SYSTEM_KEY=$key,EVAL_KIND=$kind,RUN_DIR=$run_dir"
    export_arg+=",BATCH1_DIR=$batch1_dir,OUTPUT_DIR=$output_dir,SEED=$seed"
    export_arg+=",BATCH_SIZE=$BATCH_SIZE,PREWARM_BATCHES=$PREWARM_BATCHES"
    export_arg+=",PROTOCOL_TAG=common_batch16_rtf_v1"
    export_arg+=",SELECTION_REASON=largest_stable_batch_for_no_downsampling"
    if [[ "$kind" == baseline ]]; then
      export_arg+=",BOUNDARY_SOURCE=$boundary_source"
      [[ -z "$fixed_k" ]] || export_arg+=",FIXED_K=$fixed_k"
      [[ -z "$boundary_target_dir" ]] \
        || export_arg+=",BOUNDARY_TARGET_DIR=$boundary_target_dir"
    else
      export_arg+=",BACKBONE=$backbone,EXTRA_ARGS=$extra_args"
    fi

    local command=(
      sbatch "${COMMON[@]}"
      --job-name="rtf16_${short}_s${seed: -1}"
      --output="$OUT_ROOT/slurm_logs/%x_%j.out"
      --error="$OUT_ROOT/slurm_logs/%x_%j.err"
      --export="$export_arg"
      --parsable
      eval_rtf_fixed_batch_retry.slurm
    )
    local submitted
    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'DRY-RUN:' >&2
      printf ' %q' "${command[@]}" >&2
      printf '\n' >&2
      submitted="DRY_${key}_${seed}"
    else
      submitted=$("${command[@]}")
      submitted=${submitted%%;*}
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$submitted" "$key" "$seed" "$BATCH_SIZE" "$output_dir" >> "$manifest"
    echo "$key seed=$seed batch=$BATCH_SIZE -> $submitted"
  done
}

submit_system fixed_k5 fk5 baseline fixed_rate_k5_tc100 fixed_rate 5 "" "" ""
submit_system oracle_phone orph baseline phone_alignment_tc100 alignment "" "$PHONE_DIR" "" ""
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
  echo "protocol=common_batch16_rtf_v1"
  echo "reference=no_downsampling_existing_batch16"
  echo "systems=fixed_k5,oracle_phone,cnn_ar_mean,cnn_ar_bigru,transformer_ar_local64_mean,transformer_ar_local64_bigru"
  echo "seeds=$SEEDS"
  echo "batch_size=$BATCH_SIZE"
  echo "prewarm_batches=$PREWARM_BATCHES"
  echo "allocator=expandable_segments:True"
  echo "hardware=A100_80GB"
  echo "precision=bf16"
} > "$OUT_ROOT/protocol.txt"

echo "Output root: $OUT_ROOT"
echo "Manifest: $manifest"
