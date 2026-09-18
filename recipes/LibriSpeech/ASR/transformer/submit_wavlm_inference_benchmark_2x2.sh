#!/bin/bash
# Benchmark the matched {CNN first-order AR, local-64 Transformer AR} x
# {mean, BiGRU} systems at batch size 1 and a throughput-oriented batch size 8.
# One representative seed is sufficient for systems-cost measurement; WER and
# token-rate behavior are still reported over every utterance in all test splits.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEED=${SEED:-3407}
CNN_BIGRU_DEPENDENCY=${CNN_BIGRU_DEPENDENCY:-97049}
DRY_RUN=${DRY_RUN:-0}
OUT_ROOT=$(pwd)/../../../../artifacts/segmenter/inference_benchmark_2x2

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
  local backbone=$3
  local run_dir=$4
  local extra_args=$5
  local dependency=${6:-}

  for batch_size in 1 8; do
    benchmark_dir="$OUT_ROOT/$key/seed$SEED/batch$batch_size"
    [[ ! -e "$benchmark_dir" ]] || {
      echo "Refusing to reuse benchmark output: $benchmark_dir" >&2
      exit 1
    }
    submission=(
      --export="ALL,RUN_DIR=$run_dir,BENCHMARK_DIR=$benchmark_dir,SEED=$SEED,BACKBONE=$backbone,BATCH_SIZE=$batch_size,EXTRA_ARGS=$extra_args"
      eval_segmenter_inference_benchmark.slurm
    )
    if [[ -n "$dependency" ]]; then
      submission=(--dependency="afterok:$dependency" "${submission[@]}")
    fi
    job=$(submit_job "ib_${short}_b$batch_size" "${submission[@]}")
    echo "$key batch=$batch_size -> $job${dependency:+ (afterok:$dependency)}"
  done
}

CNN_MEAN=$(pwd)/results/speechllm_segmenter_wavlm/oracleclose_bestinit/autoregressive/$SEED
CNN_BIGRU=$(pwd)/results/speechllm_segmenter_wavlm/cnn_first_order_ar_bigru_best_char_decoder/nll_mt/$SEED
TRANSFORMER_MEAN=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/nll_mt/$SEED
TRANSFORMER_BIGRU=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/$SEED

for ready in "$CNN_MEAN" "$TRANSFORMER_MEAN" "$TRANSFORMER_BIGRU"; do
  [[ -d "$ready/save" ]] || {
    echo "Missing completed source run: $ready" >&2
    exit 1
  }
done

mkdir -p slurm_logs
submit_system cnn_ar_mean carm cnn "$CNN_MEAN" "$CNN_ARGS --segment_pooling mean"
submit_system cnn_ar_bigru carb cnn "$CNN_BIGRU" "$CNN_ARGS $BIGRU_ARGS" "$CNN_BIGRU_DEPENDENCY"
submit_system transformer_ar_mean tarm transformer_ar "$TRANSFORMER_MEAN" "$TRANSFORMER_ARGS --segment_pooling mean"
submit_system transformer_ar_bigru tarb transformer_ar "$TRANSFORMER_BIGRU" "$TRANSFORMER_ARGS $BIGRU_ARGS"

echo "Benchmark output root: $OUT_ROOT"
echo "Protocol: best dev checkpoint, BF16, batch sizes 1 and 8, five warm-up batches"
