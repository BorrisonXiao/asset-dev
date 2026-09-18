#!/bin/bash
# Short real-training runs used to choose a safe dynamic-batch duration budget.
# The decoder+BiGRU lookahead arm is the larger of the two bilevel variants.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a BUDGETS <<< "${BUDGET_LIST:-100 200 300 400}"
SEED=${SEED:-3407}
DEBUG_BATCHES=${DEBUG_BATCHES:-24}
BENCH_TAG=${BENCH_TAG:-v1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
INIT_CKPT=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/3407/save/CKPT+2026-08-15+03-13-00+00
OUT_ROOT=results/speechllm_segmenter_wavlm/bilevel_memory_bench/$BENCH_TAG

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --cpus-per-task=8
  --mem=32G
  --time=02:00:00
)

die() {
  echo "$*" >&2
  exit 1
}

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

for name in segmenter.ckpt llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$INIT_CKPT/$name" ]] || die "Missing initialization file: $INIT_CKPT/$name"
done

ENCODER_ARGS="--experiment_name speechllm_segmenter_bilevel_memory_bench"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

mkdir -p slurm_logs
for budget in "${BUDGETS[@]}"; do
  [[ "$budget" =~ ^[0-9]+$ ]] || die "Invalid duration budget: $budget"
  out="$OUT_ROOT/budget_$budget/$SEED"
  [[ ! -e "$out" ]] || die "Refusing to reuse benchmark output: $out"

  extra="$ENCODER_ARGS $POLICY_ARGS $POOL_ARGS --seed $SEED --output_folder $out"
  extra+=" --stage_timing_file $out/stage_timing.jsonl"
  extra+=" --bilevel_diagnostics_file $out/bilevel_diagnostics.jsonl"
  extra+=" --segmenter_init_checkpoint $INIT_CKPT/segmenter.ckpt"
  extra+=" --decoder_init_ckpt_dir $INIT_CKPT"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
  extra+=" --rl_update_mode combined_on_policy"
  extra+=" --bilevel_mode decoder_pooler_lookahead"
  extra+=" --bilevel_support_fraction 0.5 --bilevel_inner_lr 0.01"
  extra+=" --bilevel_inner_max_grad_norm 1.0 --bilevel_support_pg_weight 1.0"
  extra+=" --bilevel_measure_support_after False --bilevel_deterministic_sdpa True"
  extra+=" --min_batch_ex_train 2 --max_batch_length_train $budget"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_segmenter 0.00005"
  extra+=" --pg_weight 1.0 --grpo_k 4 --rate_mode band"
  extra+=" --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --test_batch_size 8"
  extra+=" --debug --debug_persistently --debug_batches $DEBUG_BATCHES --debug_epochs 1"

  job=$(submit_job "bl_mem_${budget}_${SEED}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=1,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "bilevel memory benchmark budget=$budget seed=$SEED -> $job ($out)"
done

if [[ "$WARMUP_EPOCHS" -gt 0 ]]; then
  echo "Stage: decoder warmup, $DEBUG_BATCHES real train batches"
else
  echo "Stage: decoder+BiGRU lookahead joint RL, K=4, $DEBUG_BATCHES real train batches"
fi
echo "Output root: $OUT_ROOT (about 0.3 GB per completed budget)"
