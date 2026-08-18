#!/bin/bash
# Full three-seed bilevel comparison for the current WavLM Transformer-AR +
# BiGRU system.  It exactly matches the established best-char-decoder run's
# initialization and 2-warmup + 10-joint-epoch schedule.  Only the bilevel
# lookahead scope changes.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
read -r -a MODES <<< "${BILEVEL_MODES:-decoder_lookahead decoder_pooler_lookahead}"
WARMUP_EPOCHS=2
JOINT_RL_EPOCHS=10
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
INNER_LR=${INNER_LR:-0.01}
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
CONTROL_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder
COLD_DIR=$(pwd)/$CONTROL_ROOT/coldstart/3407/save
BASELINE_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt
OUT_ROOT=results/speechllm_segmenter_wavlm/bilevel_transformer_ar_local64_bigru_full

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --cpus-per-task=12
  --mem=80G
  --time=3-00:00:00
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

for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
compgen -G "$COLD_DIR/CKPT*/segmenter.ckpt" >/dev/null \
  || die "No completed Transformer-AR segmenter checkpoint under $COLD_DIR"

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_bilevel_full"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

mkdir -p slurm_logs
for mode in "${MODES[@]}"; do
  case "$mode" in
    decoder_lookahead) short=d ;;
    decoder_pooler_lookahead) short=dp ;;
    *) die "Full study only accepts decoder_lookahead or decoder_pooler_lookahead, got $mode" ;;
  esac
  for seed in "${SEEDS[@]}"; do
    [[ -d "$BASELINE_ROOT/$seed" ]] \
      || die "Missing matched non-bilevel baseline: $BASELINE_ROOT/$seed"
    out="$OUT_ROOT/$mode/lr_${INNER_LR}/$seed"
    [[ ! -e "$out" ]] || die "Refusing to reuse output: $out"

    extra="$ENCODER_ARGS $POLICY_ARGS $POOL_ARGS --seed $seed --output_folder $out"
    extra+=" --stage_timing_file $out/stage_timing.jsonl"
    extra+=" --bilevel_diagnostics_file $out/bilevel_diagnostics.jsonl"
    extra+=" --decoder_init_ckpt_dir $DECODER_CKPT"
    extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
    extra+=" --rl_update_mode combined_on_policy --bilevel_mode $mode"
    extra+=" --bilevel_support_fraction 0.5 --bilevel_inner_lr $INNER_LR"
    extra+=" --bilevel_inner_max_grad_norm 1.0 --bilevel_support_pg_weight 1.0"
    extra+=" --bilevel_measure_support_after False --bilevel_deterministic_sdpa True"
    extra+=" --initial_lr 0.0002 --lr_decoder 0.0002"
    extra+=" --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
    extra+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
    extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
    extra+=" --test_batch_size 8"

    job=$(submit_job "bl_${short}_full_${seed}" \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$extra" \
      run_segmenter.slurm)
    echo "bilevel full mode=$mode seed=$seed -> $job ($out)"
  done
done

echo "Configuration: WavLM, local-history Transformer-AR window=64, BiGRU residual pooling, K=4"
echo "Schedule: $WARMUP_EPOCHS matched warmup + $JOINT_RL_EPOCHS joint-RL epochs"
echo "Matched non-bilevel baseline: $BASELINE_ROOT/{3407,3408,3409}"
echo "Output root: $OUT_ROOT (estimated under 5 GB total)"
