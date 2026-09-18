#!/bin/bash
# Retrain the publication-critical WavLM learned-segmentation systems after the
# batch-invariant decoder-position fix.  Cold-start segmenter checkpoints are
# intentionally reused: boundary-supervised cold start never calls the decoder.
# Every output lives under a new root so pre-fix results remain untouched.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
WARMUP_EPOCHS=2
JOINT_RL_EPOCHS=10
TOTAL_EPOCHS=$((WARMUP_EPOCHS + JOINT_RL_EPOCHS))
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
CNN_SEGMENTER_CKPT=$(pwd)/results/speechllm_segmenter_wavlm/coldstart/cnn/3407/save/CKPT+2026-08-07+11-19-21+00/segmenter.ckpt
TRANSFORMER_COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save
OUT_ROOT=${OUT_ROOT:-results/speechllm_segmenter_wavlm/compact_position_retrain_v1}
MANIFEST=$OUT_ROOT/submitted_jobs.tsv

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=60G
  --time=3-00:00:00
)

die() {
  echo "$*" >&2
  exit 1
}

submit_job() {
  local priority=$1
  local nice=$2
  local system=$3
  local seed=$4
  local job_name=$5
  shift 5
  local job
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "${COMMON[@]}" --nice="$nice" --job-name="$job_name" --parsable "$@" >&2
    printf '\n' >&2
    job="DRY_$job_name"
  else
    job=$(sbatch "${COMMON[@]}" --nice="$nice" --job-name="$job_name" --parsable "$@")
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$priority" "$system" "$seed" "$job" "$nice" >> "$MANIFEST"
  echo "$job"
}

for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
[[ -s "$CNN_SEGMENTER_CKPT" ]] || die "Missing CNN segmenter checkpoint: $CNN_SEGMENTER_CKPT"
compgen -G "$TRANSFORMER_COLD_DIR/CKPT*/segmenter.ckpt" >/dev/null \
  || die "No Transformer-AR segmenter checkpoint under $TRANSFORMER_COLD_DIR"
[[ ! -e "$OUT_ROOT" ]] || die "Refusing to reuse output root: $OUT_ROOT"

mkdir -p "$OUT_ROOT" slurm_logs
printf 'priority\tsystem\tseed\tjob_id\tnice\n' > "$MANIFEST"

ENCODER_ARGS="--ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
COMMON_TRAIN_ARGS="--decoder_init_ckpt_dir $DECODER_CKPT"
COMMON_TRAIN_ARGS+=" --segmenter_reward nll --freeze_decoder_in_joint False"
COMMON_TRAIN_ARGS+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002"
COMMON_TRAIN_ARGS+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
COMMON_TRAIN_ARGS+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
COMMON_TRAIN_ARGS+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0 --test_batch_size 8"

TRANSFORMER_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
TRANSFORMER_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
TRANSFORMER_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
TRANSFORMER_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
BIGRU_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
BIGRU_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

submit_transformer_arm() {
  local priority=$1
  local nice=$2
  local system=$3
  local short=$4
  local pooling_args=$5
  local seed out extra job
  for seed in "${SEEDS[@]}"; do
    out="$OUT_ROOT/$system/nll_mt/$seed"
    extra="--experiment_name compactpos_${short} $ENCODER_ARGS $TRANSFORMER_ARGS $pooling_args"
    extra+=" --seed $seed --output_folder $out $COMMON_TRAIN_ARGS"
    extra+=" --rl_update_mode combined_on_policy --stage_timing_file $out/stage_timing.jsonl"
    job=$(submit_job "$priority" "$nice" "$system" "$seed" "cp_${short}_${seed}" \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$TRANSFORMER_COLD_DIR,EXTRA_ARGS=$extra" \
      run_segmenter.slurm)
    echo "P$priority $system seed=$seed -> $job ($out)"
  done
}

submit_cnn_arm() {
  local priority=$1
  local nice=$2
  local system=$3
  local short=$4
  local pooling_args=$5
  local seed out extra job
  for seed in "${SEEDS[@]}"; do
    out="$OUT_ROOT/$system/nll_mt/$seed"
    extra="--experiment_name compactpos_${short} $ENCODER_ARGS $pooling_args"
    extra+=" --seed $seed --output_folder $out --segmenter_init_checkpoint $CNN_SEGMENTER_CKPT"
    extra+=" --segmenter_sampling autoregressive $COMMON_TRAIN_ARGS"
    extra+=" --rl_update_mode on_policy --stage_timing_file $out/stage_timing.jsonl"
    job=$(submit_job "$priority" "$nice" "$system" "$seed" "cp_${short}_${seed}" \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=cnn,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,EXTRA_ARGS=$extra" \
      run_segmenter.slurm)
    echo "P$priority $system seed=$seed -> $job ($out)"
  done
}

# Submit in scientific priority order.  Increasing nice values ensure that the
# scheduler favors the decisive system if jobs compete for otherwise equal
# resources.
submit_transformer_arm 0 0 transformer_ar_local64_bigru tarb "$BIGRU_ARGS"
submit_transformer_arm 1 100 transformer_ar_local64_mean tarm ""
submit_cnn_arm 2 200 cnn_first_order_ar_bigru carb "$BIGRU_ARGS"
submit_cnn_arm 3 300 cnn_first_order_ar_mean carm ""

echo "Submitted corrected-training matrix. Manifest: $MANIFEST"
echo "Schedule per run: $WARMUP_EPOCHS decoder/pooler warmup + $JOINT_RL_EPOCHS joint-RL epochs; K=4"
echo "All jobs: one A100, 12 CPUs, 60 GB RAM"
