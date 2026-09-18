#!/bin/bash
# Submit the rate-matched, transcript-free BPE-CTC 100h baseline:
#   1) eight frozen BPE-CTC inference shards generate state-run boundaries;
#   2) three dependent 5-epoch decoder+BiGRU runs consume those boundaries.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
EPOCHS=${EPOCHS:-5}
DRY_RUN=${DRY_RUN:-0}

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
BPE_CTC_MODEL=$ROOT/artifacts/models/icefall_conformer_ctc_bpe500/cpu_jit.pt
BPE_CTC_TOKENS=$ROOT/artifacts/models/icefall_conformer_ctc_bpe500/tokens.txt
BOUNDARY_DIR=/export/jsalt26/omnienc/users/cxiao/boundary_targets/ctc_bpe500_state_runs
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save
OUT_ROOT=${OUT_ROOT:-results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/ctc_bpe500_state_runs_bigru_5ep}

IDENTITY=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
)

die() {
  echo "$*" >&2
  exit 1
}

submit_job() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "$@" >&2
    printf '\n' >&2
    echo "DRY_JOB"
  else
    sbatch --parsable "$@"
  fi
}

[[ -s "$BPE_CTC_MODEL" ]] || die "Missing BPE-CTC model: $BPE_CTC_MODEL"
[[ "$(head -n 1 "$BPE_CTC_TOKENS")" == "<blk> 0" ]] \
  || die "BPE-CTC token map does not declare blank ID 0"
for split in train-clean-100 dev-clean dev-other test-clean test-other; do
  [[ -d "/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech/$split" ]] \
    || die "Missing LibriSpeech split: $split"
done
for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
compgen -G "$COLD_DIR/CKPT*/segmenter.ckpt" >/dev/null \
  || die "No Transformer-AR cold-start checkpoint under $COLD_DIR"
for seed in "${SEEDS[@]}"; do
  [[ ! -e "$OUT_ROOT/$seed" ]] \
    || die "Refusing to reuse output: $OUT_ROOT/$seed"
done
mkdir -p slurm_logs

echo "Dataset: train-clean-100 (100h baseline; not LS960)"
echo "Splits: train-clean-100; evaluation on dev-clean/dev-other/test-clean/test-other"
echo "Preprocessing: 8 x A100 frozen BPE-CTC inference shards"
echo "Boundary output: $BOUNDARY_DIR; expected under 1 GB"
echo "Training: ${#SEEDS[@]} x A100; fixed BPE-CTC boundaries + residual BiGRU pooling"
echo "Training output: $OUT_ROOT; estimated 5-10 GB total"
echo "CTC compressor training: none (released LS960 BPE-500 CTC checkpoint)"
echo "Downstream adaptation: $EPOCHS epochs per seed"

gen_job=$(submit_job "${IDENTITY[@]}" generate_bpe_ctc_posterior_100h.slurm)
echo "BPE-CTC boundary generation -> $gen_job"

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_attribution_100h"
ENCODER_ARGS+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --ssl_feat_dims $WAVLM_DIM --segmenter_input_dim $WAVLM_DIM"
ENCODER_ARGS+=" --train_splits ['train-clean-100']"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
TRAIN_ARGS="--decoder_init_ckpt_dir $DECODER_CKPT"
TRAIN_ARGS+=" --boundary_source alignment --boundary_target_dir $BOUNDARY_DIR"
TRAIN_ARGS+=" --segmenter_reward nll --freeze_decoder_in_joint False"
TRAIN_ARGS+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
TRAIN_ARGS+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
TRAIN_ARGS+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
TRAIN_ARGS+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
TRAIN_ARGS+=" --test_batch_size 8 --freeze_boundary_policy True"

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  extra="$ENCODER_ARGS $POLICY_ARGS $POOL_ARGS $TRAIN_ARGS"
  extra+=" --seed $seed --output_folder $out"
  dependency=()
  if [[ "$DRY_RUN" != 1 ]]; then
    dependency=(--dependency="afterok:$gen_job")
  fi
  job=$(submit_job \
    "${IDENTITY[@]}" \
    "${dependency[@]}" \
    --cpus-per-task=12 --mem=60G --time=2-00:00:00 \
    --job-name="tc100_bpebg_${seed}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$EPOCHS,WARMUP_EPOCHS=0,COLDSTART_DIR=$COLD_DIR,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "BPE-CTC+BiGRU seed=$seed -> $job (afterok:$gen_job; $out)"
done

echo "Evaluation uses precomputed boundaries; a later RTF audit must include the live CTC encoder."
