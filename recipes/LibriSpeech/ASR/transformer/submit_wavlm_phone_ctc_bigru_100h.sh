#!/bin/bash
# Submit the short self-trained WavLM phone-CTC -> BiGRU+decoder baseline.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

DRY_RUN=${DRY_RUN:-0}
ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
ALIGN=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech_alignments
BOUNDARY=/export/jsalt26/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_tc100_best
CTC_OUT=$ROOT/artifacts/models/wavlm_phone_ctc_tc100_seed3407
OUT_ROOT=results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/wavlm_phone_ctc_best_bigru_4ep
SEEDS=(3407 3408 3409)
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
COLD_DIR=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save

IDENTITY=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
)

die() { echo "$*" >&2; exit 1; }
submit_job() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "$@" >&2
    printf '\n' >&2
    echo DRY_JOB
  else
    sbatch --parsable "$@"
  fi
}

for split in train-clean-100 dev-clean dev-other test-clean test-other; do
  [[ -d "$DATA/$split" ]] || die "Missing LibriSpeech split: $split"
  [[ -d "$ALIGN/$split" ]] || die "Missing alignment split: $split"
done
for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done
compgen -G "$COLD_DIR/CKPT*/segmenter.ckpt" >/dev/null \
  || die "No Transformer-AR cold-start checkpoint under $COLD_DIR"
[[ ! -e "$CTC_OUT" ]] || die "Refusing to reuse CTC output: $CTC_OUT"
[[ ! -e "$BOUNDARY" ]] || die "Refusing to reuse boundary output: $BOUNDARY"
for seed in "${SEEDS[@]}"; do
  [[ ! -e "$OUT_ROOT/$seed" ]] \
    || die "Refusing to reuse downstream output: $OUT_ROOT/$seed"
done
mkdir -p slurm_logs

echo "Dataset: train-clean-100 (100h baseline; not LS960)"
echo "GPU policy: A100 only; at most two simultaneous generation GPUs"
echo "CTC: frozen WavLM-Large + phone head, 3 epochs, 3 checkpoints"
echo "CTC output: $CTC_OUT; estimated under 100 MB"
echo "Boundaries: $BOUNDARY; estimated under 1 GB"
echo "Downstream: three 4-epoch BiGRU+decoder seeds, best 3 checkpoints each"
echo "Downstream outputs: $OUT_ROOT/{3407,3408,3409}; estimated under 3 GB total"
echo "Inference: all 3 CTC checkpoints on dev-clean; all 3 downstream checkpoints per seed on test/dev"

ctc_job=$(submit_job "${IDENTITY[@]}" train_wavlm_phone_ctc_100h.slurm)
echo "CTC train -> $ctc_job"

dependency=()
if [[ "$DRY_RUN" != 1 ]]; then dependency=(--dependency="afterok:$ctc_job"); fi
audit_job=$(submit_job "${IDENTITY[@]}" "${dependency[@]}" audit_wavlm_phone_ctc_checkpoints.slurm)
echo "CTC checkpoint audit -> $audit_job (afterok:$ctc_job)"
gen_job=$(submit_job "${IDENTITY[@]}" "${dependency[@]}" generate_wavlm_phone_ctc_best_100h.slurm)
echo "Best CTC boundary generation -> $gen_job (afterok:$ctc_job)"

WAVLM_ARGS="--experiment_name speechllm_segmenter_wavlm_attribution_100h"
WAVLM_ARGS+=" --ssl_hub microsoft/wavlm-large"
WAVLM_ARGS+=" --ssl_folder /export/jsalt26/omnienc/users/cxiao/hf/hub"
WAVLM_ARGS+=" --ssl_feat_dims 1024 --segmenter_input_dim 1024"
WAVLM_ARGS+=" --train_splits ['train-clean-100']"
POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
TRAIN_ARGS="--decoder_init_ckpt_dir $DECODER_CKPT"
TRAIN_ARGS+=" --coldstart_ckpt_dir $COLD_DIR"
TRAIN_ARGS+=" --boundary_source alignment --boundary_target_dir $BOUNDARY"
TRAIN_ARGS+=" --segmenter_reward nll --freeze_decoder_in_joint False"
TRAIN_ARGS+=" --rl_update_mode combined_on_policy --initial_lr 0.0002"
TRAIN_ARGS+=" --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
TRAIN_ARGS+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
TRAIN_ARGS+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
TRAIN_ARGS+=" --test_batch_size 8 --freeze_boundary_policy True"
EXTRA_ARGS="$WAVLM_ARGS $POLICY_ARGS $POOL_ARGS $TRAIN_ARGS"

train_dependency=()
if [[ "$DRY_RUN" != 1 ]]; then train_dependency=(--dependency="afterok:$gen_job"); fi
for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  seed_args="$EXTRA_ARGS --seed $seed"
  downstream_args="$seed_args --output_folder $out --checkpoints_to_keep 3"
  downstream_args+=" --test_splits [] --test_csv []"
  downstream_job=$(submit_job \
    "${IDENTITY[@]}" \
    "${train_dependency[@]}" \
    --cpus-per-task=12 --mem=80G --time=2-00:00:00 \
    --job-name="tc100_phbg_$seed" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=4,WARMUP_EPOCHS=0,EXTRA_ARGS=$downstream_args" \
    run_segmenter.slurm)
  echo "Seed $seed 4-epoch downstream train -> $downstream_job (afterok:$gen_job)"

  eval_dependency=()
  if [[ "$DRY_RUN" != 1 ]]; then eval_dependency=(--dependency="afterok:$downstream_job"); fi
  eval_job=$(submit_job \
    "${IDENTITY[@]}" \
    "${eval_dependency[@]}" \
    --job-name="tc100_phbg_e${seed}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,OUTPUT_FOLDER=$out,EXTRA_ARGS=$seed_args" \
    eval_retained_phone_ctc_bigru.slurm)
  echo "Seed $seed retained-checkpoint inference -> $eval_job (afterok:$downstream_job)"
done
