#!/bin/bash
# Matched short pilots from one completed Transformer-AR + BiGRU checkpoint.
# Each job exercises a full train/validation/test loop but debug mode limits every
# split to a small, equal number of batches.  This is an implementation/ranking
# diagnostic, not a final WER experiment.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a MODES <<< "${BILEVEL_MODES:-off heldout decoder_lookahead decoder_pooler_lookahead}"
SEED=${SEED:-3407}
DEBUG_BATCHES=${DEBUG_BATCHES:-8}
INNER_LR=${INNER_LR:-0.01}
PILOT_TAG=${PILOT_TAG:-matched_v3}
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
INIT_CKPT=$(pwd)/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/3407/save/CKPT+2026-08-15+03-13-00+00
OUT_ROOT=results/speechllm_segmenter_wavlm/bilevel_transformer_ar_local64_bigru_pilot/$PILOT_TAG

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --cpus-per-task=12
  --mem=80G
  --time=12:00:00
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

ENCODER_ARGS="--experiment_name speechllm_segmenter_wavlm_bilevel_pilot"
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
    off|heldout|decoder_lookahead|decoder_pooler_lookahead) ;;
    *) die "Unknown BILEVEL_MODES entry: $mode" ;;
  esac
  out="$OUT_ROOT/$mode/lr_${INNER_LR}/$SEED"
  [[ ! -e "$out" ]] || die "Refusing to reuse pilot output: $out"

  extra="$ENCODER_ARGS $POLICY_ARGS $POOL_ARGS --seed $SEED --output_folder $out"
  extra+=" --stage_timing_file $out/stage_timing.jsonl"
  extra+=" --bilevel_diagnostics_file $out/bilevel_diagnostics.jsonl"
  extra+=" --segmenter_init_checkpoint $INIT_CKPT/segmenter.ckpt"
  extra+=" --decoder_init_ckpt_dir $INIT_CKPT"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False"
  extra+=" --rl_update_mode combined_on_policy --bilevel_mode $mode"
  extra+=" --bilevel_support_fraction 0.5 --bilevel_inner_lr $INNER_LR"
  extra+=" --bilevel_inner_max_grad_norm 1.0 --bilevel_support_pg_weight 1.0"
  extra+=" --bilevel_measure_support_after True"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_segmenter 0.00005"
  extra+=" --pg_weight 1.0 --grpo_k 4 --rate_mode band"
  extra+=" --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --test_batch_size 8 --max_batch_length_train 100"
  extra+=" --debug --debug_persistently --debug_batches $DEBUG_BATCHES --debug_epochs 1"

  short=${mode/decoder_pooler_lookahead/dp}
  short=${short/decoder_lookahead/d}
  short=${short/heldout/h}
  short=${short/off/o}
  job=$(submit_job "bl_${short}_${PILOT_TAG}_${SEED}" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=1,WARMUP_EPOCHS=0,COLDSTART_DIR=,EXTRA_ARGS=$extra" \
    run_segmenter.slurm)
  echo "bilevel pilot mode=$mode seed=$SEED -> $job ($out)"
done

echo "Pilot: current best WavLM Transformer-AR + BiGRU checkpoint, K=4, $DEBUG_BATCHES batches/split"
echo "Output root: $OUT_ROOT (about 0.3 GB per completed arm)"
