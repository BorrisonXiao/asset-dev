#!/bin/bash
# Complete the matched three-seed WavLM RL experiment:
#   1) validated independent Bernoulli control;
#   2) first-order autoregressive boundary policy.
# Both start from the same best char segmenter and best oracle-char decoder,
# warm the decoder for two epochs on predicted boundaries, then jointly train.
# Seed 3407 is already complete; by default this submits missing seeds 3408/3409.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

TOTAL_EPOCHS=12
WARMUP_EPOCHS=2
DRY_RUN=${DRY_RUN:-0}
if (($#)); then
  SEEDS=("$@")
else
  SEEDS=(3408 3409)
fi

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
SEGMENTER_CKPT=$(pwd)/results/speechllm_segmenter_wavlm/coldstart/cnn/3407/save/CKPT+2026-08-07+11-19-21+00/segmenter.ckpt
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
OUT_ROOT=results/speechllm_segmenter_wavlm/oracleclose_bestinit

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --exclude=ga129
  --time=2-00:00:00
)

die() {
  echo "$*" >&2
  exit 1
}

[[ -s "$SEGMENTER_CKPT" ]] || die "Missing segmenter checkpoint: $SEGMENTER_CKPT"
for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done

submit_variant() {
  local policy=$1
  local short=$2
  local seed=$3
  local out="$OUT_ROOT/$policy/$seed"
  [[ ! -e "$out" ]] || die "Refusing to reuse output path: $out"

  local extra="--experiment_name speechllm_segmenter_wavlm_oracleclose"
  extra+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE --ssl_feat_dims $WAVLM_DIM"
  extra+=" --seed $seed --output_folder $out"
  extra+=" --segmenter_init_checkpoint $SEGMENTER_CKPT"
  extra+=" --decoder_init_ckpt_dir $DECODER_CKPT"
  extra+=" --segmenter_sampling $policy"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_segmenter 0.00005"
  extra+=" --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0"

  local command=(
    sbatch "${COMMON[@]}" --parsable --job-name="wl_oc_${short}_${seed}"
    --export="ALL,MODE=joint,BACKBONE=cnn,EPOCHS=$TOTAL_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,EXTRA_ARGS=$extra"
    run_segmenter.slurm
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "${command[@]}" >&2
    printf '\n' >&2
    echo "DRY_wl_oc_${short}_${seed}"
  else
    "${command[@]}"
  fi
}

echo "Best WavLM char segmenter: $SEGMENTER_CKPT"
echo "Best WavLM char decoder: $DECODER_CKPT"
echo "Schedule: $WARMUP_EPOCHS decoder-warmup + $((TOTAL_EPOCHS-WARMUP_EPOCHS)) joint-RL epochs"
echo "Output root: $OUT_ROOT"

for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Seed must be an integer: $seed"
  bernoulli_job=$(submit_variant bernoulli bern "$seed")
  autoregressive_job=$(submit_variant autoregressive ar "$seed")
  echo "Seed $seed · Bernoulli control: $bernoulli_job"
  echo "Seed $seed · Autoregressive policy: $autoregressive_job"
done
