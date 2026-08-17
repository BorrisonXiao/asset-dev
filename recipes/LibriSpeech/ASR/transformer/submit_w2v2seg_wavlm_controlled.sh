#!/bin/bash
# Controlled six-run dual-encoder study:
#   wav2vec2-base char segmenter -> boundary indices -> pooled WavLM-Large features.
# Three RL seeds compare two decoder starts while fixing the exact segmenter:
#   (1) best WavLM oracle-char decoder + 2 predicted-boundary adaptation epochs;
#   (2) fresh projection/LoRA + 6 predicted-boundary warmup epochs.
# Both arms then run the same 10-epoch Bernoulli NLL-GRPO phase.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
RL_EPOCHS=10
DRY_RUN=${DRY_RUN:-0}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
W2V2_ID=facebook/wav2vec2-base-960h
W2V2_DIM=768
W2V2_CACHE=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache

# Best wav2vec2 CNN char segmenter: seed 3407, epoch 4, dev F1=0.89453897.
SEGMENTER_CKPT=$(pwd)/results/speechllm_segmenter/coldstart/cnn/3407/save/CKPT+2026-08-02+01-25-10+00/segmenter.ckpt
# Best completed WavLM oracle-char decoder selected on dev-clean:
# seed 3408, epoch 10, dev WER=3.6929 (test-clean/other=4.62/7.57).
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
OUT_ROOT=results/speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --exclude=ga129
  --time=3-00:00:00
)

die() {
  echo "$*" >&2
  exit 1
}

[[ -s "$SEGMENTER_CKPT" ]] || die "Missing segmenter checkpoint: $SEGMENTER_CKPT"
for name in llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$DECODER_CKPT/$name" ]] || die "Missing decoder file: $DECODER_CKPT/$name"
done

submit_arm() {
  local arm=$1
  local seed=$2
  local warmup_epochs warmup_lr short decoder_arg
  case "$arm" in
    oracle_char_decoder)
      warmup_epochs=2
      warmup_lr=0.0002
      short=oc
      decoder_arg=" --decoder_init_ckpt_dir $DECODER_CKPT"
      ;;
    scratch_decoder)
      warmup_epochs=6
      warmup_lr=0.0005
      short=sc
      decoder_arg=""
      ;;
    *) die "Unknown arm: $arm" ;;
  esac

  local total_epochs=$((warmup_epochs + RL_EPOCHS))
  local out="$OUT_ROOT/$arm/$seed"
  [[ ! -e "$out" ]] || die "Refusing to reuse output path: $out"

  local extra="--experiment_name speechllm_segmenter_wavlm_hybrid_controlled"
  extra+=" --ssl_hub $WAVLM_ID --ssl_folder $WAVLM_CACHE --ssl_feat_dims $WAVLM_DIM"
  extra+=" --segmenter_ssl_hub $W2V2_ID --segmenter_ssl_folder $W2V2_CACHE"
  extra+=" --segmenter_ssl_device cuda --segmenter_input_dim $W2V2_DIM"
  extra+=" --seed $seed --output_folder $out"
  extra+=" --segmenter_init_checkpoint $SEGMENTER_CKPT$decoder_arg"
  extra+=" --segmenter_sampling bernoulli"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup $warmup_lr"
  extra+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0"

  local command=(
    sbatch "${COMMON[@]}" --parsable --job-name="hyb_${short}_${seed}"
    --export="ALL,MODE=joint,BACKBONE=cnn,EPOCHS=$total_epochs,WARMUP_EPOCHS=$warmup_epochs,EXTRA_ARGS=$extra"
    run_segmenter.slurm
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "${command[@]}" >&2
    printf '\n' >&2
    echo "DRY_hyb_${short}_${seed}"
  else
    "${command[@]}"
  fi
}

echo "Fixed wav2vec2 segmenter: $SEGMENTER_CKPT"
echo "WavLM oracle decoder: $DECODER_CKPT"
echo "Output root: $OUT_ROOT"
echo "Each arm gets $RL_EPOCHS matched joint-RL epochs; policy=bernoulli"

for seed in "${SEEDS[@]}"; do
  job=$(submit_arm oracle_char_decoder "$seed")
  echo "oracle_char_decoder seed=$seed job=$job"
done
for seed in "${SEEDS[@]}"; do
  job=$(submit_arm scratch_decoder "$seed")
  echo "scratch_decoder seed=$seed job=$job"
done
