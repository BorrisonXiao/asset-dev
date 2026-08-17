#!/bin/bash
# Long-horizon, three-seed continuation of the best wav2vec2 Transformer
# segmenter/decoder checkpoint with the first-order autoregressive boundary policy.
# The transition bias is zero-initialized, exactly preserving the source model's
# initial logits and argmax boundaries before the new structured-policy training.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

SEEDS=(3407 3408 3409)
EPOCHS=30
DRY_RUN=${DRY_RUN:-0}

W2V2_ID=facebook/wav2vec2-base-960h
W2V2_DIM=768
W2V2_CACHE=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache

# Best completed wav2vec2 Transformer NLL-MT checkpoint selected on dev-clean:
# seed 3407, epoch 10, dev WER=5.84905 (test-clean/other=7.50/11.71).
INIT_CKPT=$(pwd)/results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/nll_mt/transformer/3407/save/CKPT+2026-08-05+13-35-19+00
OUT_ROOT=results/speechllm_segmenter/w2v2_transformer_autoregressive_long/nll_mt

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

for name in segmenter.ckpt llm.ckpt proj.ckpt normalize.ckpt; do
  [[ -s "$INIT_CKPT/$name" ]] || die "Missing initialization file: $INIT_CKPT/$name"
done

submit_seed() {
  local seed=$1
  local out="$OUT_ROOT/$seed"
  [[ ! -e "$out" ]] || die "Refusing to reuse output path: $out"

  local extra="--experiment_name speechllm_segmenter_w2v2_transformer_ar_long"
  extra+=" --ssl_hub $W2V2_ID --ssl_folder $W2V2_CACHE --ssl_feat_dims $W2V2_DIM"
  extra+=" --segmenter_input_dim $W2V2_DIM --seed $seed --output_folder $out"
  extra+=" --segmenter_init_checkpoint $INIT_CKPT/segmenter.ckpt"
  extra+=" --decoder_init_ckpt_dir $INIT_CKPT"
  extra+=" --segmenter_sampling autoregressive"
  extra+=" --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy"
  extra+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002"
  extra+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
  extra+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
  extra+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0"

  local command=(
    sbatch "${COMMON[@]}" --parsable --job-name="w2v_tf_ar_${seed}"
    --export="ALL,MODE=joint,BACKBONE=transformer,EPOCHS=$EPOCHS,WARMUP_EPOCHS=0,EXTRA_ARGS=$extra"
    run_segmenter.slurm
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "${command[@]}" >&2
    printf '\n' >&2
    echo "DRY_w2v_tf_ar_${seed}"
  else
    "${command[@]}"
  fi
}

echo "Initialization: $INIT_CKPT"
echo "Output root: $OUT_ROOT"
echo "Schedule: $EPOCHS joint-RL epochs; Transformer backbone; autoregressive policy"

for seed in "${SEEDS[@]}"; do
  job=$(submit_seed "$seed")
  echo "seed=$seed job=$job"
done
