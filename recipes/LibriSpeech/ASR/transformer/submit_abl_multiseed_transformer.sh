#!/bin/bash
# Transformer-backbone replication of the corrected CNN multi-seed sweep.
# One shared six-epoch decoder warm-up is followed by 3 variants x 3 seeds.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

BB=transformer
WARMUP_SEED=3407
WARMUP_EPOCHS=6
EPOCHS=10
SEEDS=(3407 3408 3409)
BASE=results/speechllm_segmenter
COLDSTART_DIR="$(pwd)/$BASE/coldstart/$BB/$WARMUP_SEED/save"
WARMUP_OUT="$BASE/shared_warmup/$BB/$WARMUP_SEED"
WARMUP_SOURCE_DIR="$(pwd)/$WARMUP_OUT/save"
OUT_ROOT="$BASE/multiseed_shared_warmup_onpolicy"

if [[ ! -d "$COLDSTART_DIR" ]]; then
  echo "Missing Transformer cold-start checkpoint directory: $COLDSTART_DIR" >&2
  exit 1
fi
if [[ -e "$WARMUP_OUT" ]]; then
  echo "Refusing to reuse existing shared warm-up output: $WARMUP_OUT" >&2
  exit 1
fi
for variant in nll_frozen nll_mt cer_mt; do
  for seed in "${SEEDS[@]}"; do
    out="$OUT_ROOT/$variant/$BB/$seed"
    if [[ -e "$out" ]]; then
      echo "Refusing to reuse existing production output: $out" >&2
      exit 1
    fi
  done
done

# Hard policy: every GPU submission uses this exact account/partition pair.
COMMON=(
  --account=jsalt2026-lgarci27
  --partition=a100
  --comment=accept_cost
  --reservation="JSALT 2026"
  --exclude=ga129
)

warmup_extra="--seed $WARMUP_SEED --segmenter_reward nll --freeze_decoder_in_joint False --rl_update_mode on_policy --lr_segmenter 0.0001 --lr_decoder 0.0005 --pg_weight 1.0 --max_decode_ratio 3.0 --output_folder $WARMUP_OUT"
warmup_job=$(sbatch "${COMMON[@]}" \
  --job-name="tf_warm6_${WARMUP_SEED}" \
  --parsable \
  --export="ALL,MODE=joint,BACKBONE=$BB,EPOCHS=$WARMUP_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLDSTART_DIR,EXTRA_ARGS=$warmup_extra" \
  run_segmenter.slurm)
echo "shared Transformer warm-up -> $warmup_job ($WARMUP_OUT)"

submit_one() {
  local variant=$1
  local seed=$2
  local reward=$3
  local freeze_decoder=$4
  local short=$5
  local update_mode=$6
  local lr_segmenter=$7
  local pg_weight=$8
  local max_decode_ratio=$9
  local out="$OUT_ROOT/$variant/$BB/$seed"
  local extra_args
  extra_args="--seed $seed --segmenter_reward $reward --freeze_decoder_in_joint $freeze_decoder --rl_update_mode $update_mode --lr_segmenter $lr_segmenter --lr_decoder 0.0005 --pg_weight $pg_weight --max_decode_ratio $max_decode_ratio --output_folder $out"

  local jid
  jid=$(sbatch "${COMMON[@]}" \
    --dependency="afterok:$warmup_job" \
    --job-name="tf_${short}_${seed}" \
    --parsable \
    --export="ALL,MODE=joint,BACKBONE=$BB,EPOCHS=$EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLDSTART_DIR,WARMUP_SOURCE_DIR=$WARMUP_SOURCE_DIR,OUTPUT_FOLDER=$out,EXTRA_ARGS=$extra_args" \
    run_segmenter_from_shared_warmup.slurm)
  echo "$variant seed=$seed -> $jid (afterok:$warmup_job; $out)"
}

for seed in "${SEEDS[@]}"; do
  submit_one nll_frozen "$seed" nll True nf on_policy 0.0001 1.0 3.0
  submit_one nll_mt "$seed" nll False nm on_policy 0.0001 1.0 3.0
  submit_one cer_mt "$seed" cer False cm batched_on_policy 0.00005 0.5 2.0
done
