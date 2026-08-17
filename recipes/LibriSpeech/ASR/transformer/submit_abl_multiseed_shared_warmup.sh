#!/bin/bash
# Run the reward / multi-task ablations from one exact decoder-warmup model.
#
# The source is the preserved end-of-epoch-6 checkpoint from the original
# cer_mt run.  Each destination receives a hard-linked snapshot of that
# checkpoint before submission, so SpeechBrain performs ordinary exact
# recovery (model, optimizer, epoch counter, and dataloader state) and begins
# at epoch 7. NLL uses the original successful on-policy update. CER keeps the
# batched-reward speedup but uses one accumulation-aware update and a milder
# policy step (lr_segmenter=5e-5, pg_weight=0.5).
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

BB=cnn
EPOCHS=10
WARMUP_EPOCHS=6
SEEDS=(3407 3408 3409)
BASE=results/speechllm_segmenter
COLDSTART_DIR="$(pwd)/$BASE/coldstart/$BB/3407/save"
WARMUP_CKPT="$(pwd)/$BASE/abl/cer_mt/$BB/3407/save/CKPT+2026-08-02+23-35-23+00"
OUT_ROOT="$BASE/multiseed_shared_warmup_onpolicy"

required=(
  CKPT.yaml brain.ckpt counter.ckpt dataloader-TRAIN.ckpt llm.ckpt
  model_optimizer.ckpt normalize.ckpt proj.ckpt segmenter.ckpt
)
for name in "${required[@]}"; do
  if [[ ! -f "$WARMUP_CKPT/$name" ]]; then
    echo "Missing shared-warmup checkpoint file: $WARMUP_CKPT/$name" >&2
    exit 1
  fi
done
if [[ "$(tr -d '[:space:]' < "$WARMUP_CKPT/counter.ckpt")" != "$WARMUP_EPOCHS" ]]; then
  echo "Shared-warmup counter is not epoch $WARMUP_EPOCHS" >&2
  exit 1
fi
if ! grep -q '^end-of-epoch: true$' "$WARMUP_CKPT/CKPT.yaml"; then
  echo "Shared-warmup checkpoint is not an end-of-epoch checkpoint" >&2
  exit 1
fi

COMMON=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation="JSALT 2026"
  --exclude=ga129
)
DEPENDENCY_ARGS=()
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  DEPENDENCY_ARGS+=(--dependency="afterok:$AFTEROK_JOB_ID")
fi

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
  local staged="$out/save/$(basename "$WARMUP_CKPT")"

  if [[ -e "$out" ]]; then
    # Permit only the exact pre-submit state left by a controller connection
    # failure.  Any training metadata means this is a real run and must not be
    # submitted a second time.
    if [[ -e "$out/train_log.txt" || -e "$out/hyperparams.yaml" ]]; then
      echo "Refusing to reuse an initialized output path: $out" >&2
      exit 1
    fi
    for name in "${required[@]}"; do
      if [[ ! "$WARMUP_CKPT/$name" -ef "$staged/$name" ]]; then
        echo "Existing output is not the exact staged shared warm-up: $out" >&2
        exit 1
      fi
    done
    echo "Reusing exact pre-staged warm-up at $out"
  else
    mkdir -p "$out/save"
    # Hard links are safe here: recovery only reads these files, and later
    # checkpoint cleanup merely unlinks this run's directory entry.
    cp -al "$WARMUP_CKPT" "$out/save/"
  fi

  local extra_args
  extra_args="--seed $seed --segmenter_reward $reward --freeze_decoder_in_joint $freeze_decoder --rl_update_mode $update_mode --lr_segmenter $lr_segmenter --lr_decoder 0.0005 --pg_weight $pg_weight --max_decode_ratio $max_decode_ratio --output_folder $out"
  local jid
  jid=$(sbatch "${COMMON[@]}" "${DEPENDENCY_ARGS[@]}" \
    --job-name="ms2_${short}_${seed}" \
    --parsable \
    --export="ALL,MODE=joint,BACKBONE=$BB,EPOCHS=$EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$COLDSTART_DIR,EXTRA_ARGS=$extra_args" \
    run_segmenter.slurm)
  echo "$variant seed=$seed -> $jid ($out)"
}

echo "Shared warm-up: $WARMUP_CKPT"
for seed in "${SEEDS[@]}"; do
  submit_one nll_frozen "$seed" nll True nf on_policy 0.0001 1.0 3.0
  submit_one nll_mt "$seed" nll False nm on_policy 0.0001 1.0 3.0
  submit_one cer_mt "$seed" cer False cm batched_on_policy 0.00005 0.5 2.0
done
