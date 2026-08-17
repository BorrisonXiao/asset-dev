#!/bin/bash
# Job-shape only. Identity flags (--account/--partition/--comment/--reservation/
# --exclude) are passed on the sbatch command line (see submit_pilot.sh).
# Config comes via --export env vars: MODE (coldstart|joint),
# BACKBONE (cnn|transformer|transformer_ar), EPOCHS, WARMUP_EPOCHS,
# COLDSTART_DIR (joint only), EXTRA_ARGS (optional).
#SBATCH --job-name=seg
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${MODE:?set MODE=coldstart|joint}"
: "${BACKBONE:?set BACKBONE=cnn|transformer|transformer_ar}"

ARGS=(hparams/speechllm_segmenter.yaml
    --segmenter_mode "$MODE" --segmenter_backbone "$BACKBONE"
    --number_of_epochs "${EPOCHS:-10}")
if [ "$MODE" = "joint" ]; then
    ARGS+=(--warmup_epochs "${WARMUP_EPOCHS:-2}")
    if [ -n "${COLDSTART_DIR:-}" ]; then
        ARGS+=(--coldstart_ckpt_dir "$COLDSTART_DIR")
    fi
fi
# optional extra CLI overrides
if [ -n "${EXTRA_ARGS:-}" ]; then
    ARGS+=($EXTRA_ARGS)
fi

echo "=== $(date) MODE=$MODE BACKBONE=$BACKBONE EPOCHS=${EPOCHS:-10} on $(hostname) ==="
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
    train_speechllm_with_segmenter.py "${ARGS[@]}"
