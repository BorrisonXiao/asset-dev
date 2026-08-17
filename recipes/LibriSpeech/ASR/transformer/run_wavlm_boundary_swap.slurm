#!/bin/bash
# Evaluation-only oracle-decoder boundary swap. Identity/cost flags are passed
# by sbatch; BACKBONE and checkpoint paths come from --export.
#SBATCH --job-name=wl_boundary_swap
#SBATCH --time=0-06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${BACKBONE:?set BACKBONE=cnn|transformer}"
: "${COLD_SEGMENTER:?set COLD_SEGMENTER}"
: "${RL_SEGMENTER:?set RL_SEGMENTER}"
: "${OUTPUT_FOLDER:?set OUTPUT_FOLDER}"

ORACLE_DECODER=results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407/save/CKPT+2026-08-09+16-38-50+00
MANIFEST_ROOT=results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407

/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
    eval_wavlm_boundary_swap.py \
    --decoder-checkpoint "$ORACLE_DECODER" \
    --cold-segmenter "$COLD_SEGMENTER" \
    --rl-segmenter "$RL_SEGMENTER" \
    --backbone "$BACKBONE" \
    --manifest-root "$MANIFEST_ROOT" \
    --splits dev-clean \
    --summary "$OUTPUT_FOLDER/summary.json" \
    hparams/speechllm_segmenter.yaml \
    --device=cuda \
    --segmenter_mode joint \
    --segmenter_backbone "$BACKBONE" \
    --output_folder "$OUTPUT_FOLDER" \
    --experiment_name speechllm_boundary_swap_wavlm \
    --ssl_hub microsoft/wavlm-large \
    --ssl_folder "$HF_HUB_CACHE" \
    --ssl_feat_dims 1024 \
    --ssl_device cuda \
    --llm_device cuda \
    --max_decode_ratio 1.0 \
    --test_batch_size 16 \
    --precision bf16 \
    --eval_precision bf16
