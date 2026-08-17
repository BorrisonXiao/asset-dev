#!/bin/bash
#SBATCH --job-name=boundary_accuracy
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  audit_controlled_boundary_accuracy.py \
  --csv results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407/dev-clean.csv \
  --data-root /home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/datasets/LibriSpeech \
  --char-dir /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets/wavlm_boundaries/char \
  --phone-dir /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets/wavlm_boundaries/phone \
  --wav2vec2-cache /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache \
  --wavlm-cache /home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub \
  --batch-size 4 \
  --device cuda \
  --output /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/artifacts/segmenter/controlled_boundary_accuracy.json
