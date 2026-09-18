#!/bin/bash
#SBATCH --job-name=boundary_accuracy
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  audit_controlled_boundary_accuracy.py \
  --csv results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407/dev-clean.csv \
  --data-root /export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech \
  --char-dir /export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char \
  --phone-dir /export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/phone \
  --wav2vec2-cache /export/jsalt26/omnienc/users/cxiao/hf/hub \
  --wavlm-cache /export/jsalt26/omnienc/users/cxiao/hf/hub \
  --batch-size 4 \
  --device cuda \
  --output /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/controlled_boundary_accuracy.json
