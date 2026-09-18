#!/bin/bash
#SBATCH --job-name=nonasr_confidence_control
#SBATCH --account=highprio
#SBATCH --partition=gpu-a100
#SBATCH --exclude=e01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/nonasr_boundary_audit_2026-09-12/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo 'Confidence-thinning control; one A100, <10 MB output; no decoder or training'
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/score_asr_boundary_confidence.py
