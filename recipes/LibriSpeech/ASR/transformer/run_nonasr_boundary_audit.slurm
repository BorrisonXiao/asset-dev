#!/bin/bash
#SBATCH --job-name=nonasr_boundary_audit
#SBATCH --account=highprio
#SBATCH --partition=gpu-a100
#SBATCH --exclude=e01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/nonasr_boundary_audit_2026-09-12/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo 'Inference only: existing LS960 ASR and single-task non-ASR checkpoints; one A100; no training'
echo 'Outputs: artifacts/segmenter/nonasr_boundary_audit_2026-09-12; predictions expected <100 MB'
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/audit_nonasr_boundaries.py infer "$@"
