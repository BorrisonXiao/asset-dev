#!/bin/bash
#SBATCH --job-name=ls960_phone_patterns
#SBATCH --account=highprio
#SBATCH --partition=gpu-a100
#SBATCH --exclude=e01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/slurm_logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo 'Analysis only: 100h checkpoints and LS960 checkpoints (train-clean-100 train-clean-360 train-other-500); no training'
echo 'Native TIMIT TEST minus SA and five utterances per speaker from each LibriSpeech test split; three seeds'
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/audit_phone_patterns.py infer "$@"
