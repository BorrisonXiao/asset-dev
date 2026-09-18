#!/bin/bash
#SBATCH --job-name=ls960_units_audit
#SBATCH --account=highprio
#SBATCH --partition=gpu-a100
#SBATCH --exclude=e01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/slurm_logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export MPLCONFIGDIR=/tmp/ls960-units-mpl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
script=recipes/LibriSpeech/ASR/transformer/audit_segment_units.py
echo 'Inference audit of checkpoints trained on LS960: train-clean-100 train-clean-360 train-other-500'
echo 'Evaluation: 24 preselected dev-clean utterances, seeds 3407/3408/3409; one A100; no training'
$python "$script" infer
$python "$script" summarize
