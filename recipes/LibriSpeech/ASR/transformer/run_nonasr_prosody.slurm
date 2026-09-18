#!/bin/bash
#SBATCH --job-name=nonasr_prosody_audit
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/nonasr_boundary_audit_2026-09-12/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
artifacts/segmenter/nonasr_boundary_audit_2026-09-12/envs/mfa/bin/python \
    recipes/LibriSpeech/ASR/transformer/extract_nonasr_prosody.py --workers 4
