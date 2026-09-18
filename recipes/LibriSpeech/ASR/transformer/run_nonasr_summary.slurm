#!/bin/bash
#SBATCH --job-name=nonasr_boundary_summary
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/nonasr_boundary_audit_2026-09-12/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    recipes/LibriSpeech/ASR/transformer/summarize_nonasr_boundaries.py "$@"
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    recipes/LibriSpeech/ASR/transformer/analyze_nonasr_controls.py
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    recipes/LibriSpeech/ASR/transformer/audit_nonasr_gap_energy.py
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    recipes/LibriSpeech/ASR/transformer/plot_nonasr_audit.py
