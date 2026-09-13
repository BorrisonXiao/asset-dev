#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/expresso_boundary_patterns_2026-09-13/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
case "$1" in
 prepare)
  artifacts/segmenter/nonasr_boundary_audit_2026-09-12/envs/mfa/bin/python recipes/LibriSpeech/ASR/transformer/probe_expresso_acoustics.py prepare ;;
 infer)
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/probe_expresso_acoustics.py infer ;;
 analyze)
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/analyze_expresso_patterns.py ;;
 *) exit 2;;
esac
