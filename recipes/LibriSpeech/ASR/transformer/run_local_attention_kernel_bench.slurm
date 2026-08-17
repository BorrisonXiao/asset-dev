#!/bin/bash
#SBATCH --job-name=tar_attn64
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
ROOT=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm
RECIPE="$ROOT/recipes/LibriSpeech/ASR/transformer"
OUT="$RECIPE/results/speechllm_segmenter_wavlm/transformer_ar_attention_bench"
cd "$RECIPE"
mkdir -p "$OUT" slurm_logs

PYTHON_BIN=${PYTHON_BIN:-/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python}
"$PYTHON_BIN" \
  benchmark_local_attention_kernels.py \
  --output "$OUT/kernels_${SLURM_JOB_ID}.json" \
  --lengths 256 512 1024 --batch 16 --repeats 8
