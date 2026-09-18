#!/bin/bash
# Standalone policy microbenchmark. It does not import or alter the training loop.
#SBATCH --job-name=tar_opt_bench
#SBATCH --time=00:45:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE="$ROOT/recipes/LibriSpeech/ASR/transformer"
CHECKPOINT="$RECIPE/results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/coldstart/3407/save/CKPT+2026-08-11+21-43-59+00/segmenter.ckpt"
OUT_DIR="$RECIPE/results/speechllm_segmenter_wavlm/transformer_ar_optimization_bench"

cd "$RECIPE"
mkdir -p "$OUT_DIR" slurm_logs

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  benchmark_transformer_ar_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT_DIR/benchmark_${SLURM_JOB_ID}.json" \
  --batch-size 4 \
  --lengths 256 512 1024 \
  --warmup 1 \
  --repeats 3
