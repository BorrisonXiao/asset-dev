#!/bin/bash
# CPU gate between sharded CTC inference and the three downstream LS960 runs.
#SBATCH --job-name=ls960_phctc11_gate
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
: "${BOUNDARY_DIR:?set BOUNDARY_DIR}"
: "${NUM_SHARDS:?set NUM_SHARDS}"

cd "$RECIPE"
"$PY" summarize_wavlm_phone_ctc_boundaries.py \
  --boundary_dir "$BOUNDARY_DIR" \
  --num_shards "$NUM_SHARDS" \
  --split_frames 8
