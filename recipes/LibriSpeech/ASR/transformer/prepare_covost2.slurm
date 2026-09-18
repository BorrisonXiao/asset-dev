#!/bin/bash
# CoVoST 2 En->De preparation: parquet -> 16 kHz FLAC + paired ASR/ST manifests.
# CPU-only; the login node's process limit kills an 8-way decode pool.
# Resumable: utterances whose FLAC already exists are skipped.
#SBATCH --job-name=covost_prep
#SBATCH --output=slurm_logs/covost_prep_%j.out
#SBATCH --error=slurm_logs/covost_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
PARQUET_DIR=${PARQUET_DIR:-/export/jsalt26/omnienc/users/cxiao/datasets/covost2_raw/en_de}
DATA_ROOT=${DATA_ROOT:-/export/jsalt26/omnienc/users/cxiao/datasets/covost2_en_de}
SPLITS=${SPLITS:-"validation test train"}
WORKERS=${WORKERS:-8}

# Each worker builds its own torch/pyarrow state; keep BLAS single-threaded so
# 8 workers do not oversubscribe the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== CoVoST 2 En-De preparation ==="
echo "Parquet:   $PARQUET_DIR"
echo "Output:    $DATA_ROOT"
echo "Splits:    $SPLITS"
echo "Workers:   $WORKERS"
echo "Node:      $(hostname)"

"$PYTHON" covost2_prepare.py \
  --parquet-dir "$PARQUET_DIR" \
  --audio-root "$DATA_ROOT/audio" \
  --manifest-dir "$DATA_ROOT/manifests" \
  --splits $SPLITS \
  --workers "$WORKERS"

echo "=== Manifest summary ==="
for csv in "$DATA_ROOT"/manifests/*.csv; do
  echo "$(basename "$csv"): $(($(wc -l < "$csv") - 1)) rows"
done
