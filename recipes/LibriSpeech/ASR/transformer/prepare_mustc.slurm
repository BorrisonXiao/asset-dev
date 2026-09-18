#!/bin/bash
# MuST-C v1 En-De -> 16 kHz FLAC segments + paired ASR/ST manifests.
# CPU-only; segment extraction from 2043 talk WAVs. Resumable.
#SBATCH --job-name=mustc_prep
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --time=8:00:00
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
PY=${PY:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
SRC=/export/jsalt26/omnienc/users/cxiao/datasets/mustc_en_de
OUT=/export/jsalt26/omnienc/users/cxiao/datasets/mustc_en_de_prepared
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== MuST-C v1 En-De preparation ==="; echo "Node: $(hostname)"
"$PY" mustc_prepare.py --mustc-root "$SRC" --audio-root "$OUT/audio" \
  --manifest-dir "$OUT/manifests" --splits ${SPLITS:-train tst-COMMON tst-HE dev} --workers 12
echo "=== manifest summary ==="
for c in "$OUT"/manifests/*.csv; do echo "$(basename $c): $(($(wc -l < $c) - 1)) rows"; done
