#!/bin/bash
# Train the char/mora CTC aligner head on the frozen language encoder, then
# forced-align all splits and write boundary targets + reliability metrics.
# Env: LANG_CODE (zh|ja), SSL_HUB, MANIFEST_DIR, UNITS_LEVEL (char|mora),
#      OUT_ROOT (head + boundaries live under it), EXTRA_SPLITS (optional
#      "name=csv:units" specs beyond train/dev/test).
#SBATCH --job-name=xling_ctc
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${LANG_CODE:?}" "${SSL_HUB:?}" "${MANIFEST_DIR:?}" "${UNITS_LEVEL:?}" "${OUT_ROOT:?}"

ALIGN_ARGS=(
  --align "train=$MANIFEST_DIR/train.csv:$MANIFEST_DIR/train_units_${UNITS_LEVEL}.tsv"
  --align "dev=$MANIFEST_DIR/dev.csv:$MANIFEST_DIR/dev_units_${UNITS_LEVEL}.tsv"
  --align "test=$MANIFEST_DIR/test.csv:$MANIFEST_DIR/test_units_${UNITS_LEVEL}.tsv"
)
if [[ -n "${EXTRA_SPLITS:-}" ]]; then
  for spec in $EXTRA_SPLITS; do ALIGN_ARGS+=(--align "$spec"); done
fi

echo "=== $(date) xling CTC aligner lang=$LANG_CODE level=$UNITS_LEVEL on $(hostname) ==="
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  xling_ctc_boundary_align.py --mode all \
  --ssl_hub "$SSL_HUB" \
  --train_csv "$MANIFEST_DIR/train.csv" \
  --train_units "$MANIFEST_DIR/train_units_${UNITS_LEVEL}.tsv" \
  --dev_csv "$MANIFEST_DIR/dev.csv" \
  --dev_units "$MANIFEST_DIR/dev_units_${UNITS_LEVEL}.tsv" \
  --head_dir "$OUT_ROOT/ctc_head" \
  --boundary_out "$OUT_ROOT/boundaries/${UNITS_LEVEL}" \
  --stats_out "$OUT_ROOT/boundaries/${UNITS_LEVEL}_stats.json" \
  "${ALIGN_ARGS[@]}"
