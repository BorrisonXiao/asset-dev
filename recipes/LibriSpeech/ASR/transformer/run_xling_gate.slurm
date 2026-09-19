#!/bin/bash
# S1 alignment-quality gate: fails (and thereby kills dependent training jobs
# submitted with --kill-on-invalid-dep) unless coverage, aligner reliability,
# char-phone agreement, and the phone-rate initialization invariant all hold.
# Env: MANIFEST_DIR, OUT_ROOT, UNITS_LEVEL, RHO_LO, RHO_HI, EXTRA_CSVS (opt).
#SBATCH --job-name=xling_gate
#SBATCH --partition=med
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${MANIFEST_DIR:?}" "${OUT_ROOT:?}" "${UNITS_LEVEL:?}"

CSVS=("$MANIFEST_DIR/train.csv" "$MANIFEST_DIR/dev.csv" "$MANIFEST_DIR/test.csv")
if [[ -n "${EXTRA_CSVS:-}" ]]; then for c in $EXTRA_CSVS; do CSVS+=("$c"); done; fi

/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  xling_alignment_gate.py \
  --csv "${CSVS[@]}" \
  --char_dir "$OUT_ROOT/boundaries/${UNITS_LEVEL}" \
  --phone_dir "$OUT_ROOT/boundaries/phone" \
  --aligner_metrics "$OUT_ROOT/ctc_head/metrics.json" \
  --report "$OUT_ROOT/gate_report.json" \
  --rho_lo "${RHO_LO:-0.10}" --rho_hi "${RHO_HI:-0.20}"
