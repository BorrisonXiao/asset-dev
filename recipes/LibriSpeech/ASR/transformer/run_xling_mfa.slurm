#!/bin/bash
# MFA phone alignment for one language: build the corpus dir, G2P any OOV
# words into a supplementary dictionary, align with the latest v3 models, and
# convert TextGrids to per-utterance phone boundary targets.
# Env: LANG_CODE (zh|ja), MANIFEST_DIR, WORK_DIR, DICT_NAME, ACOUSTIC_NAME,
#      G2P_NAME (optional), OUT_ROOT.
# Models must already be downloaded on the login node (compute nodes are
# assumed offline): mfa model download acoustic/dictionary/g2p <name>.
#SBATCH --job-name=xling_mfa
#SBATCH --partition=med
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
# MFA shells out to OpenFst/Kaldi/postgres binaries via PATH.
export PATH=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/mfa/bin:$PATH
# MFA keeps a sqlite database under MFA_ROOT_DIR; sqlite over weka fails
# intermittently under multiprocess access ("unable to open database file"
# at TextGrid export, job 597228). Work on node-local storage and read the
# pretrained models from a copy; only the exported TextGrids land on weka.
WEKA_MFA=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/mfa
export MFA_ROOT_DIR=${SLURM_TMPDIR:-/tmp}/mfa_${SLURM_JOB_ID}
mkdir -p "$MFA_ROOT_DIR"
cp -r "$WEKA_MFA/pretrained_models" "$MFA_ROOT_DIR/"

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${LANG_CODE:?}" "${MANIFEST_DIR:?}" "${WORK_DIR:?}" "${DICT_NAME:?}" "${ACOUSTIC_NAME:?}" "${OUT_ROOT:?}"
MFA=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/mfa/bin/mfa
PY=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python

CORPUS="$WORK_DIR/corpus"
ALIGNED="$WORK_DIR/aligned"
mkdir -p "$WORK_DIR"

CSVS=("$MANIFEST_DIR/train.csv" "$MANIFEST_DIR/dev.csv" "$MANIFEST_DIR/test.csv")
if [[ -n "${EXTRA_CSVS:-}" ]]; then for c in $EXTRA_CSVS; do CSVS+=("$c"); done; fi

echo "=== $(date) MFA corpus prep lang=$LANG_CODE ==="
$PY xling_mfa_corpus.py --csv "${CSVS[@]}" --language "$LANG_CODE" --corpus_dir "$CORPUS"

# OOVs are handled by MFA itself: its tokenizer produces the dictionary's
# word conventions from the raw lab text, and --g2p_model_path generates
# pronunciations for anything still out of vocabulary.

SPEAKER_ARG=()
if [[ -n "${SPEAKER_CHARS:-}" ]]; then SPEAKER_ARG=(--speaker_characters "$SPEAKER_CHARS"); fi

echo "=== $(date) MFA align (dict=$DICT_ARG) ==="
$MFA align --clean --overwrite --use_mp -j "$SLURM_CPUS_PER_TASK" \
  "${SPEAKER_ARG[@]}" \
  --g2p_model_path "$G2P_NAME" \
  "$CORPUS" "$DICT_NAME" "$ACOUSTIC_NAME" "$ALIGNED" \
  --output_format long_textgrid

echo "=== $(date) TextGrid -> phone boundaries ==="
$PY xling_textgrid_boundaries.py --csv "${CSVS[@]}" \
  --textgrid_dir "$ALIGNED" \
  --out_dir "$OUT_ROOT/boundaries/phone" \
  --stats_out "$OUT_ROOT/boundaries/phone_stats.json"
