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
export MFA_ROOT_DIR=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/mfa

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

BASE_DICT=$(find "$MFA_ROOT_DIR/pretrained_models/dictionary" -name "${DICT_NAME}.dict" | head -1)
[[ -s "$BASE_DICT" ]] || { echo "Missing dictionary model: $DICT_NAME" >&2; exit 1; }
DICT_ARG="$BASE_DICT"
if [[ -n "${G2P_NAME:-}" ]]; then
  echo "=== $(date) G2P for OOVs ==="
  # Corpus vocabulary minus dictionary entries, then G2P the difference and
  # merge into one combined dictionary file.
  $PY - "$CORPUS" "$BASE_DICT" "$WORK_DIR/oovs.txt" <<'PYEOF'
import glob, sys
corpus, dict_path, out = sys.argv[1:4]
vocab = set()
for lab in glob.glob(corpus + "/*.lab"):
    with open(lab, encoding="utf-8") as fh:
        vocab.update(fh.read().split())
known = set()
with open(dict_path, encoding="utf-8") as fh:
    for line in fh:
        parts = line.split()
        if parts:
            known.add(parts[0])
oovs = sorted(vocab - known)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(oovs) + "\n")
print(f"vocab={len(vocab)} oov={len(oovs)}")
PYEOF
  if [[ -s "$WORK_DIR/oovs.txt" ]]; then
    $MFA g2p "$WORK_DIR/oovs.txt" "$G2P_NAME" "$WORK_DIR/oov_dict.txt" \
      --use_mp -j "$SLURM_CPUS_PER_TASK" || true
    if [[ -s "$WORK_DIR/oov_dict.txt" ]]; then
      cat "$BASE_DICT" "$WORK_DIR/oov_dict.txt" > "$WORK_DIR/combined.dict"
      DICT_ARG="$WORK_DIR/combined.dict"
    fi
  fi
fi

SPEAKER_ARG=()
if [[ -n "${SPEAKER_CHARS:-}" ]]; then SPEAKER_ARG=(--speaker_characters "$SPEAKER_CHARS"); fi

echo "=== $(date) MFA align (dict=$DICT_ARG) ==="
$MFA align --clean --overwrite --use_mp -j "$SLURM_CPUS_PER_TASK" \
  "${SPEAKER_ARG[@]}" \
  "$CORPUS" "$DICT_ARG" "$ACOUSTIC_NAME" "$ALIGNED" \
  --output_format long_textgrid

echo "=== $(date) TextGrid -> phone boundaries ==="
$PY xling_textgrid_boundaries.py --csv "${CSVS[@]}" \
  --textgrid_dir "$ALIGNED" \
  --out_dir "$OUT_ROOT/boundaries/phone" \
  --stats_out "$OUT_ROOT/boundaries/phone_stats.json"
