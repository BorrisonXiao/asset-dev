#!/bin/bash
#SBATCH --job-name=gen_word_ctc_sep
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets/logs/gen_sep_%A_%a.out
#
# Regenerate CTC word_ctc boundaries AND the parallel per-frame is_separator
# channel (word_ctc_sep) for the splits needed by the blank-removal ablations.
# Job-shape only here; identity flags (account/partition/comment/reservation/
# exclude) are passed on the sbatch command line (multi-word reservation name
# can't live in a #SBATCH line). Sharded 8 ways over the utterance list.
#
# The regenerated word_ctc is verified bit-for-bit against the shared
# wavlm_boundaries/word_ctc (see scratchpad/validate_sep.py), so word_ctc_sep is
# a trustworthy, alignment-consistent separator mask.

set -euo pipefail

RECIPE=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
DATA=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/datasets
OUT=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets
PY=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python

cd "$RECIPE"

# Splits can be overridden on the command line: `sbatch ... gen_word_ctc_sep.sh <split>...`
if [ "$#" -gt 0 ]; then
    SPLITS=("$@")
else
    SPLITS=(train-clean-100 dev-clean dev-other test-clean test-other)
fi

"$PY" ctc_boundary_align.py \
    --levels word_ctc \
    --splits "${SPLITS[@]}" \
    --audio_root "$DATA/LibriSpeech" \
    --out_root "$OUT" \
    --device cuda \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num_shards 8
