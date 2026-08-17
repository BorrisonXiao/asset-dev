#!/bin/bash
#SBATCH --job-name=blank_abl
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/results/blank_ablation/logs/%x_%j.out
#
# Blank-removal ablation runs of speechllm_fixed_pooling.yaml.
# Usage (identity flags go on the sbatch command line):
#   sbatch <identity flags> launch_blank_ablation.sh <blank_mode> <tag> [train_splits_json] [extra overrides...]
# e.g.
#   sbatch ... launch_blank_ablation.sh keep tc100
#   sbatch ... launch_blank_ablation.sh drop tc100
#
# Each run gets its own output_folder results/blank_ablation/<tag>_<blank_mode>
# so nothing collides with exp-001 (results/speechllm_fixed_pooling/alignment/3407).

set -euo pipefail

BLANK_MODE="${1:?need blank_mode (keep|drop|zero|const|shuffle)}"
TAG="${2:?need a tag (e.g. tc100, ls960)}"
TRAIN_SPLITS="${3:-[\"train-clean-100\"]}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

RECIPE=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
PY=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python
OUTPUT="$RECIPE/results/blank_ablation/${TAG}_${BLANK_MODE}"
# facebook/wav2vec2-base-960h is NOT in the shared HF cache; reuse exp-001's
# already-downloaded local copy so offline loading resolves (else a fresh
# ssl_folder + HF_HUB_OFFLINE=1 fails to find it).
SSL_FOLDER="$RECIPE/results/speechllm_fixed_pooling/alignment/3407/save/ssl_checkpoint"

# Load the gated Llama purely from the shared local HF cache (see
# launch_fixed_pooling_ctc.sh / speechllm_fixed_pooling.yaml).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd "$RECIPE"

"$PY" train_speechllm.py hparams/speechllm_fixed_pooling.yaml \
    --blank_mode "$BLANK_MODE" \
    --train_splits "$TRAIN_SPLITS" \
    --number_of_epochs 1 \
    --output_folder "$OUTPUT" \
    --ssl_folder "$SSL_FOLDER" \
    "${EXTRA[@]}"
