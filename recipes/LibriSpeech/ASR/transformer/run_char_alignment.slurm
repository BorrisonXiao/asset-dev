#!/bin/bash
# Transcript-informed character-CTC alignment baseline. The downstream model
# is trained exactly like the fixed-rate WavLM baselines, but mean-pooling
# intervals are supplied by per-frame character alignment boundaries.
#SBATCH --job-name=char_align
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${BOUNDARY_TARGET_DIR:?set BOUNDARY_TARGET_DIR to the char-alignment target directory}"

SEED=${SEED:-3407}
TRAIN_SPLITS=${TRAIN_SPLITS:-"['train-clean-100']"}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/${SEED}}
SSL_FOLDER=${SSL_FOLDER:-/export/jsalt26/omnienc/users/cxiao/hf/hub}

ARGS=(
    hparams/speechllm_fixed_pooling.yaml
    --seed "$SEED"
    --boundary_source alignment
    --boundary_target_dir "$BOUNDARY_TARGET_DIR"
    --blank_mode keep
    --train_splits "$TRAIN_SPLITS"
    --number_of_epochs "${EPOCHS:-10}"
    --ssl_folder "$SSL_FOLDER"
    --output_folder "$OUTPUT_FOLDER"
)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    ARGS+=($EXTRA_ARGS)
fi

echo "=== $(date) char-alignment seed=$SEED on $(hostname) ==="
/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    train_speechllm.py "${ARGS[@]}"
