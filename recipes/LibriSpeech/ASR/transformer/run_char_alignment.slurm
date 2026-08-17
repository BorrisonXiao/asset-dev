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
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${BOUNDARY_TARGET_DIR:?set BOUNDARY_TARGET_DIR to the char-alignment target directory}"

SEED=${SEED:-3407}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/${SEED}}
SSL_FOLDER=${SSL_FOLDER:-/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub}

ARGS=(
    hparams/speechllm_fixed_pooling.yaml
    --seed "$SEED"
    --boundary_source alignment
    --boundary_target_dir "$BOUNDARY_TARGET_DIR"
    --blank_mode keep
    --train_splits "['train-clean-100']"
    --number_of_epochs "${EPOCHS:-10}"
    --ssl_folder "$SSL_FOLDER"
    --output_folder "$OUTPUT_FOLDER"
)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    ARGS+=($EXTRA_ARGS)
fi

echo "=== $(date) char-alignment seed=$SEED on $(hostname) ==="
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
    train_speechllm.py "${ARGS[@]}"
