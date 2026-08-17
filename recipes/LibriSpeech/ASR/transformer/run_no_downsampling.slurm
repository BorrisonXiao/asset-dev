#!/bin/bash
# Native-rate (rho=1.0) SpeechLLM baseline on train-clean-100.
# The submitter supplies the required account/partition/comment/reservation flags
# and exports SEED. Optional OUTPUT_FOLDER, SSL_FOLDER, EPOCHS, and EXTRA_ARGS
# permit encoder-specific replications. All native-rate SSL frames reach the
# projection unchanged.
#SBATCH --job-name=no_ds
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

: "${SEED:?set SEED=3407|3408|3409}"

OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/speechllm_fixed_pooling/no_downsampling_tc100/${SEED}}
SSL_FOLDER=${SSL_FOLDER:-/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache}
if [[ -e "$OUTPUT_FOLDER" ]]; then
    echo "Refusing to reuse existing output path: $OUTPUT_FOLDER" >&2
    exit 1
fi

ARGS=(
    hparams/speechllm_fixed_pooling.yaml
    --seed "$SEED"
    --boundary_source none
    --train_splits "['train-clean-100']"
    --number_of_epochs "${EPOCHS:-10}"
    --ssl_folder "$SSL_FOLDER"
    --output_folder "$OUTPUT_FOLDER"
    --max_batch_length_train 50
    --max_batch_length_val 25
    --grad_accumulation_factor 20
    --test_batch_size 4
    --max_decode_ratio 0.75
)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    ARGS+=($EXTRA_ARGS)
fi

echo "=== $(date) no-downsampling rho=1.0 seed=$SEED on $(hostname) ==="
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
    train_speechllm.py "${ARGS[@]}"
