#!/bin/bash
# Evaluate the maximum-epoch checkpoint of an existing joint-segmenter run.
#
# Required environment variables:
#   RUN_DIR    Existing seed directory containing save/CKPT*/CKPT.yaml.
#   SEED       Experiment seed, used to reconstruct the model deterministically.
#   EXTRA_ARGS Architecture overrides used by the source training job.
#
# The source checkpoint is never copied or modified. A one-checkpoint symlink view
# makes the recipe's existing min-WER evaluation path unambiguously recover the
# requested maximum-epoch checkpoint.
#SBATCH --job-name=seg_last_eval
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

: "${RUN_DIR:?set RUN_DIR to the existing seed directory}"
: "${SEED:?set SEED to the source run seed}"
: "${EXTRA_ARGS:?set EXTRA_ARGS to the source model architecture overrides}"

source_save_dir="${RUN_DIR}/save"
if [[ ! -d "${source_save_dir}" ]]; then
    echo "Missing source checkpoint directory: ${source_save_dir}" >&2
    exit 1
fi

last_ckpt=""
last_epoch=-1
while IFS= read -r ckpt_dir; do
    meta="${ckpt_dir}/CKPT.yaml"
    [[ -f "${meta}" ]] || continue
    epoch=$(awk '$1 == "epoch:" {print $2; exit}' "${meta}")
    [[ "${epoch}" =~ ^[0-9]+$ ]] || continue
    if (( epoch > last_epoch )); then
        last_epoch=${epoch}
        last_ckpt=${ckpt_dir}
    fi
done < <(find "${source_save_dir}" -mindepth 1 -maxdepth 1 -type d -name 'CKPT*' -print)

if [[ -z "${last_ckpt}" ]]; then
    echo "No epoch checkpoint found under ${source_save_dir}" >&2
    exit 1
fi

eval_root="${RUN_DIR}/last_checkpoint_eval"
view_dir="${eval_root}/checkpoint_view_epoch_${last_epoch}"
wer_dir="${RUN_DIR}/wer_results_last_checkpoint_epoch_${last_epoch}"
mkdir -p "${view_dir}" "${wer_dir}"

ckpt_link="${view_dir}/$(basename "${last_ckpt}")"
if [[ -L "${ckpt_link}" ]]; then
    if [[ "$(readlink -f "${ckpt_link}")" != "$(readlink -f "${last_ckpt}")" ]]; then
        echo "Existing checkpoint link targets a different directory: ${ckpt_link}" >&2
        exit 1
    fi
elif [[ -e "${ckpt_link}" ]]; then
    echo "Checkpoint view path exists and is not a symlink: ${ckpt_link}" >&2
    exit 1
else
    ln -s "${last_ckpt}" "${ckpt_link}"
fi

args=(
    hparams/speechllm_segmenter.yaml
    --segmenter_mode joint
    --segmenter_backbone transformer_ar
    --seed "${SEED}"
    --number_of_epochs "${last_epoch}"
    --output_folder "${RUN_DIR}"
    --save_folder "${view_dir}"
    --output_wer_folder "${wer_dir}"
    --train_log "${eval_root}/eval_epoch_${last_epoch}.log"
    --stage_timing_file "${eval_root}/timing_epoch_${last_epoch}.jsonl"
    --skip_prep True
)

# Match the source run's model construction exactly.
args+=(${EXTRA_ARGS})

echo "=== $(date) last-checkpoint evaluation seed=${SEED} epoch=${last_epoch} ==="
echo "source checkpoint: ${last_ckpt}"
echo "WER output: ${wer_dir}"

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
    train_speechllm_with_segmenter.py "${args[@]}"
