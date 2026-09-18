#!/bin/bash
# Short train-clean-100 phone-CTC probe on frozen WavLM-Large.
# Identity flags are supplied by submit_wavlm_phone_ctc_bigru_100h.sh.
#SBATCH --job-name=tc100_wlmphctc
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
ALIGN=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech_alignments
CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
OUTPUT=$ROOT/artifacts/models/wavlm_phone_ctc_tc100_seed3407
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$CACHE

cd "$RECIPE"
echo "Dataset: train-clean-100 (100h baseline; not LS960); validation: dev-clean"
echo "Model: frozen microsoft/wavlm-large + trainable phone-CTC MLP head"
echo "Targets: ordered MFA phones with stress stripped; alignment times discarded"
echo "Epochs/checkpoints: 3/3; selection: dev-clean PER"
echo "Output: $OUTPUT (estimated under 100 MB)"

"$PY" wavlm_phone_ctc.py train \
  --audio_root "$DATA" \
  --alignment_root "$ALIGN" \
  --output_dir "$OUTPUT" \
  --cache_dir "$CACHE" \
  --train_split train-clean-100 \
  --valid_split dev-clean \
  --epochs 3 \
  --max_padded_seconds 120 \
  --valid_max_padded_seconds 160 \
  --max_batch_size 16 \
  --num_workers 6 \
  --seed 3407
