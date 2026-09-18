#!/bin/bash
#SBATCH --job-name=tar_phone_agree
#SBATCH --account=highprio
#SBATCH --comment=accept_cost
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs

run_root=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt
manifest=results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407/dev-clean.csv
output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/transformer_ar_bigru_phone_agreement_dev_clean.json

/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
  audit_transformer_ar_phone_agreement.py \
  --csv "$manifest" \
  --data-root /export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech \
  --phone-dir /export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/phone \
  --run-root "$run_root" \
  --ssl-cache "$HF_HUB_CACHE" \
  --batch-size 8 \
  --device cuda \
  --precision bf16 \
  --output "$output"
