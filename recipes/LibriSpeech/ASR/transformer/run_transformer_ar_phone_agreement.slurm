#!/bin/bash
#SBATCH --job-name=tar_phone_agree
#SBATCH --account=jsalt2026-lgarci27
#SBATCH --comment=accept_cost
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs

run_root=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt
manifest=results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3407/dev-clean.csv
output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/artifacts/segmenter/transformer_ar_bigru_phone_agreement_dev_clean.json

/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python \
  audit_transformer_ar_phone_agreement.py \
  --csv "$manifest" \
  --data-root /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets/LibriSpeech \
  --phone-dir /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets/wavlm_boundaries/phone \
  --run-root "$run_root" \
  --ssl-cache "$HF_HUB_CACHE" \
  --batch-size 8 \
  --device cuda \
  --precision bf16 \
  --output "$output"
