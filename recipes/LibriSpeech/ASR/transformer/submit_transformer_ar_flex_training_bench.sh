#!/bin/bash
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench

for variant in dense64_ref flex64_ref dense64_reward dense64_group flex64_reward flex64_group; do
  job=$(sbatch \
    --account=highprio \
    --comment=accept_cost \
    --partition=gpu-a100 \
 \
 \
    --job-name="tar_${variant}" \
    --export="ALL,VARIANT=$variant" \
    --parsable \
    run_transformer_ar_flex_training_bench.slurm)
  echo "$variant -> $job"
done
