#!/bin/bash
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench

for variant in dense64_ref flex64_ref dense64_reward dense64_group flex64_reward flex64_group; do
  job=$(sbatch \
    --account=jsalt2026-lgarci27 \
    --comment=accept_cost \
    --partition=a100 \
    --reservation='JSALT 2026' \
    --exclude=ga129,ga132 \
    --job-name="tar_${variant}" \
    --export="ALL,VARIANT=$variant" \
    --parsable \
    run_transformer_ar_flex_training_bench.slurm)
  echo "$variant -> $job"
done
