#!/bin/bash
# Submit matched real-batch Transformer-AR mock-training timing jobs.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs results/speechllm_segmenter_wavlm/transformer_ar_training_bench

for variant in current prealloc local64; do
  job=$(sbatch \
    --account=jsalt2026-lgarci27 \
    --comment=accept_cost \
    --partition=a100 \
    --reservation='JSALT 2026' \
    --exclude=ga129,ga132 \
    --job-name="tar_tr_${variant}" \
    --export="ALL,VARIANT=$variant" \
    --parsable \
    run_transformer_ar_training_bench.slurm)
  echo "$variant -> $job"
done
