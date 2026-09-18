#!/bin/bash
# Submit matched real-batch Transformer-AR mock-training timing jobs.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs results/speechllm_segmenter_wavlm/transformer_ar_training_bench

for variant in current prealloc local64; do
  job=$(sbatch \
    --account=highprio \
    --comment=accept_cost \
    --partition=gpu-a100 \
 \
 \
    --job-name="tar_tr_${variant}" \
    --export="ALL,VARIANT=$variant" \
    --parsable \
    run_transformer_ar_training_bench.slurm)
  echo "$variant -> $job"
done
