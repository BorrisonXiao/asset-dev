#!/bin/bash
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs results/speechllm_segmenter_wavlm/transformer_ar_combined_rollout_validation

common=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --reservation='JSALT 2026'
  --exclude=ga132
  --parsable
  run_transformer_ar_combined_full_loop_bench.slurm
)

smoke=$(sbatch \
  --job-name=tar_comb_smoke \
  --export=ALL,VARIANT=combined,RUN_SCOPE=smoke \
  "${common[@]}")
echo "combined production smoke -> $smoke"

for variant in baseline combined; do
  job=$(sbatch \
    --dependency="afterok:$smoke" \
    --job-name="tar_full_${variant}" \
    --export="ALL,VARIANT=$variant,RUN_SCOPE=full" \
    "${common[@]}")
  echo "full $variant -> $job (after smoke $smoke)"
done
