#!/bin/bash
# Cross-domain control: SeamlessM4T-v2 on the same LibriSpeech splits our
# multi-task model was scored on. Without this, a large CoVoST gap cannot be
# told apart from Seamless simply being a stronger recognizer everywhere.
#SBATCH --job-name=seamless_ls
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=results/speechllm_multitask_ls960_covost_en_de/3408
CACHE=/export/jsalt26/omnienc/users/cxiao/hf/seamless

for split in test_ls_test_other test_ls_test_clean test_ls_dev_other; do
  echo "=== SeamlessM4T-v2 on $split ==="
  "$PYTHON" eval_seamless_covost.py \
    --model-cache "$CACHE" \
    --asr-manifest "$RUN/manifests/$split.csv" \
    --out "artifacts/benchmarks/seamless_m4t_v2_${split}.json" \
    --hypothesis-dir "artifacts/benchmarks/seamless_hypotheses_${split}" \
    --batch-size "${BATCH_SIZE:-16}"
done
