#!/bin/bash
# Topline reference: SeamlessM4T-v2-large on the exact CoVoST test utterances
# our multi-task run was scored on, with the same metric implementations.
#SBATCH --job-name=seamless_eval
#SBATCH --time=12:00:00
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
OUT=${OUT:-artifacts/benchmarks/seamless_m4t_v2_covost_en_de.json}
LIMIT_ARG=""
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit $LIMIT"

echo "=== SeamlessM4T-v2-large on CoVoST En / En-De ==="
echo "Node: $(hostname)"
echo "Same manifests our model was scored on:"
wc -l "$RUN"/manifests/test_covost_{asr,st}.csv

exec "$PYTHON" eval_seamless_covost.py \
  --model-cache "$CACHE" \
  --asr-manifest "$RUN/manifests/test_covost_asr.csv" \
  --st-manifest "$RUN/manifests/test_covost_st.csv" \
  --out "$OUT" \
  --hypothesis-dir artifacts/benchmarks/seamless_hypotheses \
  --batch-size "${BATCH_SIZE:-8}" $LIMIT_ARG
