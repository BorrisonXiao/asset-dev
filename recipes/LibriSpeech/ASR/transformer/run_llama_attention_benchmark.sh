#!/bin/bash
set -euo pipefail

REPO=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
SNAPSHOT=/export/jsalt26/omnienc/users/cxiao/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
OUT_ROOT="$REPO/artifacts/benchmarks/llama_attention_a100"
FLASH_ATTN_ENV="$REPO/artifacts/benchmarks/flash_attn2_env"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$FLASH_ATTN_ENV${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_ROOT"
cd "$REPO"

exec "$PY" recipes/LibriSpeech/ASR/transformer/benchmark_llama_attention.py \
  --snapshot "$SNAPSHOT" \
  --output "$OUT_ROOT/results_${SLURM_JOB_ID:-local}.json"
