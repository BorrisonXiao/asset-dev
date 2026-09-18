#!/bin/bash
# One task-conditioned multi-task run (English ASR + En->De speech translation)
# over a shared hard segmenter, warm started from the LS960 Transformer-AR +
# BiGRU checkpoint. Submission identity/resources come from the submit script.
#SBATCH --job-name=mt_st
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
HPARAMS=${HPARAMS:-hparams/speechllm_multitask.yaml}

# The Llama base is gated for this account's own token; the shared cache is the
# only usable source, so the HF client must stay offline.
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== Task-conditioned multi-task run ==="
echo "Node:        $(hostname)"
echo "Job:         ${SLURM_JOB_ID:-none}"
echo "Hparams:     $HPARAMS"
echo "Extra args:  ${EXTRA_ARGS:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$PYTHON" train_speechllm_multitask.py "$HPARAMS" ${EXTRA_ARGS:-}
