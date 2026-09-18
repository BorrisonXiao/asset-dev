#!/bin/bash
# Pipeline gate: prove task_id=asr reproduces the single-task LS960 warm start
# before any multi-task training is launched.
#SBATCH --job-name=mt_equiv
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}

export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
LS_MANIFESTS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests

# The reference must be the ASR recipe configured exactly as the warm-started
# LS960 run was: WavLM-large encoder, Transformer-AR local-64 policy, BiGRU
# residual pooling.
ASR_OVERRIDES="
seed: 3408
segmenter_mode: joint
segmenter_backbone: transformer_ar
ssl_hub: microsoft/wavlm-large
ssl_folder: $WAVLM_CACHE
ssl_feat_dims: 1024
segmenter_input_dim: 1024
segment_pooling: bigru_residual
segment_pooling_hidden_dim: 128
segment_pooling_num_layers: 1
segmenter_ar_hidden_dim: 256
segmenter_ar_num_layers: 4
segmenter_ar_nhead: 4
segmenter_ar_ffn_dim: 1024
segmenter_ar_dropout: 0.0
segmenter_ar_history_window: 64
segmenter_ar_cache_mode: preallocated
segmenter_num_tasks: 1
boundary_source: none
warmstart_target: none
skip_prep: True
train_csv: $LS_MANIFESTS/dev-clean.csv
valid_csv: $LS_MANIFESTS/dev-clean.csv
test_csv: [$LS_MANIFESTS/dev-clean.csv]
"

echo "=== Multi-task warm-start equivalence gate ==="
echo "Node: $(hostname)"

exec "$PYTHON" verify_multitask_warm_start.py \
  --multitask-hparams hparams/speechllm_multitask.yaml \
  --asr-hparams hparams/speechllm_segmenter.yaml \
  --asr-overrides "$ASR_OVERRIDES" \
  --num-utterances "${NUM_UTTERANCES:-8}" \
  --tolerance "${TOLERANCE:-1e-4}"
