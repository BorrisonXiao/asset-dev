#!/bin/bash
# CLSP port verification, part 1 of 2: INFERENCE.
#
# Re-decodes LibriSpeech test-clean with the completed skipjack flagship
# checkpoint (WavLM Transformer-AR local-64 + BiGRU residual pooling, seed 3407,
# epoch 12) and compares the result against the WER that run recorded on
# skipjack. A match proves the migrated stack is numerically intact end to end:
# venv/torch, WavLM-Large + Llama-3.2-1B weights, LoRA/proj/segmenter/normalize
# checkpoint tensors, LibriSpeech audio, char boundary targets, and the
# batch-invariant greedy decoder.
#
# Reference (skipjack, results/.../nll_mt/3407/wer_results/wer_test-clean.txt):
#   %WER 4.91 [ 2582 / 52576, 203 ins, 459 del, 1920 sub ]
#
# The source checkpoint is never written to: evaluation recovers through a
# read-only one-checkpoint symlink view, the same pattern as
# eval_last_segmenter_checkpoint.slurm.
#SBATCH --job-name=port_verify_infer
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
SEED=${SEED:-3407}
SRC_RUN=${SRC_RUN:-results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/${SEED}}
SRC_CKPT=${SRC_CKPT:-${SRC_RUN}/save/CKPT+2026-08-15+03-13-00+00}
OUT=${OUT:-results/clsp_port_verification/inference_seed${SEED}}
SPLIT=${SPLIT:-test-clean}

test -d "${SRC_CKPT}" || { echo "Missing source checkpoint: ${SRC_CKPT}" >&2; exit 1; }
for f in segmenter.ckpt llm.ckpt proj.ckpt normalize.ckpt CKPT.yaml; do
    test -s "${SRC_CKPT}/${f}" || { echo "Missing ${SRC_CKPT}/${f}" >&2; exit 1; }
done

VIEW=${OUT}/checkpoint_view
mkdir -p "${VIEW}" "${OUT}/wer_results"
ln -sfn "$(cd "$(dirname "${SRC_CKPT}")" && pwd)/$(basename "${SRC_CKPT}")" \
    "${VIEW}/$(basename "${SRC_CKPT}")"

echo "=== $(date) CLSP port inference verification ==="
echo "node:          $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "python:        ${PYTHON}"
echo "checkpoint:    ${SRC_CKPT}"
echo "split:         ${SPLIT}"
echo "output:        ${OUT}"
echo "skipjack reference for test-clean:"
cat "${SRC_RUN}/wer_results/wer_${SPLIT}.txt" 2>/dev/null | head -2 || true
echo

# Architecture flags must match the source run exactly (see
# submit_wavlm_transformer_ar_local64_bigru_best_char_decoder.sh).
"${PYTHON}" train_speechllm_with_segmenter.py hparams/speechllm_segmenter.yaml \
    --seed "${SEED}" \
    --eval_only True \
    --segmenter_mode joint --segmenter_backbone transformer_ar \
    --ssl_hub microsoft/wavlm-large \
    --ssl_folder /export/jsalt26/omnienc/users/cxiao/hf/hub \
    --ssl_feat_dims 1024 --segmenter_input_dim 1024 \
    --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4 \
    --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024 \
    --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096 \
    --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated \
    --segment_pooling bigru_residual --segment_pooling_hidden_dim 128 \
    --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0 \
    --rl_update_mode combined_on_policy --grpo_k 4 --max_decode_ratio 3.0 \
    --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0 \
    --number_of_epochs 12 \
    --output_folder "${OUT}" \
    --save_folder "${VIEW}" \
    --output_wer_folder "${OUT}/wer_results" \
    --train_log "${OUT}/eval_log.txt" \
    --skip_prep True --csv_folder "${SRC_RUN}" \
    --test_csv "[${SRC_RUN}/${SPLIT}.csv]" \
    --test_splits "['${SPLIT}']" \
    --test_batch_size 8

echo
echo "=== CLSP result ==="
head -2 "${OUT}/wer_results/wer_${SPLIT}.txt"
echo "=== skipjack reference ==="
head -2 "${SRC_RUN}/wer_results/wer_${SPLIT}.txt"
