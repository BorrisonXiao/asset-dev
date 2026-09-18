#!/bin/bash
# CLSP port verification, part 2 of 2: TRAINING.
#
# Runs the flagship WavLM Transformer-AR local-64 + BiGRU joint-RL configuration
# for a handful of optimizer steps, warm-started exactly the way
# submit_wavlm_transformer_ar_local64_bigru_best_char_decoder.sh warm-starts it:
# the completed local-64 char cold-start segmenter plus the best char-alignment
# decoder. This exercises the parts inference cannot reach -- librispeech_prepare
# / CSV generation, the char boundary-target loader, dynamic batching, the
# cold-start and decoder checkpoint loaders, combined on-policy GRPO rollouts,
# the band rate objective, the optimizer, and checkpoint saving.
#
# It is a wiring test, not a result: 12 optimizer steps say nothing about WER.
#SBATCH --job-name=port_verify_train
#SBATCH --time=04:00:00
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
SEED=${SEED:-9901}
OUT=${OUT:-results/clsp_port_verification/training_seed${SEED}}
STEPS=${STEPS:-12}

CONTROL_ROOT=results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_best_char_decoder
COLD_DIR=$(pwd)/${CONTROL_ROOT}/coldstart/3407/save
DECODER_CKPT=$(pwd)/results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save/CKPT+2026-08-09+17-11-52+00
LS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests

compgen -G "${COLD_DIR}/CKPT*/segmenter.ckpt" >/dev/null \
    || { echo "No cold-start segmenter under ${COLD_DIR}" >&2; exit 1; }
for f in llm.ckpt proj.ckpt normalize.ckpt; do
    test -s "${DECODER_CKPT}/${f}" || { echo "Missing ${DECODER_CKPT}/${f}" >&2; exit 1; }
done

rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=== $(date) CLSP port training verification ==="
echo "node:        $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "python:      ${PYTHON}"
echo "cold start:  ${COLD_DIR}"
echo "decoder:     ${DECODER_CKPT}"
echo "steps:       ${STEPS}"
echo

"${PYTHON}" train_speechllm_with_segmenter.py hparams/speechllm_segmenter.yaml \
    --seed "${SEED}" \
    --experiment_name clsp_port_verification \
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
    --coldstart_ckpt_dir "${COLD_DIR}" \
    --decoder_init_ckpt_dir "${DECODER_CKPT}" \
    --segmenter_reward nll --freeze_decoder_in_joint False \
    --rl_update_mode combined_on_policy --initial_lr 0.0002 \
    --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005 \
    --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0 \
    --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0 \
    --output_folder "${OUT}" \
    --skip_prep True --csv_folder "${LS}" \
    --train_csv "${LS}/dev-clean.csv" --valid_csv "${LS}/dev-clean.csv" \
    --test_csv "[${LS}/test-clean.csv]" --test_splits "['test-clean']" \
    --number_of_epochs 1 --optimizer_step_limit "${STEPS}" \
    --warmup_optimizer_steps 4 --validation_interval_optimizer_steps 6 \
    --max_batch_length_train 60 --max_batch_length_val 60 \
    --num_workers 2 --test_batch_size 4

echo
echo "=== train log ==="
cat "${OUT}/train_log.txt"
echo "=== saved checkpoints ==="
ls -la "${OUT}/save"
echo "=== WER (12 steps only -- wiring check, not a result) ==="
head -2 "${OUT}/wer_results/wer_test-clean.txt" 2>/dev/null || true
