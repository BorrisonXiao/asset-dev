#!/bin/bash
#SBATCH --job-name=speechllm_align_ctc
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/results/speechllm_fixed_pooling/alignment/3407/slurm_%j.out
#
# Real (non-debug) training run of speechllm_fixed_pooling.yaml with
# boundary_source: alignment / boundary_target_dir pointed at the CTC-derived
# word_ctc boundaries. Job-shape only here -- account/partition/comment/
# reservation/exclude are passed on the sbatch command line (see submit
# instructions), since #SBATCH lines can't reliably carry quoted
# multi-word values like the reservation name.
#
# Resume: SpeechBrain's Checkpointer already saves under
# results/speechllm_fixed_pooling/alignment/3407/save every
# ckpt_interval_minutes (15) and auto-resumes from the latest checkpoint on
# next launch, so a repeat `sbatch` of this same script after a walltime
# kill continues rather than restarting.

set -euo pipefail

# meta-llama/Llama-3.2-1B-Instruct is gated and this account's HF token isn't
# authorized for it; load it from the shared team cache instead and never
# touch the network for it (see hparams/speechllm_fixed_pooling.yaml's
# llm_save_path comment).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python train_speechllm.py \
    hparams/speechllm_fixed_pooling.yaml
