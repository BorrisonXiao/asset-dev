#!/bin/bash
# Measure BPE-500 CTC collapse frequencies on a transcript-free dev-clean sample.
# Account/partition/comment/reservation/exclude are supplied by the submitter.
#SBATCH --job-name=tc100_bpe500pilot
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
MODEL=$ROOT/artifacts/models/icefall_conformer_ctc_bpe500/cpu_jit.pt
OUT=/export/jsalt26/omnienc/users/cxiao/boundary_targets/ctc_bpe500_state_runs_pilot
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export PYTHONUNBUFFERED=1

cd "$RECIPE"
echo "Dataset: first 256 sorted dev-clean utterances; frequency pilot only"
echo "Boundary source: frozen Icefall Conformer-CTC BPE-500; no transcript"
echo "Boundary rule: every argmax state run, including blank"
echo "Output: $OUT (expected under 10 MB)"

"$PY" bpe_ctc_posterior_collapse.py \
  --audio_root "$DATA" \
  --out_root "$OUT" \
  --model_path "$MODEL" \
  --splits dev-clean \
  --boundary_rule state_runs \
  --device cuda \
  --max_utterances 256
