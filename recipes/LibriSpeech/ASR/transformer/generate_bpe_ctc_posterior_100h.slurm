#!/bin/bash
# Audio-only BPE-CTC state-run preprocessing for the 100h ASR baseline.
# Account/partition/comment/reservation/exclude are supplied by the submitter.
#SBATCH --job-name=tc100_bpe500gen
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%A_%a.out

set -euo pipefail

ROOT=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
RECIPE=$ROOT/recipes/LibriSpeech/ASR/transformer
DATA=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
MODEL=$ROOT/artifacts/models/icefall_conformer_ctc_bpe500/cpu_jit.pt
OUT=/export/jsalt26/omnienc/users/cxiao/boundary_targets/ctc_bpe500_state_runs
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python

export PYTHONUNBUFFERED=1

cd "$RECIPE"
echo "Dataset: train-clean-100 plus dev/test evaluation splits (100h baseline; not LS960)"
echo "Splits: train-clean-100 dev-clean dev-other test-clean test-other"
echo "Boundary source: frozen Icefall Conformer-CTC BPE-500; no transcript"
echo "Boundary rule: every argmax state run, including blank"
echo "Output: $OUT"

"$PY" bpe_ctc_posterior_collapse.py \
  --audio_root "$DATA" \
  --out_root "$OUT" \
  --model_path "$MODEL" \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --boundary_rule state_runs \
  --device cuda \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --num_shards 8
