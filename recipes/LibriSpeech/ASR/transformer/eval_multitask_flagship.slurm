#!/bin/bash
# Final evaluation for a completed multi-task run, recovering the best-WER
# checkpoint. Used after job 171461 finished all 15000 training steps but died
# in final evaluation on two over-long CoVoST test clips; the manifest builder
# now filters utterances past the boundary policy's positional limit.
#SBATCH --job-name=mt_eval
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PYTHON=${PYTHON:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=${OUT:-results/speechllm_multitask_ls960_covost_en_de/3408}
COVOST=/export/jsalt26/omnienc/users/cxiao/datasets/covost2_en_de/manifests
LS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests

echo "=== Multi-task final evaluation (eval_only) ==="
echo "Node:   $(hostname)"
echo "Run:    $OUT"
echo "Checkpoints available:"
for d in "$OUT"/save/CKPT*; do
  echo "  $(basename "$d"): $(grep -E '^(WER|optimizer_step)' "$d/CKPT.yaml" 2>/dev/null | tr '\n' ' ')"
done

exec "$PYTHON" train_speechllm_multitask.py hparams/speechllm_multitask.yaml \
  --seed 3408 --output_folder "$OUT" --eval_only True \
  --covost_manifest_dir "$COVOST" --librispeech_manifest_dir "$LS" \
  --optimizer_step_limit 15000 \
  --bridge_optimizer_steps 3000 --film_optimizer_steps 6000
