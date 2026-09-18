#!/bin/bash
# End-to-end smoke test of the multi-task recipe: exercises every phase
# transition, both task routes, per-task validation, and the sacreBLEU path on
# a few dozen optimizer steps. Uses the small CoVoST validation split as the
# training source so it does not wait on full-corpus preparation.
#SBATCH --job-name=mt_smoke
#SBATCH --time=2:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COVOST=/export/jsalt26/omnienc/users/cxiao/datasets/covost2_en_de/manifests
LS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
OUT=${OUT:-results/smoke_multitask/$$}

# Small enough to reach the unfreeze phase quickly, large enough that each
# phase actually runs and validates at least once.
BRIDGE=${BRIDGE:-6}
FILM=${FILM:-6}
STEPS=${STEPS:-18}

# NOTE: HyperPyYAML *merges* dict overrides into the YAML value rather than
# replacing it, so the test_source_specs override below adds its keys to the six
# already in the config. The smoke run therefore evaluates the full LibriSpeech
# test splits too -- slower, but it exercises the exact flagship test path.

echo "=== Multi-task smoke test ==="
echo "Node:   $(hostname)"
echo "Output: $OUT"
echo "Phases: bridge<$BRIDGE, film<$((BRIDGE+FILM)), unfreeze -> $STEPS"

"$PYTHON" train_speechllm_multitask.py hparams/speechllm_multitask.yaml \
  --output_folder "$OUT" \
  --bridge_optimizer_steps "$BRIDGE" \
  --film_optimizer_steps "$FILM" \
  --optimizer_step_limit "$STEPS" \
  --validation_interval_optimizer_steps 6 \
  --number_of_epochs 1 \
  --max_batch_length_train 60 \
  --max_batch_length_val 60 \
  --valid_subset_size 24 \
  --num_workers 2 \
  --covost_manifest_dir "$COVOST" \
  --librispeech_manifest_dir "$LS" \
  --train_sources "[{'csv': '$COVOST/covost_st_validation.csv', 'source_key': 'covost_st', 'limit': 400}, {'csv': '$COVOST/covost_asr_validation.csv', 'source_key': 'covost_asr', 'limit': 400}, {'csv': '$LS/dev-clean.csv', 'source_key': 'librispeech_asr', 'kind': 'librispeech', 'limit': 400}]" \
  --test_source_specs "{'covost_st_test': [{'csv': '$COVOST/covost_st_test.csv', 'source_key': 'covost_st', 'limit': 24}], 'covost_asr_test': [{'csv': '$COVOST/covost_asr_test.csv', 'source_key': 'covost_asr', 'limit': 24}], 'librispeech_dev_other': [{'csv': '$LS/dev-other.csv', 'source_key': 'librispeech_asr', 'kind': 'librispeech', 'limit': 24}]}" \
  --test_sources "{'covost_st_test': '$OUT/manifests/test_covost_st.csv', 'covost_asr_test': '$OUT/manifests/test_covost_asr.csv', 'librispeech_dev_other': '$OUT/manifests/test_ls_dev_other.csv'}" \
  --test_source_keys "{'covost_st_test': 'covost_st', 'covost_asr_test': 'covost_asr', 'librispeech_dev_other': 'librispeech_asr'}"

echo "=== Per-task metrics ==="
cat "$OUT/task_metrics.jsonl"
echo "=== Train log ==="
cat "$OUT/train_log.txt"
