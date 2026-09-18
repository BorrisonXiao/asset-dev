#!/bin/bash
# Is the frozen WavLM encoder the CoVoST bottleneck?
# Frozen WavLM-large + a small character-CTC head, trained on matched hours of
# LibriSpeech and of CoVoST, each evaluated on both. No segmenter, no LLM.
#SBATCH --job-name=enc_probe
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

PY=${PY:-/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python}
CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$CACHE

COVOST=/export/jsalt26/omnienc/users/cxiao/datasets/covost2_en_de/manifests
LSMAN=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
WORK=artifacts/pilots/frozen_encoder_domain_probe
LSROOT=/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
mkdir -p "$WORK"

# LibriSpeech in the multi-task schema, so both corpora load through one path.
# Written under a lock: two parallel probe jobs would otherwise race on the
# same output files and one could read a half-written manifest.
exec 9>"$WORK/.manifest.lock"
flock 9
"$PY" - <<PYEOF
from multitask_data import build_multitask_manifest
for name, src in (("ls_train", "train.csv"), ("ls_eval", "test-other.csv")):
    build_multitask_manifest(
        [{"csv": f"$LSMAN/{src}", "source_key": "librispeech_asr",
          "kind": "librispeech"}],
        f"$WORK/{name}.csv",
        data_folder="$LSROOT",
    )
PYEOF
flock -u 9

echo "=== Frozen-encoder domain probe ==="
echo "Node: $(hostname)"
echo "Encoder: frozen microsoft/wavlm-large; head: 2-layer char-CTC (~0.8M params)"
echo "Matched budget: ${TRAIN_HOURS:-100} h per corpus, ${STEPS:-4000} steps each"

exec "$PY" probe_frozen_encoder_domains.py \
  --librispeech-train "$WORK/ls_train.csv" \
  --librispeech-eval  "$WORK/ls_eval.csv" \
  --covost-train      "$COVOST/covost_asr_train.csv" \
  --covost-eval       "$COVOST/covost_asr_test.csv" \
  --cache-dir "$CACHE" \
  --out "${OUT:-artifacts/benchmarks/frozen_encoder_domain_probe.json}" \
  --train-hours "${TRAIN_HOURS:-100}" \
  --steps "${STEPS:-4000}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --eval-batches "${EVAL_BATCHES:-200}" \
  --train-domains ${TRAIN_DOMAINS:-librispeech covost}
