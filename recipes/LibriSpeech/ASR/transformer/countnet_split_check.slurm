#!/bin/bash
# Is the speaker-count VALID split genuinely easier than TEST, or is the
# valid/test gap ours? Score the validated CountNet CRNN port on both splits.
# A model-independent reference settles it: if CountNet shows the same gap the
# splits differ; if it does not, the gap is in our eval path.
#SBATCH --job-name=countnet_split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
P=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
S=/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests_lcfull
OUT=artifacts/segmenter/countnet_split_check
mkdir -p "$OUT"
for split in valid test; do
  echo "=== CountNet on speaker_count_${split} ==="
  $P countnet_reference.py --manifest "$S/speaker_count_${split}.csv" \
      --device cpu --batch-size 16 --out "$OUT/${split}.json"
done
echo "=== summary ==="
$P - <<'PY'
import json
for s in ("valid","test"):
    d=json.load(open(f"artifacts/segmenter/countnet_split_check/{s}.json"))
    print(f"  {s:5} n={d['n']:4} accuracy={d['accuracy']:.4f} mae={d['mae']:.4f}")
PY
