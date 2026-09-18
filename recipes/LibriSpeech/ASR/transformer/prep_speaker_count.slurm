#!/bin/bash
# Speaker-count mixture generation. CPU-only but heavy: it reads and runs VAD
# over ~40k source crops to pick mostly-voiced windows, so it belongs on a
# compute node, not the shared login node.
#SBATCH --job-name=prep_spkcount
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/slurm_logs/%x_%j.out

set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
PYTHON=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
exec "$PYTHON" speaker_count_prepare.py "$@"
