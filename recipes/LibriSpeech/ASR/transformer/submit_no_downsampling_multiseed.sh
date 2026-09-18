#!/bin/bash
# Submit the native-rate (rho=1.0) three-seed baseline.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs

SEEDS=(3407 3408 3409)
OUT_ROOT=results/speechllm_fixed_pooling/no_downsampling_tc100
COMMON=(
    --account=highprio
    --partition=gpu-a100
    --comment=accept_cost
)

for seed in "${SEEDS[@]}"; do
    if [[ -e "$OUT_ROOT/$seed" ]]; then
        echo "Refusing to submit over existing output: $OUT_ROOT/$seed" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    jid=$(sbatch "${COMMON[@]}" \
        --job-name="nods_${seed}" \
        --parsable \
        --export="ALL,SEED=$seed" \
        run_no_downsampling.slurm)
    echo "no-downsampling seed=$seed -> $jid ($OUT_ROOT/$seed)"
done
