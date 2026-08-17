#!/bin/bash
# Re-run the joint phase with the collapse fix, comparing two rate objectives x two
# backbones (2x2), reusing the finished cold-starts. Only the rate objective differs
# between A and B; entropy schedule (0.05->0.01 floor) and LR are identical.
#   A: two-sided pin  (rho_bar - rho*)^2                          -> can't collapse
#   B: one-sided cap + token-tax + soft floor at rho_floor=0.08   -> rate-distortion, floored
set -euo pipefail
cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs
SEED=3407
B=results/speechllm_segmenter
COMMON=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100
        --reservation="JSALT 2026" --exclude=ga129)

for BB in cnn transformer; do
    CS="$(pwd)/$B/coldstart/$BB/$SEED/save"
    jidA=$(sbatch "${COMMON[@]}" --job-name="jA_$BB" --parsable \
        --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=2,COLDSTART_DIR=${CS},EXTRA_ARGS="--rate_two_sided True --output_folder $B/jointA/${BB}/${SEED}" \
        run_segmenter.slurm)
    echo "  jointA (two-sided pin) $BB -> job $jidA"
    jidB=$(sbatch "${COMMON[@]}" --job-name="jB_$BB" --parsable \
        --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=2,COLDSTART_DIR=${CS},EXTRA_ARGS="--rate_two_sided False --lambda_floor 1.0 --output_folder $B/jointB/${BB}/${SEED}" \
        run_segmenter.slurm)
    echo "  jointB (cap+tax+floor) $BB -> job $jidB"
done
echo "### submitted 4 joint runs. Monitor rho at epoch 3-4 (collapse point) ###"
