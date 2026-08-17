#!/bin/bash
# Submit the v1 dynamic-segmenter pilot (train-clean-100):
#   - fixed-rate baselines k in {3,4,5,6,8}                  (WER-vs-tokens frontier)
#   - cold-start segmenter (cnn, transformer)                (init + B1 char-oracle)
#   - joint GRPO RL (cnn, transformer), each afterok its cold-start
# Gumbel/CIF differentiable arm deferred (see plans/dynamic_segmenter_rl_v1.md).
set -euo pipefail
cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
mkdir -p slurm_logs
SEED=3407
COMMON=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100
        --reservation="JSALT 2026" --exclude=ga129)

echo "### fixed-rate baselines ###"
for K in 3 4 5 6 8; do
    jid=$(sbatch "${COMMON[@]}" --job-name="fr_k${K}" --parsable \
        --export=ALL,FIXED_K=${K},EPOCHS=10 run_fixed_rate.slurm)
    echo "  fixed_rate k=$K -> job $jid"
done

echo "### cold-start (segmenter init + B1) ###"
declare -A CS_ID
for BB in cnn transformer; do
    jid=$(sbatch "${COMMON[@]}" --job-name="cs_${BB}" --parsable \
        --export=ALL,MODE=coldstart,BACKBONE=${BB},EPOCHS=10 run_segmenter.slurm)
    CS_ID[$BB]=$jid
    echo "  coldstart $BB -> job $jid"
done

echo "### joint GRPO RL (afterok cold-start) ###"
for BB in cnn transformer; do
    CSDIR="$(pwd)/results/speechllm_segmenter/coldstart/${BB}/${SEED}/save"
    jid=$(sbatch "${COMMON[@]}" --job-name="joint_${BB}" --parsable \
        --dependency=afterok:${CS_ID[$BB]} \
        --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=2,COLDSTART_DIR=${CSDIR} \
        run_segmenter.slurm)
    echo "  joint $BB -> job $jid (after ${CS_ID[$BB]})"
done
echo "### submitted. watch with: squeue -u \$USER ###"
