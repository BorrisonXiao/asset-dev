#!/bin/bash
# 3 reward / multi-task ablations (CNN, warmup 6, band rate objective, entropy dropped),
# reusing the finished cold-start CNN segmenter. Compares:
#   (1 vs 2) single-task (frozen decoder) vs multi-task (CE-co-trained decoder), at NLL reward
#   (2 vs 3) NLL reward vs CER reward (free-running decode), at multi-task
set -euo pipefail
cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
SEED=3407; BB=cnn; B=results/speechllm_segmenter
CS="$(pwd)/$B/coldstart/$BB/$SEED/save"
COMMON=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100 --reservation="JSALT 2026" --exclude=ga129)

j1=$(sbatch "${COMMON[@]}" --job-name=abl_nll_frozen --parsable \
  --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=6,COLDSTART_DIR=${CS},EXTRA_ARGS="--segmenter_reward nll --freeze_decoder_in_joint True --output_folder $B/abl/nll_frozen/${BB}/${SEED}" \
  run_segmenter.slurm)
echo "  1) NLL, frozen decoder (single-task) -> $j1"

j2=$(sbatch "${COMMON[@]}" --job-name=abl_nll_mt --parsable \
  --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=6,COLDSTART_DIR=${CS},EXTRA_ARGS="--segmenter_reward nll --freeze_decoder_in_joint False --output_folder $B/abl/nll_mt/${BB}/${SEED}" \
  run_segmenter.slurm)
echo "  2) NLL, multi-task (co-trained dec)  -> $j2"

j3=$(sbatch "${COMMON[@]}" --job-name=abl_cer_mt --parsable \
  --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=6,COLDSTART_DIR=${CS},EXTRA_ARGS="--segmenter_reward cer --freeze_decoder_in_joint False --output_folder $B/abl/cer_mt/${BB}/${SEED}" \
  run_segmenter.slurm)
echo "  3) CER, multi-task (co-trained dec)  -> $j3"
echo "### 3 ablations submitted (warmup 6 / joint 4, band [0.1,0.3], entropy off) ###"
