#!/bin/bash
# Relaunch the CER-reward multi-task ablation with the OPTIMIZED GRPO loop:
#   * batched K-sample decode (one wide generation instead of K sequential decodes)
#   * off-policy sample reuse (num_iterations=3, PPO-clipped ratio) -> amortize the decode
#   * max_decode_ratio 3->2
# Resumes the finished 6-epoch decoder-warmup of the cancelled cer_mt run (epoch-6 ckpt),
# so it runs only the 4 joint epochs (7-10) -- same warmup as the nll ablations, but skips
# re-doing the warmup wall-time. The aborted intra-epoch-7 (old slow loop) ckpt is moved
# aside so recovery starts cleanly from end-of-warmup.
set -euo pipefail
cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer
SEED=3407; BB=cnn; B=results/speechllm_segmenter
CS="$(pwd)/$B/coldstart/$BB/$SEED/save"
SAVE="$B/abl/cer_mt/$BB/$SEED/save"

# Move the aborted intra-epoch-7 checkpoint aside (non-destructive) so the checkpointer
# recovers from the clean end-of-epoch-6 warmup checkpoint.
for c in "$SAVE"/CKPT*; do
  if grep -q "brain_intra_epoch_ckpt: true" "$c/CKPT.yaml" 2>/dev/null; then
    mv "$c" "$SAVE/_ABORTED_$(basename "$c")"
    echo "moved aborted intra-epoch ckpt aside: $c"
  fi
done

COMMON=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100 --reservation="JSALT 2026" --exclude=ga129)
j=$(sbatch "${COMMON[@]}" --job-name=abl_cer_mt_opt --parsable \
  --export=ALL,MODE=joint,BACKBONE=${BB},EPOCHS=10,WARMUP_EPOCHS=6,COLDSTART_DIR=${CS},EXTRA_ARGS="--segmenter_reward cer --freeze_decoder_in_joint False --output_folder $B/abl/cer_mt/${BB}/${SEED}" \
  run_segmenter.slurm)
echo "CER (optimized, resume from warmup) -> $j"
echo "$j" > /tmp/cer_opt_jid.txt
