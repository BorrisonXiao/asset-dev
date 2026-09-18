#!/bin/bash
# Evaluate trained checkpoints on the speaker-count transfer pair: our synthetic
# mixtures and Dynamic-SUPERB's LibriCount folds, scored in one job so the
# checkpoint is held fixed.
#
# Both checkpoints matter and for different reasons: step 1500 is what ASR-WER
# selection actually reported (policy still frozen), step 15000 has a trained
# segmenter. Comparing them says whether policy training helped or hurt transfer.
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

ARM_ROOT=${ARM_ROOT:-results/speechllm_multitask_ls960_nonasr_rateaux}
SEEDS=${SEEDS:-"3407 3408 3409"}
STEPS=${STEPS:-"1500 15000"}
DRY_RUN=${DRY_RUN:-0}
die() { echo "$*" >&2; exit 1; }

[[ -s /export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests/libricount_test_0to4.csv ]] \
  || die "LibriCount manifest missing; run libricount_prepare.py first"

for seed in $SEEDS; do
  OUT="$ARM_ROOT/$seed"
  [[ -d "$OUT/save" ]] || die "No checkpoints under $OUT/save"
  for step in $STEPS; do
    grep -lq "optimizer_step: $step" "$OUT"/save/CKPT*/CKPT.yaml 2>/dev/null \
      || die "seed $seed has no checkpoint at step $step"
    ARGS="--output_folder $OUT --seed $seed --eval_checkpoint_step $step"
    if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: seed=$seed step=$step $ARGS"; continue; fi
    jid=$(EXTRA_ARGS="$ARGS" HPARAMS=hparams/speechllm_multitask_nonasr_eval.yaml \
      sbatch --account=highprio --comment=accept_cost --partition=gpu-a100 \
        --gpus=1 --cpus-per-task=8 --mem=48G --time=03:00:00 \
        --job-name="lceval_${seed}_${step}" --parsable --export=ALL run_multitask.slurm)
    echo "submitted seed=$seed step=$step job=$jid"
  done
done
