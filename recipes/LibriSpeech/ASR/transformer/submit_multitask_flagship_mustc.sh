#!/bin/bash
# Task-conditioned multi-task flagship on MuST-C v1 En-De (TED) + LS960 replay.
#
# MuST-C is the corpus the plan originally specified. FBK withdrew public
# distribution, so this copy came from the HLTCOE corpora mirror. TED speech is
# far closer to WavLM's pretraining distribution than Common Voice, which the
# frozen-encoder probe identified as the cause of the CoVoST degradation
# (in-domain CER 5.98 LibriSpeech vs 14.69 CoVoST, matched hours).
#
# Set DEPEND_ON to a prep job id to queue this behind data preparation.
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEED=${SEED:-3408}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/speechllm_multitask_ls960_mustc_en_de}
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-15000}
BRIDGE_STEPS=${BRIDGE_STEPS:-3000}
FILM_STEPS=${FILM_STEPS:-6000}
VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-1500}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-2-00:00:00}
DEPEND_ON=${DEPEND_ON:-}
DRY_RUN=${DRY_RUN:-0}

MUSTC=/export/jsalt26/omnienc/users/cxiao/datasets/mustc_en_de_prepared/manifests
LS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
WARM_START=${WARM_START:-$(pwd)/results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00}
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"

die() { echo "$*" >&2; exit 1; }

[[ "$TRAIN_SPLITS" == *train-clean-100* && "$TRAIN_SPLITS" == *train-clean-360* && "$TRAIN_SPLITS" == *train-other-500* ]] \
  || die "LS960 split invariant failed"
[[ "$OUTPUT_ROOT" == *ls960* ]] || die "Output root must contain ls960"
[[ "$MAX_OPTIMIZER_STEPS" -gt $((BRIDGE_STEPS + FILM_STEPS)) ]] || die "step budget inconsistent"
for f in segmenter.ckpt proj.ckpt llm.ckpt normalize.ckpt; do
  [[ -s "$WARM_START/$f" ]] || die "Warm start missing $f"
done
for csv in train dev-clean dev-other test-clean test-other; do
  [[ -s "$LS/$csv.csv" ]] || die "Missing LibriSpeech manifest: $csv"
done
[[ $(wc -l < "$LS/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

# MuST-C manifests may not exist yet when this is queued behind prep; the
# in-job validation below runs at start time and is the real gate.
if [[ -z "$DEPEND_ON" ]]; then
  for csv in mustc_st_train mustc_asr_train mustc_st_dev mustc_asr_dev \
             mustc_st_tst-COMMON mustc_asr_tst-COMMON; do
    [[ -s "$MUSTC/$csv.csv" ]] || die "Missing MuST-C manifest: $MUSTC/$csv.csv"
  done
fi

OUTPUT="$OUTPUT_ROOT/$SEED"
[[ ! -e "$OUTPUT" ]] || die "Refusing to reuse output folder: $OUTPUT"
mkdir -p "$OUTPUT_ROOT" slurm_logs

EXTRA_ARGS="--seed $SEED --output_folder $OUTPUT --warm_start_ckpt_dir $WARM_START"
EXTRA_ARGS+=" --train_splits $TRAIN_SPLITS --bridge_optimizer_steps $BRIDGE_STEPS"
EXTRA_ARGS+=" --film_optimizer_steps $FILM_STEPS --optimizer_step_limit $MAX_OPTIMIZER_STEPS"
EXTRA_ARGS+=" --validation_interval_optimizer_steps $VALIDATION_INTERVAL_STEPS"
EXTRA_ARGS+=" --mustc_manifest_dir $MUSTC --librispeech_manifest_dir $LS"

SBATCH_ARGS=(--account=highprio --comment=accept_cost --partition=gpu-a100
 --gpus=1 --cpus-per-task=12 --mem=64G --time="$JOB_TIME_LIMIT"
  --job-name="mt_mustc_${SEED}")
[[ -n "$DEPEND_ON" ]] && SBATCH_ARGS+=(--dependency=afterok:"$DEPEND_ON" --kill-on-invalid-dep=yes)

echo "=== MULTI-TASK FLAGSHIP on MuST-C En-De ==="
echo "Seed:        $SEED"
echo "ST corpus:   MuST-C v1 En-De (TED); tst-COMMON is the reported test set"
echo "Replay:      $TRAIN_SPLITS ($(($(wc -l < "$LS/train.csv") - 1)) utts)"
echo "Mixture:     0.50 ST / 0.25 MuST-C ASR / 0.25 LibriSpeech-960h replay"
echo "Steps:       bridge 0-$BRIDGE_STEPS, film -$((BRIDGE_STEPS+FILM_STEPS)), unfreeze -> $MAX_OPTIMIZER_STEPS"
echo "Warm start:  $WARM_START"
echo "Output:      $OUTPUT"
[[ -n "$DEPEND_ON" ]] && echo "Depends on:  job $DEPEND_ON (data prep) — starts only if prep succeeds"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: sbatch ${SBATCH_ARGS[*]} --export=ALL run_multitask.slurm"; exit 0
fi
job_id=$(EXTRA_ARGS="$EXTRA_ARGS" HPARAMS=hparams/speechllm_multitask_mustc.yaml \
  sbatch "${SBATCH_ARGS[@]}" --parsable --export=ALL run_multitask.slurm)
echo "Submitted MuST-C flagship: $job_id"
printf 'seed\tjob_id\tdepends_on\tmax_steps\toutput\n' > "$OUTPUT_ROOT/submitted_jobs.tsv"
printf '%s\t%s\t%s\t%s\t%s\n' "$SEED" "$job_id" "${DEPEND_ON:-none}" "$MAX_OPTIMIZER_STEPS" "$OUTPUT" >> "$OUTPUT_ROOT/submitted_jobs.tsv"
