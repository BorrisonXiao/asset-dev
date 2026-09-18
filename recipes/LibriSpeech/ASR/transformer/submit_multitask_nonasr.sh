#!/bin/bash
# Launch the non-ASR task-conditioned multi-task run.
#
# Warm start: LS960 local-64 Transformer-AR + BiGRU, seed 3408 (best dev-clean
#             WER, 2.879, of the completed three-seed cohort).
# Tasks:      LibriSpeech-960h ASR anchor, CREMA-D emotion, on-the-fly
#             LibriSpeech speaker-count mixtures, Fluent Speech Commands intent.
# Phases:     frozen-policy bridge -> FiLM-only GRPO -> last-block unfreezing,
#             all controlled by optimizer steps. Per-task rate bands apply only
#             in the last phase, so the FiLM phase stays matched-rate.
#
# Every training hyperparameter is inherited from the validated ASR+ST flagship
# config; only the data sources and the mixture differ.
#
# SMOKE=1 runs a short shakedown into results/smoke_multitask_nonasr instead:
# few steps, tiny validation, no LS960 name requirement.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SMOKE=${SMOKE:-0}
SEED=${SEED:-3408}
DRY_RUN=${DRY_RUN:-0}
# auto|reward|aux|both. auto == reward for the full-AR policy, so the
# default arm and an explicit reward arm are the same experiment.
RATE_CHANNEL=${RATE_CHANNEL:-auto}

if [[ "$SMOKE" == 1 ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-results/smoke_multitask_nonasr}
  MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-24}
  BRIDGE_STEPS=${BRIDGE_STEPS:-8}
  FILM_STEPS=${FILM_STEPS:-8}
  VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-8}
  VALID_SUBSET=${VALID_SUBSET:-32}
  # The final test pass is exercised, not skipped: a broken test spec would
  # otherwise surface only after a full training run has already been paid for.
  TEST_SUBSET=${TEST_SUBSET:-48}
  JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-02:00:00}
  JOB_NAME="smoke_mt_nonasr"
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-results/speechllm_multitask_ls960_nonasr}
  MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-15000}
  BRIDGE_STEPS=${BRIDGE_STEPS:-3000}
  FILM_STEPS=${FILM_STEPS:-6000}
  VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-1500}
  VALID_SUBSET=${VALID_SUBSET:-1000}
  TEST_SUBSET=${TEST_SUBSET:-}          # empty = full test sets
  JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-2-00:00:00}
  JOB_NAME="mt_nonasr_${SEED}"
fi

NONASR_MANIFESTS=/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests
SPKCOUNT_MANIFESTS=${SPKCOUNT_MANIFESTS:-/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests_lcfull}
LS_MANIFESTS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
WARM_START=${WARM_START:-$(pwd)/results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00}

# LibriSpeech replay must be the full 960 h corpus; this is the project's
# scale-up invariant, not a formality.
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"

die() { echo "$*" >&2; exit 1; }

[[ "$TRAIN_SPLITS" == *train-clean-100* && "$TRAIN_SPLITS" == *train-clean-360* && "$TRAIN_SPLITS" == *train-other-500* ]] \
  || die "LS960 split invariant failed: $TRAIN_SPLITS"
if [[ "$SMOKE" != 1 ]]; then
  [[ "$OUTPUT_ROOT" == *ls960* ]] || die "Output root must contain ls960: $OUTPUT_ROOT"
fi
[[ "$MAX_OPTIMIZER_STEPS" -gt $((BRIDGE_STEPS + FILM_STEPS)) ]] \
  || die "max steps ($MAX_OPTIMIZER_STEPS) must exceed bridge+film ($((BRIDGE_STEPS + FILM_STEPS)))"

for f in segmenter.ckpt proj.ckpt llm.ckpt normalize.ckpt; do
  [[ -s "$WARM_START/$f" ]] || die "Warm start missing $f: $WARM_START/$f"
done

for csv in cremad_emotion_train cremad_emotion_valid cremad_emotion_test \
           fsc_intent_train fsc_intent_valid fsc_intent_test; do
  [[ -s "$NONASR_MANIFESTS/$csv.csv" ]] || die "Missing manifest: $NONASR_MANIFESTS/$csv.csv"
done
for csv in speaker_count_train speaker_count_valid speaker_count_test; do
  [[ -s "$SPKCOUNT_MANIFESTS/$csv.csv" ]] || die "Missing manifest: $SPKCOUNT_MANIFESTS/$csv.csv"
done
for csv in train dev-clean dev-other test-clean test-other; do
  [[ -s "$LS_MANIFESTS/$csv.csv" ]] || die "Missing LibriSpeech manifest: $LS_MANIFESTS/$csv.csv"
done
[[ $(wc -l < "$LS_MANIFESTS/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

# The speaker-count mixtures are recipes, not audio: the manifest points into a
# JSONL that has to be there, or every mixture batch fails at load time.
for split in train valid test; do
  [[ -s "$SPKCOUNT_MANIFESTS/speaker_count_${split}_recipes.jsonl" ]] \
    || die "Missing mixture recipes: $SPKCOUNT_MANIFESTS/speaker_count_${split}_recipes.jsonl"
done
# The generator profile decides what a speaker-count number means, so refuse to
# launch on mixtures whose provenance is not recorded.
[[ -s "$SPKCOUNT_MANIFESTS/speaker_count_prep.json" ]] \
  || die "No speaker_count_prep.json in $SPKCOUNT_MANIFESTS; regenerate it"

# Sampled audio must actually resolve. A missing corpus file would otherwise
# surface thousands of steps in, on an A100 that has already been queued for.
missing=0
for csv in cremad_emotion_train fsc_intent_train; do
  while IFS=, read -r _ _ wav _; do
    [[ -f "$wav" ]] || missing=$((missing + 1))
  done < <(tail -n +2 "$NONASR_MANIFESTS/$csv.csv" | shuf -n 100 --random-source=<(yes))
done
[[ "$missing" -eq 0 ]] || die "$missing of 200 sampled task audio files are missing"

emotion_rows=$(($(wc -l < "$NONASR_MANIFESTS/cremad_emotion_train.csv") - 1))
count_rows=$(($(wc -l < "$SPKCOUNT_MANIFESTS/speaker_count_train.csv") - 1))
intent_rows=$(($(wc -l < "$NONASR_MANIFESTS/fsc_intent_train.csv") - 1))

OUTPUT="$OUTPUT_ROOT/$SEED"
[[ ! -e "$OUTPUT" ]] || die "Refusing to reuse output folder: $OUTPUT"
mkdir -p "$OUTPUT_ROOT" slurm_logs

EXTRA_ARGS="--seed $SEED --output_folder $OUTPUT"
EXTRA_ARGS+=" --warm_start_ckpt_dir $WARM_START"
EXTRA_ARGS+=" --train_splits $TRAIN_SPLITS"
EXTRA_ARGS+=" --bridge_optimizer_steps $BRIDGE_STEPS"
EXTRA_ARGS+=" --film_optimizer_steps $FILM_STEPS"
EXTRA_ARGS+=" --optimizer_step_limit $MAX_OPTIMIZER_STEPS"
EXTRA_ARGS+=" --validation_interval_optimizer_steps $VALIDATION_INTERVAL_STEPS"
# An "epoch" here is min_over_sources(batches / probability), and CREMA-D is by
# far the smallest source, so the derived epoch is short (255 batches for the
# 0.40/0.20/0.20/0.20 mixture). number_of_epochs then silently caps training far
# below optimizer_step_limit -- job 1780790 stopped at step 765 of 15000 that
# way. Pin the epoch length and derive the bound from it instead of inheriting
# the ST flagship's value, which was safe only because CoVoST is large.
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-$VALIDATION_INTERVAL_STEPS}
NUM_EPOCHS=$(( (MAX_OPTIMIZER_STEPS + BATCHES_PER_EPOCH - 1) / BATCHES_PER_EPOCH + 1 ))
[[ $((NUM_EPOCHS * BATCHES_PER_EPOCH)) -ge "$MAX_OPTIMIZER_STEPS" ]] \
  || die "epoch bound $NUM_EPOCHS x $BATCHES_PER_EPOCH < max steps $MAX_OPTIMIZER_STEPS"
EXTRA_ARGS+=" --multitask_batches_per_epoch $BATCHES_PER_EPOCH"
EXTRA_ARGS+=" --number_of_epochs $NUM_EPOCHS"
EXTRA_ARGS+=" --valid_subset_size $VALID_SUBSET"
[[ -n "$TEST_SUBSET" ]] && EXTRA_ARGS+=" --test_subset_size $TEST_SUBSET"
EXTRA_ARGS+=" --nonasr_manifest_dir $NONASR_MANIFESTS"
EXTRA_ARGS+=" --speaker_count_manifest_dir $SPKCOUNT_MANIFESTS"
EXTRA_ARGS+=" --librispeech_manifest_dir $LS_MANIFESTS"
EXTRA_ARGS+=" --rate_channel $RATE_CHANNEL"

SBATCH_ARGS=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=64G
  --time="$JOB_TIME_LIMIT"
  --job-name="$JOB_NAME"
)
# Node exclusion, for working around a faulty GPU. Job 1780854 died on e01 with
# "CUDA error: unspecified launch failure" 32 min in; a sibling job with the same
# code passed on e02, which isolated it to the node rather than the code.
# Note --requeue does NOT cover this: SLURM requeues on node failure and
# preemption, not on a non-zero application exit, so such a fault costs the run.
[[ -n "${EXCLUDE_NODES:-}" ]] && SBATCH_ARGS+=(--exclude="$EXCLUDE_NODES")

echo "=== NON-ASR TASK-CONDITIONED MULTI-TASK RUN ==="
[[ "$SMOKE" == 1 ]] && echo "MODE:            SMOKE (short shakedown, not a result)"
echo "Seed:            $SEED"
echo "Warm start:      $WARM_START"
echo "Output:          $OUTPUT"
echo "LibriSpeech:     $TRAIN_SPLITS (ASR anchor, $(($(wc -l < "$LS_MANIFESTS/train.csv") - 1)) utts)"
echo "CREMA-D:         $emotion_rows clips (emotion, voice-vote labels)"
echo "Speaker count:   $count_rows mixtures from $SPKCOUNT_MANIFESTS"
echo "                 profile: $(python3 -c "import json,sys;d=json.load(open(\"$SPKCOUNT_MANIFESTS/speaker_count_prep.json\"));print(d.get(\"profile\"), d.get(\"placement\"), \"vad\" if d.get(\"vad_labels\") else \"placement-labels\")" 2>/dev/null || echo unknown)"
echo "FSC:             $intent_rows clips (intent, 31 candidates)"
echo "Mixture:         0.40 LS960 ASR / 0.20 emotion / 0.20 speaker_count / 0.20 intent"
echo "Steps:           bridge 0-$BRIDGE_STEPS, film $BRIDGE_STEPS-$((BRIDGE_STEPS + FILM_STEPS)), unfreeze -> $MAX_OPTIMIZER_STEPS"
echo "Epoch bound:     $NUM_EPOCHS epochs x $BATCHES_PER_EPOCH batches >= $MAX_OPTIMIZER_STEPS steps"
echo "Validate every:  $VALIDATION_INTERVAL_STEPS steps (WER for ASR, accuracy/macro-F1 for the rest)"
echo "Rate bands:      global 7.5-12.5 Hz until the unfreeze phase, then per task"
echo "Rate channel:    $RATE_CHANNEL"
echo "Test sets:       ${TEST_SUBSET:-full}"
echo "Resources:       1 A100, 12 CPUs, 64 GB, $JOB_TIME_LIMIT"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: sbatch ${SBATCH_ARGS[*]} --export=ALL run_multitask.slurm"
  echo "DRY-RUN: HPARAMS=hparams/speechllm_multitask_nonasr.yaml"
  echo "DRY-RUN: EXTRA_ARGS=$EXTRA_ARGS"
  exit 0
fi

# EXTRA_ARGS reaches the job through --export=ALL and the prefix assignment
# below. It must NOT be listed inside --export=, because SLURM splits that list
# on commas and the LS960 split list contains them.
job_id=$(EXTRA_ARGS="$EXTRA_ARGS" HPARAMS=hparams/speechllm_multitask_nonasr.yaml \
  sbatch "${SBATCH_ARGS[@]}" --parsable --export=ALL run_multitask.slurm)
echo "Submitted non-ASR multi-task job: $job_id"
TSV="$OUTPUT_ROOT/submitted_jobs.tsv"
[[ -s "$TSV" ]] || printf 'seed\tjob_id\trate_channel\tmax_steps\tbridge\tfilm\toutput\n' > "$TSV"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$SEED" "$job_id" "$RATE_CHANNEL" "$MAX_OPTIMIZER_STEPS" "$BRIDGE_STEPS" \
  "$FILM_STEPS" "$OUTPUT" >> "$TSV"
