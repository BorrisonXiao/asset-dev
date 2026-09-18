#!/bin/bash
# Launch the one-seed task-conditioned multi-task flagship run.
#
# Warm start: LS960 local-64 Transformer-AR + BiGRU, seed 3408 (best dev-clean
# WER, 2.879, of the completed three-seed cohort).
# Tasks:      English ASR (CoVoST + LibriSpeech-960h replay) and En->De ST.
# Phases:     frozen-policy bridge -> FiLM-only GRPO -> last-block unfreezing,
#             all controlled by optimizer steps.
#
# MuST-C v1 is distributed behind an FBK registration form, so the paired
# ASR/ST corpus is CoVoST 2 En->De: same English audio, English transcript and
# German translation, which is the property the same-audio analysis needs.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SEED=${SEED:-3408}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/speechllm_multitask_ls960_covost_en_de}
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-15000}
BRIDGE_STEPS=${BRIDGE_STEPS:-3000}
FILM_STEPS=${FILM_STEPS:-6000}
VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-1500}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-2-00:00:00}
DRY_RUN=${DRY_RUN:-0}

COVOST_MANIFESTS=/export/jsalt26/omnienc/users/cxiao/datasets/covost2_en_de/manifests
LS_MANIFESTS=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
WARM_START=${WARM_START:-$(pwd)/results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00}

# LibriSpeech replay must be the full 960 h corpus; this is the project's
# scale-up invariant, not a formality.
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"

die() { echo "$*" >&2; exit 1; }

[[ "$TRAIN_SPLITS" == *train-clean-100* && "$TRAIN_SPLITS" == *train-clean-360* && "$TRAIN_SPLITS" == *train-other-500* ]] \
  || die "LS960 split invariant failed: $TRAIN_SPLITS"
[[ "$OUTPUT_ROOT" == *ls960* ]] || die "Output root must contain ls960: $OUTPUT_ROOT"
[[ "$MAX_OPTIMIZER_STEPS" -gt $((BRIDGE_STEPS + FILM_STEPS)) ]] \
  || die "max steps ($MAX_OPTIMIZER_STEPS) must exceed bridge+film ($((BRIDGE_STEPS + FILM_STEPS)))"

for f in segmenter.ckpt proj.ckpt llm.ckpt normalize.ckpt; do
  [[ -s "$WARM_START/$f" ]] || die "Warm start missing $f: $WARM_START/$f"
done

for csv in covost_st_train covost_asr_train covost_st_validation covost_asr_validation \
           covost_st_test covost_asr_test; do
  [[ -s "$COVOST_MANIFESTS/$csv.csv" ]] || die "Missing CoVoST manifest: $COVOST_MANIFESTS/$csv.csv"
done
for csv in train dev-clean dev-other test-clean test-other; do
  [[ -s "$LS_MANIFESTS/$csv.csv" ]] || die "Missing LibriSpeech manifest: $LS_MANIFESTS/$csv.csv"
done
[[ $(wc -l < "$LS_MANIFESTS/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

# The audio referenced by the CoVoST manifests must actually exist; a partially
# prepared corpus would fail thousands of steps into the run.
st_rows=$(($(wc -l < "$COVOST_MANIFESTS/covost_st_train.csv") - 1))
asr_rows=$(($(wc -l < "$COVOST_MANIFESTS/covost_asr_train.csv") - 1))
[[ "$st_rows" -eq "$asr_rows" ]] || die "Paired views disagree: st=$st_rows asr=$asr_rows"
[[ "$st_rows" -gt 200000 ]] || die "CoVoST train looks incomplete ($st_rows rows); is prep still running?"
missing=0
while IFS=, read -r _ _ wav _; do
  [[ -f "$wav" ]] || missing=$((missing + 1))
done < <(tail -n +2 "$COVOST_MANIFESTS/covost_st_train.csv" | shuf -n 200 --random-source=<(yes))
[[ "$missing" -eq 0 ]] || die "$missing of 200 sampled CoVoST audio files are missing"

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
EXTRA_ARGS+=" --covost_manifest_dir $COVOST_MANIFESTS"
EXTRA_ARGS+=" --librispeech_manifest_dir $LS_MANIFESTS"

SBATCH_ARGS=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=64G
  --time="$JOB_TIME_LIMIT"
  --job-name="mt_st_${SEED}"
)

echo "=== TASK-CONDITIONED MULTI-TASK FLAGSHIP ==="
echo "Seed:            $SEED"
echo "Warm start:      $WARM_START"
echo "Output:          $OUTPUT"
echo "LibriSpeech:     $TRAIN_SPLITS (replay, $(($(wc -l < "$LS_MANIFESTS/train.csv") - 1)) utts)"
echo "CoVoST En-De:    $st_rows paired utterances (ASR view + ST view)"
echo "Mixture:         0.50 ST / 0.25 CoVoST ASR / 0.25 LibriSpeech-960h replay"
echo "Steps:           bridge 0-$BRIDGE_STEPS, film $BRIDGE_STEPS-$((BRIDGE_STEPS + FILM_STEPS)), unfreeze -> $MAX_OPTIMIZER_STEPS"
echo "Validate every:  $VALIDATION_INTERVAL_STEPS steps (per-task WER + sacreBLEU/chrF++)"
echo "Resources:       1 A100, 12 CPUs, 64 GB, $JOB_TIME_LIMIT"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: sbatch ${SBATCH_ARGS[*]} --export=ALL run_multitask.slurm"
  echo "DRY-RUN: EXTRA_ARGS=$EXTRA_ARGS"
  exit 0
fi

# EXTRA_ARGS reaches the job through --export=ALL and the prefix assignment
# below. It must NOT be listed inside --export=, because SLURM splits that list
# on commas and the LS960 split list contains them.
job_id=$(EXTRA_ARGS="$EXTRA_ARGS" sbatch "${SBATCH_ARGS[@]}" --parsable \
  --export=ALL run_multitask.slurm)
echo "Submitted flagship multi-task job: $job_id"
printf 'seed\tjob_id\twarm_start\tmax_steps\tbridge\tfilm\toutput\n' > "$OUTPUT_ROOT/submitted_jobs.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$SEED" "$job_id" "$WARM_START" "$MAX_OPTIMIZER_STEPS" "$BRIDGE_STEPS" \
  "$FILM_STEPS" "$OUTPUT" >> "$OUTPUT_ROOT/submitted_jobs.tsv"
