#!/bin/bash
# Single-task non-ASR baselines: is the bottleneck the rate, the multi-task
# mixture, or the pooled-prefix interface itself?
#
# The multi-task runs cannot separate those three. Every arm here trains ONE
# task over ONE segmentation, and differs from the other arms in NOTHING but
# the segmentation:
#
#   ARM              segmentation                        f_audio
#   nods_50hz        fixed grid, k=1 (every frame)       50.0 Hz
#   fixed_25hz       fixed grid, k=2                     25.0 Hz
#   fixed_10hz       fixed grid, k=5                     10.0 Hz
#   fixed_5hz        fixed grid, k=10                     5.0 Hz
#   learned_frozen   the warm start's Transformer-AR policy, argmax, frozen
#   learned_grpo     the same policy, TRAINED with K=4 GRPO on this one task
#
# Why these four rates. 10 Hz is what the reported multi-task checkpoint
# actually emitted (7.6-10.6 Hz across tasks) and 5 Hz is what its Phase C
# compresses to (5.4-5.6 Hz), so two arms sit exactly on the learned operating
# points. 50 Hz is the no-compression ceiling for this interface and 25 Hz fills
# the gap, giving a 4-point rate-quality curve at k = 1, 2, 5, 10.
#
# Why both learned arms. Held frozen, learned_frozen is comparable in both
# directions: to the fixed arms (segmentation is then the only variable) AND to
# the published multi-task number, whose selected checkpoint is itself a
# frozen-policy bridge checkpoint. It isolates multi-task interference.
# learned_grpo then adds exactly one thing on top -- the policy is trained --
# so frozen vs GRPO at an identical step budget is what says whether training
# the boundary policy buys anything. It mirrors the multi-task schedule
# (bridge 20% / FiLM 40% / unfreeze 40%, per-task band in Phase C) and by
# default waits for its own learned_frozen counterpart to finish, so the two
# do not contend for the same GPUs.
#
# Matched effective batch. All arms optimize over 300 s of audio per step. At
# 50 Hz that is 15k audio tokens, so the long-sequence arms split the batch and
# recover it with gradient accumulation instead of shrinking it, which would
# confound rate with batch size.
#
# Checkpoint selection is on each task's own validation metric, never ASR WER:
# there is no ASR source in these runs, and ASR-only selection is what made the
# multi-task runs report pre-policy checkpoints.
#
# Usage:
#   ./submit_nonasr_singletask_baselines.sh                 # all 45 jobs
#   TASKS=speaker_count ./submit_nonasr_singletask_baselines.sh
#   ARMS="nods_50hz fixed_10hz" SEEDS=3407 ./submit_nonasr_singletask_baselines.sh
#   SMOKE=1 TASKS=speaker_count ARMS=nods_50hz SEEDS=3407 ./...   # shakedown
#   DRY_RUN=1 ./...                                         # print, submit nothing
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
TASKS=${TASKS:-"emotion speaker_count intent"}
ARMS=${ARMS:-"nods_50hz fixed_25hz fixed_10hz fixed_5hz learned_frozen"}
# learned_grpo is not in the default list: it waits on its learned_frozen
# counterpart, so it is launched as its own pass once those are queued.
# WAIT_FOR_FROZEN=0 submits it without the dependency.
WAIT_FOR_FROZEN=${WAIT_FOR_FROZEN:-1}
SEEDS=${SEEDS:-"3407 3408 3409"}

NONASR_MANIFESTS=${NONASR_MANIFESTS:-/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests}
SPKCOUNT_MANIFESTS=${SPKCOUNT_MANIFESTS:-/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests_lcfull}
WARM_START=${WARM_START:-$(pwd)/results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00}
HPARAMS=hparams/speechllm_singletask_nonasr.yaml

if [[ "$SMOKE" == 1 ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-results/smoke_singletask_nonasr}
  JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-02:00:00}
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-results/speechllm_singletask_nonasr}
  JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-12:00:00}
fi

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -s "$HPARAMS" ]] || die "missing hparams: $HPARAMS"
for f in segmenter.ckpt proj.ckpt llm.ckpt normalize.ckpt; do
  [[ -s "$WARM_START/$f" ]] || die "warm start missing $f: $WARM_START/$f"
done
# The mixtures are recipes, not audio; a missing JSONL fails at load time.
[[ -s "$SPKCOUNT_MANIFESTS/speaker_count_prep.json" ]] \
  || die "no speaker_count_prep.json in $SPKCOUNT_MANIFESTS; regenerate it"

# ---------------------------------------------------------------- task table
# source_key | manifest dir | csv prefix | selection metric | steps | valid every | valid subset
task_spec() {
  case "$1" in
    emotion)
      # 4,702 clips / 3.3 h -> ~40 batches per data epoch, so 1200 steps is
      # ~30 data epochs. That is more than the task needs, but an emotion job
      # is only ~17 min and the first six were already done or running when the
      # budget was revisited, so cutting it would have discarded more compute
      # than it saved. Held at 1200 so all 15 emotion runs share one budget.
      # Accuracy is uninformative here (61.7% of test is neutral); macro-F1 is
      # the metric the task is judged on, so it is also what selects.
      # Its valid split is 665 clips, so a 500 subset is nearly exhaustive.
      echo "cremad_emotion|$NONASR_MANIFESTS|cremad_emotion|macro_f1_emotion|1200|120|500" ;;
    speaker_count)
      # 20,000 mixtures / 27.8 h -> ~333 batches per epoch; 2600 steps is ~7.8
      # data epochs, cut from 12 for time. Still the largest budget in the
      # sweep, since this is the task the study is about. Classes are balanced,
      # so accuracy is the right metric. If the selected checkpoint lands on
      # the LAST validation point the budget was binding and should be
      # extended -- the collector prints best-VALID so that is visible.
      echo "speaker_count|$SPKCOUNT_MANIFESTS|speaker_count|acc_speaker_count|2600|260|600" ;;
    intent)
      # 23,132 clips / 14.7 h -> ~176 batches per epoch; 900 steps is ~5.1 data
      # epochs, cut from 8.5 for time. The task already saturates at 0.994, so
      # the extra epochs bought overfitting, not headroom.
      # The small valid subset matters more here than anywhere else: with 31
      # candidates, validating 1000 utterances cost ~18 min per job against
      # ~14 min of training. 400 is ample to rank checkpoints on a task scoring
      # ~0.99.
      echo "fsc_intent|$NONASR_MANIFESTS|fsc_intent|acc_intent|900|90|400" ;;
    *) die "unknown task: $1" ;;
  esac
}

# ----------------------------------------------------------------- arm table
# fixed_rate_k | max_batch_length_train | grad_accum | candidate_chunk_size
#
# The first three keep the effective batch at 300 s of audio in every arm.
# The fourth bounds evaluation memory: closed-set scoring runs the decoder on
# (chunk x sequence) and its logits are (rows, positions, 128256 vocab), so the
# chunk has to shrink as the sequence grows or the arm OOMs at test time --
# which is exactly how the first k=1 smoke job (1784456) died.
arm_spec() {
  case "$1" in
    nods_50hz)      echo "1|75|4|32"    ;;
    fixed_25hz)     echo "2|150|2|64"   ;;
    fixed_10hz)     echo "5|300|1|256"  ;;
    fixed_5hz)      echo "10|300|1|256" ;;
    learned_frozen) echo "null|300|1|256" ;;
    learned_grpo)   echo "null|300|1|256" ;;
    *) die "unknown arm: $1" ;;
  esac
}

mkdir -p "$OUTPUT_ROOT" slurm_logs
submitted=0
TSV="$OUTPUT_ROOT/submitted_jobs.tsv"
[[ -s "$TSV" ]] || printf 'task\tarm\tseed\tjob_id\tfixed_rate_k\tsteps\toutput\n' > "$TSV"

echo "=== SINGLE-TASK NON-ASR BASELINES ==="
[[ "$SMOKE" == 1 ]] && echo "MODE:          SMOKE (shakedown, not a result)"
echo "Warm start:    $WARM_START"
echo "Tasks:         $TASKS"
echo "Arms:          $ARMS"
echo "Seeds:         $SEEDS"
echo "Output root:   $OUTPUT_ROOT"
echo

for task in $TASKS; do
  IFS='|' read -r src_key man_dir csv_pfx sel_metric steps valid_every task_valid_subset <<< "$(task_spec "$task")"
  for csv in train valid test; do
    [[ -s "$man_dir/${csv_pfx}_${csv}.csv" ]] || die "missing manifest: $man_dir/${csv_pfx}_${csv}.csv"
  done
  train_rows=$(($(wc -l < "$man_dir/${csv_pfx}_train.csv") - 1))

  for arm in $ARMS; do
    IFS='|' read -r k mbl accum chunk <<< "$(arm_spec "$arm")"

    if [[ "$SMOKE" == 1 ]]; then
      steps=24; valid_every=8; valid_subset=32; test_subset=48
    else
      valid_subset=${VALID_SUBSET:-$task_valid_subset}; test_subset=""
    fi

    # batches_per_epoch counts DATALOADER batches, not optimizer steps, so the
    # epoch bound has to be multiplied by the accumulation factor or the run
    # silently stops early (this is how job 1780790 ended at step 765/15000).
    batches_per_epoch=$((valid_every * accum))
    num_epochs=$(( (steps * accum + batches_per_epoch - 1) / batches_per_epoch + 1 ))
    [[ $((num_epochs * batches_per_epoch)) -ge $((steps * accum)) ]] \
      || die "epoch bound $num_epochs x $batches_per_epoch < $((steps * accum)) batches"

    for seed in $SEEDS; do
      out="$OUTPUT_ROOT/$task/$arm/$seed"
      if [[ -e "$out" ]]; then
        echo "  SKIP (exists) $task/$arm/$seed"
        continue
      fi

      args="--seed $seed --output_folder $out"
      args+=" --warm_start_ckpt_dir $WARM_START"
      args+=" --task_source_key $src_key"
      args+=" --task_train_csv $man_dir/${csv_pfx}_train.csv"
      args+=" --task_valid_csv $man_dir/${csv_pfx}_valid.csv"
      args+=" --task_test_csv $man_dir/${csv_pfx}_test.csv"
      args+=" --selection_metric $sel_metric"
      args+=" --fixed_rate_k $k"
      args+=" --max_batch_length_train $mbl"
      args+=" --grad_accumulation_factor $accum"
      args+=" --candidate_chunk_size $chunk"
      args+=" --optimizer_step_limit $steps"
      args+=" --validation_interval_optimizer_steps $valid_every"
      args+=" --multitask_batches_per_epoch $batches_per_epoch"
      args+=" --number_of_epochs $num_epochs"
      args+=" --valid_subset_size $valid_subset"
      [[ -n "$test_subset" ]] && args+=" --test_subset_size $test_subset"
      dep=""
      if [[ "$arm" == learned_grpo ]]; then
        # Mirror the multi-task schedule at this task's budget: 20% frozen
        # bridge, 40% FiLM-only GRPO, 40% with the last policy block unfrozen.
        # The total is identical to the frozen arm's, so the two differ only in
        # whether the policy is trained.
        bridge=$((steps / 5)); film=$((steps * 2 / 5))
        [[ $((bridge + film)) -lt "$steps" ]] \
          || die "phase split $bridge+$film does not leave an unfreeze phase in $steps"
        args+=" --bridge_optimizer_steps $bridge --film_optimizer_steps $film"
        args+=" --freeze_boundary_policy False"
        if [[ "$WAIT_FOR_FROZEN" == 1 ]]; then
          # Start when this task/seed's frozen run terminates (afterany, so a
          # frozen failure does not strand the GRPO job forever).
          frozen_id=$(awk -F'\t' -v t="$task" -v s="$seed" \
            '$1==t && $2=="learned_frozen" && $3==s {print $4}' "$TSV" 2>/dev/null | tail -1)
          [[ -n "$frozen_id" ]] && dep="--dependency=afterany:$frozen_id"
        fi
      else
        # The whole run is the frozen-policy bridge: no GRPO, no rate pressure.
        # Every fixed arm shares this, so segmentation is the only variable.
        args+=" --bridge_optimizer_steps 1000000 --film_optimizer_steps 0"
        args+=" --freeze_boundary_policy True"
      fi

      if [[ "$DRY_RUN" == 1 ]]; then
        printf '  DRY %-14s %-15s %s  k=%-4s mbl=%-3s accum=%s steps=%s epochs=%s %s\n' \
          "$task" "$arm" "$seed" "$k" "$mbl" "$accum" "$steps" "$num_epochs" "$dep"
        continue
      fi

      job_id=$(EXTRA_ARGS="$args" HPARAMS="$HPARAMS" \
        sbatch --account=highprio --comment=accept_cost --partition=gpu-a100 \
               --gpus=1 --cpus-per-task=12 --mem=64G --time="$JOB_TIME_LIMIT" \
               --job-name="st_${task}_${arm}_${seed}" \
               ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} ${dep} \
               --parsable --export=ALL run_multitask.slurm)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$task" "$arm" "$seed" "$job_id" "$k" "$steps" "$out" >> "$TSV"
      printf '  %-14s %-15s seed %s -> job %s (k=%s, %s steps, %s rows)%s\n' \
        "$task" "$arm" "$seed" "$job_id" "$k" "$steps" "$train_rows" \
        "${dep:+ after ${dep#--dependency=afterany:}}"
      submitted=$((submitted + 1))
    done
  done
done

echo
echo "Submitted $submitted job(s); ledger: $TSV"
