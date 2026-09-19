#!/bin/bash
# Cross-lingual 24k-step replication: full dependency-chained pipeline for
# Mandarin (AISHELL-1 + chinese-hubert-large) and Japanese (ReazonSpeech small
# + japanese-hubert-large). Per language:
#
#   ctc_aligner (a100)  ─┐
#   mfa + phone targets ─┴─> gate (med) ─> coldstart (a100, seed 3407)
#        (med)                              ├─> fixed k=5   x3 seeds (CE)
#                                           ├─> char oracle x3 seeds (CE)
#                                           ├─> phone oracle x3 seeds (CE)
#                                           └─> learned 24k x3 seeds
#                                               (each afterok its seed's
#                                                phone-oracle arm: decoder init)
#
# All GPU jobs: a100 partition only. Manifests must exist (prep runs on the
# login node before this script). Band: rho in [0.10, 0.20] (5-10 Hz).
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

DRY_RUN=${DRY_RUN:-0}
LANGS=${LANGS:-"zh ja"}
read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
ROOT=results/speechllm_xling_step24k
HF_CACHE=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub
DATASETS=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets
RHO_LO_DEFAULT=0.10
RHO_HI=0.20
MAX_STEPS=24000
VALID_INTERVAL=4000
WARMUP_STEPS=2400
CE_EPOCHS=10
MANIFEST=$ROOT/submitted_jobs.tsv

GPU_IDENTITY=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100 --exclude=ga129)
CPU_IDENTITY=(--account=jsalt2026-lgarci27 --comment=accept_cost --partition=med)

die() { echo "$*" >&2; exit 1; }
submit_job() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2; printf ' %q' sbatch "$@" >&2; printf '\n' >&2
    echo "DRY_$RANDOM"
  else
    sbatch --parsable "$@"
  fi
}
log_job() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$MANIFEST"; }

lang_config() {
  local lang=$1
  case "$lang" in
    zh)
      SSL_HUB=TencentGameMate/chinese-hubert-large
      UNITS_LEVEL=char
      DICT_NAME=mandarin_china_mfa
      ACOUSTIC_NAME=mandarin_mfa
      G2P_NAME=mandarin_china_mfa
      DATA_FOLDER=$DATASETS/aishell/data_aishell
      MFA_SPEAKER_CHARS=11
      CTC_EPOCHS=5
      RHO_LO=0.065   # measured zh char rate 0.069 (3.5 Hz) < 0.10 floor; S1 rule
      SAFETY_EPOCHS=16
      COLDSTART_EPOCHS=6
      VALID_CSV_BASE=dev_sub.csv
      TEST_SPLITS="['test']"
      EXTRA_TEST_CSVS=""
      EXTRA_GATE_CSVS=""
      EXTRA_ALIGN_SPLITS=""
      ;;
    ja)
      SSL_HUB=yky-h/japanese-hubert-large
      UNITS_LEVEL=mora
      DICT_NAME=japanese_mfa
      ACOUSTIC_NAME=japanese_mfa
      G2P_NAME=japanese_mfa
      DATA_FOLDER=$DATASETS/reazonspeech_small
      MFA_SPEAKER_CHARS=
      CTC_EPOCHS=6
      RHO_LO=0.095   # measured ja mora rate 0.098 (4.9 Hz), a hair under 0.10
      SAFETY_EPOCHS=32
      COLDSTART_EPOCHS=10
      VALID_CSV_BASE=dev.csv
      TEST_SPLITS="['test','jsut_test']"
      EXTRA_TEST_CSVS="jsut_test.csv"
      EXTRA_GATE_CSVS="jsut_test.csv"
      EXTRA_ALIGN_SPLITS="jsut_test"
      ;;
    *) die "unknown language: $lang" ;;
  esac
}

for lang in $LANGS; do
  lang_config "$lang"
  M=$(pwd)/$ROOT/$lang/manifests
  for f in train.csv dev.csv test.csv "$VALID_CSV_BASE" \
      "train_units_${UNITS_LEVEL}.tsv" "dev_units_${UNITS_LEVEL}.tsv" \
      "test_units_${UNITS_LEVEL}.tsv"; do
    [[ -s "$M/$f" ]] || die "[$lang] missing manifest: $M/$f"
  done
  snap=$HF_CACHE/models--${SSL_HUB//\//--}
  [[ -d "$snap" ]] || die "[$lang] encoder not cached: $snap"
done
[[ -d "$HF_CACHE/models--meta-llama--Llama-3.2-1B-Instruct" ]] || die "LLM not cached"
mkdir -p slurm_logs "$ROOT"
printf 'lang\tstage\tjob_id\tdetail\n' > "$MANIFEST"

for lang in $LANGS; do
  lang_config "$lang"
  L=$ROOT/$lang
  M=$(pwd)/$L/manifests
  T=$(pwd)/$L/targets
  VALID_CSV=$M/$VALID_CSV_BASE

  TEST_CSV="[$M/test.csv"
  for extra in $EXTRA_TEST_CSVS; do TEST_CSV+=",$M/$extra"; done
  TEST_CSV+="]"

  ALIGN_EXTRA=""
  for split in $EXTRA_ALIGN_SPLITS; do
    ALIGN_EXTRA+="$split=$M/$split.csv:$M/${split}_units_${UNITS_LEVEL}.tsv "
  done
  GATE_EXTRA=""
  for extra in $EXTRA_GATE_CSVS; do GATE_EXTRA+="$M/$extra "; done

  if [[ "${RESUBMIT_LEARNED_ONLY:-0}" == 1 ]]; then
    # Learned-only resubmission (e.g. band change): reuse the queued pipeline.
    # PHONE_JOBS_<LANG> holds "seed:job_id" pairs for the afterok deps.
    cold_out=$L/coldstart/3407
    declare -A phone_jobs=()
    pj_var="PHONE_JOBS_${lang^^}"
    for pair in ${!pj_var}; do
      phone_jobs[${pair%%:*}]=${pair##*:}
    done
  fi
  if [[ "${RESUBMIT_LEARNED_ONLY:-0}" != 1 ]]; then
  # ---- alignment stage -----------------------------------------------------
  # NOTE: values that may contain commas/spaces are passed as environment
  # variables on the submit command (inherited via --export=ALL), never inside
  # the --export list itself — sbatch splits that list on commas (see the
  # 2026-08 LS960 split-list bug in docs/project_notes/bugs.md).
  ctc_done_var="REUSE_CTC_DONE_${lang^^}"
  if [[ -n "${!ctc_done_var:-}" ]]; then
    # A completed aligner's targets are on disk; a purged job id cannot be a
    # dependency, so the gate depends on MFA alone.
    ctc_job=""
    echo "[$lang] CTC aligner already complete; targets reused"
  else
    ctc_job=$(EXTRA_SPLITS="$ALIGN_EXTRA" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_ctc" \
      --export="ALL,LANG_CODE=$lang,SSL_HUB=$SSL_HUB,MANIFEST_DIR=$M,UNITS_LEVEL=$UNITS_LEVEL,OUT_ROOT=$T,CTC_EPOCHS=$CTC_EPOCHS" \
      run_xling_ctc_aligner.slurm)
    log_job "$lang" ctc_aligner "$ctc_job" "$UNITS_LEVEL"
    echo "[$lang] CTC aligner -> $ctc_job"
  fi

  reuse_var="REUSE_MFA_JOB_${lang^^}"
  if [[ -n "${!reuse_var:-}" ]]; then
    # Resubmission path: an earlier MFA job survived a failed sibling stage.
    mfa_job=${!reuse_var}
  else
    mfa_job=$(EXTRA_CSVS="$GATE_EXTRA" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_mfa" \
      --gpus=1 --partition=a100 --cpus-per-task=32 --mem=180G --time=1-00:00:00 \
      --export="ALL,LANG_CODE=$lang,MANIFEST_DIR=$M,WORK_DIR=$T/mfa,DICT_NAME=$DICT_NAME,ACOUSTIC_NAME=$ACOUSTIC_NAME,G2P_NAME=$G2P_NAME,OUT_ROOT=$T,SPEAKER_CHARS=$MFA_SPEAKER_CHARS" \
      run_xling_mfa.slurm)
  fi
  log_job "$lang" mfa "$mfa_job" "$ACOUSTIC_NAME"
  echo "[$lang] MFA + phone targets -> $mfa_job"

  dep() { if [[ "$DRY_RUN" == 1 ]]; then :; else printf -- '--dependency=afterok:%s' "$1"; fi; }

  gate_dep=""
  dep_ids="$mfa_job"
  [[ -n "$ctc_job" ]] && dep_ids="${ctc_job}:${mfa_job}"
  [[ "$DRY_RUN" == 1 ]] || gate_dep="--dependency=afterok:${dep_ids} --kill-on-invalid-dep=yes"
  gate_job=$(EXTRA_CSVS="$GATE_EXTRA" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_gate" $gate_dep \
    --gpus=1 --partition=a100 --cpus-per-task=4 --mem=16G --time=02:00:00 \
    --export="ALL,MANIFEST_DIR=$M,OUT_ROOT=$T,UNITS_LEVEL=$UNITS_LEVEL,RHO_LO=$RHO_LO,RHO_HI=$RHO_HI" \
    run_xling_gate.slurm)
  log_job "$lang" gate "$gate_job" "afterok:$ctc_job,$mfa_job"
  echo "[$lang] gate -> $gate_job"

  fi
  # ---- shared training argument groups --------------------------------------
  DATA_ARGS="--skip_prep True --data_folder $DATA_FOLDER --csv_folder $M"
  DATA_ARGS+=" --train_csv $M/train.csv --valid_csv $VALID_CSV"
  DATA_ARGS+=" --test_splits $TEST_SPLITS --test_csv $TEST_CSV"
  DATA_ARGS+=" --train_splits ['train'] --dev_splits ['dev']"
  SSL_ARGS="--ssl_hub $SSL_HUB --ssl_folder $HF_CACHE --ssl_feat_dims 1024 --segmenter_input_dim 1024"
  POLICY_ARGS="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
  POLICY_ARGS+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
  POLICY_ARGS+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
  POLICY_ARGS+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
  POOL_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
  POOL_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"
  LOADER_ARGS="--num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2"
  MISC_ARGS="--selection_metric cer --precision bf16 --eval_precision bf16"
  MISC_ARGS+=" --ckpt_interval_minutes 30 --test_batch_size 8 --checkpoints_to_keep 1"
  BASE_ARGS="$DATA_ARGS $SSL_ARGS $POLICY_ARGS $POOL_ARGS $LOADER_ARGS $MISC_ARGS"

  if [[ "${RESUBMIT_LEARNED_ONLY:-0}" != 1 ]]; then
  # ---- cold start (phone-oracle supervision; shared by all seeds) -----------
  cold_out=$L/coldstart/3407
  cold_args="$BASE_ARGS --experiment_name xling_${lang}_coldstart --seed 3407"
  cold_args+=" --output_folder $cold_out"
  cold_args+=" --warmstart_target alignment --boundary_target_dir $T/boundaries/phone"
  cold_args+=" --max_batch_length_train 400 --max_batch_length_val 100 --grad_accumulation_factor 1"
  cold_dep=""
  [[ "$DRY_RUN" == 1 ]] || cold_dep="--dependency=afterok:$gate_job --kill-on-invalid-dep=yes"
  cold_job=$(EXTRA_ARGS="$cold_args" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_cold" $cold_dep \
    --cpus-per-task=12 --mem=64G --time=1-00:00:00 \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=coldstart,BACKBONE=transformer_ar,EPOCHS=$COLDSTART_EPOCHS" \
    run_segmenter.slurm)
  log_job "$lang" coldstart "$cold_job" "afterok:$gate_job"
  echo "[$lang] cold start -> $cold_job"

  # ---- CE baseline arms ------------------------------------------------------
  CE_COMMON="$BASE_ARGS --freeze_boundary_policy True --coldstart_ckpt_dir $(pwd)/$cold_out/save"
  CE_COMMON+=" --max_batch_length_train 500 --max_batch_length_val 100 --grad_accumulation_factor 2"
  CE_COMMON+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
  ce_dep=""
  [[ "$DRY_RUN" == 1 ]] || ce_dep="--dependency=afterok:$cold_job --kill-on-invalid-dep=yes"

  declare -A phone_jobs=()
  for seed in "${SEEDS[@]}"; do
    for arm in fixed_k5 char_oracle phone_oracle; do
      case "$arm" in
        fixed_k5)
          arm_args="--boundary_source none --fixed_rate_k 5"
          arm_args+=" --warmstart_target alignment --boundary_target_dir $T/boundaries/phone" ;;
        char_oracle)
          arm_args="--boundary_source alignment --warmstart_target alignment"
          arm_args+=" --boundary_target_dir $T/boundaries/$UNITS_LEVEL" ;;
        phone_oracle)
          arm_args="--boundary_source alignment --warmstart_target alignment"
          arm_args+=" --boundary_target_dir $T/boundaries/phone" ;;
      esac
      out=$L/$arm/$seed
      args="$CE_COMMON $arm_args --experiment_name xling_${lang}_${arm} --seed $seed --output_folder $out"
      job=$(EXTRA_ARGS="$args" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_${arm}_$seed" $ce_dep \
        --cpus-per-task=12 --mem=64G --time=1-00:00:00 \
        --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$CE_EPOCHS,WARMUP_EPOCHS=0" \
        run_segmenter.slurm)
      log_job "$lang" "$arm/$seed" "$job" "afterok:$cold_job"
      echo "[$lang] $arm seed=$seed -> $job"
      [[ "$arm" == phone_oracle ]] && phone_jobs[$seed]=$job
    done
  done

  fi
  # ---- learned 24k-step arm ---------------------------------------------------
  for seed in "${SEEDS[@]}"; do
    out=$L/transformer_ar_local64_bigru/$seed
    args="$BASE_ARGS --experiment_name xling_${lang}_learned24k --seed $seed --output_folder $out"
    args+=" --coldstart_ckpt_dir $(pwd)/$cold_out/save"
    args+=" --warmstart_target alignment --boundary_target_dir $T/boundaries/$UNITS_LEVEL"
    args+=" --segmenter_reward nll --freeze_decoder_in_joint False --bilevel_mode off"
    args+=" --rl_update_mode combined_on_policy --grpo_k 4 --pg_weight 1.0 --max_decode_ratio 3.0"
    args+=" --rate_mode band --rho_lo $RHO_LO --rho_hi $RHO_HI --lambda_cap 1.0"
    args+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0"
    args+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002 --lr_segmenter 0.00005"
    args+=" --max_batch_length_train 300 --max_batch_length_val 100 --grad_accumulation_factor 1 --min_batch_ex_train 2"
    args+=" --optimizer_step_limit $MAX_STEPS --validation_interval_optimizer_steps $VALID_INTERVAL"
    args+=" --validate_at_warmup_end True --warmup_optimizer_steps $WARMUP_STEPS"
    args+=" --stage_timing_file $out/stage_timing.jsonl"
    joint_dep=""
    [[ "$DRY_RUN" == 1 ]] || joint_dep="--dependency=afterok:${phone_jobs[$seed]} --kill-on-invalid-dep=yes"
    job=$(EXTRA_ARGS="$args" submit_job "${GPU_IDENTITY[@]}" --job-name="x${lang}_learn_$seed" $joint_dep \
      --cpus-per-task=12 --mem=64G --time=3-00:00:00 \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$SAFETY_EPOCHS,WARMUP_EPOCHS=0,DECODER_SAVE_DIR=$(pwd)/$L/phone_oracle/$seed/save" \
      run_xling_joint.slurm)
    log_job "$lang" "learned24k/$seed" "$job" "afterok:${phone_jobs[$seed]}"
    echo "[$lang] learned 24k seed=$seed -> $job"
  done
  unset phone_jobs
done

echo "Submitted cross-lingual step24k pipeline. Manifest: $MANIFEST"
echo "Band: rho in [$RHO_LO, $RHO_HI] (5-10 Hz); warmup $WARMUP_STEPS/$MAX_STEPS steps"
