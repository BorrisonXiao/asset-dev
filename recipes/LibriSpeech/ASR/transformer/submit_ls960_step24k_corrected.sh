#!/bin/bash
# Corrected LS960 learned-system runs controlled by optimizer steps rather than
# epoch boundaries. Uses completed LS960 cold starts and oracle-char decoders.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"
SOURCE_ROOT=${SOURCE_ROOT:-results/speechllm_ls960_scaleup_v2_optimized}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/speechllm_ls960_step24k_corrected}
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-24000}
VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-4000}
WARMUP_FRACTION=${WARMUP_FRACTION:-0.20}
SAFETY_EPOCH_LIMIT=${SAFETY_EPOCH_LIMIT:-3}
DRY_RUN=${DRY_RUN:-0}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-2-00:00:00}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
BOUNDARY_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char
CSV_DIR=$SOURCE_ROOT/manifests
MANIFEST=$OUTPUT_ROOT/submitted_jobs.tsv

die() {
  echo "$*" >&2
  exit 1
}

[[ "$TRAIN_SPLITS" == *train-clean-100* && "$TRAIN_SPLITS" == *train-clean-360* && "$TRAIN_SPLITS" == *train-other-500* ]] \
  || die "LS960 split invariant failed: $TRAIN_SPLITS"
[[ "$OUTPUT_ROOT" == *ls960* ]] || die "Output root must contain ls960"
[[ "$MAX_OPTIMIZER_STEPS" -gt "$VALIDATION_INTERVAL_STEPS" ]] \
  || die "max optimizer steps must exceed the validation interval"
[[ ! -e "$OUTPUT_ROOT" ]] || die "Refusing to reuse output root: $OUTPUT_ROOT"
[[ -d "$BOUNDARY_DIR" ]] || die "Missing char boundary targets: $BOUNDARY_DIR"

for csv in train-clean-100 train-clean-360 train-other-500 train dev-clean test-clean test-other; do
  [[ -s "$CSV_DIR/$csv.csv" ]] || die "Missing LS960 manifest: $CSV_DIR/$csv.csv"
done
[[ $(wc -l < "$CSV_DIR/train-clean-100.csv") -eq 28540 ]] || die "Bad train-clean-100 manifest"
[[ $(wc -l < "$CSV_DIR/train-clean-360.csv") -eq 104015 ]] || die "Bad train-clean-360 manifest"
[[ $(wc -l < "$CSV_DIR/train-other-500.csv") -eq 148689 ]] || die "Bad train-other-500 manifest"
[[ $(wc -l < "$CSV_DIR/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

tar_cold=$SOURCE_ROOT/coldstart/transformer_ar_local64/3407/save
cnn_cold=$SOURCE_ROOT/coldstart/cnn/3407/save
find "$tar_cold" -maxdepth 2 -name segmenter.ckpt -size +0c -print -quit | grep -q . \
  || die "Missing Transformer cold-start checkpoint under $tar_cold"
find "$cnn_cold" -maxdepth 2 -name segmenter.ckpt -size +0c -print -quit | grep -q . \
  || die "Missing CNN cold-start checkpoint under $cnn_cold"
for seed in "${SEEDS[@]}"; do
  decoder_save=$SOURCE_ROOT/oracle_char_decoder/$seed/save
  find "$decoder_save" -maxdepth 2 -name llm.ckpt -size +0c -print -quit | grep -q . \
    || die "Missing oracle-char decoder checkpoint for seed $seed"
done

mkdir -p "$OUTPUT_ROOT" slurm_logs
printf 'system\tseed\tjob_id\ttrain_splits\tmax_optimizer_steps\tvalidation_interval_steps\twarmup_fraction\toutput\n' > "$MANIFEST"

COMMON_SBATCH=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=64G
  --time="$JOB_TIME_LIMIT"
)

DATA_ARGS="--skip_prep True --csv_folder $(pwd)/$CSV_DIR"
DATA_ARGS+=" --train_csv $(pwd)/$CSV_DIR/train.csv --valid_csv $(pwd)/$CSV_DIR/dev-clean.csv"
DATA_ARGS+=" --test_splits ['test-clean','test-other']"
DATA_ARGS+=" --test_csv [$(pwd)/$CSV_DIR/test-clean.csv,$(pwd)/$CSV_DIR/test-other.csv]"
DATA_ARGS+=" --train_splits $TRAIN_SPLITS"
LOADER_ARGS="--num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2"
TRAIN_ARGS="--max_batch_length_train 300 --grad_accumulation_factor 1 --min_batch_ex_train 2"
STEP_ARGS="--optimizer_step_limit $MAX_OPTIMIZER_STEPS"
STEP_ARGS+=" --validation_interval_optimizer_steps $VALIDATION_INTERVAL_STEPS"
STEP_ARGS+=" --validate_at_warmup_end True --warmup_fraction_of_epoch $WARMUP_FRACTION"
ENCODER_ARGS="--ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM --ssl_folder $WAVLM_CACHE"
ENCODER_ARGS+=" --segmenter_input_dim $WAVLM_DIM --boundary_target_dir $BOUNDARY_DIR"
COMMON_LEARNED="--segmenter_reward nll --freeze_decoder_in_joint False --bilevel_mode off"
COMMON_LEARNED+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002"
COMMON_LEARNED+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
COMMON_LEARNED+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
COMMON_LEARNED+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0 --test_batch_size 8"
COMMON_LEARNED+=" --ckpt_interval_minutes 30 --precision bf16 --eval_precision bf16"
COMMON_LEARNED+=" $DATA_ARGS $LOADER_ARGS $TRAIN_ARGS $STEP_ARGS $ENCODER_ARGS"

TRANSFORMER_POLICY="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
TRANSFORMER_POLICY+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
TRANSFORMER_POLICY+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
TRANSFORMER_POLICY+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
BIGRU_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
BIGRU_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

submit_one() {
  local system=$1
  local seed=$2
  local job_name=$3
  local output=$4
  local coldstart_dir=$5
  local backbone=$6
  local system_args=$7
  local decoder_save=$SOURCE_ROOT/oracle_char_decoder/$seed/save
  local extra_args
  extra_args="--experiment_name speechllm_ls960_step24k_${system} --seed $seed --output_folder $output"
  extra_args+=" --stage_timing_file $output/stage_timing.jsonl $COMMON_LEARNED $BIGRU_ARGS $system_args"
  local job_id
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' env "EXTRA_ARGS=$extra_args" sbatch "${COMMON_SBATCH[@]}" --job-name="$job_name" \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=$backbone,EPOCHS=$SAFETY_EPOCH_LIMIT,WARMUP_EPOCHS=0,COLDSTART_DIR=$(pwd)/$coldstart_dir,DECODER_SAVE_DIR=$(pwd)/$decoder_save" \
      run_segmenter_ls960_joint.slurm >&2
    printf '\n' >&2
    job_id="DRY_$job_name"
  else
    job_id=$(EXTRA_ARGS="$extra_args" sbatch "${COMMON_SBATCH[@]}" --job-name="$job_name" --parsable \
      --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=$backbone,EPOCHS=$SAFETY_EPOCH_LIMIT,WARMUP_EPOCHS=0,COLDSTART_DIR=$(pwd)/$coldstart_dir,DECODER_SAVE_DIR=$(pwd)/$decoder_save" \
      run_segmenter_ls960_joint.slurm)
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$system" "$seed" "$job_id" "$TRAIN_SPLITS" "$MAX_OPTIMIZER_STEPS" \
    "$VALIDATION_INTERVAL_STEPS" "$WARMUP_FRACTION" "$output" >> "$MANIFEST"
  echo "$system seed=$seed -> $job_id"
}

echo "=== LS960 STEP-CONTROLLED CORRECTED RUNS ==="
echo "Training splits: $TRAIN_SPLITS"
echo "Output root: $OUTPUT_ROOT"
echo "Stop: $MAX_OPTIMIZER_STEPS optimizer steps (~2 LS960 passes)"
echo "Validate: warmup end plus every $VALIDATION_INTERVAL_STEPS steps"
echo "Warmup: first $WARMUP_FRACTION of one LS960 pass; then joint RL"
echo "Resources/job: 1 A100, 12 CPUs, 64 GB RAM"
echo "Time limit: $JOB_TIME_LIMIT"

for seed in "${SEEDS[@]}"; do
  submit_one \
    transformer_ar_local64_bigru "$seed" "ls960s_tarb_$seed" \
    "$OUTPUT_ROOT/transformer_ar_local64_bigru/nll_mt/$seed" "$tar_cold" transformer_ar \
    "$TRANSFORMER_POLICY --rl_update_mode combined_on_policy"
  submit_one \
    cnn_first_order_ar_bigru "$seed" "ls960s_carb_$seed" \
    "$OUTPUT_ROOT/cnn_first_order_ar_bigru/nll_mt/$seed" "$cnn_cold" cnn \
    "--segmenter_sampling autoregressive --rl_update_mode batched_on_policy"
done

echo "Submitted corrected LS960 step-controlled queue. Manifest: $MANIFEST"
