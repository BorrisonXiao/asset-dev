#!/bin/bash
# Compute-matched LibriSpeech-960h experiment queue. Every training job uses all
# three official LibriSpeech training splits. Jobs share verified CSV manifests,
# use BF16 dynamic batching, and preserve approximately the optimizer-update
# exposure of the established 100h recipes without repeating ten data passes.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

read -r -a SEEDS <<< "${SEED_LIST:-3407 3408 3409}"
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"
export TRAIN_SPLITS
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
BASE=${OUT_ROOT:-results/speechllm_ls960_scaleup_v2_optimized}
if [[ "$BASE" == /* ]]; then
  BASE_ABS=$BASE
else
  BASE_ABS=$(pwd)/$BASE
fi
MANIFEST=$BASE/submitted_jobs.tsv
CSV_SOURCE=${CSV_SOURCE:-results/speechllm_fixed_pooling/alignment/3407}
CSV_DIR=$BASE/manifests
CSV_DIR_ABS=$BASE_ABS/manifests
DRY_RUN=${DRY_RUN:-0}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-2-00:00:00}

WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
BOUNDARY_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char

# One pass over 960h is approximately the example exposure of ten 100h epochs.
# The learned run uses one pass total: 20% decoder adaptation, then 80% joint RL.
# With a 300 s dynamic-batch budget and no accumulation this yields approximately
# the same number of optimizer updates as the former 100 s x accumulation-4 run.
BASELINE_EPOCHS=1
COLDSTART_EPOCHS=1
WARMUP_EPOCHS=0
WARMUP_FRACTION=0.20
JOINT_RL_EPOCHS=1
TOTAL_JOINT_EPOCHS=$JOINT_RL_EPOCHS

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=64G
  --time="$JOB_TIME_LIMIT"
)

die() {
  echo "$*" >&2
  exit 1
}

submit_job() {
  local priority=$1
  local nice=$2
  local system=$3
  local seed=$4
  local job_name=$5
  local extra_args=$6
  shift 6
  local epochs batch_budget grad_accum warmup_fraction
  case "$system" in
    oracle_char_decoder|fixed_k5)
      epochs=$BASELINE_EPOCHS; batch_budget=500; grad_accum=2; warmup_fraction=0 ;;
    no_downsampling)
      epochs=$BASELINE_EPOCHS; batch_budget=50; grad_accum=20; warmup_fraction=0 ;;
    *_coldstart)
      epochs=$COLDSTART_EPOCHS; batch_budget=400; grad_accum=1; warmup_fraction=0 ;;
    *)
      epochs=$TOTAL_JOINT_EPOCHS; batch_budget=300; grad_accum=1; warmup_fraction=$WARMUP_FRACTION ;;
  esac
  local job
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' env "EXTRA_ARGS=$extra_args" sbatch "${COMMON[@]}" --nice="$nice" --job-name="$job_name" --parsable "$@" >&2
    printf '\n' >&2
    job="DRY_$job_name"
  else
    job=$(EXTRA_ARGS="$extra_args" sbatch "${COMMON[@]}" --nice="$nice" --job-name="$job_name" --parsable "$@")
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$priority" "$system" "$seed" "$job" "$nice" "$TRAIN_SPLITS" \
    "$epochs" "$batch_budget" "$grad_accum" "$warmup_fraction" >> "$MANIFEST"
  echo "$job"
}

[[ "$TRAIN_SPLITS" == *train-clean-100* && "$TRAIN_SPLITS" == *train-clean-360* && "$TRAIN_SPLITS" == *train-other-500* ]] \
  || die "LS960 invariant failed: $TRAIN_SPLITS"
[[ -d "$BOUNDARY_DIR" ]] || die "Missing character-boundary directory: $BOUNDARY_DIR"
target_count=$(find "$BOUNDARY_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l)
(( target_count >= 292367 )) || die "Incomplete full-corpus char targets: found $target_count"
[[ ! -e "$BASE" ]] || die "Refusing to reuse LS960 output root: $BASE"

for csv in train-clean-100 train-clean-360 train-other-500 train dev-clean test-clean test-other; do
  [[ -s "$CSV_SOURCE/$csv.csv" ]] || die "Missing shared LS960 manifest: $CSV_SOURCE/$csv.csv"
done
[[ $(wc -l < "$CSV_SOURCE/train-clean-100.csv") -eq 28540 ]] || die "Bad train-clean-100 manifest"
[[ $(wc -l < "$CSV_SOURCE/train-clean-360.csv") -eq 104015 ]] || die "Bad train-clean-360 manifest"
[[ $(wc -l < "$CSV_SOURCE/train-other-500.csv") -eq 148689 ]] || die "Bad train-other-500 manifest"
[[ $(wc -l < "$CSV_SOURCE/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

mkdir -p "$BASE" "$CSV_DIR" slurm_logs
cp "$CSV_SOURCE"/{train-clean-100,train-clean-360,train-other-500,train,dev-clean,test-clean,test-other}.csv "$CSV_DIR"/
printf 'priority\tsystem\tseed\tjob_id\tnice\ttrain_splits\tepochs\tbatch_budget_s\tgrad_accum\twarmup_fraction\n' > "$MANIFEST"

echo "=== LIBRISPEECH-960H SCALE-UP ==="
echo "Training splits: $TRAIN_SPLITS"
echo "Output root: $BASE"
echo "Shared manifests: $CSV_DIR (281241 training utterances)"
echo "A100 only; one GPU per job"
echo "Time limit: $JOB_TIME_LIMIT"

FIXED_ENCODER_ARGS="--ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM"
SEGMENTER_ENCODER_ARGS="$FIXED_ENCODER_ARGS --ssl_folder $WAVLM_CACHE"
SEGMENTER_ENCODER_ARGS+=" --segmenter_input_dim $WAVLM_DIM"
SCALE_ARGS="--train_splits $TRAIN_SPLITS"
DATA_ARGS="--skip_prep True --csv_folder $CSV_DIR_ABS"
DATA_ARGS+=" --train_csv $CSV_DIR_ABS/train.csv --valid_csv $CSV_DIR_ABS/dev-clean.csv"
DATA_ARGS+=" --test_splits ['test-clean','test-other']"
DATA_ARGS+=" --test_csv [$CSV_DIR_ABS/test-clean.csv,$CSV_DIR_ABS/test-other.csv]"
LOADER_ARGS="--num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2"
COMMON_IO_ARGS="$DATA_ARGS $LOADER_ARGS --ckpt_interval_minutes 30"
COMPRESSED_TRAIN_ARGS="--max_batch_length_train 500 --grad_accumulation_factor 2 --min_batch_ex_train 2"
COMPRESSED_TRAIN_ARGS+=" --initial_lr 0.00025 --final_lr 0.00025"
COLDSTART_TRAIN_ARGS="--max_batch_length_train 400 --grad_accumulation_factor 1 --min_batch_ex_train 2"
JOINT_TRAIN_ARGS="--max_batch_length_train 300 --grad_accumulation_factor 1 --min_batch_ex_train 2"
JOINT_TRAIN_ARGS+=" --warmup_fraction_of_epoch $WARMUP_FRACTION"

declare -A CHAR_JOBS
for seed in "${SEEDS[@]}"; do
  if [[ "$seed" == 3407 ]]; then char_priority=0; char_nice=0; else char_priority=2; char_nice=80; fi
  out="$BASE/oracle_char_decoder/$seed"
  extra="--experiment_name speechllm_ls960_oracle_char $FIXED_ENCODER_ARGS"
  extra+=" $COMMON_IO_ARGS $COMPRESSED_TRAIN_ARGS"
  extra+=" --test_batch_size 8 --stage_timing_file $out/stage_timing.jsonl"
  job=$(submit_job "$char_priority" "$char_nice" oracle_char_decoder "$seed" "l960o_char_$seed" "$extra" \
    --export="ALL,SEED=$seed,EPOCHS=$BASELINE_EPOCHS,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,BOUNDARY_TARGET_DIR=$BOUNDARY_DIR" \
    run_char_alignment.slurm)
  CHAR_JOBS[$seed]=$job
  echo "P$char_priority LS960 oracle-char decoder seed=$seed -> $job"
done

TRANSFORMER_POLICY="--segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4"
TRANSFORMER_POLICY+=" --segmenter_ar_nhead 4 --segmenter_ar_ffn_dim 1024"
TRANSFORMER_POLICY+=" --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096"
TRANSFORMER_POLICY+=" --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated"
BIGRU_ARGS="--segment_pooling bigru_residual --segment_pooling_hidden_dim 128"
BIGRU_ARGS+=" --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0"

tar_cold_out="$BASE/coldstart/transformer_ar_local64/3407"
tar_cold_extra="--experiment_name speechllm_ls960_tar_cold $SEGMENTER_ENCODER_ARGS $TRANSFORMER_POLICY $SCALE_ARGS"
tar_cold_extra+=" $COMMON_IO_ARGS $COLDSTART_TRAIN_ARGS"
tar_cold_extra+=" --seed 3407 --output_folder $tar_cold_out --test_batch_size 8"
tar_cold_job=$(submit_job 0 0 transformer_ar_local64_coldstart 3407 l960o_cs_tar "$tar_cold_extra" \
  --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=coldstart,BACKBONE=transformer_ar,EPOCHS=$COLDSTART_EPOCHS" \
  run_segmenter.slurm)
echo "P0 LS960 Transformer-AR cold start -> $tar_cold_job"

cnn_cold_out="$BASE/coldstart/cnn/3407"
cnn_cold_extra="--experiment_name speechllm_ls960_cnn_cold $SEGMENTER_ENCODER_ARGS $SCALE_ARGS"
cnn_cold_extra+=" $COMMON_IO_ARGS $COLDSTART_TRAIN_ARGS"
cnn_cold_extra+=" --seed 3407 --output_folder $cnn_cold_out --test_batch_size 8"
cnn_cold_job=$(submit_job 0 0 cnn_coldstart 3407 l960o_cs_cnn "$cnn_cold_extra" \
  --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=coldstart,BACKBONE=cnn,EPOCHS=$COLDSTART_EPOCHS" \
  run_segmenter.slurm)
echo "P0 LS960 CNN cold start -> $cnn_cold_job"

COMMON_LEARNED="--segmenter_reward nll --freeze_decoder_in_joint False"
COMMON_LEARNED+=" --initial_lr 0.0002 --lr_decoder 0.0002 --lr_decoder_warmup 0.0002"
COMMON_LEARNED+=" --lr_segmenter 0.00005 --pg_weight 1.0 --grpo_k 4 --max_decode_ratio 3.0"
COMMON_LEARNED+=" --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0"
COMMON_LEARNED+=" --entropy_coeff_init 0.0 --entropy_coeff_final 0.0 --test_batch_size 8"
COMMON_LEARNED+=" $COMMON_IO_ARGS $JOINT_TRAIN_ARGS"

for seed in "${SEEDS[@]}"; do
  if [[ "$seed" == 3407 ]]; then learned_priority=1; learned_nice=20; else learned_priority=2; learned_nice=120; fi
  decoder_save=$BASE_ABS/oracle_char_decoder/$seed/save

  tar_out="$BASE/transformer_ar_local64_bigru/nll_mt/$seed"
  tar_extra="--experiment_name speechllm_ls960_tarb $SEGMENTER_ENCODER_ARGS $TRANSFORMER_POLICY $BIGRU_ARGS $SCALE_ARGS"
  tar_extra+=" --seed $seed --output_folder $tar_out $COMMON_LEARNED"
  tar_extra+=" --rl_update_mode combined_on_policy --stage_timing_file $tar_out/stage_timing.jsonl"
  tar_job=$(submit_job "$learned_priority" "$learned_nice" transformer_ar_local64_bigru "$seed" "l960o_tarb_$seed" "$tar_extra" \
    --dependency="afterok:${CHAR_JOBS[$seed]}:$tar_cold_job" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=transformer_ar,EPOCHS=$TOTAL_JOINT_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$BASE_ABS/coldstart/transformer_ar_local64/3407/save,DECODER_SAVE_DIR=$decoder_save" \
    run_segmenter_ls960_joint.slurm)
  echo "P$learned_priority LS960 Transformer-AR + BiGRU seed=$seed -> $tar_job"

  cnn_out="$BASE/cnn_first_order_ar_bigru/nll_mt/$seed"
  cnn_extra="--experiment_name speechllm_ls960_carb $SEGMENTER_ENCODER_ARGS $BIGRU_ARGS $SCALE_ARGS"
  cnn_extra+=" --seed $seed --output_folder $cnn_out --segmenter_sampling autoregressive $COMMON_LEARNED"
  cnn_extra+=" --rl_update_mode batched_on_policy --stage_timing_file $cnn_out/stage_timing.jsonl"
  cnn_job=$(submit_job "$learned_priority" "$learned_nice" cnn_first_order_ar_bigru "$seed" "l960o_carb_$seed" "$cnn_extra" \
    --dependency="afterok:${CHAR_JOBS[$seed]}:$cnn_cold_job" \
    --export="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,MODE=joint,BACKBONE=cnn,EPOCHS=$TOTAL_JOINT_EPOCHS,WARMUP_EPOCHS=$WARMUP_EPOCHS,COLDSTART_DIR=$BASE_ABS/coldstart/cnn/3407/save,DECODER_SAVE_DIR=$decoder_save" \
    run_segmenter_ls960_joint.slurm)
  echo "P$learned_priority LS960 CNN first-order AR + BiGRU seed=$seed -> $cnn_job"
done

for seed in "${SEEDS[@]}"; do
  if [[ "$seed" == 3407 ]]; then fixed_priority=1; fixed_nice=40; native_priority=1; native_nice=60; else fixed_priority=3; fixed_nice=200; native_priority=4; native_nice=300; fi
  fixed_out="$BASE/fixed_k5/$seed"
  fixed_extra="--experiment_name speechllm_ls960_fixed_k5 $FIXED_ENCODER_ARGS"
  fixed_extra+=" $COMMON_IO_ARGS $COMPRESSED_TRAIN_ARGS"
  fixed_extra+=" --test_batch_size 8 --stage_timing_file $fixed_out/stage_timing.jsonl"
  fixed_job=$(submit_job "$fixed_priority" "$fixed_nice" fixed_k5 "$seed" "l960o_k5_$seed" "$fixed_extra" \
    --export="ALL,FIXED_K=5,SEED=$seed,EPOCHS=$BASELINE_EPOCHS,OUTPUT_FOLDER=$fixed_out,SSL_FOLDER=$WAVLM_CACHE" \
    run_fixed_rate.slurm)
  echo "P$fixed_priority LS960 fixed k=5 seed=$seed -> $fixed_job"

  native_out="$BASE/no_downsampling/$seed"
  native_extra="--experiment_name speechllm_ls960_native $FIXED_ENCODER_ARGS"
  native_extra+=" $COMMON_IO_ARGS --initial_lr 0.00025 --final_lr 0.00025"
  native_extra+=" --stage_timing_file $native_out/stage_timing.jsonl"
  native_job=$(submit_job "$native_priority" "$native_nice" no_downsampling "$seed" "l960o_native_$seed" "$native_extra" \
    --export="ALL,SEED=$seed,EPOCHS=$BASELINE_EPOCHS,OUTPUT_FOLDER=$native_out,SSL_FOLDER=$WAVLM_CACHE" \
    run_no_downsampling.slurm)
  echo "P$native_priority LS960 no-downsampling seed=$seed -> $native_job"
done

echo "Submitted the LS960 scale-up queue. Manifest: $MANIFEST"
echo "Compressed baselines: 1 epoch, 500 s batches x accumulation 2, lr=2.5e-4"
echo "Cold starts: 1 epoch, 400 s batches x accumulation 1"
echo "Learned: 1 epoch total, 20% warmup + 80% joint RL, 300 s batches, K=4"
echo "Native 50 Hz: validated 50 s batches x accumulation 20"
