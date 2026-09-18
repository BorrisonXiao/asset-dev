#!/bin/bash
# One-time repair for LS960 jobs submitted before the fixed/char/native wrappers
# accepted TRAIN_SPLITS through the environment. Replaces only those wrappers
# and rewires the already queued learned-system dependencies.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"
BASE=results/speechllm_ls960_scaleup_v1
WAVLM_ID=microsoft/wavlm-large
WAVLM_DIM=1024
WAVLM_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
BOUNDARY_DIR=/export/jsalt26/omnienc/users/cxiao/datasets/wavlm_boundaries/char

COMMON=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
  --gpus=1
  --cpus-per-task=12
  --mem=60G
  --time=3-00:00:00
)

declare -A OLD_CHAR=([3407]=105254 [3408]=105255 [3409]=105256)
declare -A TAR_JOB=([3407]=105259 [3408]=105261 [3409]=105263)
declare -A CNN_JOB=([3407]=105260 [3408]=105262 [3409]=105264)
TAR_COLD_JOB=105257
CNN_COLD_JOB=105258

# Cancel only wrapper jobs whose old command contains the duplicate split key.
scancel 105254 105255 105256 105265 105266 105267 105268 105269 105270 || true

declare -A NEW_CHAR
for seed in 3407 3408 3409; do
  out="$BASE/oracle_char_decoder/$seed"
  extra="--experiment_name speechllm_ls960_oracle_char --ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM"
  extra+=" --test_batch_size 8 --stage_timing_file $out/stage_timing.jsonl"
  job=$(TRAIN_SPLITS="$TRAIN_SPLITS" EXTRA_ARGS="$extra" sbatch "${COMMON[@]}" \
    --nice=0 --job-name="l960r_char_$seed" --parsable \
    --export="ALL,SEED=$seed,EPOCHS=1,OUTPUT_FOLDER=$out,SSL_FOLDER=$WAVLM_CACHE,BOUNDARY_TARGET_DIR=$BOUNDARY_DIR" \
    run_char_alignment.slurm)
  NEW_CHAR[$seed]=$job
  echo "replacement oracle-char seed=$seed: ${OLD_CHAR[$seed]} -> $job"
done

for seed in 3407 3408 3409; do
  scontrol update JobId="${TAR_JOB[$seed]}" \
    Dependency="afterok:${NEW_CHAR[$seed]}:$TAR_COLD_JOB"
  scontrol update JobId="${CNN_JOB[$seed]}" \
    Dependency="afterok:${NEW_CHAR[$seed]}:$CNN_COLD_JOB"
  echo "rewired seed=$seed learned jobs to oracle-char ${NEW_CHAR[$seed]}"
done

for seed in 3407 3408 3409; do
  fixed_out="$BASE/fixed_k5/$seed"
  fixed_extra="--experiment_name speechllm_ls960_fixed_k5 --ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM"
  fixed_extra+=" --test_batch_size 8 --stage_timing_file $fixed_out/stage_timing.jsonl"
  fixed_job=$(TRAIN_SPLITS="$TRAIN_SPLITS" EXTRA_ARGS="$fixed_extra" sbatch "${COMMON[@]}" \
    --nice=200 --job-name="l960r_k5_$seed" --parsable \
    --export="ALL,FIXED_K=5,SEED=$seed,EPOCHS=1,OUTPUT_FOLDER=$fixed_out,SSL_FOLDER=$WAVLM_CACHE" \
    run_fixed_rate.slurm)
  echo "replacement fixed-k5 seed=$seed -> $fixed_job"

  native_out="$BASE/no_downsampling/$seed"
  native_extra="--experiment_name speechllm_ls960_native --ssl_hub $WAVLM_ID --ssl_feat_dims $WAVLM_DIM"
  native_extra+=" --stage_timing_file $native_out/stage_timing.jsonl"
  native_job=$(TRAIN_SPLITS="$TRAIN_SPLITS" EXTRA_ARGS="$native_extra" sbatch "${COMMON[@]}" \
    --nice=300 --job-name="l960r_native_$seed" --parsable \
    --export="ALL,SEED=$seed,EPOCHS=1,OUTPUT_FOLDER=$native_out,SSL_FOLDER=$WAVLM_CACHE" \
    run_no_downsampling.slurm)
  echo "replacement no-downsampling seed=$seed -> $native_job"
done
