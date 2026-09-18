#!/bin/bash
# One-pass LS960 CE baseline using frozen ~11-Hz phone-CTC boundaries.
set -euo pipefail

cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer

NUM_SHARDS=${NUM_SHARDS:-16}
MAX_GENERATORS=${MAX_GENERATORS:-4}
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"
BOUNDARY_DIR=/export/jsalt26/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_ls960_best_longsplit8
OUT_ROOT=results/speechllm_ls960_phone_ctc_longsplit8_mean_ce_1ep
OUT_ROOT_ABS=$(pwd)/$OUT_ROOT
CSV_SOURCE=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
CSV_DIR=$OUT_ROOT_ABS/manifests
MANIFEST=$OUT_ROOT_ABS/submitted_jobs.tsv
SEEDS=(3407 3408 3409)
CTC_RANKING=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/models/wavlm_phone_ctc_tc100_seed3407/ranking.json

die() { echo "$*" >&2; exit 1; }
[[ "$TRAIN_SPLITS" == *train-clean-100* ]] || die "Missing train-clean-100"
[[ "$TRAIN_SPLITS" == *train-clean-360* ]] || die "Missing train-clean-360"
[[ "$TRAIN_SPLITS" == *train-other-500* ]] || die "Missing train-other-500"
[[ -s "$CTC_RANKING" ]] \
  || die "Missing trained phone-CTC checkpoint ranking"
[[ ! -e "$BOUNDARY_DIR" ]] || die "Refusing to overwrite $BOUNDARY_DIR"
[[ ! -e "$OUT_ROOT" ]] || die "Refusing to overwrite $OUT_ROOT"
for csv in train-clean-100 train-clean-360 train-other-500 train dev-clean dev-other test-clean test-other; do
  [[ -s "$CSV_SOURCE/$csv.csv" ]] || die "Missing LS960 manifest: $CSV_SOURCE/$csv.csv"
done
[[ $(wc -l < "$CSV_SOURCE/train-clean-100.csv") -eq 28540 ]] || die "Bad train-clean-100 manifest"
[[ $(wc -l < "$CSV_SOURCE/train-clean-360.csv") -eq 104015 ]] || die "Bad train-clean-360 manifest"
[[ $(wc -l < "$CSV_SOURCE/train-other-500.csv") -eq 148689 ]] || die "Bad train-other-500 manifest"
[[ $(wc -l < "$CSV_SOURCE/train.csv") -eq 281242 ]] || die "Bad merged LS960 manifest"

mkdir -p "$BOUNDARY_DIR" "$CSV_DIR" slurm_logs
cp "$CSV_SOURCE"/{train-clean-100,train-clean-360,train-other-500,train,dev-clean,dev-other,test-clean,test-other}.csv "$CSV_DIR"/
printf 'stage\tseed\tjob_id\tdependency\ttrain_splits\tepochs\tobjective\n' > "$MANIFEST"

echo "=== WAVLM PHONE-CTC LONG-SPLIT8 LS960 CE ==="
echo "Training splits: $TRAIN_SPLITS"
echo "Boundary inference splits: train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
echo "Boundary output: $BOUNDARY_DIR (estimated 1.2 GB)"
echo "Training outputs: $OUT_ROOT/{3407,3408,3409} (estimated 4.2 GB total)"
echo "Total estimated new storage including logs/manifests: under 7 GB"
echo "GPU plan: up to $MAX_GENERATORS one-A100 boundary shards, then three one-A100 training seeds"
echo "Training: one full LS960 pass, regular CE, parameter-free mean pooling, decoder projection + LoRA only"

gpu_identity=(
  --account=highprio
  --comment=accept_cost
  --partition=gpu-a100
)

gen_job=$(sbatch --parsable \
  "${gpu_identity[@]}" \
  --array="0-$((NUM_SHARDS - 1))%$MAX_GENERATORS" \
  --export="ALL,BOUNDARY_DIR=$BOUNDARY_DIR,NUM_SHARDS=$NUM_SHARDS" \
  generate_wavlm_phone_ctc_longsplit8_ls960.slurm)
printf 'boundary_generation\t-\t%s\t-\t%s\t0\tfrozen_phone_ctc_audio_only\n' \
  "$gen_job" "$TRAIN_SPLITS" >> "$MANIFEST"

gate_job=$(sbatch --parsable \
  --account=highprio \
  --comment=accept_cost \
  --partition=cpu \
  --dependency="afterok:$gen_job" \
  --export="ALL,BOUNDARY_DIR=$BOUNDARY_DIR,NUM_SHARDS=$NUM_SHARDS" \
  validate_wavlm_phone_ctc_longsplit8_ls960.slurm)
printf 'boundary_validation\t-\t%s\t%s\t%s\t0\tcount_and_rate_gate\n' \
  "$gate_job" "$gen_job" "$TRAIN_SPLITS" >> "$MANIFEST"

export TRAIN_SPLITS TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/$seed"
  extra="--experiment_name speechllm_ls960_phone_ctc_longsplit8_mean_ce"
  extra+=" --ssl_hub microsoft/wavlm-large --ssl_feat_dims 1024"
  extra+=" --skip_prep True --csv_folder $CSV_DIR"
  extra+=" --train_csv $CSV_DIR/train.csv --valid_csv $CSV_DIR/dev-clean.csv"
  extra+=" --test_splits ['test-clean','test-other','dev-other']"
  extra+=" --test_csv [$CSV_DIR/test-clean.csv,$CSV_DIR/test-other.csv,$CSV_DIR/dev-other.csv]"
  extra+=" --num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2"
  extra+=" --max_batch_length_train 500 --grad_accumulation_factor 2 --min_batch_ex_train 2"
  extra+=" --initial_lr 0.00025 --final_lr 0.00025 --test_batch_size 8"
  extra+=" --ckpt_interval_minutes 30 --stage_timing_file $out/stage_timing.jsonl"
  train_job=$(EXTRA_ARGS="$extra" sbatch --parsable \
    "${gpu_identity[@]}" \
    --dependency="afterok:$gate_job" \
    --cpus-per-task=12 --mem=64G --time=3-00:00:00 \
    --job-name="l960_ph11_$seed" \
    --export="ALL,SEED=$seed,EPOCHS=1,OUTPUT_FOLDER=$out,SSL_FOLDER=/export/jsalt26/omnienc/users/cxiao/hf/hub,BOUNDARY_TARGET_DIR=$BOUNDARY_DIR" \
    run_char_alignment.slurm)
  printf 'decoder_ce\t%s\t%s\t%s\t%s\t1\tregular_ce_projection_plus_lora\n' \
    "$seed" "$train_job" "$gate_job" "$TRAIN_SPLITS" >> "$MANIFEST"
  echo "seed=$seed train=$train_job dependency=$gate_job"
done

echo "boundary_generation=$gen_job boundary_validation=$gate_job"
echo "Submitted manifest: $MANIFEST"
