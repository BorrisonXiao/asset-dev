#!/bin/bash
# Resume the three completed LS960 phone-CTC decoder-CE runs for one more pass.
set -euo pipefail

cd /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer

RESERVATION=${RESERVATION:-JSALT 2026}
TRAIN_SPLITS="['train-clean-100','train-clean-360','train-other-500']"
BOUNDARY_DIR=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets/wavlm_phone_ctc_ls960_best_longsplit8
OUT_ROOT=results/speechllm_ls960_phone_ctc_longsplit8_mean_ce_1ep
OUT_ROOT_ABS=$(pwd)/$OUT_ROOT
CSV_DIR=$OUT_ROOT_ABS/manifests
MANIFEST=$OUT_ROOT_ABS/submitted_jobs.tsv
SEEDS=(3407 3408 3409)

die() { echo "$*" >&2; exit 1; }
[[ "$TRAIN_SPLITS" == *train-clean-100* ]] || die "Missing train-clean-100"
[[ "$TRAIN_SPLITS" == *train-clean-360* ]] || die "Missing train-clean-360"
[[ "$TRAIN_SPLITS" == *train-other-500* ]] || die "Missing train-other-500"
[[ -s "$BOUNDARY_DIR/_stats/combined_summary.json" ]] \
  || die "Missing validated LS960 boundary summary"
for csv in train dev-clean dev-other test-clean test-other; do
  [[ -s "$CSV_DIR/$csv.csv" ]] || die "Missing staged manifest: $CSV_DIR/$csv.csv"
done
for seed in "${SEEDS[@]}"; do
  save="$OUT_ROOT/$seed/save"
  [[ $(find "$save" -maxdepth 1 -type d -name 'CKPT*' | wc -l) -eq 1 ]] \
    || die "Expected exactly one resumable checkpoint for seed $seed"
  ckpt=$(find "$save" -maxdepth 1 -type d -name 'CKPT*' -print -quit)
  for file in CKPT.yaml counter.ckpt model_optimizer.ckpt llm.ckpt proj.ckpt normalize.ckpt; do
    [[ -s "$ckpt/$file" ]] || die "Incomplete checkpoint for seed $seed: $file"
  done
  grep -q '^epoch: 1$' "$ckpt/CKPT.yaml" \
    || die "Seed $seed checkpoint is not the completed first-stage checkpoint"
done

echo "=== CONTINUE WAVLM PHONE-CTC LONG-SPLIT8 LS960 CE ==="
echo "Training splits: $TRAIN_SPLITS"
echo "Resume source/output: $OUT_ROOT/{3407,3408,3409}"
echo "Estimated additional storage: under 1 GB because best-checkpoint retention replaces superseded checkpoints"
echo "Compute: three one-A100 jobs; ga129 excluded"
echo "Training objective: regular CE; parameter-free mean pooling; decoder projection + LoRA only"

identity=(
  --account=jsalt2026-lgarci27
  --comment=accept_cost
  --partition=a100
  --exclude=ga129
)
if [[ -n "$RESERVATION" ]]; then
  identity+=(--reservation="$RESERVATION")
fi

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
  job=$(EXTRA_ARGS="$extra" sbatch --parsable \
    "${identity[@]}" \
    --cpus-per-task=12 --mem=64G --time=1-00:00:00 \
    --job-name="l960_ph11x_$seed" \
    --export="ALL,SEED=$seed,EPOCHS=2,OUTPUT_FOLDER=$out,SSL_FOLDER=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub,BOUNDARY_TARGET_DIR=$BOUNDARY_DIR" \
    run_char_alignment.slurm)
  printf 'decoder_ce_continuation\t%s\t%s\t-\t%s\t1\tregular_ce_projection_plus_lora\n' \
    "$seed" "$job" "$TRAIN_SPLITS" >> "$MANIFEST"
  echo "seed=$seed continuation=$job"
done
