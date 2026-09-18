#!/usr/bin/env bash
# Runs from a frozen source snapshot. Submission supplies resources and logging.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm copies this script into its spool directory; --chdir is the source.
    recipe_dir=$PWD
else
    recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
[[ -f "$recipe_dir/train_speechllm_adaptive.py" ]] || { echo 'Missing frozen recipe' >&2; exit 1; }
snapshot_root=$(cd "$recipe_dir/../../../.." && pwd)
cd "$recipe_dir"
arm=${1:?adaptive, original, or fixed_lower}
output=${2:?absolute output directory containing ls960}
mode=${3:-production}
[[ "$output" = /*ls960* ]] || { echo 'Expected absolute LS960 output path' >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HOME=/export/jsalt26/omnienc/users/cxiao/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$snapshot_root:$recipe_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
unset SEGMENTER_EXPERIMENTAL_RUNTIME
python_bin=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
csv_dir=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
source_ckpt=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3407/save/CKPT+2026-08-25+03-01-20+00
config=hparams/adaptive_frequency_ls960.yaml
max_steps=8000
interval=500
smoke_args=()
if [[ "$mode" = smoke ]]; then
    config=hparams/adaptive_frequency_ls960_smoke.yaml
    max_steps=4
    interval=2
    smoke_args=(--continuation-smoke-dev-limit 16)
elif [[ "$mode" != production ]]; then
    echo 'mode must be production or smoke' >&2
    exit 1
fi
train_splits="['train-clean-100','train-clean-360','train-other-500']"
echo "LS960 splits: $train_splits"
echo "arm=$arm seed=3407 mode=$mode output=$output new_updates=$max_steps"
echo "Source checkpoint: $source_ckpt"
echo "Runtime snapshot: $snapshot_root"
args=(
    hparams/speechllm_segmenter.yaml
    --continuation-config "$config" --continuation-arm "$arm"
    --seed 3407 --segmenter_mode joint --segmenter_backbone transformer_ar
    --output_folder "$output" --experiment_name speechllm_ls960_adaptive_frequency
    --data_folder /export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech
    --skip_prep True --csv_folder "$csv_dir" --train_csv "$csv_dir/train.csv"
    --valid_csv "$csv_dir/dev-clean.csv" --test_csv "[$csv_dir/dev-other.csv]"
    --train_splits "$train_splits" --dev_splits "['dev-clean']" --test_splits "['dev-other']"
    --ssl_hub microsoft/wavlm-large --ssl_feat_dims 1024 --segmenter_input_dim 1024
    --ssl_folder "$HF_HUB_CACHE" --llm_save_path "$HF_HUB_CACHE"
    --segmenter_init_checkpoint "$source_ckpt/segmenter.ckpt" --decoder_init_ckpt_dir "$source_ckpt"
    --warmstart_target none --boundary_source none --coldstart_ckpt_dir null
    --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4 --segmenter_ar_nhead 4
    --segmenter_ar_ffn_dim 1024 --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096
    --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated
    --segment_pooling bigru_residual --segment_pooling_hidden_dim 128
    --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0
    --rl_update_mode combined_on_policy --segmenter_reward nll --grpo_k 4
    --grpo_normalize_std True --pg_weight 1.0 --bilevel_mode off
    --freeze_boundary_policy False --freeze_decoder_in_joint False
    --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0
    --entropy_coeff_init 0.0 --entropy_coeff_final 0.0
    --initial_lr 0.00002 --lr_decoder 0.00002 --lr_decoder_warmup 0.00002
    --lr_segmenter 0.000005 --weight_decay 0.0 --max_grad_norm 1.0
    --warmup_epochs 0 --warmup_optimizer_steps 0 --warmup_fraction_of_epoch null
    --validate_at_warmup_end False --optimizer_step_limit "$max_steps"
    --validation_interval_optimizer_steps "$interval" --number_of_epochs 10000
    --max_batch_length_train 300 --grad_accumulation_factor 1 --min_batch_ex_train 2
    --num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2
    --test_batch_size 8 --precision bf16 --eval_precision bf16
    --ckpt_interval_minutes 30 --noprogressbar
    "${smoke_args[@]}"
)
if [[ "$mode" = smoke ]]; then
    # Exit after a real mid-epoch checkpoint, restart the process, finish at 4.
    "$python_bin" train_speechllm_adaptive.py "${args[@]}" --continuation-smoke-pause-step 2
    "$python_bin" train_speechllm_adaptive.py "${args[@]}"
    "$python_bin" verify_adaptive_frequency_smoke.py "$output"
else
    exec "$python_bin" train_speechllm_adaptive.py "${args[@]}"
fi
