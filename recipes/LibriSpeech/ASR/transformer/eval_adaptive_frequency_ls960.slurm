#!/usr/bin/env bash
# Resources and per-arm output paths are supplied at submission.
set -euo pipefail
training_root=${1:?absolute training root}
output=${2:?absolute evaluation directory}
entrypoint=${3:?frozen evaluation entrypoint}
[[ "$training_root" = /*ls960* && "$output" = /*ls960* ]]
[[ -d "$output/checkpoint_view/CKPT+continuation-step-008000" ]]
export ADAPTIVE_EVAL_SOURCE="$training_root/source"
recipe_dir="$ADAPTIVE_EVAL_SOURCE/recipes/LibriSpeech/ASR/transformer"
cd "$recipe_dir"
export PYTHONPATH="$ADAPTIVE_EVAL_SOURCE:$recipe_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub
export HF_HOME=/export/jsalt26/omnienc/users/cxiao/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
unset SEGMENTER_EXPERIMENTAL_RUNTIME
python_bin=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
csv_dir=/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests
train_splits="['train-clean-100','train-clean-360','train-other-500']"
echo "Training provenance: LS960 splits=$train_splits; seed=3407; 24000+8000 updates"
echo "TEST ONLY: test-clean + test-other; batch=8; precision=bf16; output=$output"
echo "Runtime snapshot=$ADAPTIVE_EVAL_SOURCE; checkpoint step=8000"
exec "$python_bin" "$entrypoint" hparams/speechllm_segmenter.yaml \
    --seed 3407 --segmenter_mode joint --segmenter_backbone transformer_ar \
    --output_folder "$output" --save_folder "$output/checkpoint_view" \
    --train_log "$output/eval.log" --output_wer_folder "$output/wer_results" \
    --stage_timing_file "$output/stage_timing.jsonl" \
    --inference_benchmark_file "$output/benchmark.jsonl" --inference_warmup_batches 5 \
    --experiment_name speechllm_ls960_adaptive_frequency_test --eval_only True \
    --data_folder /export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech \
    --skip_prep True --csv_folder "$csv_dir" --train_csv "$csv_dir/train.csv" \
    --valid_csv "$csv_dir/dev-clean.csv" \
    --test_csv "[$csv_dir/test-clean.csv,$csv_dir/test-other.csv]" \
    --train_splits "$train_splits" --dev_splits "['dev-clean']" \
    --test_splits "['test-clean','test-other']" \
    --ssl_hub microsoft/wavlm-large --ssl_feat_dims 1024 --segmenter_input_dim 1024 \
    --ssl_folder "$HF_HUB_CACHE" --llm_save_path "$HF_HUB_CACHE" \
    --segmenter_init_checkpoint null --decoder_init_ckpt_dir null --coldstart_ckpt_dir null \
    --warmstart_target none --boundary_source none \
    --segmenter_ar_hidden_dim 256 --segmenter_ar_num_layers 4 --segmenter_ar_nhead 4 \
    --segmenter_ar_ffn_dim 1024 --segmenter_ar_dropout 0.0 --segmenter_ar_max_positions 4096 \
    --segmenter_ar_history_window 64 --segmenter_ar_cache_mode preallocated \
    --segment_pooling bigru_residual --segment_pooling_hidden_dim 128 \
    --segment_pooling_num_layers 1 --segment_pooling_dropout 0.0 \
    --rl_update_mode combined_on_policy --segmenter_reward nll --grpo_k 4 \
    --grpo_normalize_std True --pg_weight 1.0 --bilevel_mode off \
    --freeze_boundary_policy False --freeze_decoder_in_joint False \
    --rate_mode band --rho_lo 0.15 --rho_hi 0.25 --lambda_cap 1.0 \
    --entropy_coeff_init 0.0 --entropy_coeff_final 0.0 \
    --initial_lr 0.00002 --lr_decoder 0.00002 --lr_decoder_warmup 0.00002 \
    --lr_segmenter 0.000005 --weight_decay 0.0 --max_grad_norm 1.0 \
    --warmup_epochs 0 --warmup_optimizer_steps 0 --warmup_fraction_of_epoch null \
    --validate_at_warmup_end False --number_of_epochs 10000 \
    --max_batch_length_train 300 --grad_accumulation_factor 1 --min_batch_ex_train 2 \
    --num_workers 4 --pin_memory True --persistent_workers True --prefetch_factor 2 \
    --test_batch_size 8 --precision bf16 --eval_precision bf16 --noprogressbar
