#!/bin/bash
# Evaluate each retained downstream checkpoint by validation-WER rank.
# Configuration is supplied by the submitter via OUTPUT_FOLDER and EXTRA_ARGS.
#SBATCH --job-name=tc100_phbg_eval
#SBATCH --array=0-2%1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%A_%a.out

set -euo pipefail

: "${OUTPUT_FOLDER:?set OUTPUT_FOLDER}"
: "${EXTRA_ARGS:?set EXTRA_ARGS}"

RECIPE=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/recipes/LibriSpeech/ASR/transformer
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
RANK=$SLURM_ARRAY_TASK_ID
NUMBER_OF_EPOCHS=${NUMBER_OF_EPOCHS:-4}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE=/export/jsalt26/omnienc/users/cxiao/hf/hub

cd "$RECIPE"
ARGS=(hparams/speechllm_segmenter.yaml
  --segmenter_mode joint
  --segmenter_backbone transformer_ar
  --number_of_epochs "$NUMBER_OF_EPOCHS"
  --output_folder "$OUTPUT_FOLDER"
  --save_folder "$OUTPUT_FOLDER/save"
  --output_wer_folder "$OUTPUT_FOLDER/wer_results_rank_$RANK"
  --train_log "$OUTPUT_FOLDER/eval_rank_$RANK.log"
  --inference_benchmark_file "$OUTPUT_FOLDER/benchmark_rank_$RANK.jsonl"
  --eval_only True
  --eval_checkpoint_rank "$RANK"
  --checkpoints_to_keep 3)
ARGS+=($EXTRA_ARGS)

echo "Evaluating downstream retained checkpoint rank=$RANK from $OUTPUT_FOLDER/save"
"$PY" train_speechllm_with_segmenter.py "${ARGS[@]}"
