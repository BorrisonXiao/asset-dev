#!/bin/bash
#SBATCH --job-name=nonasr_mfa_alignment
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/artifacts/segmenter/nonasr_boundary_audit_2026-09-12/logs/%x-%j.out
set -euo pipefail
cd /export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
audit_dir="$PWD/artifacts/segmenter/nonasr_boundary_audit_2026-09-12"
export PATH="$audit_dir/envs/mfa/bin:$PATH"
export MFA_ROOT_DIR="$audit_dir/tools/mfa_home"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
dictionary_path="${1:-$MFA_ROOT_DIR/pretrained_models/dictionary/english_us_arpa.dict}"
alignment_output="${2:-$audit_dir/alignments}"
alignment_work="${3:-$audit_dir/mfa_work}"
echo 'Independent MFA phone/word alignment; CPU only, eight cores; outputs inside the analysis directory'
mfa align "$audit_dir/data/mfa_corpus" \
    "$dictionary_path" \
    "$MFA_ROOT_DIR/pretrained_models/acoustic/english_us_arpa.zip" \
    "$alignment_output" \
    --temporary_directory "$alignment_work" --num_jobs 8 --clean --overwrite \
    --output_format long_textgrid
