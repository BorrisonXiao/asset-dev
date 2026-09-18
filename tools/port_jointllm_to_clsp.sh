#!/usr/bin/env bash

set -euo pipefail

source_root=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm
environment_root=/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm
uv_binary=/home/jhu/jsalt2026-ext-cxiao7/.local/bin/uv
dataset_root=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/datasets
hf_root=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/hf/hub
user_root=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao
phone_ctc_archive_root=${source_root}/artifacts/portability/wavlm_boundaries
destination_host=ctl2
destination_root=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
bwlimit_kib=25600

rsync_base=(
  ionice -c2 -n7
  nice -n 10
  rsync -a
  --human-readable
  --stats
  --partial
  --partial-dir=.rsync-partial
  --bwlimit="${bwlimit_kib}"
)

remote_mkdir() {
  local remote_dir=$1
  case "${remote_dir}" in
    "${destination_root}"|"${destination_root}"/*) ;;
    *)
      printf 'Refusing destination outside %s: %s\n' \
        "${destination_root}" "${remote_dir}" >&2
      return 2
      ;;
  esac
  ssh "${destination_host}" mkdir -p -- "${remote_dir}"
}

sync_tree() {
  local source_dir=$1
  local remote_dir=$2
  test -d "${source_dir}"
  remote_mkdir "${remote_dir}"
  "${rsync_base[@]}" "${source_dir}/" \
    "${destination_host}:${remote_dir}/"
}

sync_core() {
  remote_mkdir "${destination_root}"
  "${rsync_base[@]}" \
    --exclude '/recipes/LibriSpeech/ASR/transformer/results/' \
    --exclude '/.publish_staging/' \
    --exclude '/.pytest_cache/' \
    --exclude '/.ruff_cache/' \
    --exclude '**/__pycache__/' \
    --exclude '*.pyc' \
    "${source_root}/" "${destination_host}:${destination_root}/"
}

sync_environment() {
  sync_tree "${environment_root}" \
    "${destination_root}/_portability/envs/jointllm"
  sync_tooling
}

sync_tooling() {
  local remote_dir=${destination_root}/_portability/bin
  test -x "${uv_binary}"
  remote_mkdir "${remote_dir}"
  "${rsync_base[@]}" "${uv_binary}" \
    "${destination_host}:${remote_dir}/uv"
}

sync_data_archives() {
  local remote_dir=${destination_root}/_portability/data
  remote_mkdir "${remote_dir}/wavlm_boundaries"
  "${rsync_base[@]}" \
    "${dataset_root}/wavlm_boundaries/char.tgz" \
    "${dataset_root}/wavlm_boundaries/phone.tgz" \
    "${dataset_root}/wavlm_boundaries/syllable.tgz" \
    "${dataset_root}/wavlm_boundaries/word.tgz" \
    "${dataset_root}/wavlm_boundaries/word_ctc.tgz" \
    "${destination_host}:${remote_dir}/wavlm_boundaries/"
  "${rsync_base[@]}" \
    "${dataset_root}/temp/alignments_tar/librispeech_alignments.tar.gz" \
    "${destination_host}:${remote_dir}/"
}

sync_phone_ctc_boundary_archives() {
  local remote_dir=${destination_root}/_portability/data/wavlm_boundaries
  test -f "${phone_ctc_archive_root}/phone_ctc_tc100_best_longsplit8.tgz"
  test -f "${phone_ctc_archive_root}/phone_ctc_ls960_best_longsplit8.tgz"
  remote_mkdir "${remote_dir}"
  "${rsync_base[@]}" \
    "${phone_ctc_archive_root}/phone_ctc_tc100_best_longsplit8.tgz" \
    "${phone_ctc_archive_root}/phone_ctc_ls960_best_longsplit8.tgz" \
    "${destination_host}:${remote_dir}/"
}

sync_models() {
  local remote_dir=${destination_root}/_portability/hf/hub
  sync_tree "${hf_root}/models--microsoft--wavlm-large" \
    "${remote_dir}/models--microsoft--wavlm-large"
  sync_tree "${hf_root}/models--meta-llama--Llama-3.2-1B-Instruct" \
    "${remote_dir}/models--meta-llama--Llama-3.2-1B-Instruct"
  sync_tree "${hf_root}/models--facebook--wav2vec2-base" \
    "${remote_dir}/models--facebook--wav2vec2-base"
  sync_tree "${user_root}/ssl_cache/models--facebook--wav2vec2-base-960h" \
    "${remote_dir}/models--facebook--wav2vec2-base-960h"
}

sync_ls960_completed() {
  sync_tree \
    "${source_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_scaleup_v2_optimized" \
    "${destination_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_scaleup_v2_optimized"
}

sync_corrected_ls960() {
  if [[ ${CONFIRM_QUIESCENT:-0} != 1 ]]; then
    printf '%s\n' \
      'Set CONFIRM_QUIESCENT=1 only after the corrected LS960 jobs stop writing.' >&2
    return 2
  fi
  sync_tree \
    "${source_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_step24k_corrected" \
    "${destination_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_step24k_corrected"
}

sync_best_100h() {
  local source_parent=${source_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter_wavlm
  local remote_parent=${destination_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter_wavlm
  sync_tree "${source_parent}/compact_position_retrain_v1" \
    "${remote_parent}/compact_position_retrain_v1"
  sync_tree "${source_parent}/oracleclose_bestinit" \
    "${remote_parent}/oracleclose_bestinit"
}

sync_broad_sweep() {
  sync_tree \
    "${source_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy" \
    "${destination_root}/recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy"
}

sync_segmenters() {
  local results_root=${source_root}/recipes/LibriSpeech/ASR/transformer/results
  local remote_results=${destination_root}/recipes/LibriSpeech/ASR/transformer/results
  sync_tree "${results_root}/speechllm_segmenter_wavlm" \
    "${remote_results}/speechllm_segmenter_wavlm"
  sync_tree "${results_root}/speechllm_segmenter" \
    "${remote_results}/speechllm_segmenter"
}

sync_result_metadata() {
  local results_root=${source_root}/recipes/LibriSpeech/ASR/transformer/results
  local remote_results=${destination_root}/recipes/LibriSpeech/ASR/transformer/results
  remote_mkdir "${remote_results}"
  "${rsync_base[@]}" \
    --exclude '*.ckpt' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.safetensors' \
    --exclude '*.bin' \
    "${results_root}/" "${destination_host}:${remote_results}/"
}

sync_all_results() {
  sync_tree \
    "${source_root}/recipes/LibriSpeech/ASR/transformer/results" \
    "${destination_root}/recipes/LibriSpeech/ASR/transformer/results"
}

sync_raw_librispeech() {
  sync_tree "${dataset_root}/LibriSpeech" \
    "${destination_root}/_portability/data/LibriSpeech"
}

phase=${1:-}
case "${phase}" in
  core) sync_core ;;
  environment) sync_environment ;;
  tooling) sync_tooling ;;
  data-archives) sync_data_archives ;;
  phone-ctc-boundaries) sync_phone_ctc_boundary_archives ;;
  models) sync_models ;;
  ls960-completed) sync_ls960_completed ;;
  corrected-ls960) sync_corrected_ls960 ;;
  best-100h) sync_best_100h ;;
  broad-sweep) sync_broad_sweep ;;
  segmenters) sync_segmenters ;;
  result-metadata) sync_result_metadata ;;
  all-results) sync_all_results ;;
  raw-librispeech) sync_raw_librispeech ;;
  priority-static)
    sync_core
    sync_tooling
    sync_data_archives
    sync_models
    sync_ls960_completed
    sync_best_100h
    sync_broad_sweep
    ;;
  *)
    printf 'Usage: %s {%s}\n' "$0" \
      'core|environment|tooling|data-archives|phone-ctc-boundaries|models|ls960-completed|corrected-ls960|best-100h|broad-sweep|segmenters|result-metadata|all-results|raw-librispeech|priority-static' >&2
    exit 2
    ;;
esac
