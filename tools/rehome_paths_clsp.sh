#!/usr/bin/env bash
# Rewrite skipjack (JSALT/weka) absolute paths to their CLSP equivalents.
#
# The 2026-08-24..30 port (tools/port_jointllm_to_clsp.sh) copied bytes only:
# every launcher, SLURM wrapper and hparams file still pointed at
# /weka/scratch/jhu/jsalt2026-lgarci27/... and /home/jhu/jsalt2026-ext-cxiao7/...,
# neither of which exists on CLSP. This script performs the one-time rewrite.
#
# Idempotent: re-running it is a no-op once every path has been rewritten.
# Scope: executable/config files only. Project notes, reports and results/
# keep their original paths because they are historical records of runs that
# really did execute under those paths.
#
# Usage:
#   tools/rehome_paths_clsp.sh --dry-run    # list files that would change
#   tools/rehome_paths_clsp.sh              # rewrite in place

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

new_user=/export/jsalt26/omnienc/users/cxiao
new_repo=${new_user}/skipjack/jointllm
new_env=${new_user}/envs/jointllm
new_hf=${new_user}/hf/hub
new_datasets=${new_user}/datasets

old_weka=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc
old_home=/home/jhu/jsalt2026-ext-cxiao7

# Longest prefixes first: sed applies these in order per line.
mappings=(
  "${old_weka}/users/cxiao/jointllm|${new_repo}"
  "${old_weka}/users/cxiao/envs/jointllm|${new_env}"
  "${old_weka}/users/cxiao/ssl_cache|${new_hf}"
  "${old_weka}/users/cxiao|${new_user}"
  "${old_weka}/datasets|${new_datasets}"
  "${old_weka}|${new_user}"
  "${old_home}/scratch_jsalt2026-lgarci27/omnienc/hf/hub|${new_hf}"
  "${old_home}/scratch_jsalt2026-lgarci27/omnienc/datasets|${new_datasets}"
  "${old_home}/scratch_jsalt2026-lgarci27/omnienc/users/cxiao|${new_user}"
  "${old_home}/cxiao/envs/jointllm|${new_env}"
  "${old_home}/.local/bin/uv|${new_repo}/_portability/bin/uv"
  "${old_home}/.cache|${new_user}/.cache"
  "${old_home}|${new_user}"
  # $HOME-relative interpreter references (skipjack $HOME was ${old_home}).
  "\$HOME/cxiao/envs/jointllm|${new_env}"
  "~/cxiao/envs/jointllm|${new_env}"
)

sed_args=()
for mapping in "${mappings[@]}"; do
  sed_args+=(-e "s|${mapping%%|*}|${mapping##*|}|g")
done

# tools/port_jointllm_to_clsp.sh names the *source* cluster on purpose.
mapfile -t targets < <(
  find recipes speechbrain tests templates tools \
    -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.sh' -o -name '*.slurm' \) \
    -not -path 'recipes/LibriSpeech/ASR/transformer/results/*' \
    -not -path '*/__pycache__/*' \
    -not -path 'tools/port_jointllm_to_clsp.sh' \
    -not -path 'tools/rehome_paths_clsp.sh' \
    -print0 |
    xargs -0 grep -l -e "${old_weka}" -e "${old_home}" \
      -e '\$HOME/cxiao/envs/jointllm' -e '~/cxiao/envs/jointllm' || true
)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No skipjack paths left to rewrite."
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${targets[@]}"
  printf '%d file(s) would be rewritten.\n' "${#targets[@]}"
  exit 0
fi

sed -i "${sed_args[@]}" "${targets[@]}"
printf 'Rewrote %d file(s).\n' "${#targets[@]}"

remaining=$(grep -rl -e "${old_weka}" -e "${old_home}" \
  -e '\$HOME/cxiao/envs/jointllm' -e '~/cxiao/envs/jointllm' "${targets[@]}" || true)
if [[ -n "${remaining}" ]]; then
  echo "Unmapped skipjack paths remain in:" >&2
  printf '%s\n' "${remaining}" >&2
  exit 1
fi
echo "Verified: no skipjack paths remain in the rewritten files."
