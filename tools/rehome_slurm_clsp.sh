#!/usr/bin/env bash
# Rewrite skipjack Slurm identity flags to their CLSP equivalents.
#
# skipjack (JSALT)            CLSP
#   --account=jsalt2026-lgarci27  --account=highprio   (required by gpu-a100)
#   --partition=a100              --partition=gpu-a100 (e02-e05, 8x A100)
#   --partition=med               --partition=cpu
#   --reservation="JSALT 2026"    (dropped; CLSP has no such reservation)
#   --exclude=ga129[,ga132]       (dropped; those are skipjack nodes)
#   --comment=accept_cost         (kept; free-form and harmless on CLSP)
#
# Reservation/exclude flags are deleted as tokens rather than as whole lines so
# that `\` line continuations and `COMMON=( ... )` arrays stay syntactically
# intact. The now-unused RESERVATION/RESERVATION_NAME variables are left in
# place so the mapping can be reversed if the work ever moves back.
#
# Idempotent. Usage: tools/rehome_slurm_clsp.sh [--dry-run]

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

mapfile -t targets < <(
  find recipes tools -type f \( -name '*.sh' -o -name '*.slurm' \) \
    -not -path 'recipes/LibriSpeech/ASR/transformer/results/*' \
    -not -path 'tools/rehome_slurm_clsp.sh' \
    -not -path 'tools/rehome_paths_clsp.sh' \
    -not -path 'tools/port_jointllm_to_clsp.sh' \
    -print0 |
    xargs -0 grep -l -e 'jsalt2026-lgarci27' -e 'partition=a100' \
      -e 'partition=med' -e 'reservation=' -e 'exclude=ga1' \
      -e 'EXCLUDE_NODES' || true
)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No skipjack Slurm flags left to rewrite."
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${targets[@]}"
  printf '%d file(s) would be rewritten.\n' "${#targets[@]}"
  exit 0
fi

sed -i \
  -e 's/--account=jsalt2026-lgarci27/--account=highprio/g' \
  -e 's/--partition=a100/--partition=gpu-a100/g' \
  -e 's/--partition=med/--partition=cpu/g' \
  -e 's/[[:space:]]*--reservation="JSALT 2026"//g' \
  -e "s/[[:space:]]*--reservation='JSALT 2026'//g" \
  -e 's/[[:space:]]*--reservation="\$RESERVATION_NAME"//g' \
  -e 's/[[:space:]]*--reservation="\$RESERVATION"//g' \
  -e 's/[[:space:]]*--exclude=ga129,ga132//g' \
  -e 's/[[:space:]]*--exclude=ga129//g' \
  -e 's/[[:space:]]*--exclude=ga132//g' \
  -e 's/[[:space:]]*--exclude="\${EXCLUDE_NODES:-ga129}"//g' \
  "${targets[@]}"
printf 'Rewrote %d file(s).\n' "${#targets[@]}"

remaining=$(grep -rln -e 'jsalt2026-lgarci27' -e '--partition=a100' \
  -e '--partition=med' -e 'reservation=' -e 'exclude=ga1' \
  "${targets[@]}" || true)
if [[ -n "${remaining}" ]]; then
  echo "Unmapped skipjack Slurm flags remain in:" >&2
  printf '%s\n' "${remaining}" >&2
  exit 1
fi
echo "Verified: no skipjack Slurm identity flags remain."
