#!/usr/bin/env bash
# Rewrite skipjack absolute audio paths inside the ported CoVoST 2 / MuST-C
# manifests to their CLSP equivalents.
#
# The LibriSpeech manifests use SpeechBrain's `$data_root/` placeholder and are
# cluster-portable as-is. The ST manifests, produced by covost2_prepare.py /
# mustc_prepare.py, instead store one absolute path per utterance in the `wav`
# column, so they need this pass after the corpora tars are extracted.
#
# Idempotent. Usage: tools/rehome_manifests_clsp.sh [--dry-run]

set -euo pipefail

datasets=/export/jsalt26/omnienc/users/cxiao/datasets
old_weka=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/datasets
old_home=/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/users/cxiao/datasets

mapfile -t targets < <(
  find "${datasets}" -type f -name '*.csv' -print0 |
    xargs -0 grep -l -e "${old_weka}" -e "${old_home}" || true
)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No skipjack paths left in the manifests."
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${targets[@]}"
  printf '%d manifest(s) would be rewritten.\n' "${#targets[@]}"
  exit 0
fi

sed -i \
  -e "s|${old_weka}|${datasets}|g" \
  -e "s|${old_home}|${datasets}|g" \
  "${targets[@]}"
printf 'Rewrote %d manifest(s).\n' "${#targets[@]}"

# Sample one audio path per rewritten manifest and confirm it now resolves.
status=0
for csv in "${targets[@]}"; do
  wav=$(tail -n +2 "${csv}" | head -1 | cut -d, -f3)
  if [[ -f "${wav}" ]]; then
    echo "ok   $(basename "${csv}") -> ${wav}"
  else
    echo "FAIL $(basename "${csv}") -> ${wav}" >&2
    status=1
  fi
done
exit "${status}"
