#!/usr/bin/env bash
# Single-change diagnostic: current LSSC ISTA budget 16 -> 64, isolated source.
set -Eeuo pipefail
deadline=$((SECONDS + 10800))
while [[ ! -f /tmp/nss-paper-confirm128/results.json ]]; do
  if (( SECONDS > deadline )); then
    echo 'Reference confirmation did not finish before the ISTA diagnostic deadline' >&2
    exit 1
  fi
  sleep 10
done
sudo unshare -n bash /tmp/guest_build.sh \
  --source-tar /tmp/nss-paper-ista64-source.tar.gz \
  --work-root /tmp/nss-paper-ista64-build --revision c3d08d2-ista64-ablation \
  > /tmp/nss-paper-ista64-build.log 2>&1
source /opt/nss-c4/bin/guest_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
script=/tmp/nss-paper-harness-next/tests/paper_compare.py
fixtures=/tmp/nss-paper-harness/artifacts/paper-compare-20260908/fixtures128
cases=(house-128-s5 barbara-128-s5 peppers256-128-s5 house-128-s25 barbara-128-s25 peppers256-128-s25)
common=(--fixtures "$fixtures" --reference /tmp/nss-paper-reference --algorithms LSSC --variants default dense --repeats 1)
for image in "${cases[@]}"; do
  for repeat in 0 1 2; do
    order=(16 64)
    if (( repeat % 2 )); then order=(64 16); fi
    for budget in "${order[@]}"; do
      if [[ $budget == 16 ]]; then
        plugin=/tmp/nss-paper-build/build/libnss.so
        output=/tmp/nss-paper-ista16-confirm
      else
        plugin=/tmp/nss-paper-ista64-build/build/libnss.so
        output=/tmp/nss-paper-ista64
      fi
      taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign "${common[@]}" \
        --cases "$image" --repeat-start "$repeat" --plugin "$plugin" --out "$output"
    done
  done
done
echo 'ISTA DIAGNOSTIC COMPLETE'
