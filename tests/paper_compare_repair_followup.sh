#!/usr/bin/env bash
set -Eeuo pipefail
deadline=$((SECONDS + 7200))
until grep -q 'ISTA DIAGNOSTIC COMPLETE' /tmp/nss-paper-ista-followup.log; do
  if (( SECONDS > deadline )); then exit 1; fi
  sleep 10
done
source /opt/nss-c4/bin/guest_env.sh
export LD_LIBRARY_PATH="/opt/nss-c4/runtime/vsenv/lib:${LD_LIBRARY_PATH:-}:/tmp/nss-paper-reference/denoise_iccv09/libs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=true
script=/tmp/paper_compare-final.py
fixtures=/tmp/nss-paper-harness/artifacts/paper-compare-20260908/fixtures128
common=(--fixtures "$fixtures" --reference /tmp/nss-paper-reference --plugin /tmp/nss-paper-build/build/libnss.so --algorithms NCSR --timeout 1200)
taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign "${common[@]}" \
  --cases barbara-128-s50 --variants default dense fast full --out /tmp/nss-paper-timing-repair
taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign "${common[@]}" \
  --cases house-128-s50 --variants dense --out /tmp/nss-paper-timing-repair
taskset -c 0 "$NSS_C4_PYTHON" "$script" reference --fixtures "$fixtures" \
  --reference /tmp/nss-paper-reference --case house-128-s25 --algorithm LSSC --variant fast \
  --output /tmp/nss-paper-input-check.f32 --timeout 1200
echo 'TIMING REPAIR AND INPUT CHECK COMPLETE'
