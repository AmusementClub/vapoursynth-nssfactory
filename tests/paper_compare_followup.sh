#!/usr/bin/env bash
# Run on this campaign's isolated C4 after the main matrix finishes.
set -Eeuo pipefail
source /opt/nss-c4/bin/guest_env.sh
export LD_LIBRARY_PATH="/opt/nss-c4/runtime/vsenv/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/tmp/nss-paper-harness-next
script="$root/tests/paper_compare.py"
fixtures128=/tmp/nss-paper-harness/artifacts/paper-compare-20260908/fixtures128
fixtures256="$root/artifacts/paper-compare-20260908/fixtures256"
common=(--reference /tmp/nss-paper-reference --plugin /tmp/nss-paper-build/build/libnss.so --timeout 1200)
deadline=$((SECONDS + 7200))
while [[ ! -f /tmp/nss-paper-matrix128/results.json ]]; do
  if (( SECONDS > deadline )); then
    echo 'Main matrix did not finish within the followup deadline' >&2
    exit 1
  fi
  sleep 10
done
echo 'START NCSR matched public parameters'
taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign --fixtures "$fixtures128" \
  --out /tmp/nss-paper-matched128 --algorithms NCSR --variants matched \
  --cases house-128-s25 barbara-128-s25 peppers256-128-s25 --repeats 3 "${common[@]}"
echo 'START 256x256 size check'
taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign --fixtures "$fixtures256" \
  --out /tmp/nss-paper-size256 "${common[@]}"
echo 'START 128x128 independent confirmation'
taskset -c 0 "$NSS_C4_PYTHON" "$script" campaign --fixtures "$fixtures128" \
  --out /tmp/nss-paper-confirm128 --cases house-128-s25 --variants full fast dense default "${common[@]}"
echo 'FOLLOWUP COMPLETE'
