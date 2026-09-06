#!/usr/bin/env bash
set -euo pipefail
# Explicit same-guest source, immutable baseline, and bounded candidate list.
source /opt/nss-c4/bin/guest_env.sh
root=$1
base=$2
shift 2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for mask in "$@"; do
  out="${NSS_CAMPAIGN_OUT:-/tmp/nss-campaign}/mask-$mask"
  mkdir -p "$out"
  build="$root/build-$mask"
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0-1
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0-1
  cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DNSS_BM_EXPERIMENT="$mask" \
    -DNSS_GIT_DESCRIBE="bm-correct-exp-$mask" -DNSS_ENABLE_CUDA=OFF -DNSS_ENABLE_VULKAN=OFF \
    -DVapourSynth_INCLUDE_DIR="$NSS_C4_VS_PREFIX/lib/python3/dist-packages/vapoursynth/include" \
    -DFETCHCONTENT_SOURCE_DIR_HIGHWAY="$NSS_C4_HIGHWAY_SOURCE" > "$out/build.log" 2>&1
  if ! cmake --build "$build" -j2 >> "$out/build.log" 2>&1; then
    echo "build_failed" > "$out/status"; continue
  fi
  sha256sum "$build/libnss.so" > "$out/plugin.sha256"
  if ! ctest --test-dir "$build" --output-on-failure > "$out/ctest.log" 2>&1; then
    echo "ctest_failed" > "$out/status"; continue
  fi
  if ! NSS_SO="$build/libnss.so" taskset -c 0 "$NSS_C4_PYTHON" "$root/tests/test_bm3d_semantics.py" > "$out/semantics.log" 2>&1; then
    echo "semantics_failed" > "$out/status"; continue
  fi
  if ! "$NSS_C4_PYTHON" "$root/tests/c4_bm_numeric_matrix.py" --baseline "$base" --candidate "$build/libnss.so" --out "$out/numerics" > "$out/numerics.log" 2>&1; then
    echo "numeric_replay_required" > "$out/status"; continue
  fi
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0 > "$out/quiet.log" 2>&1
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0 >> "$out/quiet.log" 2>&1
  "$NSS_C4_PYTHON" "$root/tests/c4_paired_bm.py" run --baseline "$base" --candidate "$build/libnss.so" \
    --configs "$root/tests/c4_configs_$mask.json" --out "$out/screen" --pairs 7 --group-seconds 40 > "$out/screen.log" 2>&1 || { echo "screen_error" > "$out/status"; continue; }
  "$NSS_C4_PYTHON" - "$out" <<'PYCODE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'screen/decision.json').read_text());(p/'status').write_text('screen_passed' if v['passed'] else 'screen_not_admitted')
PYCODE
  echo "candidate $mask $(cat "$out/status")"
done
