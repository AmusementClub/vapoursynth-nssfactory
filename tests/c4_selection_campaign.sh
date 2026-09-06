#!/usr/bin/env bash
set -euo pipefail
source /opt/nss-c4/bin/guest_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
lane=$1
root=/tmp/nss-selection
out=/tmp/nss-selection-evidence
mkdir "$out"
sudo /opt/nss-c4/bin/guest_verify.sh > "$out/guest-verify.log" 2>&1
sudo unshare -n /tmp/guest_build.sh --source-tar /tmp/selection-source.tar.gz --work-root "$root" --revision selection-20260906 > "$out/baseline-build.log" 2>&1
cmake -S "$root" -B "$root/build" -DNSS_BM_EXPERIMENT=0 >> "$out/baseline-build.log" 2>&1
cmake --build "$root/build" -j2 >> "$out/baseline-build.log" 2>&1
ctest --test-dir "$root/build" --output-on-failure >> "$out/baseline-build.log" 2>&1
cp "$root/build/libnss.so" "$out/baseline.so"
cp /tmp/selection-source.tar.gz.sha256 "$out/"
cp "$root/tests/c4_selection_campaign.sh" "$out/"
"$NSS_C4_PYTHON" "$root/tests/c4_selection_configs.py" "$out/configs"
"$NSS_C4_PYTHON" "$root/tests/test_c4_bm_budget.py" > "$out/budget-tests.log" 2>&1
case "$lane" in
  tlv1) masks='1 2'; matrix=topk ;;
  tlv2) masks='2048 4096 8192'; matrix=buffers ;;
  yul) masks='16384 32768 256'; matrix=ring ;;
  *) exit 2 ;;
esac
for mask in $masks; do
  phase="$out/mask-$mask"
  mkdir "$phase"
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0-1
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0-1
  cmake -S "$root" -B "$root/build" -DNSS_BM_EXPERIMENT="$mask" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$phase/build.log" 2>&1
  cmake --build "$root/build" -j2 >> "$phase/build.log" 2>&1
  cp "$root/build/libnss.so" "$phase/candidate.so"
  cp "$root/build/compile_commands.json" "$phase/"
  cp "$root/build/CMakeCache.txt" "$phase/"
  ctest --test-dir "$root/build" --output-on-failure > "$phase/ctest.log" 2>&1
  NSS_SO="$phase/candidate.so" "$NSS_C4_PYTHON" "$root/tests/test_bm3d_semantics.py" > "$phase/semantics.log" 2>&1
  if ! "$NSS_C4_PYTHON" "$root/tests/c4_bm_numeric_matrix.py" --baseline "$out/baseline.so" --candidate "$phase/candidate.so" --out "$phase/numerics" > "$phase/numerics.log" 2>&1; then
    echo numerical_failure > "$phase/status"; continue
  fi
  sudo bash "$root/tests/c4_bm_quiet_kernel.sh" "$phase/kernel-affinity" > "$phase/quiet.log" 2>&1
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0
  "$NSS_C4_PYTHON" "$root/tests/c4_paired_bm.py" run --baseline "$out/baseline.so" --candidate "$phase/candidate.so" --configs "$out/configs/$matrix.json" --out "$phase/bench" --pairs 7 --group-seconds 45 --selection-threshold 1.02 > "$phase/bench.log" 2>&1
  echo completed > "$phase/status"
  cat "$phase/bench/selection.json"
done
echo completed > "$out/status"
