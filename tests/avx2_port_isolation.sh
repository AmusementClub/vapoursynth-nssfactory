#!/usr/bin/env bash
set -euo pipefail
source /opt/nss-c4/bin/guest_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=/tmp/nss-avx2-evidence/avx3-isolation
mkdir "$out"
sudo systemctl set-property --runtime system.slice AllowedCPUs=0-1
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-1
for side in baseline port; do
  root=/tmp/nss-avx2-$side
  mask=0
  if [[ $side == port ]]; then mask=2047; fi
  cmake -S "$root" -B "$root/build-avx3" -DCMAKE_BUILD_TYPE=Release -DNSS_BM_EXPERIMENT=2305 -DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT="$mask" -DNSS_HWY_TARGET_MODE=avx3 -DNSS_GIT_DESCRIBE=avx2-port -DNSS_ENABLE_CUDA=OFF -DNSS_ENABLE_VULKAN=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVapourSynth_INCLUDE_DIR="$NSS_C4_VS_PREFIX/lib/python3/dist-packages/vapoursynth/include" -DFETCHCONTENT_SOURCE_DIR_HIGHWAY="$NSS_C4_HIGHWAY_SOURCE" > "$out/$side-build.log" 2>&1
  cmake --build "$root/build-avx3" -j2 >> "$out/$side-build.log" 2>&1
  cp "$root/build-avx3/libnss.so" "$out/$side.so"
  cp "$root/build-avx3/compile_commands.json" "$out/$side-compile.json"
  ctest --test-dir "$root/build-avx3" --output-on-failure > "$out/$side-ctest.log" 2>&1
done
if cmp -s "$out/baseline.so" "$out/port.so"; then
  echo byte_identical > "$out/identity.txt"
else
  echo 'machine_code_differs; inspect codecheck and whole-filter regression' > "$out/identity.txt"
  "$NSS_C4_PYTHON" /tmp/nss-avx2-port/tests/avx2_avx3_codecheck.py "$out/baseline.so" "$out/port.so" "$out/codecheck"
  "$NSS_C4_PYTHON" /tmp/nss-avx2-port/tests/c4_bm_numeric_matrix.py --baseline "$out/baseline.so" --candidate "$out/port.so" --out "$out/numeric"
  NSS_SO="$out/port.so" "$NSS_C4_PYTHON" /tmp/nss-avx2-port/tests/test_bm3d_semantics.py
  "$NSS_C4_PYTHON" /tmp/nss-avx2-port/tests/avx2_final_configs.py "$out/configs"
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0
  "$NSS_C4_PYTHON" /tmp/nss-avx2-port/tests/c4_paired_bm.py run --baseline "$out/baseline.so" --candidate "$out/port.so" --configs "$out/configs/avx3.json" --out "$out/bench" --pairs 7 --group-seconds 45 --selection-threshold 1.02
fi
echo completed > "$out/status"
