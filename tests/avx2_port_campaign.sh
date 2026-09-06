#!/usr/bin/env bash
# Sequential build/verify/time on one host. Archive scripts before running.
set -euo pipefail
lane=$1
source /opt/nss-c4/bin/guest_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/tmp/nss-avx2-port
out=/tmp/nss-avx2-evidence
base=/tmp/nss-avx2-baseline/build/libnss.so
test -s "$base"
cd /tmp
sha256sum -c nss-avx2-source-a.tar.gz.sha256 > "$out/source-check.txt"
mkdir "$root"
tar -xzf nss-avx2-source-a.tar.gz -C "$root"
cp /tmp/avx2_port_campaign.sh "$out/"
cp "$base" "$out/baseline.so"
cp /tmp/nss-avx2-source-a.tar.gz.sha256 /tmp/nss-avx2-baseline-e639c8b.tar.gz.sha256 "$out/"
"$NSS_C4_PYTHON" "$root/tests/avx2_port_configs.py" "$out/configs"
build() {
  local mask=$1 phase=$2
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0-1
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0-1
  cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DNSS_BM_EXPERIMENT=2305 -DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT="$mask" -DNSS_HWY_TARGET_MODE=avx2 -DNSS_GIT_DESCRIBE=avx2-port -DNSS_ENABLE_CUDA=OFF -DNSS_ENABLE_VULKAN=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVapourSynth_INCLUDE_DIR="$NSS_C4_VS_PREFIX/lib/python3/dist-packages/vapoursynth/include" -DFETCHCONTENT_SOURCE_DIR_HIGHWAY="$NSS_C4_HIGHWAY_SOURCE" > "$phase/build.log" 2>&1
  cmake --build "$root/build" -j2 >> "$phase/build.log" 2>&1
}
mkdir "$out/mask-0"
build 0 "$out/mask-0"
ctest --test-dir "$root/build" --output-on-failure > "$out/mask-0/ctest.log" 2>&1
cp "$root/build/libnss.so" "$out/mask-0/candidate.so"
if cmp -s "$base" "$root/build/libnss.so"; then echo identical > "$out/baseline-identity.txt"; else echo differs > "$out/baseline-identity.txt"; fi
case "$lane" in
  tlv1) masks='1 2 256' ;;
  tlv2) masks='4 8 512 1024' ;;
  yul) masks='16 32 64 96 128' ;;
  *) exit 2 ;;
esac
for mask in $masks; do
  phase="$out/mask-$mask"
  mkdir "$phase"
  if ! build "$mask" "$phase"; then echo build_failed > "$phase/status"; continue; fi
  cp "$root/build/libnss.so" "$phase/candidate.so"
  cp "$root/build/compile_commands.json" "$root/build/CMakeCache.txt" "$phase/"
  objdump -d -C "$phase/candidate.so" > "$phase/assembly.txt"
  if ! ctest --test-dir "$root/build" --output-on-failure > "$phase/ctest.log" 2>&1; then echo ctest_failed > "$phase/status"; continue; fi
  if ! NSS_SO="$phase/candidate.so" "$NSS_C4_PYTHON" "$root/tests/test_bm3d_semantics.py" > "$phase/semantics.log" 2>&1; then echo semantics_failed > "$phase/status"; continue; fi
  if ! "$NSS_C4_PYTHON" "$root/tests/c4_bm_numeric_matrix.py" --baseline "$base" --candidate "$phase/candidate.so" --out "$phase/numeric" > "$phase/numeric.log" 2>&1; then echo numeric_failed > "$phase/status"; continue; fi
  if ! "$NSS_C4_PYTHON" "$root/tests/c4_bm_numeric_matrix.py" --baseline "$base" --candidate "$phase/candidate.so" --configs "$out/configs/$mask-numeric.json" --out "$phase/extra-numeric" > "$phase/extra-numeric.log" 2>&1; then echo extra_numeric_failed > "$phase/status"; continue; fi
  sudo bash "$root/tests/c4_bm_quiet_kernel.sh" "$phase/kernel-affinity" > "$phase/quiet.log" 2>&1
  sudo systemctl set-property --runtime system.slice AllowedCPUs=0
  sudo systemctl set-property --runtime user.slice AllowedCPUs=0
  if ! "$NSS_C4_PYTHON" "$root/tests/c4_paired_bm.py" run --baseline "$base" --candidate "$phase/candidate.so" --configs "$out/configs/$mask.json" --out "$phase/bench" --pairs 7 --group-seconds 45 --selection-threshold 1.02 > "$phase/bench.log" 2>&1; then echo bench_failed > "$phase/status"; continue; fi
  echo screened > "$phase/status"
  cat "$phase/bench/selection.json"
done
echo screened > "$out/status"
