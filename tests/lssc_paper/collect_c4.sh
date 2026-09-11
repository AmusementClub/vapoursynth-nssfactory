#!/usr/bin/env bash
# Task-specific evidence collection; never bundles author assets or a venv.
set -Eeuo pipefail

case "$(hostname)" in
  nss-c4-lssc-scale-20260908-a1) ;;
  *) echo 'refusing collection on an unexpected host' >&2; exit 2 ;;
esac
source_root=/tmp/nss-lssc-scale-build
result_root=/tmp/nss-lssc-scale-results
paper_python=/tmp/nss-lssc-python/bin/python
archive=/tmp/nss-lssc-scale-evidence.tar.gz
test ! -e "$archive"
test -f "$result_root/python-house256/result.json"
test -f "$result_root/budgets-barbara512/result.json"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/tmp/nss-lssc-python/lib:/tmp/denoise_iccv09/libs
cd "$source_root"
taskset -c 0 "$paper_python" -m pytest -q tests/lssc_paper > "$result_root/final-pytest.log" 2>&1
sudo /opt/nss-c4/bin/guest_verify.sh > "$result_root/final-guest-verify.log" 2>&1
uname -a > "$result_root/kernel.txt"
lscpu > "$result_root/lscpu.txt"
"$paper_python" -m pip freeze > "$result_root/python-packages.txt"
dpkg-query -W octave libblas3 liblapack3 libc6 libstdc++6 > "$result_root/system-packages.txt"
systemctl is-enabled apt-daily.timer apt-daily-upgrade.timer > "$result_root/apt-timer-state.txt" || true
cp /var/log/apt/history.log "$result_root/apt-history.log"
ldd /tmp/denoise_iccv09/mexDenoise.mexa64 > "$result_root/author-runtime-resolution.txt"
sha256sum /tmp/denoise_iccv09/mexDenoise.mexa64 /tmp/denoise_iccv09/dicts/dict_n9.mat \
  /tmp/nss-lssc-python/lib/libimf.so /tmp/nss-lssc-python/lib/libintlc.so.5 \
  "$source_root/build/libnss.so" > "$result_root/external-binary-hashes.txt"
cp "$source_root/build/libnss.so" "$result_root/clean-base-libnss.so"
cp "$source_root/build/Testing/Temporary/LastTest.log" "$result_root/native-ctest.log"
mkdir "$result_root/harnesses"
cp "$source_root"/tests/lssc_paper/*.py "$result_root/harnesses/"
cp "$source_root"/tests/lssc_paper/collect_c4.sh "$result_root/harnesses/"
sha256sum "$result_root"/harnesses/* > "$result_root/harness-hashes-guest-paths.txt"

cd /tmp
tar -czf "$archive" nss-lssc-scale-results nss-lssc-build.log \
  nss-lssc-octave-install.log nss-lssc-python-install.log \
  house-128-s25-noisy.f32 house-clean.f32 house-256-s25-noisy.f32 house256-clean.f32 \
  noisy.f32 clean.f32
sha256sum nss-lssc-scale-evidence.tar.gz > nss-lssc-scale-evidence.tar.gz.sha256
cat "$result_root/final-pytest.log"
cat nss-lssc-scale-evidence.tar.gz.sha256
