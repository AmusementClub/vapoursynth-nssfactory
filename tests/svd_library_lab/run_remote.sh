#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/nss-c4/bin/guest_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/tmp/nss-svd-20260919
lab="$root/source-lab/tests/svd_library_lab"
trap 'rc=$?; printf "%s\n" "$rc" > "$root/run-exit.txt"' EXIT
{
    date -u
    uname -a
    lscpu
    "$root/lab-build/tests/test_backend"
    dpkg-query -W libopenblas0-pthread libgfortran5
    sha256sum /lib/x86_64-linux-gnu/libopenblas.so.0 "$root/lab-build/libnss.so" "$root/baseline/build/libnss.so"
    ldd "$root/lab-build/libnss.so"
    "$NSS_C4_PYTHON" -m pip freeze
} > "$root/environment.txt"
cp /tmp/nss-svd-baseline-build.log "$root/baseline-build.log"
"$NSS_C4_PYTHON" "$lab/campaign.py" run --source "$root/source-lab" \
    --plugin "$root/lab-build/libnss.so" --stock "$root/baseline/build/libnss.so" \
    --fixtures "$root/fixtures" --out "$root/frames"
"$NSS_C4_PYTHON" "$lab/kernel_campaign.py" --frames "$root/frames" \
    --bench "$root/lab-build/svd_lab_bench" --out "$root/kernels"
"$NSS_C4_PYTHON" "$lab/analyze.py" --frames "$root/frames" --kernels "$root/kernels" \
    --fixtures "$root/fixtures" --out "$root/analysis.json" > "$root/analysis.log"
