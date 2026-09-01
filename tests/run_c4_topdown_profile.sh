#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE=${NSS_PROFILE_SCRIPT:?set NSS_PROFILE_SCRIPT}
PLUGIN=${NSS_SO:?set NSS_SO}
PYTHON=${NSS_PROFILE_PYTHON:?set NSS_PROFILE_PYTHON}
OUT=${NSS_PROFILE_OUT:?set NSS_PROFILE_OUT}

mkdir -p "$OUT"
test -x "$PYTHON"
test -f "$PROFILE"
test -f "$PLUGIN"
test "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0

export NSS_SO="$PLUGIN"
export PYTHONPATH=${NSS_PROFILE_PYTHONPATH:?set NSS_PROFILE_PYTHONPATH}
export LD_LIBRARY_PATH=${NSS_PROFILE_LIBRARY_PATH:?set NSS_PROFILE_LIBRARY_PATH}
export NSS_PROFILE_WIDTH=1920
export NSS_PROFILE_HEIGHT=1080

specs=(
    "nlm 20"
    "bm3d 40"
    "wnnm 12"
    "twsc 6"
    "ncsr 6"
    "lssc 5"
    "nlh 1"
    "mcwnnm 1"
)

{
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    lscpu
    perf --version
    c++ --version | head -n 1
    cmake --version | head -n 1
    sha256sum "$PROFILE" "$PLUGIN"
    echo "candidate_revision=${NSS_PROFILE_REVISION:-unknown}"
    echo "source_sha256=${NSS_PROFILE_SOURCE_SHA256:-unknown}"
    echo "cpu_affinity=0"
    echo "sibling_cpu=1"
    printf 'spec=%s\n' "${specs[@]}"
} >"$OUT/metadata.txt"

pgrep -af 'bench_cpu|cc1plus|ninja|profile_cpu_all' >"$OUT/processes_before.txt" || true
ps -eLo psr,pcpu,pid,comm,args --sort=-pcpu | head -20 >"$OUT/cpu_before.txt"

for level in TopdownL1 TopdownL2 TopdownL3; do
    for spec in "${specs[@]}"; do
        read -r algorithm frames <<<"$spec"
        export NSS_PROFILE_ALGORITHM=$algorithm
        export NSS_PROFILE_FRAMES=$frames
        taskset -c 0 perf stat -M "$level" -o "$OUT/${level}_${algorithm}.txt" -- \
            "$PYTHON" "$PROFILE" >"$OUT/timing_${level}_${algorithm}.json"
    done
done

for spec in "${specs[@]}"; do
    read -r algorithm frames <<<"$spec"
    export NSS_PROFILE_ALGORITHM=$algorithm
    export NSS_PROFILE_FRAMES=$frames
    data="$OUT/perf_${algorithm}.data"
    taskset -c 0 perf record -q -F 997 -e cycles:u -o "$data" -- \
        "$PYTHON" "$PROFILE" >"$OUT/timing_record_${algorithm}.json"
    test -s "$data"
    perf report --stdio --no-children --percent-limit 0.25 --sort symbol,dso -i "$data" \
        >"$OUT/hot_${algorithm}.txt"
done

pgrep -af 'bench_cpu|cc1plus|ninja|profile_cpu_all' >"$OUT/processes_after.txt" || true
ps -eLo psr,pcpu,pid,comm,args --sort=-pcpu | head -20 >"$OUT/cpu_after.txt"
sha256sum "$OUT"/* >"$OUT/SHA256SUMS"
