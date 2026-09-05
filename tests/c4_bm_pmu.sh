#!/usr/bin/env bash
# C4 diagnostics only. Never run concurrently with the paired wall-time gate.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 PLUGIN OUTPUT_DIR [GRAY8 [FRAMES]]" >&2
    exit 2
fi
source /opt/nss-c4/bin/guest_env.sh
plugin=$(realpath "$1")
output=$2
sample=${3:-/opt/nss-c4/samples/gray8/MAPPA.gray8}
frames=${4:-40}
test -f "$plugin"
test -f "$sample"
test ! -e "$output"
[[ "$frames" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$output"
script_dir=$(cd "$(dirname "$0")" && pwd)
sha256sum "$plugin" "$sample" "$script_dir/c4_paired_bm.py" "$script_dir/profile_cpu_all.py" > "$output/inputs.sha256"
cat /proc/stat > "$output/proc-before.txt"
perf version > "$output/perf-version.txt"
lscpu > "$output/lscpu.txt"
for stage in basic wiener; do
    config=$("$NSS_C4_PYTHON" -c 'import json,sys; print(json.dumps(dict(stage=sys.argv[1],sample=sys.argv[2],frames=int(sys.argv[3]))))' "$stage" "$sample" "$frames")
    command=("$NSS_C4_PYTHON" "$script_dir/c4_paired_bm.py" worker --plugin "$plugin" --config "$config")
    taskset -c 0 perf stat -M TopdownL1,TopdownL2 -o "$output/$stage-topdown.txt" -- "${command[@]}" > "$output/$stage-topdown.json"
    taskset -c 0 perf stat -e cycles:u,instructions:u,branches:u,branch-misses:u -o "$output/$stage-counters.txt" -- "${command[@]}" > "$output/$stage-counters.json"
    taskset -c 0 perf record -q -e cycles:u -F 997 -o "$output/$stage.data" -- "${command[@]}" > "$output/$stage-record.json"
    perf report --stdio --no-children --percent-limit 0.2 --sort symbol,dso -i "$output/$stage.data" > "$output/$stage-hot.txt"
    perf annotate --stdio -i "$output/$stage.data" > "$output/$stage-annotate.txt"
done
cat /proc/stat > "$output/proc-after.txt"
objdump -d -C "$plugin" > "$output/assembly.asm"
sha256sum -c "$output/inputs.sha256" > "$output/hash-check.txt"
"$NSS_C4_PYTHON" - "$output" <<'PY'
from pathlib import Path
import json
import re
import sys
root = Path(sys.argv[1])
for stage in ['basic', 'wiener']:
    counters = (root / f'{stage}-counters.txt').read_text()
    cycles = re.search(r'([\d,]+)\s+cycles:u', counters)
    assert cycles and int(cycles[1].replace(',', '')) > 0, 'missing hardware cycles'
    assert 'not counted' not in counters and 'not supported' not in counters
    assert re.search(r'Total Lost Samples:\s+0\b', (root / f'{stage}-hot.txt').read_text())
    assert (root / f'{stage}.data').stat().st_size > 0
print(json.dumps({'hardware_events_valid': True, 'lost_samples': 0, 'stages': ['basic', 'wiener']}))
PY
