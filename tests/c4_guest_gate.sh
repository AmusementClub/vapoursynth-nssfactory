#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
usage: c4_guest_gate.sh --root DIR --baseline-tar FILE --candidate-tar FILE --vs-venv-tar FILE --out FILE --baseline-revision REV --candidate-revision REV
EOF
}

ROOT=
BASELINE_TAR=
CANDIDATE_TAR=
VENV_TAR=
OUT=
BASELINE_REVISION=
CANDIDATE_REVISION=
while (($#)); do
    case "$1" in
        --root) ROOT=${2:?missing value for --root}; shift 2 ;;
        --baseline-tar) BASELINE_TAR=${2:?missing value for --baseline-tar}; shift 2 ;;
        --candidate-tar) CANDIDATE_TAR=${2:?missing value for --candidate-tar}; shift 2 ;;
        --vs-venv-tar) VENV_TAR=${2:?missing value for --vs-venv-tar}; shift 2 ;;
        --out) OUT=${2:?missing value for --out}; shift 2 ;;
        --baseline-revision) BASELINE_REVISION=${2:?missing value for --baseline-revision}; shift 2 ;;
        --candidate-revision) CANDIDATE_REVISION=${2:?missing value for --candidate-revision}; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ -z "$ROOT" || -z "$BASELINE_TAR" || -z "$CANDIDATE_TAR" || -z "$VENV_TAR" || -z "$OUT" ||
      -z "$BASELINE_REVISION" || -z "$CANDIDATE_REVISION" ]]; then
    usage
    exit 2
fi
for command_name in cmake tar python3 taskset find mktemp; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "c4 guest: required command is missing: $command_name" >&2
        exit 1
    }
done
for archive in "$BASELINE_TAR" "$CANDIDATE_TAR" "$VENV_TAR"; do
    [[ -f "$archive" && -r "$archive" ]] || {
        echo "c4 guest: archive is not readable: $archive" >&2
        exit 2
    }
done

if [[ "$ROOT" == */ ]]; then
    ROOT=${ROOT%/}
fi
# This helper removes only its own short-lived work directories. Requiring a
# dedicated path below /tmp prevents an accidental broad cleanup target.
case "$ROOT" in
    ""|/|/tmp|*/../*|*/..|*/./*|*/.)
        echo "c4 guest: --root must be a dedicated non-traversing directory below /tmp" >&2
        exit 2
        ;;
    /tmp/*) ;;
    *)
        echo "c4 guest: --root must be below /tmp: $ROOT" >&2
        exit 2
        ;;
esac
root_parent=${ROOT%/*}
[[ -n "$root_parent" && "$root_parent" != "$ROOT" ]] || {
    echo "c4 guest: refusing an unsafe root: $ROOT" >&2
    exit 2
}
mkdir -p -- "$root_parent"
parent_real=$(cd -- "$root_parent" && pwd -P)
tmp_real=$(cd /tmp && pwd -P)
case "$parent_real" in
    "$tmp_real"|"$tmp_real"/*) ;;
    *)
        echo "c4 guest: root parent resolves outside /tmp: $parent_real" >&2
        exit 2
        ;;
esac
if [[ -e "$ROOT" || -L "$ROOT" ]]; then
    [[ -d "$ROOT" && ! -L "$ROOT" ]] || {
        echo "c4 guest: root exists but is not a real directory: $ROOT" >&2
        exit 2
    }
fi
mkdir -p -- "$ROOT"
[[ ! -L "$ROOT" ]] || {
    echo "c4 guest: root became a symlink: $ROOT" >&2
    exit 2
}

mkdir -p -- "$(dirname -- "$OUT")"
for path in "$ROOT/baseline" "$ROOT/candidate" "$ROOT/venv" "$ROOT/venv-extract" \
            "$ROOT/build-baseline" "$ROOT/build-candidate" "$ROOT/results"; do
    [[ ! -L "$path" ]] || {
        echo "c4 guest: refusing symlink cleanup target: $path" >&2
        exit 2
    }
    rm -rf -- "$path"
done
mkdir -p -- "$ROOT/baseline" "$ROOT/candidate" "$ROOT/venv-extract"
tar -xf "$BASELINE_TAR" -C "$ROOT/baseline"
tar -xf "$CANDIDATE_TAR" -C "$ROOT/candidate"
tar -xf "$VENV_TAR" -C "$ROOT/venv-extract"

resolve_source_root() {
    local base=$1
    if [[ -f "$base/CMakeLists.txt" ]]; then
        printf '%s\n' "$base"
        return
    fi
    local candidates=()
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && candidates+=("$candidate")
    done < <(find "$base" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/CMakeLists.txt' \; -print)
    if ((${#candidates[@]} != 1)); then
        echo "c4 guest: source archive must have a unique CMakeLists.txt root: $base" >&2
        exit 2
    fi
    printf '%s\n' "${candidates[0]}"
}

resolve_venv_root() {
    local base=$1
    if [[ -x "$base/bin/python" || -x "$base/bin/python3" ]]; then
        printf '%s\n' "$base"
        return
    fi
    # A tarball made with `tar -C parent venv` normally extracts as
    # `base/venv/bin/python` (the file is three levels below base). Discover
    # executable candidates first and deduplicate python/python3 aliases that
    # belong to the same virtual-environment root.
    local roots=()
    local candidates=()
    local candidate root existing duplicate
    while IFS= read -r candidate; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        candidates+=("$candidate")
    done < <(find "$base" -mindepth 1 -maxdepth 6 \( -type f -o -type l \) \
        \( -path '*/bin/python' -o -path '*/bin/python3' \) -print)
    for candidate in "${candidates[@]}"; do
        case "$candidate" in
            */bin/python) root=${candidate%/bin/python} ;;
            */bin/python3) root=${candidate%/bin/python3} ;;
            *) continue ;;
        esac
        duplicate=0
        if [[ ${roots+x} ]]; then
            for existing in "${roots[@]}"; do
                if [[ "$existing" == "$root" ]]; then
                    duplicate=1
                    break
                fi
            done
        fi
        ((duplicate)) || roots+=("$root")
    done
    if ((${#roots[@]} != 1)); then
        echo "c4 guest: venv archive must contain one runnable bin/python: $base" >&2
        exit 2
    fi
    printf '%s\n' "${roots[0]}"
}

BASELINE_SOURCE=$(resolve_source_root "$ROOT/baseline")
CANDIDATE_SOURCE=$(resolve_source_root "$ROOT/candidate")
VENV_ROOT=$(resolve_venv_root "$ROOT/venv-extract")
if [[ -x "$VENV_ROOT/bin/python" ]]; then
    PYTHON_BIN="$VENV_ROOT/bin/python"
elif [[ -x "$VENV_ROOT/bin/python3" ]]; then
    PYTHON_BIN="$VENV_ROOT/bin/python3"
else
    echo "c4 guest: resolved venv has no executable Python: $VENV_ROOT" >&2
    exit 2
fi
venv_bin=$(dirname "$PYTHON_BIN")
PATH="$venv_bin:$PATH"
export PATH
"$PYTHON_BIN" -c 'import json, statistics' >/dev/null
taskset -c 0 true

build_one() {
    local source=$1
    local build=$2
    local revision=$3
    cmake -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DNSS_ENABLE_CUDA=OFF -DNSS_ENABLE_VULKAN=OFF \
        -DNSS_GIT_DESCRIBE="$revision"
    cmake --build "$build" -j2 --target bench_cpu
}

build_one "$BASELINE_SOURCE" "$ROOT/build-baseline" "$BASELINE_REVISION"
build_one "$CANDIDATE_SOURCE" "$ROOT/build-candidate" "$CANDIDATE_REVISION"

baseline_exe="$ROOT/build-baseline/tests/bench_cpu"
candidate_exe="$ROOT/build-candidate/tests/bench_cpu"
[[ -x "$baseline_exe" && -x "$candidate_exe" ]] || {
    echo "c4 guest: bench_cpu was not built" >&2
    exit 1
}

json_time() {
    local exe=$1
    local seed=$2
    local threads=$3
    local width=$4
    local height=$5
    local frame_iters=$6
    local frame=${7:-0}
    local output
    output=$(env NSS_BENCH_JSON=1 NSS_NO_WARMUP=1 NSS_BENCH_SEED="$seed" NSS_BENCH_FRAME="$frame" \
        NSS_BENCH_THREADS="$threads" \
        taskset -c 0 "$exe" nlm "$width" "$height" "$frame_iters" --json)
    "$PYTHON_BIN" - "$output" "$frame_iters" <<'PY'
import json
import sys

raw = sys.argv[1]
requested_iterations = int(sys.argv[2])
for line in reversed(raw.splitlines()):
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    for result in payload.get("results", []):
        if result.get("name") == "nlm":
            iterations = int(result.get("iterations", requested_iterations))
            if iterations < 1:
                raise SystemExit("bench result reported non-positive iterations")
            print(format(float(result["wall_time_ms"]) / iterations, ".17g"))
            raise SystemExit(0)
import re
for line in reversed(raw.splitlines()):
    if not line.lstrip().startswith("nlm"):
        continue
    # The historical baseline can predate JSON v2; retain the same
    # per-iteration process-wall timing boundary for its plain text line.
    match = re.search(r"([0-9]+(?:\.[0-9]*)?)\s+ms/iter\b", line)
    if match:
        print(format(float(match.group(1)), ".17g"))
        raise SystemExit(0)
    match = re.search(r"([0-9]+(?:\.[0-9]*)?)\s+ms\b", line)
    if match:
        print(format(float(match.group(1)) / requested_iterations, ".17g"))
        raise SystemExit(0)
raise SystemExit("bench output did not contain an nlm timing")
PY
}

result_dir="$ROOT/results"
mkdir -p "$result_dir"
formal_tsv="$result_dir/formal.tsv"
raw_tsv="$result_dir/paired.tsv"

run_formal_pairs() {
    local pair_count=$1
    local run_name=$2
    printf 'pair\trun\tshape\tthreads\tseed\tframe\tbaseline_ms\tcandidate_ms\tratio\n' >"$formal_tsv"
    for ((pair = 1; pair <= pair_count; ++pair)); do
        local seed=$((9000 + pair))
        local frame=$((100 + pair))
        local iters=$((1 + pair % 3))
        local baseline_ms candidate_ms ratio
        # Alternate process order as well as input seed to reduce systematic
        # thermal/order bias while retaining a paired sample per row.
        if ((pair % 2 == 1)); then
            baseline_ms=$(json_time "$baseline_exe" "$seed" 1 1920 1080 "$iters" "$frame")
            candidate_ms=$(json_time "$candidate_exe" "$seed" 1 1920 1080 "$iters" "$frame")
        else
            candidate_ms=$(json_time "$candidate_exe" "$seed" 1 1920 1080 "$iters" "$frame")
            baseline_ms=$(json_time "$baseline_exe" "$seed" 1 1920 1080 "$iters" "$frame")
        fi
        ratio=$("$PYTHON_BIN" - "$baseline_ms" "$candidate_ms" <<'PY'
import sys
b = float(sys.argv[1])
c = float(sys.argv[2])
if not (b > 0.0 and c > 0.0):
    raise SystemExit("non-positive benchmark timing")
print(format(b / c, ".17g"))
PY
)
        printf '%s\t%s\tformal-1080p-gray\t1\t%s\t%s\t%s\t%s\t%s\n' \
            "$pair" "$run_name" "$seed" "$frame" "$baseline_ms" "$candidate_ms" "$ratio" >>"$formal_tsv"
    done
}

formal_ratio() {
    "$PYTHON_BIN" - "$formal_tsv" <<'PY'
import csv
import statistics
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit("formal timing file is empty")
print(format(statistics.median(float(row["ratio"]) for row in rows), ".17g"))
PY
}

critical_margin=${NSS_C4_CRITICAL_MARGIN:-0.03}
"$PYTHON_BIN" - "$critical_margin" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0.0:
    raise SystemExit("NSS_C4_CRITICAL_MARGIN must be a finite non-negative number")
PY

# Warm each executable in an independent process. The formal gate below uses
# only the subsequent paired processes.
json_time "$baseline_exe" 7000 1 1920 1080 1 >/dev/null
json_time "$candidate_exe" 7000 1 1920 1080 1 >/dev/null

run_formal_pairs 7 initial
initial_ratio=$(formal_ratio)
critical_rerun=0
if "$PYTHON_BIN" - "$initial_ratio" "$critical_margin" <<'PY'
import sys
ratio = float(sys.argv[1])
margin = float(sys.argv[2])
raise SystemExit(0 if abs(ratio - 1.30) <= margin else 1)
PY
then
    critical_rerun=1
    echo "c4 guest: initial ratio $initial_ratio is within +/-$critical_margin of the gate; rerunning 15 pairs" >&2
    run_formal_pairs 15 critical-rerun
fi

cp "$formal_tsv" "$raw_tsv"

run_aux() {
    local shape=$1
    local width=$2
    local height=$3
    local frame_iters=$4
    local threads=$5
    local seed=$6
    local frame=$7
    local baseline_ms candidate_ms ratio
    baseline_ms=$(json_time "$baseline_exe" "$seed" "$threads" "$width" "$height" "$frame_iters" "$frame")
    candidate_ms=$(json_time "$candidate_exe" "$seed" "$threads" "$width" "$height" "$frame_iters" "$frame")
    ratio=$("$PYTHON_BIN" - "$baseline_ms" "$candidate_ms" <<'PY'
import sys
b = float(sys.argv[1])
c = float(sys.argv[2])
print(format(b / c if c > 0 else 0.0, ".17g"))
PY
)
    printf 'aux-%s\tauxiliary\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$shape" "$shape" "$threads" "$seed" "$frame" "$baseline_ms" "$candidate_ms" "$ratio" >>"$raw_tsv"
}

# Auxiliary shapes exercise different content sizes, frame counts and the
# benchmark's multi-thread metadata contract. They do not enter the formal
# 1.30x decision.
run_aux 1280x720 1280 720 2 2 15001 201
run_aux 640x360 640 360 1 2 15002 202
run_aux 1920x1080 1920 1080 1 2 15003 203

cpu_model=unknown
if [[ -r /proc/cpuinfo ]]; then
    cpu_model=$(awk -F: '/model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)
elif command -v sysctl >/dev/null 2>&1; then
    cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo unknown)
fi
compiler=$(c++ --version | head -n 1)
"$PYTHON_BIN" - "$raw_tsv" "$OUT" "$initial_ratio" "$critical_rerun" "$critical_margin" \
    "$BASELINE_REVISION" "$CANDIDATE_REVISION" "$cpu_model" "$compiler" <<'PY'
import json
import pathlib
import statistics
import sys

raw_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
initial_ratio = float(sys.argv[3])
critical_rerun = bool(int(sys.argv[4]))
critical_margin = float(sys.argv[5])
baseline_revision = sys.argv[6]
candidate_revision = sys.argv[7]
cpu_model = sys.argv[8]
compiler = sys.argv[9]
rows = []
with raw_path.open(encoding="utf-8") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    for line in handle:
        if not line.strip():
            continue
        values = line.rstrip("\n").split("\t")
        row = dict(zip(header, values))
        row["baseline_ms"] = float(row["baseline_ms"])
        row["candidate_ms"] = float(row["candidate_ms"])
        row["ratio"] = float(row["ratio"])
        rows.append(row)
formal = [row for row in rows if row["shape"] == "formal-1080p-gray"]
baseline = [row["baseline_ms"] for row in formal]
candidate = [row["candidate_ms"] for row in formal]
if not formal:
    raise SystemExit("formal timing rows are missing")
paired_median = statistics.median(row["ratio"] for row in formal)
payload = {
    "schema": "nssfactory.c4.cpu-gate.v2",
    "machine_type": "c4-highcpu-2",
    "cpu": cpu_model,
    "compiler": compiler,
    "baseline_revision": baseline_revision,
    "candidate_revision": candidate_revision,
    "cpu_affinity": "0",
    "formal_pairs": len(formal),
    "initial_formal_pairs": 7,
    "critical_rerun": critical_rerun,
    "critical_margin": critical_margin,
    "initial_paired_median_ratio": initial_ratio,
    "formal_shape": {"width": 1920, "height": 1080, "channels": "Gray", "d": 1},
    "paired_median_baseline_ms": statistics.median(baseline),
    "paired_median_candidate_ms": statistics.median(candidate),
    "paired_median_ratio": paired_median,
    "ratio_of_medians": statistics.median(baseline) / statistics.median(candidate),
    "gate": ">=1.30x",
    "passed": paired_median >= 1.30,
    "rows": rows,
}
out_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"paired_median_ratio": paired_median, "passed": paired_median >= 1.30}, sort_keys=True))
if paired_median < 1.30:
    raise SystemExit(3)
PY
