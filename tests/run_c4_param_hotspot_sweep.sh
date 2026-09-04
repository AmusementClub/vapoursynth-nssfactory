#!/usr/bin/env bash
# One-at-a-time legal parameter sweep: 1 timed frame + 1 perf record per config.
# Requires the same guest env as run_c4_topdown_profile.sh.
set -Eeuo pipefail

PROFILE=${NSS_PROFILE_SCRIPT:?set NSS_PROFILE_SCRIPT}
PLUGIN=${NSS_SO:?set NSS_SO}
PYTHON=${NSS_PROFILE_PYTHON:?set NSS_PROFILE_PYTHON}
OUT=${NSS_SWEEP_OUT:?set NSS_SWEEP_OUT}

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
export NSS_PROFILE_FRAMES=1
export NSS_PROFILE_SEED=42

# name|algorithm|env assignments (space-separated KEY=VAL)
configs=(
  "nlm/default|nlm|"
  "nlm/d0|nlm|NSS_PROFILE_NLM_D=0"
  "nlm/d2|nlm|NSS_PROFILE_NLM_D=2"
  "nlm/a1|nlm|NSS_PROFILE_NLM_A=1"
  "nlm/a4|nlm|NSS_PROFILE_NLM_A=4"
  "nlm/s1|nlm|NSS_PROFILE_NLM_S=1"
  "nlm/s8|nlm|NSS_PROFILE_NLM_S=8"

  "bm3d/default|bm3d|"
  "bm3d/range3|bm3d|NSS_PROFILE_BM3D_RANGE=3"
  "bm3d/range20|bm3d|NSS_PROFILE_BM3D_RANGE=20"
  "bm3d/group4|bm3d|NSS_PROFILE_BM3D_GROUP=4"
  "bm3d/group16|bm3d|NSS_PROFILE_BM3D_GROUP=16"
  "bm3d/block4|bm3d|NSS_PROFILE_BM3D_BLOCK=4 NSS_PROFILE_BM3D_STEP=4"
  "bm3d/block16|bm3d|NSS_PROFILE_BM3D_BLOCK=16 NSS_PROFILE_BM3D_STEP=8"
  "bm3d/radius1|bm3d|NSS_PROFILE_BM3D_RADIUS=1"

  "wnnm/default|wnnm|"
  "wnnm/residual1|wnnm|NSS_PROFILE_WNNM_RESIDUAL=1"
  "wnnm/range3|wnnm|NSS_PROFILE_WNNM_RANGE=3"
  "wnnm/range20|wnnm|NSS_PROFILE_WNNM_RANGE=20"
  "wnnm/group4|wnnm|NSS_PROFILE_WNNM_GROUP=4"
  "wnnm/group16|wnnm|NSS_PROFILE_WNNM_GROUP=16"
  "wnnm/block4|wnnm|NSS_PROFILE_WNNM_BLOCK=4 NSS_PROFILE_WNNM_STEP=4"
  "wnnm/block16|wnnm|NSS_PROFILE_WNNM_BLOCK=16 NSS_PROFILE_WNNM_STEP=8"
  "wnnm/radius1|wnnm|NSS_PROFILE_WNNM_RADIUS=1"

  "twsc/default|twsc|"
  "twsc/range3|twsc|NSS_PROFILE_TWSC_RANGE=3"
  "twsc/range20|twsc|NSS_PROFILE_TWSC_RANGE=20"
  "twsc/group4|twsc|NSS_PROFILE_TWSC_GROUP=4"
  "twsc/group16|twsc|NSS_PROFILE_TWSC_GROUP=16"
  "twsc/block4|twsc|NSS_PROFILE_TWSC_BLOCK=4 NSS_PROFILE_TWSC_STEP=4"
  "twsc/block16|twsc|NSS_PROFILE_TWSC_BLOCK=16 NSS_PROFILE_TWSC_STEP=8"
  "twsc/radius1|twsc|NSS_PROFILE_TWSC_RADIUS=1"
  "twsc/iters1|twsc|NSS_PROFILE_TWSC_ITERS=1"

  "ncsr/default|ncsr|"
  "ncsr/range3|ncsr|NSS_PROFILE_NCSR_RANGE=3"
  "ncsr/range20|ncsr|NSS_PROFILE_NCSR_RANGE=20"
  "ncsr/group4|ncsr|NSS_PROFILE_NCSR_GROUP=4"
  "ncsr/group16|ncsr|NSS_PROFILE_NCSR_GROUP=16"
  "ncsr/block4|ncsr|NSS_PROFILE_NCSR_BLOCK=4 NSS_PROFILE_NCSR_STEP=4"
  "ncsr/block16|ncsr|NSS_PROFILE_NCSR_BLOCK=16 NSS_PROFILE_NCSR_STEP=8"
  "ncsr/radius1|ncsr|NSS_PROFILE_NCSR_RADIUS=1"
  "ncsr/iters1|ncsr|NSS_PROFILE_NCSR_ITERS=1"

  "lssc/default|lssc|"
  "lssc/block4|lssc|NSS_PROFILE_LSSC_BLOCK=4 NSS_PROFILE_LSSC_STEP=4"
  "lssc/block16|lssc|NSS_PROFILE_LSSC_BLOCK=16 NSS_PROFILE_LSSC_STEP=8"

  "nlh/default|nlh|"
  "nlh/range7|nlh|NSS_PROFILE_NLH_RANGE=7"
  "nlh/group8|nlh|NSS_PROFILE_NLH_GROUP=8"
  "nlh/group4|nlh|NSS_PROFILE_NLH_GROUP=4"
  "nlh/q2|nlh|NSS_PROFILE_NLH_Q=2"
  "nlh/q8|nlh|NSS_PROFILE_NLH_Q=8"
  "nlh/block4|nlh|NSS_PROFILE_NLH_BLOCK=4 NSS_PROFILE_NLH_STEP=4"
  "nlh/radius1|nlh|NSS_PROFILE_NLH_RADIUS=1"

  "mcwnnm/default|mcwnnm|"
  "mcwnnm/range3|mcwnnm|NSS_PROFILE_MCWNNM_RANGE=3"
  "mcwnnm/range20|mcwnnm|NSS_PROFILE_MCWNNM_RANGE=20"
  "mcwnnm/group4|mcwnnm|NSS_PROFILE_MCWNNM_GROUP=4"
  "mcwnnm/group16|mcwnnm|NSS_PROFILE_MCWNNM_GROUP=16"
  "mcwnnm/block4|mcwnnm|NSS_PROFILE_MCWNNM_BLOCK=4 NSS_PROFILE_MCWNNM_STEP=4"
  "mcwnnm/block9|mcwnnm|NSS_PROFILE_MCWNNM_BLOCK=9 NSS_PROFILE_MCWNNM_STEP=8"
  "mcwnnm/residual0|mcwnnm|NSS_PROFILE_MCWNNM_RESIDUAL=0"
  "mcwnnm/admm1|mcwnnm|NSS_PROFILE_MCWNNM_ADMM_ITER=1"
  "mcwnnm/iters1|mcwnnm|NSS_PROFILE_MCWNNM_ITERS=1"
  "mcwnnm/radius1|mcwnnm|NSS_PROFILE_MCWNNM_RADIUS=1"
)

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  lscpu | sed -n '1,30p'
  sha256sum "$PROFILE" "$PLUGIN"
  echo "cpu_affinity=0"
  echo "gray8=${NSS_PROFILE_GRAY8:-synthetic}"
  echo "revision=${NSS_PROFILE_REVISION:-unknown}"
  echo "source_sha256=${NSS_PROFILE_SOURCE_SHA256:-unknown}"
} >"$OUT/metadata.txt"

printf 'config\talgorithm\tms\thash\tlost\ttop1\tpct1\ttop2\tpct2\ttop3\tpct3\n' >"$OUT/summary.tsv"

run_one() {
  local slug=$1 algo=$2
  local dir="$OUT/$slug"
  mkdir -p "$dir"
  export NSS_PROFILE_ALGORITHM=$algo
  taskset -c 0 "$PYTHON" "$PROFILE" >"$dir/time.json"
  local data="$dir/perf.data"
  taskset -c 0 perf record -q -F 997 -e cycles:u -o "$data" -- \
    "$PYTHON" "$PROFILE" >"$dir/perf-time.json"
  test -s "$data"
  perf report --stdio --no-children --percent-limit 0.25 --sort symbol,dso -i "$data" \
    >"$dir/perf-report.txt"
  python3 - "$slug" "$algo" "$dir" "$OUT/summary.tsv" <<'PY'
import json, re, sys
slug, algo, d, tsv = sys.argv[1:5]
time = json.load(open(f"{d}/time.json"))
text = open(f"{d}/perf-report.txt", errors="replace").read()
lost = "unknown"
m = re.search(r"Total Lost Samples:\s*(\d+)", text)
if m:
    lost = m.group(1)
hots = []
for line in text.splitlines():
    mm = re.match(r"\s*([0-9.]+)%\s+.*?\[.\]\s+(\S+)", line)
    if not mm:
        continue
    name = mm.group(2)
    if name.startswith("[") or name in {"perf", "python3"}:
        continue
    hots.append((float(mm.group(1)), name))
    if len(hots) == 3:
        break
while len(hots) < 3:
    hots.append((0.0, "-"))
with open(tsv, "a") as f:
    f.write(
        "\t".join(
            [
                slug,
                algo,
                f"{time['milliseconds_per_frame']:.3f}",
                time["output_sha256"],
                lost,
                hots[0][1],
                f"{hots[0][0]:.2f}",
                hots[1][1],
                f"{hots[1][0]:.2f}",
                hots[2][1],
                f"{hots[2][0]:.2f}",
            ]
        )
        + "\n"
    )
print(f"{slug}\t{time['milliseconds_per_frame']:.3f}\t{hots[0][1]} {hots[0][0]:.1f}%", flush=True)
PY
}

for spec in "${configs[@]}"; do
  IFS='|' read -r slug algo envspec <<<"$spec"
  unset NSS_PROFILE_NLM_D NSS_PROFILE_NLM_A NSS_PROFILE_NLM_S
  unset NSS_PROFILE_BM3D_BLOCK NSS_PROFILE_BM3D_STEP NSS_PROFILE_BM3D_GROUP
  unset NSS_PROFILE_BM3D_RANGE NSS_PROFILE_BM3D_RADIUS
  unset NSS_PROFILE_WNNM_BLOCK NSS_PROFILE_WNNM_STEP NSS_PROFILE_WNNM_GROUP
  unset NSS_PROFILE_WNNM_RANGE NSS_PROFILE_WNNM_RADIUS NSS_PROFILE_WNNM_RESIDUAL
  unset NSS_PROFILE_TWSC_BLOCK NSS_PROFILE_TWSC_STEP NSS_PROFILE_TWSC_GROUP
  unset NSS_PROFILE_TWSC_RANGE NSS_PROFILE_TWSC_RADIUS NSS_PROFILE_TWSC_ITERS
  unset NSS_PROFILE_NCSR_BLOCK NSS_PROFILE_NCSR_STEP NSS_PROFILE_NCSR_GROUP
  unset NSS_PROFILE_NCSR_RANGE NSS_PROFILE_NCSR_RADIUS NSS_PROFILE_NCSR_ITERS
  unset NSS_PROFILE_LSSC_BLOCK NSS_PROFILE_LSSC_STEP NSS_PROFILE_LSSC_RADIUS
  unset NSS_PROFILE_NLH_BLOCK NSS_PROFILE_NLH_STEP NSS_PROFILE_NLH_GROUP
  unset NSS_PROFILE_NLH_RANGE NSS_PROFILE_NLH_RADIUS NSS_PROFILE_NLH_Q
  unset NSS_PROFILE_MCWNNM_BLOCK NSS_PROFILE_MCWNNM_STEP NSS_PROFILE_MCWNNM_GROUP
  unset NSS_PROFILE_MCWNNM_RANGE NSS_PROFILE_MCWNNM_RADIUS NSS_PROFILE_MCWNNM_RESIDUAL
  unset NSS_PROFILE_MCWNNM_ADMM_ITER NSS_PROFILE_MCWNNM_ITERS
  if [[ -n "$envspec" ]]; then
    # shellcheck disable=SC2086
    eval export $envspec
  fi
  run_one "$slug" "$algo"
done

sha256sum "$OUT"/summary.tsv "$OUT"/metadata.txt >"$OUT/SHA256SUMS"
find "$OUT" -type f -name 'perf.data' -print0 | sort -z | xargs -0 sha256sum >>"$OUT/SHA256SUMS"
cat "$OUT/summary.tsv"
