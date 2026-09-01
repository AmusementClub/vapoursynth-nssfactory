#!/usr/bin/env bash
# shellcheck disable=SC2016
set -Eeuo pipefail

# Offline contract tests for the C4 scripts. No gcloud credentials or compute
# resources are needed: every cloud/build command is replaced by a fixture
# executable in a private PATH entry.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_ROOT=$(mktemp -d "/tmp/nss-c4-script-test.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

FAKE_BIN="$TEST_ROOT/bin"
mkdir -p "$FAKE_BIN"
REAL_TAR=$(command -v tar)
REAL_PYTHON=$(command -v python3)
export NSS_TEST_REAL_TAR="$REAL_TAR"

write_executable() {
    local path=$1
    shift
    printf '%s\n' "$@" >"$path"
    chmod 0755 "$path"
}

write_executable "$FAKE_BIN/cmake" \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    'if [[ ${1:-} == -S ]]; then' \
    '    source=$2; build=' \
    '    for ((i = 1; i <= $#; ++i)); do' \
    '        arg=${!i}' \
    '        if [[ $arg == -B ]]; then j=$((i + 1)); build=${!j}; fi' \
    '    done' \
    '    [[ -n $build ]] || { echo "fake cmake: missing build directory" >&2; exit 2; }' \
    '    mkdir -p "$build/tests"' \
    '    if [[ $source == *baseline* ]]; then ms=${FAKE_BASELINE_MS:-2}; else ms=${FAKE_CANDIDATE_MS:-1}; fi' \
    '    printf "%s\\n" "#!/usr/bin/env bash" "printf '\''{\"schema\":\"nssfactory.bench.v2\",\"git_revision\":\"fake\",\"compiler\":\"fake\",\"cpu\":\"fake\",\"width\":1920,\"height\":1080,\"seed\":1,\"frame_number\":1,\"thread_count\":1,\"results\":[{\"name\":\"nlm\",\"wall_time_ms\":${ms}}]}\\n'\''" >"$build/tests/bench_cpu"' \
    '    chmod 0755 "$build/tests/bench_cpu"' \
    'elif [[ ${1:-} == --build ]]; then' \
    '    exit 0' \
    'else' \
    '    echo "fake cmake: unsupported invocation" >&2; exit 2' \
    'fi'

write_executable "$FAKE_BIN/taskset" \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    'if [[ ${1:-} == -c ]]; then shift 2; fi' \
    'exec "$@"'

write_executable "$FAKE_BIN/tar" \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    'if [[ ${FAKE_REQUIRE_COPYFILE_DISABLE:-0} == 1 && ${COPYFILE_DISABLE:-0} != 1 ]]; then' \
    '    echo "macOS source archive did not disable AppleDouble metadata" >&2; exit 96' \
    'fi' \
    'if [[ ${FAKE_TAR_MODE:-} == separate ]]; then' \
    '    for arg in "$@"; do' \
    '        [[ $arg == --exclude=* ]] && { echo "BSD tar rejected equals exclude syntax" >&2; exit 97; }' \
    '    done' \
    'fi' \
    'exec "${NSS_TEST_REAL_TAR:?}" "$@"'

write_executable "$FAKE_BIN/gcloud" \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    'state=${FAKE_GCLOUD_STATE:?}' \
    'mkdir -p "$state"' \
    'printf "%s\\n" "$*" >>"$state/calls"' \
    'command_value=' \
    'previous=' \
    'for arg in "$@"; do' \
    '    if [[ $arg == --command=* ]]; then command_value=${arg#--command=}; fi' \
    '    if [[ $previous == --command ]]; then command_value=$arg; fi' \
    '    previous=$arg' \
    'done' \
    'if [[ ${1:-} == config && ${2:-} == get-value ]]; then echo fake-project; exit 0; fi' \
    'if [[ ${1:-} == compute && ${2:-} == zones && ${3:-} == describe ]]; then echo "${4:-us-central1-a}"; exit 0; fi' \
    'if [[ ${1:-} == compute && ${2:-} == regions && ${3:-} == describe ]]; then echo '\''{"quotas":[{"metric":"CPUS","limit":8,"usage":0}]}'\''; exit 0; fi' \
    'if [[ ${1:-} == compute && ${2:-} == machine-types && ${3:-} == describe ]]; then echo c4-highcpu-2; exit 0; fi' \
    'if [[ ${1:-} == auth && ${2:-} == list ]]; then echo runner@example.invalid; exit 0; fi' \
    'if [[ ${1:-} == policy-intelligence && ${2:-} == troubleshoot-policy && ${3:-} == iam ]]; then' \
    '    if [[ ${FAKE_PERMISSION_DENY:-0} == 1 ]]; then echo DENIED; else echo GRANTED; fi' \
    '    exit 0' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == instances && ${3:-} == list ]]; then' \
    '    if [[ ${FAKE_SIBLING:-0} == 1 ]]; then echo existing-sibling; fi' \
    '    exit 0' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == instances && ${3:-} == create ]]; then' \
    '    if [[ ${FAKE_CREATE_FAIL:-0} == 1 ]]; then echo "Spot admission failed" >&2; exit 1; fi' \
    '    touch "$state/created"; echo created; exit 0' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == instances && ${3:-} == describe ]]; then' \
    '    if [[ ! -f "$state/created" || -f "$state/deleted" ]]; then echo "ERROR: resource was not found" >&2; exit 1; fi' \
    '    if [[ -f "$state/deleting" ]]; then' \
    '        case ${FAKE_DELETE_DESCRIBE:-notfound} in' \
    '            notfound) echo "ERROR: resource was not found" >&2; exit 1 ;;' \
    '            generic) echo "ERROR: permission denied: resource not found" >&2; exit 1 ;;' \
    '            *) echo "still present"; exit 0 ;;' \
    '        esac' \
    '    fi' \
    '    echo instance; exit 0' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == instances && ${3:-} == delete ]]; then' \
    '    case ${FAKE_DELETE_API:-ok} in' \
    '        notfound) touch "$state/deleted"; echo "ERROR: resource was not found" >&2; exit 1 ;;' \
    '        generic) echo "ERROR: permission denied: resource not found" >&2; exit 1 ;;' \
    '        *) touch "$state/deleting"; exit 0 ;;' \
    '    esac' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == ssh ]]; then' \
    '    if [[ $command_value == true ]]; then exit 0; fi' \
    '    if [[ $command_value == *c4_guest_gate.sh* ]]; then' \
    '        echo '\''{"schema":"nssfactory.c4.cpu-gate.v2","machine_type":"c4-highcpu-2","cpu":"fake-cpu","compiler":"fake-compiler","baseline_revision":"fake-baseline","candidate_revision":"fake-candidate","formal_pairs":7,"paired_median_ratio":2.0,"passed":true,"rows":[]}'\'' >"$state/artifact.json"' \
    '        if [[ ${FAKE_GUEST_RC:-0} != 0 ]]; then exit "$FAKE_GUEST_RC"; fi' \
    '    fi' \
    '    exit 0' \
    'fi' \
    'if [[ ${1:-} == compute && ${2:-} == scp ]]; then' \
    '    args=("$@"); remote_idx=-1' \
    '    for ((i = 0; i < ${#args[@]}; ++i)); do' \
    '        if [[ ${args[$i]} == *:* && ${args[$i]} != *://* ]]; then remote_idx=$i; fi' \
    '    done' \
    '    if (( remote_idx >= 0 )) && [[ ${args[$remote_idx]} == */results/c4.json ]]; then' \
    '        destination=${args[$((remote_idx + 1))]}' \
    '        mkdir -p "$(dirname "$destination")"' \
    '        cp "$state/artifact.json" "$destination"' \
    '    fi' \
    '    exit 0' \
    'fi' \
    'echo "fake gcloud: unsupported invocation: $*" >&2' \
    'exit 2'

chmod 0755 "$FAKE_BIN"/*

mkdir -p "$TEST_ROOT/baseline-source" "$TEST_ROOT/candidate-source" \
    "$TEST_ROOT/venv-package/venv/bin"
printf 'cmake_minimum_required(VERSION 3.24)\nproject(fake LANGUAGES CXX)\n' >"$TEST_ROOT/baseline-source/CMakeLists.txt"
printf 'cmake_minimum_required(VERSION 3.24)\nproject(fake LANGUAGES CXX)\n' >"$TEST_ROOT/candidate-source/CMakeLists.txt"
ln -s "$REAL_PYTHON" "$TEST_ROOT/venv-package/venv/bin/python"
ln -s "$REAL_PYTHON" "$TEST_ROOT/venv-package/venv/bin/python3"

BASELINE_TAR="$TEST_ROOT/baseline.tar.gz"
CANDIDATE_TAR="$TEST_ROOT/candidate.tar.gz"
VENV_TAR="$TEST_ROOT/vs-venv.tar.gz"
"$REAL_TAR" -czf "$BASELINE_TAR" -C "$TEST_ROOT/baseline-source" .
"$REAL_TAR" -czf "$CANDIDATE_TAR" -C "$TEST_ROOT/candidate-source" .
"$REAL_TAR" -czf "$VENV_TAR" -C "$TEST_ROOT/venv-package" venv

assert_contains() {
    local needle=$1
    local file=$2
    if command -v rg >/dev/null 2>&1; then
        rg -F -- "$needle" "$file" >/dev/null
    else
        grep -F -- "$needle" "$file" >/dev/null
    fi || {
        echo "test_c4_scripts: expected '$needle' in $file" >&2
        return 1
    }
}

assert_json() {
    "$REAL_PYTHON" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["schema"] == "nssfactory.c4.cpu-gate.v2"
assert payload["formal_pairs"] == 7
assert payload["passed"] is True
assert payload["baseline_revision"] == "fake-baseline"
assert payload["candidate_revision"] == "fake-candidate"
assert payload["cpu"]
assert payload["compiler"]
PY
}

# The nested venv archive contains both python aliases. The guest helper must
# resolve one root, preserve the symlink, build both fixtures, and emit JSON.
GUEST_ROOT="$TEST_ROOT/guest-work"
GUEST_OUT="$TEST_ROOT/guest.json"
PATH="$FAKE_BIN:$PATH" FAKE_BASELINE_MS=2 FAKE_CANDIDATE_MS=1 \
    "$SCRIPT_DIR/c4_guest_gate.sh" --root "$GUEST_ROOT" --baseline-tar "$BASELINE_TAR" \
    --candidate-tar "$CANDIDATE_TAR" --vs-venv-tar "$VENV_TAR" --out "$GUEST_OUT" \
    --baseline-revision fake-baseline --candidate-revision fake-candidate >/dev/null
assert_json "$GUEST_OUT"

run_runner_case() {
    local name=$1
    local expected=$2
    shift 2
    local state="$TEST_ROOT/state-$name"
    local log="$TEST_ROOT/runner-$name.log"
    mkdir -p "$state"
    set +e
    PATH="$FAKE_BIN:$PATH" FAKE_GCLOUD_STATE="$state" NSS_TEST_REAL_TAR="$REAL_TAR" \
        NSS_C4_BASELINE_DIR="$TEST_ROOT/baseline-source" NSS_C4_CANDIDATE_DIR="$TEST_ROOT/candidate-source" \
        NSS_C4_BASELINE_REVISION=fake-baseline NSS_C4_CANDIDATE_REVISION=fake-candidate \
        NSS_C4_ARTIFACT_DIR="$state/artifacts" NSS_C4_DELETE_POLL_ATTEMPTS=1 NSS_C4_DELETE_POLL_SECONDS=0 \
        "$@" "$SCRIPT_DIR/run_c4_cpu_gate.sh" --project fake-project --zone us-central1-a --vs-venv-tar "$VENV_TAR" \
        >"$log" 2>&1
    local got=$?
    set -e
    if [[ "$got" != "$expected" ]]; then
        echo "test_c4_scripts: $name returned $got, expected $expected" >&2
        sed -n '1,120p' "$log" >&2
        return 1
    fi
    printf '%s\n' "$state"
}

missing_baseline_state="$TEST_ROOT/state-missing-baseline"
mkdir -p "$missing_baseline_state"
set +e
PATH="$FAKE_BIN:$PATH" FAKE_GCLOUD_STATE="$missing_baseline_state" \
    NSS_C4_CANDIDATE_DIR="$TEST_ROOT/candidate-source" \
    env -u NSS_C4_BASELINE_REF -u NSS_C4_BASELINE_DIR \
    "$SCRIPT_DIR/run_c4_cpu_gate.sh" --project fake-project --zone us-central1-a --vs-venv-tar "$VENV_TAR" \
    >"$TEST_ROOT/runner-missing-baseline.log" 2>&1
missing_baseline_rc=$?
set -e
if [[ "$missing_baseline_rc" != 2 || -s "$missing_baseline_state/calls" ]]; then
    echo 'test_c4_scripts: missing corrected baseline did not fail before gcloud access' >&2
    sed -n '1,120p' "$TEST_ROOT/runner-missing-baseline.log" >&2
    exit 1
fi

# Successful run: separate --exclude arguments are accepted by the BSD-tar
# fixture, labels are present, and a real NOT_FOUND confirms deletion.
success_state=$(run_runner_case success 0 env FAKE_TAR_MODE=separate FAKE_REQUIRE_COPYFILE_DISABLE=1 \
    FAKE_DELETE_DESCRIBE=notfound)
assert_contains '--labels=nss-c4-runner=true' "$success_state/calls"
assert_contains '--no-service-account --no-scopes' "$success_state/calls"
assert_contains '--filter=name=(' "$success_state/calls"
assert_contains '--permission=compute.instances.create' "$success_state/calls"
assert_contains '--permission=compute.instances.delete' "$success_state/calls"
assert_json "$success_state/artifacts"/*.json

# A guest failure must still fetch its artifact and return the guest status.
failure_state=$(run_runner_case guest-failure 7 env FAKE_GUEST_RC=7)
failure_artifact=$(find "$failure_state/artifacts" -maxdepth 1 -type f -name '*.json' -print -quit 2>/dev/null || true)
if [[ -z "$failure_artifact" || ! -s "$failure_artifact" ]]; then
    echo 'test_c4_scripts: guest failure artifact was not fetched' >&2
    sed -n '1,160p' "$TEST_ROOT/runner-guest-failure.log" >&2
    sed -n '1,200p' "$failure_state/calls" >&2
    exit 1
fi
assert_contains 'instances delete' "$failure_state/calls"

# A sibling blocks creation before any new instance exists.
sibling_state=$(run_runner_case sibling 2 env FAKE_SIBLING=1)
if assert_contains 'instances create' "$sibling_state/calls" 2>/dev/null; then
    echo 'test_c4_scripts: sibling check happened after create' >&2
    exit 1
fi
assert_contains 'labels.nss-c4-runner=true' "$sibling_state/calls"

# IAM must be positively confirmed before any create call.
permission_state=$(run_runner_case permission-denied 2 env FAKE_PERMISSION_DENY=1)
if assert_contains 'instances create' "$permission_state/calls" 2>/dev/null; then
    echo 'test_c4_scripts: permission failure happened after create' >&2
    exit 1
fi

assumed_permission_state=$(run_runner_case permission-assumed 0 env FAKE_PERMISSION_DENY=1 \
    NSS_C4_ASSUME_PERMISSIONS=1)
if assert_contains 'policy-intelligence troubleshoot-policy' "$assumed_permission_state/calls" 2>/dev/null; then
    echo 'test_c4_scripts: explicit permission confirmation still called the IAM API' >&2
    exit 1
fi

# A generic permission/API error containing the words "resource not found"
# must not be mistaken for successful deletion.
run_runner_case delete-error 1 env FAKE_DELETE_DESCRIBE=generic >/dev/null

# A delete API NOT_FOUND is an idempotent success, while generic delete errors
# remain failures.
run_runner_case delete-notfound 0 env FAKE_DELETE_API=notfound >/dev/null
run_runner_case delete-api-error 1 env FAKE_DELETE_API=generic >/dev/null

printf '%s\n' 'C4 script contract tests passed'
