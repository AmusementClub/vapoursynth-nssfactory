#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
usage: tests/run_c4_cpu_gate.sh --project PROJECT --zone ZONE --vs-venv-tar PATH

Exactly one of NSS_C4_BASELINE_REF or NSS_C4_BASELINE_DIR is required and must
identify the corrected CPU baseline. NSS_C4_ARTIFACT_DIR defaults to a temporary
directory; NSS_C4_CANDIDATE_DIR can override the candidate source tree.
NSS_C4_CRITICAL_MARGIN is forwarded to the guest (default 0.03 speedup units).
EOF
}

PROJECT=
ZONE=
VS_VENV_TAR=
while (($#)); do
    case "$1" in
        --project) PROJECT=${2:?missing value for --project}; shift 2 ;;
        --zone) ZONE=${2:?missing value for --zone}; shift 2 ;;
        --vs-venv-tar) VS_VENV_TAR=${2:?missing value for --vs-venv-tar}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# Validate all local inputs before any compute API that can create a resource.
[[ -n "$ZONE" ]] || { echo "C4 gate: --zone is required; refusing to guess a zone" >&2; exit 2; }
[[ -n "$VS_VENV_TAR" ]] || { echo "C4 gate: --vs-venv-tar is required" >&2; exit 2; }
[[ -f "$VS_VENV_TAR" && -r "$VS_VENV_TAR" ]] || {
    echo "C4 gate: venv tar is not a readable file: $VS_VENV_TAR" >&2
    exit 2
}
command -v gcloud >/dev/null 2>&1 || { echo "C4 gate: gcloud is required" >&2; exit 2; }
command -v tar >/dev/null 2>&1 || { echo "C4 gate: tar is required" >&2; exit 2; }
command -v gzip >/dev/null 2>&1 || { echo "C4 gate: gzip is required" >&2; exit 2; }

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE_REF=${NSS_C4_BASELINE_REF:-}
BASELINE_DIR=${NSS_C4_BASELINE_DIR:-}
CANDIDATE_DIR=${NSS_C4_CANDIDATE_DIR:-$REPO_DIR}
BASELINE_REVISION=${NSS_C4_BASELINE_REVISION:-}
CANDIDATE_REVISION=${NSS_C4_CANDIDATE_REVISION:-}
ARTIFACT_DIR=${NSS_C4_ARTIFACT_DIR:-}
TMP_DIR=
INSTANCE=
INSTANCE_CREATED=0
QUOTA_JSON=
DELETE_POLL_ATTEMPTS=${NSS_C4_DELETE_POLL_ATTEMPTS:-30}
DELETE_POLL_SECONDS=${NSS_C4_DELETE_POLL_SECONDS:-2}
ASSUME_PERMISSIONS=${NSS_C4_ASSUME_PERMISSIONS:-0}

if [[ -n "$BASELINE_REF" && -n "$BASELINE_DIR" ]]; then
    echo "C4 gate: set only one of NSS_C4_BASELINE_REF or NSS_C4_BASELINE_DIR" >&2
    exit 2
fi
if [[ -z "$BASELINE_REF" && -z "$BASELINE_DIR" ]]; then
    echo "C4 gate: corrected baseline is required via NSS_C4_BASELINE_REF or NSS_C4_BASELINE_DIR" >&2
    exit 2
fi
if [[ -n "$BASELINE_DIR" ]]; then
    [[ -d "$BASELINE_DIR" ]] || { echo "C4 gate: baseline directory is missing: $BASELINE_DIR" >&2; exit 2; }
    [[ -n "$BASELINE_REVISION" ]] || {
        echo "C4 gate: NSS_C4_BASELINE_REVISION is required with NSS_C4_BASELINE_DIR" >&2
        exit 2
    }
else
    git -C "$REPO_DIR" rev-parse --verify "$BASELINE_REF" >/dev/null 2>&1 || {
        echo "C4 gate: baseline ref does not resolve: $BASELINE_REF" >&2
        exit 2
    }
    BASELINE_REVISION=$(git -C "$REPO_DIR" rev-parse --short=12 "$BASELINE_REF")
fi
if [[ -z "$CANDIDATE_REVISION" ]]; then
    CANDIDATE_REVISION=$(git -C "$CANDIDATE_DIR" describe --tags --always --dirty 2>/dev/null || true)
fi
[[ -n "$CANDIDATE_REVISION" ]] || {
    echo "C4 gate: candidate revision is unavailable; set NSS_C4_CANDIDATE_REVISION" >&2
    exit 2
}

if [[ ! "$DELETE_POLL_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "C4 gate: NSS_C4_DELETE_POLL_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$DELETE_POLL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "C4 gate: NSS_C4_DELETE_POLL_SECONDS must be a non-negative number" >&2
    exit 2
fi
if [[ "$ASSUME_PERMISSIONS" != 0 && "$ASSUME_PERMISSIONS" != 1 ]]; then
    echo "C4 gate: NSS_C4_ASSUME_PERMISSIONS must be 0 or 1" >&2
    exit 2
fi

is_resource_not_found() {
    local message=$1
    # Keep this deliberately narrow. A generic "not found" substring can
    # occur in permission/API errors and must not be treated as confirmation
    # that cleanup completed.
    [[ "$message" == *"was not found"* || "$message" == *"NOT_FOUND"* ||
       "$message" == *"404 Not Found"* || "$message" == *"HTTP 404"* ]]
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if (( INSTANCE_CREATED )); then
        echo "C4 gate: deleting only instance $INSTANCE" >&2
        local delete_output
        local delete_rc=0
        if delete_output=$(gcloud compute instances delete "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --quiet 2>&1); then
            :
        else
            delete_rc=$?
            if is_resource_not_found "$delete_output"; then
                echo "C4 gate: $INSTANCE was already absent during deletion" >&2
            else
                echo "C4 gate: deletion request failed (status $delete_rc): $delete_output" >&2
                rc=1
            fi
        fi
        if (( rc == 0 )); then
            local gone=0
            for ((attempt = 1; attempt <= DELETE_POLL_ATTEMPTS; ++attempt)); do
                local describe_output
                if describe_output=$(gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" 2>&1); then
                    sleep "$DELETE_POLL_SECONDS"
                else
                    if is_resource_not_found "$describe_output"; then
                        gone=1
                        break
                    fi
                    echo "C4 gate: deletion verification failed for $INSTANCE: $describe_output" >&2
                    rc=1
                    break
                fi
            done
            if (( ! gone )); then
                echo "C4 gate: deletion did not complete for $INSTANCE in $ZONE" >&2
                rc=1
            else
                echo "C4 gate: $INSTANCE deletion confirmed" >&2
            fi
        fi
    fi
    if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
        rm -rf -- "$TMP_DIR"
    fi
    if [[ -n "$QUOTA_JSON" && -f "$QUOTA_JSON" ]]; then
        rm -f -- "$QUOTA_JSON"
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

if [[ -z "$PROJECT" ]]; then
    PROJECT=$(gcloud config get-value project 2>/dev/null || true)
fi
if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" || "$PROJECT" == "(unset)"* ]]; then
    echo "C4 gate: no gcloud project configured; pass --project" >&2
    exit 2
fi

command -v python3 >/dev/null 2>&1 || { echo "C4 gate: python3 is required for quota parsing" >&2; exit 2; }
REGION=${ZONE%-[a-z]}
quota_json=$(mktemp "${TMPDIR:-/tmp}/nss-c4-quota.XXXXXX")
QUOTA_JSON=$quota_json
if ! gcloud compute zones describe "$ZONE" --project "$PROJECT" --format='value(name)' >/dev/null; then
    echo "C4 gate: zone is unavailable or not accessible: $ZONE" >&2
    exit 2
fi
if ! gcloud compute regions describe "$REGION" --project "$PROJECT" --format=json >"$quota_json"; then
    echo "C4 gate: could not inspect quota for region $REGION" >&2
    exit 2
fi
python3 - "$quota_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
quotas = payload.get("quotas", [])
seen = False
for quota in quotas:
    metric = str(quota.get("metric", "")).upper()
    if metric not in {"CPUS", "C4_CPUS"}:
        continue
    seen = True
    limit = float(quota.get("limit", 0))
    usage = float(quota.get("usage", 0))
    if limit <= usage:
        raise SystemExit(f"CPU quota exhausted: {metric} usage={usage} limit={limit}")
if not seen:
    raise SystemExit("region response did not expose CPUS/C4_CPUS quota")
PY
rm -f "$quota_json"

# Machine-type lookup is the local Spot-capacity/shape admission check. Spot
# stock itself is finalized by the create call; its exact API error is retained
# in the shell log and never replaced with a guessed fallback zone/type.
if ! gcloud compute machine-types describe c4-highcpu-2 --project "$PROJECT" --zone "$ZONE" --format='value(name)' >/dev/null; then
    echo "C4 gate: c4-highcpu-2 is unavailable in $ZONE" >&2
    exit 2
fi
if [[ "$ASSUME_PERMISSIONS" == 1 ]]; then
    echo "C4 gate: operator confirmed compute.instances.create/delete permissions; skipping IAM API preflight" >&2
else
    PRINCIPAL=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)
    [[ -n "$PRINCIPAL" ]] || { echo "C4 gate: no active gcloud principal" >&2; exit 2; }
    permission_resource="//compute.googleapis.com/projects/$PROJECT/zones/$ZONE/instances/nss-c4-permission-probe"
    for permission in compute.instances.create compute.instances.delete; do
        access=$(gcloud policy-intelligence troubleshoot-policy iam \
            "//cloudresourcemanager.googleapis.com/projects/$PROJECT" \
            --principal-email="$PRINCIPAL" --permission="$permission" \
            --resource-name="$permission_resource" --resource-service=compute.googleapis.com \
            --resource-type=compute.googleapis.com/Instance --format='value(access)' 2>/dev/null || true)
        [[ "$access" == "GRANTED" ]] || {
            echo "C4 gate: $permission was not confirmed as GRANTED for $PRINCIPAL" >&2
            exit 2
        }
    done
fi

[[ -d "$CANDIDATE_DIR" ]] || { echo "C4 gate: candidate directory is missing: $CANDIDATE_DIR" >&2; exit 2; }
[[ -f "$REPO_DIR/tests/c4_guest_gate.sh" ]] || {
    echo "C4 gate: guest helper is missing" >&2
    exit 2
}

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nss-c4-run.XXXXXX")
candidate_tar="$TMP_DIR/candidate.tar.gz"
baseline_tar="$TMP_DIR/baseline.tar.gz"
guest_helper="$TMP_DIR/c4_guest_gate.sh"
cp "$REPO_DIR/tests/c4_guest_gate.sh" "$guest_helper"
chmod 0755 "$guest_helper"

archive_tree() {
    local source=$1
    local destination=$2
    # Both BSD tar (macOS) and GNU tar (guest/dev Linux) accept these long
    # exclude forms; keep build products out of the upload explicitly.
    COPYFILE_DISABLE=1 tar --exclude '.git' --exclude 'build' --exclude 'build-*' --exclude '.codegraph' \
        -czf "$destination" -C "$source" .
}

archive_tree "$CANDIDATE_DIR" "$candidate_tar"
if [[ -n "$BASELINE_DIR" ]]; then
    archive_tree "$BASELINE_DIR" "$baseline_tar"
else
    git -C "$REPO_DIR" archive --format=tar "$BASELINE_REF" | gzip -c >"$baseline_tar"
fi

suffix=$(date -u +%Y%m%d%H%M%S)-$RANDOM
INSTANCE="nss-c4-$suffix"
if gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
    echo "C4 gate: generated instance name already exists: $INSTANCE" >&2
    exit 2
fi
# Check for an exact-name sibling immediately before creation as well as any
# still-running instance created by this runner; no existing instance is ever
# a cleanup target.
siblings=$(gcloud compute instances list --project "$PROJECT" \
    --filter="name=('$INSTANCE') OR labels.nss-c4-runner=true" --format='value(name)' || true)
if [[ -n "$siblings" ]]; then
    echo "C4 gate: sibling instance exists: $INSTANCE" >&2
    exit 2
fi

echo "C4 gate: creating $INSTANCE (c4-highcpu-2 Spot) in $ZONE" >&2
if ! gcloud compute instances create "$INSTANCE" --project "$PROJECT" --zone "$ZONE" \
    --machine-type=c4-highcpu-2 --provisioning-model=SPOT --instance-termination-action=DELETE \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=50GB \
    --no-service-account --no-scopes --labels=nss-c4-runner=true; then
    if gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
        INSTANCE_CREATED=1
    fi
    echo "C4 gate: Spot admission failed; no fallback resource will be created" >&2
    exit 1
fi
INSTANCE_CREATED=1

ssh_ready=0
for _ in {1..60}; do
    if gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command='true' --quiet >/dev/null 2>&1; then
        ssh_ready=1
        break
    fi
    sleep 5
done
(( ssh_ready )) || { echo "C4 gate: SSH did not become ready for $INSTANCE" >&2; exit 1; }

remote_root="/tmp/$INSTANCE"
venv_name=$(basename "$VS_VENV_TAR")
remote_root_q=$(printf '%q' "$remote_root")
gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command="mkdir -p -- $remote_root_q" --quiet
gcloud compute scp "$guest_helper" "$candidate_tar" "$baseline_tar" "$VS_VENV_TAR" \
    "$INSTANCE:$remote_root/" --project "$PROJECT" --zone "$ZONE"
remote_out="$remote_root/results/c4.json"
guest_rc=0
guest_command=$(printf 'bash %q --root %q --baseline-tar %q --candidate-tar %q --vs-venv-tar %q --out %q --baseline-revision %q --candidate-revision %q' \
    "$remote_root/c4_guest_gate.sh" "$remote_root/work" "$remote_root/baseline.tar.gz" \
    "$remote_root/candidate.tar.gz" "$remote_root/$venv_name" "$remote_out" "$BASELINE_REVISION" \
    "$CANDIDATE_REVISION")
if [[ -n "${NSS_C4_CRITICAL_MARGIN:-}" ]]; then
    guest_command="NSS_C4_CRITICAL_MARGIN=$(printf '%q' "$NSS_C4_CRITICAL_MARGIN") $guest_command"
fi
if gcloud compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --command "$guest_command"; then
    :
else
    guest_rc=$?
    echo "C4 gate: guest run failed (status $guest_rc); instance cleanup remains armed" >&2
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
    ARTIFACT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nss-c4-artifact.XXXXXX")
else
    mkdir -p "$ARTIFACT_DIR"
fi
local_out="$ARTIFACT_DIR/$INSTANCE.json"
if ! gcloud compute scp "$INSTANCE:$remote_out" "$local_out" --project "$PROJECT" --zone "$ZONE"; then
    echo "C4 gate: guest artifact is unavailable at $remote_out" >&2
    if (( guest_rc != 0 )); then
        exit "$guest_rc"
    fi
    exit 1
fi
echo "C4 gate artifact: $local_out" >&2
if (( guest_rc != 0 )); then
    exit "$guest_rc"
fi
python3 - "$local_out" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(json.dumps({
    "paired_median_ratio": payload.get("paired_median_ratio"),
    "passed": payload.get("passed"),
    "formal_pairs": payload.get("formal_pairs"),
}, sort_keys=True))
if not payload.get("passed", False):
    raise SystemExit(3)
PY
