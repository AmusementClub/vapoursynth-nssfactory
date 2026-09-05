#!/usr/bin/env python3
"""Paired C4 baseline/candidate gate for the public VAggregate filter."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


BASELINE_SO = Path(os.environ["NSS_BASELINE_SO"])
CANDIDATE_SO = Path(os.environ["NSS_CANDIDATE_SO"])
PROFILE = Path(os.environ.get("NSS_PROFILE_SCRIPT", Path(__file__).with_name("profile_vaggregate.py")))
OUT = Path(os.environ.get("NSS_GATE_OUT", "/tmp/nss-vaggregate-gate"))
RADII = tuple(int(value) for value in os.environ.get("NSS_GATE_RADII", "0,1,2,4,8,16").split(","))
PAIRS = int(os.environ.get("NSS_GATE_PAIRS", "7"))
MAX_REGRESSION = float(os.environ.get("NSS_GATE_MAX_REGRESSION", "0.02"))


def run_one(plugin: Path, radius: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["NSS_SO"] = str(plugin)
    env["NSS_PROFILE_RADIUS"] = str(radius)
    completed = subprocess.run(
        [sys.executable, str(PROFILE)],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> int:
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError("gate process must be pinned exclusively to CPU 0")
    OUT.mkdir(parents=True, exist_ok=True)
    workloads: list[dict[str, Any]] = []
    passed = True
    for radius in RADII:
        rows: list[dict[str, Any]] = []
        for pair in range(1, PAIRS + 1):
            order = (("baseline", BASELINE_SO), ("candidate", CANDIDATE_SO))
            if pair % 2 == 0:
                order = tuple(reversed(order))
            results = {name: run_one(plugin, radius) for name, plugin in order}
            baseline = results["baseline"]
            candidate = results["candidate"]
            hash_equal = baseline["output_sha256"] == candidate["output_sha256"]
            ratio = baseline["milliseconds_per_frame"] / candidate["milliseconds_per_frame"]
            rows.append(
                {
                    "pair": pair,
                    "baseline_ms": baseline["milliseconds_per_frame"],
                    "candidate_ms": candidate["milliseconds_per_frame"],
                    "ratio": ratio,
                    "hash_equal": hash_equal,
                }
            )
        median_ratio = statistics.median(row["ratio"] for row in rows)
        row_passed = median_ratio >= 1.0 - MAX_REGRESSION and all(row["hash_equal"] for row in rows)
        passed = passed and row_passed
        workload = {
            "radius": radius,
            "pairs": PAIRS,
            "median_baseline_ms": statistics.median(row["baseline_ms"] for row in rows),
            "median_candidate_ms": statistics.median(row["candidate_ms"] for row in rows),
            "median_ratio": median_ratio,
            "passed": row_passed,
            "rows": rows,
        }
        workloads.append(workload)
        print(json.dumps({key: workload[key] for key in workload if key != "rows"}, sort_keys=True), flush=True)

    payload = {
        "schema": "nssfactory.c4.vaggregate-gate.v1",
        "cpu": cpu_model(),
        "baseline_source_sha256": os.environ.get("NSS_BASELINE_SOURCE_SHA256", ""),
        "candidate_source_sha256": os.environ.get("NSS_CANDIDATE_SOURCE_SHA256", ""),
        "max_regression": MAX_REGRESSION,
        "passed": passed,
        "workloads": workloads,
    }
    destination = OUT / "summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "wrote": str(destination)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
