#!/usr/bin/env python3
"""Run the paired C4 gate for the generated DCT and NCSR FastExp round."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--profile-script", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--current-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(
    plugin: Path,
    script: Path,
    algorithm: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    parameters: dict[str, str],
    output: Path | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "NSS_SO": str(plugin),
            "NSS_PROFILE_ALGORITHM": algorithm,
            "NSS_PROFILE_WIDTH": str(width),
            "NSS_PROFILE_HEIGHT": str(height),
            "NSS_PROFILE_FRAMES": str(frames),
            "NSS_PROFILE_SEED": str(seed),
        }
    )
    env.update(parameters)
    if output is not None:
        env["NSS_PROFILE_OUTPUT"] = str(output)
    completed = subprocess.run(
        ["taskset", "-c", "0", sys.executable, str(script)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("schema") == "nssfactory.profile.cpu.v1":
            return payload
    raise RuntimeError(f"profile output did not contain JSON: {completed.stdout}")


def paired_rows(
    left: tuple[str, Path],
    right: tuple[str, Path],
    script: Path,
    algorithm: str,
    width: int,
    height: int,
    frames: int,
    pairs: int,
    seed_base: int,
    parameters: dict[str, str],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in range(1, pairs + 1):
        seed = seed_base + pair
        order = (left, right) if pair % 2 else (right, left)
        results: dict[str, dict[str, Any]] = {}
        for name, plugin in order:
            results[name] = profile(plugin, script, algorithm, width, height, frames, seed, parameters)
        left_ms = float(results[left[0]]["milliseconds_per_frame"])
        right_ms = float(results[right[0]]["milliseconds_per_frame"])
        rows.append(
            {
                "workload": label,
                "pair": pair,
                "seed": seed,
                "left": left[0],
                "right": right[0],
                "left_ms": left_ms,
                "right_ms": right_ms,
                "ratio": left_ms / right_ms,
            }
        )
    return rows


def compare_outputs(left: Path, right: Path, left_hash: str, right_hash: str) -> dict[str, Any]:
    a = np.fromfile(left, dtype=np.float32)
    b = np.fromfile(right, dtype=np.float32)
    if a.size == 0 or a.size != b.size:
        raise RuntimeError(f"quality outputs have incompatible sizes: {a.size} and {b.size}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    psnr = math.inf if rmse == 0.0 else 20.0 * math.log10(1.0 / rmse)
    return {
        "samples": int(a.size),
        "finite": finite,
        "left_sha256": left_hash,
        "right_sha256": right_hash,
        "hash_equal": left_hash == right_hash,
        "max_abs": max_abs,
        "rmse": rmse,
        "psnr_peak_1_db": psnr,
    }


def quality_case(
    left: tuple[str, Path],
    right: tuple[str, Path],
    script: Path,
    algorithm: str,
    width: int,
    height: int,
    seed: int,
    parameters: dict[str, str],
    out_dir: Path,
    label: str,
) -> dict[str, Any]:
    left_out = out_dir / f"{label}-{left[0]}.f32"
    right_out = out_dir / f"{label}-{right[0]}.f32"
    left_result = profile(left[1], script, algorithm, width, height, 1, seed, parameters, left_out)
    right_result = profile(right[1], script, algorithm, width, height, 1, seed, parameters, right_out)
    result = compare_outputs(
        left_out,
        right_out,
        str(left_result["output_sha256"]),
        str(right_result["output_sha256"]),
    )
    result.update({"workload": label, "left": left[0], "right": right[0]})
    left_out.unlink()
    right_out.unlink()
    return result


def median_ratio(rows: list[dict[str, Any]]) -> float:
    return statistics.median(float(row["ratio"]) for row in rows)


def main() -> int:
    args = parse_args()
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError("gate process must be pinned exclusively to CPU 0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plugins = {
        "current": args.current_build / "libnss.so",
        "candidate": args.candidate_build / "libnss.so",
    }
    for path in [*plugins.values(), args.profile_script]:
        if not path.is_file():
            raise FileNotFoundError(path)

    current = ("current", plugins["current"])
    candidate = ("candidate", plugins["candidate"])
    timing_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    bm3d_rows: list[dict[str, Any]] = []

    for block in (8, 16, 32):
        for group in (8, 16, 32, 64):
            label = f"bm3d-b{block}-g{group}"
            params = {
                "NSS_PROFILE_BM3D_BLOCK": str(block),
                "NSS_PROFILE_BM3D_STEP": str(min(block, 8)),
                "NSS_PROFILE_BM3D_GROUP": str(group),
                "NSS_PROFILE_BM3D_RANGE": "7",
                "NSS_PROFILE_BM3D_RADIUS": "0",
            }
            rows = paired_rows(current, candidate, args.profile_script, "bm3d", 256, 256, 3, 5,
                               10000 + block * 100 + group, params, label)
            timing_rows.extend(rows)
            ratio = median_ratio(rows)
            bm3d_rows.append({"block": block, "group": group, "pairs": len(rows), "paired_median_ratio": ratio})
            quality_rows.append(
                quality_case(current, candidate, args.profile_script, "bm3d", 256, 256, 20000 + block + group,
                             params, args.out_dir, label)
            )

    ncsr_params = {
        "NSS_PROFILE_NCSR_BLOCK": "8",
        "NSS_PROFILE_NCSR_STEP": "8",
        "NSS_PROFILE_NCSR_GROUP": "8",
        "NSS_PROFILE_NCSR_RANGE": "7",
        "NSS_PROFILE_NCSR_RADIUS": "0",
        "NSS_PROFILE_NCSR_ITERS": "2",
    }
    ncsr_incremental = paired_rows(current, candidate, args.profile_script, "ncsr", 1920, 1080, 3, 7, 31000,
                                   ncsr_params, "ncsr-1080p-incremental")
    timing_rows.extend(ncsr_incremental)
    quality_rows.append(
        quality_case(current, candidate, args.profile_script, "ncsr", 256, 256, 32000, ncsr_params,
                     args.out_dir, "ncsr-default")
    )

    lssc_results: list[dict[str, Any]] = []
    for block in (8, 16):
        label = f"lssc-b{block}"
        params = {
            "NSS_PROFILE_LSSC_BLOCK": str(block),
            "NSS_PROFILE_LSSC_STEP": str(block),
        }
        rows = paired_rows(current, candidate, args.profile_script, "lssc", 256, 256, 2, 3,
                           40000 + block, params, label)
        timing_rows.extend(rows)
        lssc_results.append({"block": block, "pairs": len(rows), "paired_median_ratio": median_ratio(rows)})
        quality_rows.append(
            quality_case(current, candidate, args.profile_script, "lssc", 256, 256, 41000 + block, params,
                         args.out_dir, label)
        )

    repeat_params = {
        "NSS_PROFILE_BM3D_BLOCK": "32",
        "NSS_PROFILE_BM3D_STEP": "8",
        "NSS_PROFILE_BM3D_GROUP": "64",
        "NSS_PROFILE_BM3D_RANGE": "7",
        "NSS_PROFILE_BM3D_RADIUS": "0",
    }
    repeat_a = profile(plugins["candidate"], args.profile_script, "bm3d", 256, 256, 1, 50000, repeat_params)
    repeat_b = profile(plugins["candidate"], args.profile_script, "bm3d", 256, 256, 1, 50000, repeat_params)
    ncsr_repeat_a = profile(plugins["candidate"], args.profile_script, "ncsr", 256, 256, 1, 50001, ncsr_params)
    ncsr_repeat_b = profile(plugins["candidate"], args.profile_script, "ncsr", 256, 256, 1, 50001, ncsr_params)
    repeat_exact = (
        repeat_a["output_sha256"] == repeat_b["output_sha256"]
        and ncsr_repeat_a["output_sha256"] == ncsr_repeat_b["output_sha256"]
    )

    default_quality = next(row for row in quality_rows if row["workload"] == "bm3d-b8-g8")
    lssc_default_quality = next(row for row in quality_rows if row["workload"] == "lssc-b8")
    approximate_quality = [row for row in quality_rows if row not in (default_quality, lssc_default_quality)]
    quality_passed = (
        default_quality["hash_equal"]
        and lssc_default_quality["hash_equal"]
        and all(
            row["finite"]
            and float(row["psnr_peak_1_db"]) >= 110.0
            and float(row["max_abs"]) <= 1.0e-4
            for row in approximate_quality
        )
    )
    minimum_matrix_ratio = min(float(row["paired_median_ratio"]) for row in bm3d_rows)
    affected_ratios = [
        float(row["paired_median_ratio"])
        for row in bm3d_rows
        if not (row["block"] == 8 and row["group"] == 8)
    ]
    affected_median = statistics.median(affected_ratios)
    ncsr_incremental_ratio = median_ratio(ncsr_incremental)
    minimum_lssc_ratio = min(float(row["paired_median_ratio"]) for row in lssc_results)
    passed = (
        quality_passed
        and repeat_exact
        and minimum_matrix_ratio >= 0.98
        and affected_median >= 1.02
        and ncsr_incremental_ratio >= 0.98
        and minimum_lssc_ratio >= 0.98
    )

    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    payload = {
        "schema": "nssfactory.c4.cpu-gate.v2",
        "gate_kind": "dct-ncsr-round",
        "machine_type": "c4-highcpu-2",
        "cpu": cpu,
        "cpu_affinity": "0",
        "smt_sibling": Path("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list").read_text().strip(),
        "current_revision": args.current_revision,
        "candidate_revision": args.candidate_revision,
        "plugin_sha256": {name: sha256(path) for name, path in plugins.items()},
        "formal_pairs": 7,
        "paired_median_ratio": ncsr_incremental_ratio,
        "bm3d_matrix": {
            "width": 256,
            "height": 256,
            "rows": bm3d_rows,
            "minimum_configuration_ratio": minimum_matrix_ratio,
            "affected_median_ratio": affected_median,
        },
        "ncsr": {
            "width": 1920,
            "height": 1080,
            "incremental_paired_median_ratio": ncsr_incremental_ratio,
        },
        "lssc": {"rows": lssc_results, "minimum_configuration_ratio": minimum_lssc_ratio},
        "quality": quality_rows,
        "determinism_passed": repeat_exact,
        "quality_passed": quality_passed,
        "timing_rows": timing_rows,
        "gates": {
            "minimum_configuration_ratio": 0.98,
            "bm3d_affected_median_ratio": 1.02,
            "ncsr_incremental_paired_median_ratio": 0.98,
            "quality_psnr_peak_1_db": 110.0,
            "quality_max_abs": 1.0e-4,
        },
        "passed": passed,
    }
    artifact = args.out_dir / "c4.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(artifact), "passed": passed, "bm3d_affected_median_ratio": affected_median,
                      "ncsr_incremental_paired_median_ratio": ncsr_incremental_ratio}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
