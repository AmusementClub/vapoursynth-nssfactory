#!/usr/bin/env python3
"""Validate the final C4 CPU Highway campaign artifact."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


GATES = {
    "nlm": 1.30,
    "bm3d": 1.25,
    "wnnm": 1.15,
    "twsc": 1.15,
    "ncsr": 1.15,
    "lssc": 1.15,
    "nlh": 1.30,
    "mcwnnm": 1.15,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: object, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} must be finite")
    return result


def validate(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    require(payload.get("schema") == "nssfactory.c4.cpu-campaign.v1", "unexpected campaign schema")
    require(payload.get("machine_type") == "c4-highcpu-2", "campaign was not run on c4-highcpu-2")
    require(isinstance(payload.get("candidate_revision"), str) and payload["candidate_revision"],
            "candidate_revision is required")
    require(isinstance(payload.get("baseline_revision"), str) and payload["baseline_revision"],
            "baseline_revision is required")

    correctness = payload.get("correctness")
    require(isinstance(correctness, dict), "correctness object is required")
    for field in ("ctest_passed", "output_semantics_passed", "determinism_passed", "quality_passed"):
        require(correctness.get(field) is True, f"correctness.{field} must be true")

    algorithms = payload.get("algorithms")
    require(isinstance(algorithms, dict), "algorithms object is required")
    require(set(algorithms) == set(GATES), "campaign must contain exactly the eight CPU algorithms")
    for name, gate in GATES.items():
        row = algorithms[name]
        require(isinstance(row, dict), f"algorithms.{name} must be an object")
        pairs = row.get("pairs")
        require(isinstance(pairs, int) and not isinstance(pairs, bool) and pairs >= 7,
                f"algorithms.{name}.pairs must be at least 7")
        ratio = finite_number(row.get("paired_median_ratio"), f"algorithms.{name}.paired_median_ratio")
        minimum = finite_number(row.get("minimum_configuration_ratio"),
                                f"algorithms.{name}.minimum_configuration_ratio")
        require(ratio >= gate, f"{name} paired median {ratio:.6f}x is below {gate:.2f}x")
        require(minimum >= 0.98, f"{name} has a tested configuration regressing by more than 2 percent")
        require(row.get("quality_passed") is True, f"algorithms.{name}.quality_passed must be true")
        require(row.get("determinism_passed") is True, f"algorithms.{name}.determinism_passed must be true")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <campaign.json>", file=sys.stderr)
        return 2
    try:
        validate(Path(argv[1]))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"CPU campaign validation failed: {exc}", file=sys.stderr)
        return 1
    print("CPU campaign validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
