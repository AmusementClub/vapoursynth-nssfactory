#!/usr/bin/env python3
"""Model exact staged BM3D SSD pruning on deterministic gray8 samples.

The pair strategy mirrors SpatialMatch8's adjacent-x scheduling.  It skips the
last four rows only when both candidates in a pair are already unable to enter
the stable top-8 set, so the accepted matches and their traversal order remain
unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from analyze_bm_pruning import raster_positions


@dataclass
class Totals:
    candidates: int = 0
    eligible: int = 0
    pruned: int = 0
    rows: int = 0
    checks: int = 0

    def add(self, other: "Totals") -> None:
        self.candidates += other.candidates
        self.eligible += other.eligible
        self.pruned += other.pruned
        self.rows += other.rows
        self.checks += other.checks


def insert_top8(topk: list[np.float32], distance: np.float32) -> None:
    if len(topk) < 8:
        topk.append(distance)
        return
    worst_index = int(np.argmax(topk))
    if distance < topk[worst_index]:
        topk[worst_index] = distance


def thresholds_before(full: np.ndarray, self_index: int) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.full(full.size, np.float32(np.inf), dtype=np.float32)
    eligible = np.zeros(full.size, dtype=bool)
    topk = [np.float32(0.0)]
    for index, distance in enumerate(full):
        if index == self_index:
            continue
        if len(topk) == 8:
            thresholds[index] = max(topk)
            eligible[index] = True
        insert_top8(topk, distance)
    return thresholds, eligible


def simulate_individual(lower: np.ndarray, full: np.ndarray, self_index: int, prefix_rows: int) -> Totals:
    thresholds, eligible = thresholds_before(full, self_index)
    candidates = np.arange(full.size) != self_index
    prunable = candidates & eligible & ~(lower < thresholds)
    count = int(np.count_nonzero(candidates))
    pruned = int(np.count_nonzero(prunable))
    return Totals(
        candidates=count,
        eligible=int(np.count_nonzero(eligible)),
        pruned=pruned,
        rows=prefix_rows * pruned + 8 * (count - pruned),
        checks=int(np.count_nonzero(eligible)),
    )


def simulate_pair(lower: np.ndarray, full: np.ndarray, self_index: int, span: int, prefix_rows: int) -> Totals:
    topk = [np.float32(0.0)]
    totals = Totals(candidates=full.size - 1)
    for row_start in range(0, full.size, span):
        for column in range(0, span, 2):
            indices = list(range(row_start + column, min(row_start + column + 2, row_start + span)))
            active = [index for index in indices if index != self_index]
            if not active:
                continue
            can_check = len(topk) == 8
            if can_check:
                worst = max(topk)
                totals.eligible += len(active)
                totals.checks += 1
                if all(not lower[index] < worst for index in active):
                    totals.pruned += len(active)
                    totals.rows += prefix_rows * len(active)
                    continue
            totals.rows += 8 * len(active)
            for index in active:
                insert_top8(topk, full[index])
    return totals


def simulate_pair_common(
    prefix_columns: np.ndarray,
    full: np.ndarray,
    self_index: int,
    span: int,
    prefix_rows: int,
    reduction: str,
) -> Totals:
    """Reject a pair from a lower bound shared by both candidates."""
    topk = [np.float32(0.0)]
    totals = Totals(candidates=full.size - 1)
    for row_start in range(0, full.size, span):
        for column in range(0, span, 2):
            indices = list(range(row_start + column, min(row_start + column + 2, row_start + span)))
            active = [index for index in indices if index != self_index]
            if not active:
                continue
            can_check = len(topk) == 8
            if can_check:
                worst = max(topk)
                totals.eligible += len(active)
                totals.checks += 1
                common = prefix_columns[active[0]]
                for index in active[1:]:
                    common = np.minimum(common, prefix_columns[index])
                if reduction == "sum":
                    common_lower = common.sum(dtype=np.float32)
                elif reduction == "max":
                    common_lower = common.max()
                else:
                    raise ValueError(f"unknown reduction: {reduction}")
                if not common_lower < worst:
                    totals.pruned += len(active)
                    totals.rows += prefix_rows * len(active)
                    continue
            totals.rows += 8 * len(active)
            for index in active:
                insert_top8(topk, full[index])
    return totals


def simulate_checkpoints(
    row_sums: np.ndarray,
    full: np.ndarray,
    self_index: int,
    row_order: Iterable[int],
    checkpoints: tuple[int, ...],
) -> Totals:
    thresholds, eligible = thresholds_before(full, self_index)
    candidates = np.arange(full.size) != self_index
    ordered = row_sums[:, list(row_order)]
    cumulative = np.cumsum(ordered, axis=1, dtype=np.float32)
    totals = Totals(candidates=int(np.count_nonzero(candidates)), eligible=int(np.count_nonzero(eligible)))
    for index in np.flatnonzero(candidates):
        if not eligible[index]:
            totals.rows += 8
            continue
        stopped = False
        for checkpoint in checkpoints:
            totals.checks += 1
            if not cumulative[index, checkpoint - 1] < thresholds[index]:
                totals.pruned += 1
                totals.rows += checkpoint
                stopped = True
                break
        if not stopped:
            totals.rows += 8
    return totals


def print_totals(sample: str, strategy: str, totals: Totals) -> None:
    prune_rate = totals.pruned / totals.eligible if totals.eligible else 0.0
    average_rows = totals.rows / totals.candidates if totals.candidates else 0.0
    row_savings = 1.0 - average_rows / 8.0
    checks = totals.checks / totals.candidates if totals.candidates else 0.0
    print(
        f"{sample}\t{strategy}\t{totals.candidates}\t{totals.eligible}\t{totals.pruned}\t"
        f"{prune_rate:.6f}\t{average_rows:.6f}\t{row_savings:.6f}\t{checks:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()

    block = 8
    bm_range = 7
    expected = args.width * args.height
    xs = raster_positions(args.width, block, block, args.sample_every)
    ys = raster_positions(args.height, block, block, args.sample_every)
    aggregate: dict[str, Totals] = {}

    print("sample\tstrategy\tcandidates\teligible\tpruned\tprune_rate\taverage_rows\trow_savings\tchecks_per_candidate")
    for path in sorted(args.sample_dir.glob("*.gray8")):
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} bytes, expected {expected}")
        base = raw.reshape(args.height, args.width).astype(np.float32) * np.float32(1.0 / 255.0)
        rng = np.random.RandomState(args.seed)
        noisy = base + rng.randn(args.height, args.width).astype(np.float32) * np.float32(3.0 / 255.0)
        windows = np.lib.stride_tricks.sliding_window_view(noisy, (block, block))
        sample_totals: dict[str, Totals] = {}

        for cy in ys:
            top = max(cy - bm_range, 0)
            bottom = min(cy + bm_range, args.height - block)
            for cx in xs:
                left = max(cx - bm_range, 0)
                right = min(cx + bm_range, args.width - block)
                reference = windows[cy, cx]
                candidates = windows[top : bottom + 1, left : right + 1]
                delta = candidates - reference
                squared = np.square(delta, dtype=np.float32)
                row_sums = squared.sum(axis=3, dtype=np.float32).reshape(-1, block)
                full = row_sums.sum(axis=1, dtype=np.float32)
                span = right - left + 1
                self_index = (cy - top) * span + (cx - left)

                prefix_columns1 = squared[:, :, :1, :].sum(axis=2, dtype=np.float32).reshape(-1, block)
                prefix_columns2 = squared[:, :, :2, :].sum(axis=2, dtype=np.float32).reshape(-1, block)
                prefix_columns4 = squared[:, :, :4, :].sum(axis=2, dtype=np.float32).reshape(-1, block)
                first4 = prefix_columns4.sum(axis=1, dtype=np.float32)
                column4 = prefix_columns4.max(axis=1)
                variance_order = tuple(np.argsort(reference.var(axis=1))[::-1].tolist())
                oracle_order = np.argsort(row_sums, axis=1)[:, ::-1]
                oracle_rows = np.take_along_axis(row_sums, oracle_order, axis=1)

                cases = {
                    "first4-individual": simulate_individual(first4, full, self_index, 4),
                    "first4-pair-all": simulate_pair(first4, full, self_index, span, 4),
                    "common4-pair-min-columns": simulate_pair_common(
                        prefix_columns4, full, self_index, span, 4, "sum"
                    ),
                    "common1-pair-max-column": simulate_pair_common(
                        prefix_columns1, full, self_index, span, 1, "max"
                    ),
                    "common2-pair-max-column": simulate_pair_common(
                        prefix_columns2, full, self_index, span, 2, "max"
                    ),
                    "common4-pair-max-column": simulate_pair_common(
                        prefix_columns4, full, self_index, span, 4, "max"
                    ),
                    "column4-individual": simulate_individual(column4, full, self_index, 4),
                    "column4-pair-all": simulate_pair(column4, full, self_index, span, 4),
                    "natural-each-row": simulate_checkpoints(row_sums, full, self_index, range(8), tuple(range(1, 8))),
                    "natural-after4-6": simulate_checkpoints(row_sums, full, self_index, range(8), (4, 6)),
                    "alternating-each-row": simulate_checkpoints(
                        row_sums, full, self_index, (0, 2, 4, 6, 1, 3, 5, 7), tuple(range(1, 8))
                    ),
                    "variance-each-row": simulate_checkpoints(
                        row_sums, full, self_index, variance_order, tuple(range(1, 8))
                    ),
                    "oracle-each-row": simulate_checkpoints(
                        oracle_rows, full, self_index, range(8), tuple(range(1, 8))
                    ),
                }
                for name, totals in cases.items():
                    sample_totals.setdefault(name, Totals()).add(totals)

        for name, totals in sample_totals.items():
            print_totals(path.stem, name, totals)
            aggregate.setdefault(name, Totals()).add(totals)

    for name, totals in aggregate.items():
        print_totals("ALL", name, totals)


if __name__ == "__main__":
    main()
