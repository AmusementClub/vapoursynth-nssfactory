#!/usr/bin/env python3
"""Reassess exact first-four-row BM3D pruning on real gray8 samples.

This models four threshold schedules while preserving the final stable top-8:

* raster: the current candidate order;
* nearest7-seed: seven spatial neighbours establish an initial upper bound,
  then candidates are still committed in raster order;
* nearest7-preload: the same seven neighbours are inserted up front into an
  ordinal-aware top-8, so later candidates can tighten the threshold;
* center-order: candidates are evaluated center-out, with raster ordinals used
  to resolve equal-distance matches.

The final-threshold case is deliberately unimplementable.  It is the strongest
possible threshold schedule for a given lower bound and therefore separates a
weak bound from a late threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_bm_pruning import raster_positions


@dataclass
class Totals:
    references: int = 0
    candidates: int = 0
    checked: int = 0
    pruned: int = 0
    rows: int = 0
    seed_rows_if_duplicated: int = 0

    def add(self, other: "Totals") -> None:
        self.references += other.references
        self.candidates += other.candidates
        self.checked += other.checked
        self.pruned += other.pruned
        self.rows += other.rows
        self.seed_rows_if_duplicated += other.seed_rows_if_duplicated


def better(a: tuple[np.float32, int], b: tuple[np.float32, int]) -> bool:
    return bool(a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]))


def add_top8(topk: list[tuple[np.float32, int]], item: tuple[np.float32, int]) -> None:
    if len(topk) < 8:
        topk.append(item)
        return
    worst = max(range(8), key=lambda i: (topk[i][0], topk[i][1]))
    if better(item, topk[worst]):
        topk[worst] = item


def can_reject(lower: np.float32, ordinal: int, worst: tuple[np.float32, int]) -> bool:
    """True only if every full distance represented by lower loses to worst."""
    return bool(lower > worst[0] or (lower == worst[0] and ordinal > worst[1]))


def reduce8_like_spatial_match(values: np.ndarray) -> np.ndarray:
    """Match SpatialMatch8's HSum8 reduction tree for float32 lanes."""
    halves = np.add(values[:, :4], values[:, 4:], dtype=np.float32)
    outer = np.add(halves[:, 0], halves[:, 3], dtype=np.float32)
    inner = np.add(halves[:, 1], halves[:, 2], dtype=np.float32)
    return np.add(outer, inner, dtype=np.float32)


def fma_square_add(diff: np.ndarray, accumulator: np.ndarray) -> np.ndarray:
    """Correctly round diff*diff+accumulator once, as the Highway FMA does."""
    exact_enough = np.multiply(diff.astype(np.float64), diff.astype(np.float64))
    exact_enough = np.add(exact_enough, accumulator.astype(np.float64))
    return exact_enough.astype(np.float32)


def spatial_match8_distances(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact first4 checkpoint and full SSD arithmetic schedule."""
    flat = delta.reshape(-1, 8, 8)
    squared = np.square(flat, dtype=np.float32)
    partial_vector = np.add(
        np.add(squared[:, 0], squared[:, 2], dtype=np.float32),
        np.add(squared[:, 1], squared[:, 3], dtype=np.float32),
        dtype=np.float32,
    )
    first4 = reduce8_like_spatial_match(partial_vector)

    accumulators = [
        fma_square_add(flat[:, row + 4], squared[:, row])
        for row in range(4)
    ]
    full_vector = np.add(
        np.add(accumulators[0], accumulators[2], dtype=np.float32),
        np.add(accumulators[1], accumulators[3], dtype=np.float32),
        dtype=np.float32,
    )
    full = reduce8_like_spatial_match(full_vector)
    if np.any(first4 > full):
        raise AssertionError("SpatialMatch8 first4 checkpoint exceeded its full SSD")
    return first4, full


def nearest_order(xs: np.ndarray, ys: np.ndarray, cx: int, cy: int, self_index: int) -> np.ndarray:
    ordinal = np.arange(xs.size, dtype=np.int32)
    radius = np.maximum(np.abs(xs - cx), np.abs(ys - cy))
    distance2 = np.square(xs - cx) + np.square(ys - cy)
    order = np.lexsort((ordinal, distance2, radius))
    return order[order != self_index]


def final_top8(full: np.ndarray, self_index: int) -> list[tuple[np.float32, int]]:
    topk = [(np.float32(0.0), self_index)]
    for ordinal, distance in enumerate(full):
        if ordinal != self_index:
            add_top8(topk, (distance, ordinal))
    return topk


def assert_same_top8(topk: list[tuple[np.float32, int]], full: np.ndarray, self_index: int) -> None:
    expected = sorted(final_top8(full, self_index), key=lambda item: (item[0], item[1]))
    actual = sorted(topk, key=lambda item: (item[0], item[1]))
    if actual != expected:
        raise AssertionError(f"pruning changed stable top-8: {actual!r} != {expected!r}")


def simulate_raster(lower: np.ndarray, full: np.ndarray, self_index: int) -> Totals:
    topk = [(np.float32(0.0), self_index)]
    totals = Totals(references=1, candidates=full.size - 1)
    for ordinal, distance in enumerate(full):
        if ordinal == self_index:
            continue
        if len(topk) == 8:
            totals.checked += 1
            worst = max(topk, key=lambda item: (item[0], item[1]))
            if can_reject(lower[ordinal], ordinal, worst):
                totals.pruned += 1
                totals.rows += 4
                continue
        totals.rows += 8
        add_top8(topk, (distance, ordinal))
    assert_same_top8(topk, full, self_index)
    return totals


def simulate_seeded(
    lower: np.ndarray,
    full: np.ndarray,
    self_index: int,
    seed_order: np.ndarray,
) -> Totals:
    seeds = seed_order[:7]
    seed_topk = [(np.float32(0.0), self_index), *((full[i], int(i)) for i in seeds)]
    seed_worst = max(seed_topk, key=lambda item: (item[0], item[1]))
    topk = [(np.float32(0.0), self_index)]
    totals = Totals(
        references=1,
        candidates=full.size - 1,
        seed_rows_if_duplicated=8 * len(seeds),
    )
    for ordinal, distance in enumerate(full):
        if ordinal == self_index:
            continue
        totals.checked += 1
        reject = can_reject(lower[ordinal], ordinal, seed_worst)
        if len(topk) == 8:
            dynamic_worst = max(topk, key=lambda item: (item[0], item[1]))
            reject = reject or can_reject(lower[ordinal], ordinal, dynamic_worst)
        if reject:
            totals.pruned += 1
            totals.rows += 4
            continue
        totals.rows += 8
        add_top8(topk, (distance, ordinal))
    assert_same_top8(topk, full, self_index)
    return totals


def simulate_preloaded(
    lower: np.ndarray,
    full: np.ndarray,
    self_index: int,
    seed_order: np.ndarray,
) -> Totals:
    """Pre-evaluate seven seeds and maintain one ordinal-aware top-8.

    Seed distances are charged once up front and reused, so their raster slots
    incur no second SSD.  Inserting them before raster traversal is exact only
    because every comparison includes the original raster ordinal.
    """
    seeds = {int(index) for index in seed_order[:7]}
    topk = [(np.float32(0.0), self_index), *((full[i], i) for i in seeds)]
    totals = Totals(references=1, candidates=full.size - 1, rows=8 * len(seeds))
    for ordinal, distance in enumerate(full):
        if ordinal == self_index or ordinal in seeds:
            continue
        totals.checked += 1
        worst = max(topk, key=lambda item: (item[0], item[1]))
        if can_reject(lower[ordinal], ordinal, worst):
            totals.pruned += 1
            totals.rows += 4
            continue
        totals.rows += 8
        add_top8(topk, (distance, ordinal))
    assert_same_top8(topk, full, self_index)
    return totals


def simulate_ordered(
    lower: np.ndarray,
    full: np.ndarray,
    self_index: int,
    order: np.ndarray,
) -> Totals:
    topk = [(np.float32(0.0), self_index)]
    totals = Totals(references=1, candidates=full.size - 1)
    for raw_ordinal in order:
        ordinal = int(raw_ordinal)
        if len(topk) == 8:
            totals.checked += 1
            worst = max(topk, key=lambda item: (item[0], item[1]))
            if can_reject(lower[ordinal], ordinal, worst):
                totals.pruned += 1
                totals.rows += 4
                continue
        totals.rows += 8
        add_top8(topk, (full[ordinal], ordinal))
    assert_same_top8(topk, full, self_index)
    return totals


def simulate_final_threshold(lower: np.ndarray, full: np.ndarray, self_index: int) -> Totals:
    topk = final_top8(full, self_index)
    worst = max(topk, key=lambda item: (item[0], item[1]))
    totals = Totals(references=1, candidates=full.size - 1, checked=full.size - 1)
    for ordinal in range(full.size):
        if ordinal == self_index:
            continue
        if can_reject(lower[ordinal], ordinal, worst):
            totals.pruned += 1
            totals.rows += 4
        else:
            totals.rows += 8
    return totals


def print_totals(sample: str, strategy: str, totals: Totals) -> None:
    prune_rate = totals.pruned / totals.checked if totals.checked else 0.0
    average_rows = totals.rows / totals.candidates if totals.candidates else 0.0
    row_savings = 1.0 - average_rows / 8.0
    duplicated_rows = (totals.rows + totals.seed_rows_if_duplicated) / totals.candidates if totals.candidates else 0.0
    print(
        f"{sample}\t{strategy}\t{totals.references}\t{totals.candidates}\t{totals.checked}\t"
        f"{totals.pruned}\t{prune_rate:.6f}\t{average_rows:.6f}\t{row_savings:.6f}\t{duplicated_rows:.6f}"
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
    ref_xs = raster_positions(args.width, block, block, args.sample_every)
    ref_ys = raster_positions(args.height, block, block, args.sample_every)
    aggregate: dict[str, Totals] = {}

    print(
        "sample\tstrategy\treferences\tcandidates\tchecked\tpruned\tprune_rate\t"
        "average_rows\trow_savings\taverage_rows_if_seed_duplicated"
    )
    for path in sorted(args.sample_dir.glob("*.gray8")):
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} bytes, expected {expected}")
        base = raw.reshape(args.height, args.width).astype(np.float32) * np.float32(1.0 / 255.0)
        rng = np.random.RandomState(args.seed)
        noisy = base + rng.randn(args.height, args.width).astype(np.float32) * np.float32(3.0 / 255.0)
        windows = np.lib.stride_tricks.sliding_window_view(noisy, (block, block))
        sample_totals: dict[str, Totals] = {}

        for cy in ref_ys:
            top = max(cy - bm_range, 0)
            bottom = min(cy + bm_range, args.height - block)
            for cx in ref_xs:
                left = max(cx - bm_range, 0)
                right = min(cx + bm_range, args.width - block)
                reference = windows[cy, cx]
                candidates = windows[top : bottom + 1, left : right + 1]
                delta = candidates - reference
                first4, full = spatial_match8_distances(delta)
                tail_sum = delta[:, :, 4:, :].sum(axis=(2, 3), dtype=np.float32).reshape(-1)
                first4_tail_dc = first4 + np.square(tail_sum, dtype=np.float32) / np.float32(32.0)
                if np.any(first4_tail_dc > full):
                    raise AssertionError("rounded first4 + tail-DC exceeded the full SSD on this sample")

                span = right - left + 1
                self_index = (cy - top) * span + (cx - left)
                grid_y, grid_x = np.indices((bottom - top + 1, span), dtype=np.int32)
                candidate_x = grid_x.reshape(-1) + left
                candidate_y = grid_y.reshape(-1) + top
                order = nearest_order(candidate_x, candidate_y, cx, cy, self_index)

                cases = {
                    "raster-first4": simulate_raster(first4, full, self_index),
                    "nearest7-seed-first4": simulate_seeded(first4, full, self_index, order),
                    "nearest7-preload-first4": simulate_preloaded(first4, full, self_index, order),
                    "center-order-first4": simulate_ordered(first4, full, self_index, order),
                    "final-threshold-first4": simulate_final_threshold(first4, full, self_index),
                    "raster-first4-tail-dc": simulate_raster(first4_tail_dc, full, self_index),
                    "nearest7-seed-first4-tail-dc": simulate_seeded(first4_tail_dc, full, self_index, order),
                    "nearest7-preload-first4-tail-dc": simulate_preloaded(
                        first4_tail_dc, full, self_index, order
                    ),
                    "center-order-first4-tail-dc": simulate_ordered(first4_tail_dc, full, self_index, order),
                    "final-threshold-first4-tail-dc": simulate_final_threshold(first4_tail_dc, full, self_index),
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
