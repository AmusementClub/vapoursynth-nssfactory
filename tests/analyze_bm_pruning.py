#!/usr/bin/env python3
"""Estimate exact BM3D SSD pruning rates on deterministic gray8 samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def raster_positions(length: int, block: int, step: int, sample_every: int) -> list[int]:
    positions = [min(value, length - block) for value in range(0, length - block + step, step)]
    return positions[::sample_every]


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
    width = args.width
    height = args.height
    expected = width * height
    xs = raster_positions(width, block, block, args.sample_every)
    ys = raster_positions(height, block, block, args.sample_every)
    fixed_subsets = {
        "first4": np.array([0, 1, 2, 3]),
        "alternating": np.array([0, 2, 4, 6]),
        "middle4": np.array([2, 3, 4, 5]),
        "last4": np.array([4, 5, 6, 7]),
    }

    print("sample\tstrategy\tpruned\teligible\tprune_rate\testimated_full_rows")
    for path in sorted(args.sample_dir.glob("*.gray8")):
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} bytes, expected {expected}")
        base = raw.reshape(height, width).astype(np.float32) * np.float32(1.0 / 255.0)
        rng = np.random.RandomState(args.seed)
        noisy = base + rng.randn(height, width).astype(np.float32) * np.float32(3.0 / 255.0)
        windows = np.lib.stride_tricks.sliding_window_view(noisy, (block, block))
        stats = {
            name: [0, 0]
            for name in (
                *fixed_subsets,
                "variance4",
                "first4_tail_dc",
                "alternating_tail_dc",
                "variance4_tail_dc",
                "dc",
                "row_dc",
                "col_dc",
                "row_col_dc",
                "oracle4",
            )
        }

        for cy in ys:
            top = max(cy - bm_range, 0)
            bottom = min(cy + bm_range, height - block)
            for cx in xs:
                left = max(cx - bm_range, 0)
                right = min(cx + bm_range, width - block)
                reference = windows[cy, cx]
                candidates = windows[top : bottom + 1, left : right + 1]
                delta = candidates - reference
                squared = np.square(delta, dtype=np.float32)
                row_sums = squared.sum(axis=3, dtype=np.float32)
                full = row_sums.sum(axis=2, dtype=np.float32).reshape(-1)
                subsets = dict(fixed_subsets)
                subsets["variance4"] = np.argsort(reference.var(axis=1))[-4:]
                bounds = {
                    name: row_sums[:, :, rows].sum(axis=2, dtype=np.float32).reshape(-1)
                    for name, rows in subsets.items()
                }
                for name in ("first4", "alternating", "variance4"):
                    rows = subsets[name]
                    tail = np.setdiff1d(np.arange(block), rows, assume_unique=True)
                    tail_sum = delta[:, :, tail, :].sum(axis=(2, 3), dtype=np.float32)
                    bounds[f"{name}_tail_dc"] = bounds[name] + (
                        np.square(tail_sum, dtype=np.float32) / np.float32(tail.size * block)
                    ).reshape(-1)
                total_sum = delta.sum(axis=(2, 3), dtype=np.float32)
                dc = np.square(total_sum, dtype=np.float32) / np.float32(block * block)
                row_sum = delta.sum(axis=3, dtype=np.float32)
                col_sum = delta.sum(axis=2, dtype=np.float32)
                row_dc = np.square(row_sum, dtype=np.float32).sum(axis=2, dtype=np.float32) / np.float32(block)
                col_dc = np.square(col_sum, dtype=np.float32).sum(axis=2, dtype=np.float32) / np.float32(block)
                bounds["dc"] = dc.reshape(-1)
                bounds["row_dc"] = row_dc.reshape(-1)
                bounds["col_dc"] = col_dc.reshape(-1)
                # Row-constant and column-constant subspaces intersect only at
                # the all-constant component, so subtract DC once.
                bounds["row_col_dc"] = (row_dc + col_dc - dc).reshape(-1)
                bounds["oracle4"] = np.sort(row_sums, axis=2)[:, :, -4:].sum(axis=2, dtype=np.float32).reshape(-1)
                span = right - left + 1
                self_index = (cy - top) * span + (cx - left)

                for name, lower_bounds in bounds.items():
                    topk = [np.float32(0.0)]
                    for index, (lower_bound, distance) in enumerate(zip(lower_bounds, full)):
                        if index == self_index:
                            continue
                        if len(topk) == 8:
                            worst = max(topk)
                            stats[name][1] += 1
                            if not lower_bound < worst:
                                stats[name][0] += 1
                                continue
                        if len(topk) < 8:
                            topk.append(distance)
                        else:
                            worst_index = int(np.argmax(topk))
                            if distance < topk[worst_index]:
                                topk[worst_index] = distance

        four_row_strategies = {*fixed_subsets, "variance4", "first4_tail_dc", "alternating_tail_dc",
                               "variance4_tail_dc", "oracle4"}
        for name, (pruned, eligible) in stats.items():
            rate = pruned / eligible if eligible else 0.0
            initial_rows = 4.0 if name in four_row_strategies else 0.0
            estimated_rows = initial_rows + (8.0 - initial_rows) * (1.0 - rate)
            print(f"{path.stem}\t{name}\t{pruned}\t{eligible}\t{rate:.6f}\t{estimated_rows:.3f}")


if __name__ == "__main__":
    main()
