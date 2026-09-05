#!/usr/bin/env python3
"""Paired C4 A/B: BM3D rolling vs explicit BM3D+VAggregate."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

PLUGIN = Path(os.environ["NSS_SO"])
GRAY8 = os.environ.get("NSS_PROFILE_GRAY8", "/opt/nss-c4/samples/gray8/MAPPA.gray8")
WIDTH = int(os.environ.get("NSS_PROFILE_WIDTH", "1920"))
HEIGHT = int(os.environ.get("NSS_PROFILE_HEIGHT", "1080"))
OUT = Path(os.environ.get("NSS_GATE_OUT", "/tmp/nss-bm3d-rolling-out"))
_PLUGIN_LOADED = False


def make_source(core: vs.Core, frames: int, seed: int) -> vs.VideoNode:
    rng = np.random.RandomState(seed)
    raw = np.fromfile(GRAY8, dtype=np.uint8)
    expected = WIDTH * HEIGHT
    if raw.size != expected:
        raise ValueError(f"gray8 sample has {raw.size} bytes, expected {expected}")
    base = raw.reshape(HEIGHT, WIDTH).astype(np.float32) * np.float32(1.0 / 255.0)
    noise = rng.randn(HEIGHT, WIDTH).astype(np.float32) * np.float32(3.0 / 255.0)
    plane = base + noise
    blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=vs.GRAYS, length=frames, color=[0.0])

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        output = f.copy()
        np.asarray(output[0])[:, :] = plane
        return output

    return core.std.ModifyFrame(clip=blank, clips=blank, selector=fill)


def make_filter(core: vs.Core, source: vs.VideoNode, mode: str, radius: int) -> vs.VideoNode:
    kwargs = dict(
        sigma=3,
        radius=radius,
        block_size=8,
        block_step=8,
        group_size=8,
        bm_range=7,
    )
    if mode == "rolling":
        return core.nss.BM3D(
            source,
            temporal_mode="rolling",
            rolling_chunk=4,
            rolling_cache_limit=16,
            **kwargs,
        )
    fat = core.nss.BM3D(source, temporal_mode="legacy", **kwargs)
    return core.nss.VAggregate(fat, source, radius=radius)


def run_pass(mode: str, radius: int, frames: int, seed: int, hash_last: bool) -> dict[str, Any]:
    global _PLUGIN_LOADED
    core = vs.core
    core.num_threads = 1
    if not _PLUGIN_LOADED:
        core.std.LoadPlugin(path=str(PLUGIN))
        _PLUGIN_LOADED = True
    source = make_source(core, frames, seed)
    output = make_filter(core, source, mode, radius)
    start = time.perf_counter()
    last = None
    for n in range(frames):
        last = output.get_frame(n)
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    assert last is not None
    values = np.ascontiguousarray(np.asarray(last[0])[:, :WIDTH], dtype=np.float32)
    digest.update(values.tobytes())
    return {
        "mode": mode,
        "radius": radius,
        "frames": frames,
        "seed": seed,
        "milliseconds_per_frame": elapsed * 1000.0 / frames,
        "output_sha256": digest.hexdigest() if hash_last else "",
    }


def paired(radius: int, frames: int, pairs: int, seed_base: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in range(1, pairs + 1):
        seed = seed_base + pair
        order = ("legacy", "rolling") if pair % 2 else ("rolling", "legacy")
        results = {mode: run_pass(mode, radius, frames, seed, hash_last=False) for mode in order}
        legacy_ms = float(results["legacy"]["milliseconds_per_frame"])
        rolling_ms = float(results["rolling"]["milliseconds_per_frame"])
        rows.append(
            {
                "pair": pair,
                "seed": seed,
                "legacy_ms": legacy_ms,
                "rolling_ms": rolling_ms,
                "ratio": legacy_ms / rolling_ms,
            }
        )
    ratios = [row["ratio"] for row in rows]
    return {
        "radius": radius,
        "frames": frames,
        "pairs": pairs,
        "median_ratio": statistics.median(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "median_legacy_ms": statistics.median(row["legacy_ms"] for row in rows),
        "median_rolling_ms": statistics.median(row["rolling_ms"] for row in rows),
        "rows": rows,
    }


def quality(radius: int, frames: int, seed: int) -> dict[str, Any]:
    left = run_pass("legacy", radius, frames, seed, hash_last=True)
    right = run_pass("rolling", radius, frames, seed, hash_last=True)
    return {
        "radius": radius,
        "legacy_sha256": left["output_sha256"],
        "rolling_sha256": right["output_sha256"],
        "hash_equal": left["output_sha256"] == right["output_sha256"],
    }


def main() -> int:
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError("gate process must be pinned exclusively to CPU 0")
    if os.environ.get("NSS_GATE_SINGLE"):
        payload = run_pass(
            os.environ["NSS_PROFILE_BM3D_TEMPORAL_MODE"],
            int(os.environ["NSS_PROFILE_BM3D_RADIUS"]),
            int(os.environ["NSS_PROFILE_FRAMES"]),
            int(os.environ.get("NSS_PROFILE_SEED", "42")),
            hash_last=True,
        )
        print(json.dumps({"schema": "nssfactory.profile.cpu.v1", **payload}, sort_keys=True))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": "nssfactory.c4.bm3d-rolling.v1",
        "sample": Path(GRAY8).name,
        "workloads": [],
        "quality": [],
    }
    for radius, frames in ((1, 12), (4, 8)):
        timing = paired(radius, frames, 5, 44000 + radius * 100)
        quality_row = quality(radius, frames, 55000 + radius)
        summary["workloads"].append(timing)
        summary["quality"].append(quality_row)
        print(
            json.dumps(
                {
                    "radius": radius,
                    "median_ratio": timing["median_ratio"],
                    "median_legacy_ms": timing["median_legacy_ms"],
                    "median_rolling_ms": timing["median_rolling_ms"],
                    "hash_equal": quality_row["hash_equal"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": str(OUT / "summary.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
