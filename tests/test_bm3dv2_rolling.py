#!/usr/bin/env python3
"""Compare rolling BM3Dv2 against legacy BM3D+VAggregate."""

from __future__ import annotations

import os
import sys

import numpy as np
import vapoursynth as vs


def make_clip(core: vs.Core, frames: int, width: int, height: int, seed: int = 0) -> vs.VideoNode:
    rng = np.random.RandomState(seed)
    base = rng.rand(height, width).astype(np.float32)
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=frames, color=[0.0])

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        np.asarray(out[0])[:, :] = (base + 0.02 * n).astype(np.float32)
        return out

    return core.std.ModifyFrame(clip=blank, clips=blank, selector=fill)


def frame_plane(clip: vs.VideoNode, n: int) -> np.ndarray:
    f = clip.get_frame(n)
    return np.asarray(f[0]).copy()


def main() -> int:
    plugin = os.environ.get("NSS_SO")
    if not plugin:
        print("NSS_SO is required", file=sys.stderr)
        return 2
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=plugin)
    src = make_clip(core, frames=7, width=48, height=32, seed=3)
    radius = 1
    kwargs = dict(sigma=3, radius=radius, block_size=8, group_size=8, block_step=8, bm_range=7)
    legacy = core.nss.BM3Dv2(src, temporal_mode="legacy", **kwargs)
    rolling = core.nss.BM3Dv2(src, temporal_mode="rolling", rolling_chunk=4, rolling_cache_limit=2, **kwargs)
    defaulted = core.nss.BM3Dv2(src, **kwargs)
    max_abs = 0.0
    for n in range(src.num_frames):
        a = frame_plane(legacy, n)
        b = frame_plane(rolling, n)
        c = frame_plane(defaulted, n)
        max_abs = max(max_abs, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        max_abs = max(max_abs, float(np.max(np.abs(a.astype(np.float64) - c.astype(np.float64)))))
    again = frame_plane(rolling, 0)
    max_abs = max(max_abs, float(np.max(np.abs(frame_plane(legacy, 0).astype(np.float64) - again.astype(np.float64)))))
    print(f"rolling vs legacy max_abs={max_abs:.8g}")
    if max_abs > 1e-5:
        print("rolling/legacy mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
