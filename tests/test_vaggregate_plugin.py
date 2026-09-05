#!/usr/bin/env python3
"""Plugin-level VAggregate and temporal BM3D regression checks."""

from __future__ import annotations

import os
import sys

import numpy as np
import vapoursynth as vs


def main() -> int:
    plugin = os.environ.get("NSS_SO")
    if not plugin:
        print("NSS_SO is required", file=sys.stderr)
        return 2

    core = vs.core
    core.num_threads = 2
    core.std.LoadPlugin(path=plugin)
    width, height, radius, frames = 48, 32, 1, 3
    slices = 2 * radius + 1
    src = core.std.BlankClip(
        width=width,
        height=height,
        format=vs.YUV444PS,
        length=frames,
        color=[0.2, 0.3, 0.4],
    )

    fat_blank = core.std.BlankClip(
        width=width,
        height=height * slices * 2,
        format=vs.YUV444PS,
        length=frames,
        color=[0.0, 0.0, 0.0],
    )

    def fill_fat(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        out = f.copy()
        for plane in range(3):
            values = np.asarray(out[plane])
            for t in range(slices):
                values[(t * 2) * height : (t * 2 + 1) * height, :width] = (plane + 1) * (t + 1)
                values[(t * 2 + 1) * height : (t * 2 + 2) * height, :width] = t + 2
            values[0, 0] = 9.0
            values[height, 0] = 0.0
            values[2 * height, 0] = 9.0
            values[3 * height, 0] = 0.0
            values[4 * height, 0] = 9.0
            values[5 * height, 0] = 0.0
        return out

    fat = core.std.ModifyFrame(clip=fat_blank, clips=fat_blank, selector=fill_fat)
    selected = core.nss.VAggregate(fat, src, radius=radius, planes=[1])
    frame = selected.get_frame(0)
    expected_plane1 = (2.0 * (1.0 + 2.0 + 3.0)) / (2.0 + 3.0 + 4.0)
    if not np.allclose(np.asarray(frame[0])[:, :width], 0.2, atol=1e-6):
        raise AssertionError("planes=[1] did not copy unselected plane 0")
    if not np.allclose(np.asarray(frame[2])[:, :width], 0.4, atol=1e-6):
        raise AssertionError("planes=[1] did not copy unselected plane 2")
    plane1 = np.asarray(frame[1])[:, :width]
    if abs(float(plane1[1, 1]) - expected_plane1) > 1e-6:
        raise AssertionError("planes=[1] did not aggregate plane 1")
    if abs(float(plane1[0, 0]) - 0.3) > 1e-6:
        raise AssertionError("zero-denominator fallback did not copy src")

    bm_fat = core.nss.BM3D(
        src,
        sigma=[3.0, 0.0, 0.0],
        radius=radius,
        block_size=8,
        block_step=8,
        group_size=8,
        bm_range=3,
    )
    bm_out = core.nss.VAggregate(bm_fat, src, radius=radius)
    bm_frame = bm_out.get_frame(1)
    for plane, expected in ((1, 0.3), (2, 0.4)):
        if not np.allclose(np.asarray(bm_frame[plane])[:, :width], expected, atol=1e-6):
            raise AssertionError(f"BM3D sigma=0 plane {plane} was not preserved")

    for removed in ("BM3Dv2", "WNNMv2", "NLHv2", "MCWNNMv2"):
        if hasattr(core.nss, removed):
            raise AssertionError(f"removed API is still registered: {removed}")

    try:
        core.nss.BM3D(src, radius=1, temporal_mode="typo")
    except vs.Error:
        pass
    else:
        raise AssertionError("BM3D accepted an unknown temporal_mode")

    print("VAggregate plugin regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
