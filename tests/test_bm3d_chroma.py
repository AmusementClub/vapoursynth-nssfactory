#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""BM3D chroma=1 (CBM3D): groups matched on plane 0, every plane filtered with them.

Checks, on a small YUV 4:4:4 clip, spatial and radius 1/2, basic and with ref:
plane 0 is the plane the separate filter gives (same groups, same filter);
planes 1 and 2 change and are denoised; a plane with sigma 0 is copied and
does not change the others; rolling equals legacy + VAggregate; chroma needs
YUV 4:4:4 and 0 or 1.

usage: test_bm3d_chroma.py --plugin PATH
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"bm3d chroma SKIP: {error}")
    raise SystemExit(77)

WIDTH, HEIGHT, FRAMES = 64, 48, 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(path=args.plugin)
    rng = np.random.default_rng(11)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    # The planes share their edges (as luma and chroma of a picture do) at different amplitudes.
    clean = [[(0.5 + a * np.sign(np.sin((x + 2 * t) / 6) * np.cos((y + t) / 7))).astype(np.float32) for a in (0.2, 0.1, -0.08)]
             for t in range(FRAMES)]
    noisy = [[(c + rng.normal(0, 10 / 255, c.shape)).astype(np.float32) for c in frame] for frame in clean]
    blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=vs.YUV444PS, length=FRAMES)

    def fill(n, f):
        out = f.copy()
        for p in range(3):
            np.asarray(out[p])[:] = noisy[n][p]
        return out
    src = core.std.ModifyFrame(blank, blank, fill)

    def planes(node, n):
        f = node.get_frame(n)
        return [np.array(f[p], dtype=np.float64) for p in range(3)]

    def psnr(a, b):
        return 10 * np.log10(1 / np.mean((a - b) ** 2))

    failures = []
    bm3d = core.nss.BM3D
    for kw in ({}, dict(block_size=4), dict(block_size=16, group_size=16), dict(group_size=16, block_step=3)):
        separate, shared = bm3d(src, sigma=10, **kw), bm3d(src, sigma=10, chroma=1, **kw)
        final = bm3d(src, sigma=10, chroma=1, ref=shared, **kw)
        a, b, c = planes(separate, 2), planes(shared, 2), planes(final, 2)
        if not np.array_equal(a[0], b[0]):
            failures.append(f"{kw}: plane 0 differs from the separate filter")
        for p in (1, 2):
            if np.array_equal(a[p], b[p]):
                failures.append(f"{kw}: plane {p} is the separate filter's")
            before = psnr(noisy[2][p].astype(np.float64), clean[2][p])
            if psnr(b[p], clean[2][p]) < before + 3 or psnr(c[p], clean[2][p]) < before + 3:
                failures.append(f"{kw}: plane {p} is not denoised")
    both = planes(bm3d(src, sigma=10, chroma=1), 1)
    luma_kept = planes(bm3d(src, sigma=[0, 10, 10], chroma=1), 1)
    if not np.array_equal(luma_kept[0], noisy[1][0].astype(np.float64)):
        failures.append("a plane with sigma 0 is not copied")
    if not np.array_equal(luma_kept[1], both[1]) or not np.array_equal(luma_kept[2], both[2]):
        failures.append("planes 1 and 2 depend on sigma[0]")
    for radius in (1, 2):
        legacy = core.nss.VAggregate(bm3d(src, sigma=10, radius=radius, chroma=1), src, radius=radius)
        rolling = bm3d(src, sigma=10, radius=radius, chroma=1, temporal_mode="rolling", rolling_chunk=4)
        separate = core.nss.VAggregate(bm3d(src, sigma=10, radius=radius), src, radius=radius)
        for n in range(FRAMES):
            a, b = planes(legacy, n), planes(rolling, n)
            if any(not np.array_equal(u, v) for u, v in zip(a, b)):
                failures.append(f"radius {radius} frame {n}: rolling differs from legacy + VAggregate")
            if not np.array_equal(a[0], planes(separate, n)[0]):
                failures.append(f"radius {radius} frame {n}: plane 0 differs from the separate filter")
    for clip, kw, message in ((src, dict(chroma=2), "chroma must be 0 or 1"), (src, dict(chroma=-1), "chroma must be 0 or 1"),
                              *((core.std.BlankClip(format=fmt, width=64, height=64), dict(chroma=1),
                                 "chroma requires a YUV 4:4:4 clip") for fmt in (vs.GRAYS, vs.RGBS, vs.YUV420PS))):
        try:
            bm3d(clip, **kw)
            failures.append(f"{clip.format.name} {kw}: no error")
        except vs.Error as error:
            if message not in str(error):
                failures.append(f"{clip.format.name} {kw}: {error}")
    for failure in failures:
        print("FAIL:", failure)
    print(f"bm3d chroma: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
