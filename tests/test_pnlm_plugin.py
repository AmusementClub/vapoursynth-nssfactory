#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss.PNLM through VapourSynth: denoising, the temporal window (distinct
frames n - d .. n + d, fewer at the ends of the clip), plane handling and
sigma=0, determinism under concurrent requests, argument errors and a memory
limit that holds.

usage: test_pnlm_plugin.py --plugin PATH
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"PNLM plugin SKIP: {error}")
    raise SystemExit(77)

failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)
        print(f"FAIL: {what}")


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def scene(w, h, n_planes, shift=0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = x + shift
    clean = []
    for p in range(n_planes):
        v = 0.3 + 0.05 * p + 0.3 * (((x // 19) + (y // 13)) % 2) + 0.1 * np.sin(0.07 * x + 0.05 * y * (p + 1))
        v[(x - 50) ** 2 + (y - 40) ** 2 < 300] = 0.85 - 0.1 * p
        clean.append(v.astype(np.float32))
    return clean


def clip_of(core, frames, fmt):
    """A clip whose frame n holds frames[n] (a list of planes)."""
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=len(frames))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))

    sigma = 8.0
    w, h, length = 150, 110, 5
    rng = np.random.default_rng(1)
    clean = [scene(w, h, 3, shift=t) for t in range(length)]
    noisy = [[(c + rng.standard_normal(c.shape) * sigma / 255).astype(np.float32) for c in f] for f in clean]
    gray = clip_of(core, [[f[0]] for f in noisy], vs.GRAYS)

    # Denoises; the neighbouring frames help (default d = 1).
    spatial = planes(core.nss.PNLM(gray, sigma=sigma, d=0).get_frame(2))[0]
    temporal = planes(core.nss.PNLM(gray, sigma=sigma).get_frame(2))[0]
    m_noisy, m_spatial, m_temporal = mse(noisy[2][0], clean[2][0]), mse(spatial, clean[2][0]), mse(temporal, clean[2][0])
    print(f"gray: noisy MSE {m_noisy:.3g}, d=0 {m_spatial:.3g}, d=1 {m_temporal:.3g}")
    expect(m_spatial < 0.5 * m_noisy, "PNLM denoises")
    expect(m_temporal < m_spatial, "d=1 improves on d=0 for a slowly moving scene")

    # The window: d = 0 reads only frame n; frame 0 with d = 1 reads frames 0 and 1 only.
    flipped = clip_of(core, [[1 - f[0]] if t != 2 else [f[0]] for t, f in enumerate(noisy)], vs.GRAYS)
    expect(np.array_equal(planes(core.nss.PNLM(flipped, sigma=sigma, d=0).get_frame(2))[0], spatial),
           "d=0 reads only the frame itself")
    far = clip_of(core, [[f[0]] if t < 2 else [1 - f[0]] for t, f in enumerate(noisy)], vs.GRAYS)
    expect(np.array_equal(planes(core.nss.PNLM(far, sigma=sigma).get_frame(0))[0],
                          planes(core.nss.PNLM(gray, sigma=sigma).get_frame(0))[0]),
           "frame 0 with d=1 reads frames 0 and 1 only")
    short = clip_of(core, [[noisy[0][0]]], vs.GRAYS)
    expect(np.array_equal(planes(core.nss.PNLM(short, sigma=sigma, d=3).get_frame(0))[0],
                          planes(core.nss.PNLM(short, sigma=sigma, d=0).get_frame(0))[0]),
           "a one-frame clip is filtered alone whatever d")

    # Determinism: frames requested concurrently and out of order equal a serial run.
    node = core.nss.PNLM(gray, sigma=sigma, a=3)
    serial = [planes(node.get_frame(n))[0] for n in range(length)]
    again = core.nss.PNLM(gray, sigma=sigma, a=3)
    futures = [again.get_frame_async(n) for n in reversed(range(length))]
    concurrent = [planes(fut.result())[0] for fut in reversed(futures)]
    expect(all(np.array_equal(a, b) for a, b in zip(serial, concurrent)), "concurrent frames equal serial ones")

    # YUV420: each plane on its own; luma equals the gray result; sigma=0 copies the plane.
    yuv_frames = [[f[0], f[1][::2, ::2] - 0.35, f[2][::2, ::2] - 0.4] for f in noisy]
    yuv = planes(core.nss.PNLM(clip_of(core, yuv_frames, vs.YUV420PS), sigma=[sigma, 0, sigma]).get_frame(2))
    expect(np.array_equal(yuv[0], temporal), "YUV420 luma equals the gray result")
    expect(np.array_equal(yuv[1], yuv_frames[2][1]), "sigma=0 passes the plane through")
    expect(not np.array_equal(yuv[2], yuv_frames[2][2]), "a filtered chroma plane changes")
    rgb = planes(core.nss.PNLM(clip_of(core, noisy, vs.RGBS), sigma=sigma).get_frame(2))
    expect(np.array_equal(rgb[0], temporal), "RGB planes are filtered on their own")
    for block in (4, 6, 8):
        out = planes(core.nss.PNLM(gray, sigma=sigma, block_size=block, group_size=24, a=5).get_frame(1))[0]
        expect(mse(out, clean[1][0]) < 0.5 * m_noisy, f"block_size={block} denoises")

    # Argument errors at creation.
    bad = [dict(block_size=5), dict(block_size=10), dict(group_size=1), dict(group_size=33), dict(d=-1), dict(d=32),
           dict(a=0), dict(a=65), dict(sigma=-1.0), dict(sigma=float("nan"))]
    for kw in bad:
        try:
            core.nss.PNLM(gray, **kw)
            expect(False, f"{kw} is rejected")
        except vs.Error as error:
            expect(str(error).startswith("nss.PNLM:") or str(error).startswith("nss:"), f"{kw}: {error}")
    tiny = core.std.BlankClip(width=5, height=40, format=vs.GRAYS, length=1)
    try:
        core.nss.PNLM(tiny, block_size=6)
        expect(False, "a plane smaller than block_size is rejected")
    except vs.Error:
        pass
    expect(planes(core.nss.PNLM(tiny, block_size=4).get_frame(0))[0].shape == (40, 5),
           "a 5-wide plane is filtered with block_size=4")

    # The workspace (four grid rows of estimates) fits a small limit; a limit
    # never changes the output.
    limited = planes(core.nss.PNLM(gray, sigma=sigma, memory_limit_mb=1).get_frame(2))[0]
    expect(np.array_equal(limited, temporal), "memory_limit_mb=1 holds and leaves the output unchanged")

    if failures:
        print(f"test_pnlm_plugin: {len(failures)} failure(s)")
        return 1
    print("test_pnlm_plugin: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
