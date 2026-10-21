#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss.BM4D through VapourSynth: denoising, random access and concurrency
giving the same bits as a sequential run, chunk geometry, sigma=0, short
clips, YUV420 planes and the memory limit.

usage: test_bm4d_plugin.py --plugin PATH
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"BM4D plugin SKIP: {error}")
    raise SystemExit(77)

failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)
        print(f"FAIL: {what}")


def clip_of(core, frames, fmt):
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=len(frames))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def video(t, h, w, sigma, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = [(0.35 + 0.3 * ((((x + f) // 9) + (y // 7)) % 2) + 0.05 * np.sin(0.2 * (x + f))).astype(np.float32)
             for f in range(t)]
    noisy = [(c + rng.standard_normal(c.shape) * sigma / 255).astype(np.float32) for c in clean]
    return clean, noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))

    sigma = 12.0
    t, h, w = 40, 48, 64  # two chunks
    clean, noisy = video(t, h, w, sigma, 1)
    clip = clip_of(core, [[f] for f in noisy], vs.GRAYS)

    serial = [planes(core.nss.BM4D(clip, sigma=sigma).get_frame(n))[0] for n in range(t)]
    mse_out = np.mean([(serial[n] - clean[n]) ** 2 for n in range(t)])
    mse_in = np.mean([(noisy[n] - clean[n]) ** 2 for n in range(t)])
    print(f"gray: noisy MSE {mse_in:.3g}, BM4D {mse_out:.3g}")
    expect(mse_out < 0.1 * mse_in, "BM4D denoises")

    node = core.nss.BM4D(clip, sigma=sigma)
    order = [35, 0, 33, 31, 39, 1] + list(range(t))
    random_access = {n: planes(node.get_frame(n))[0] for n in order}
    expect(all(np.array_equal(random_access[n], serial[n]) for n in range(t)), "random access equals sequential")
    node = core.nss.BM4D(clip, sigma=sigma)
    futures = [node.get_frame_async(n) for n in reversed(range(t))]
    concurrent = [planes(f.result())[0] for f in reversed(futures)]
    expect(all(np.array_equal(a, b) for a, b in zip(concurrent, serial)), "concurrent frames equal sequential ones")

    # A chunk is its frames plus 8 on each side: frames 0..31 do not see frames 40 and later.
    longer_clean, longer = video(56, h, w, sigma, 1)
    expect(all(np.array_equal(longer[n], noisy[n]) for n in range(t)), "the longer clip extends the same video")
    first = planes(core.nss.BM4D(clip_of(core, [[f] for f in longer], vs.GRAYS), sigma=sigma).get_frame(5))[0]
    expect(np.array_equal(first, serial[5]), "frames past the chunk's padding do not change it")

    # YUV420: planes on their own; sigma=0 passes the plane through.
    yuv = [[f, (f[::2, ::2] - 0.35), (0.4 - f[::2, ::2])] for f in noisy]
    out = planes(core.nss.BM4D(clip_of(core, yuv, vs.YUV420PS), sigma=[sigma, 0, sigma]).get_frame(3))
    expect(np.array_equal(out[0], serial[3]), "YUV420 luma equals the gray result")
    expect(np.array_equal(out[1], yuv[3][1]), "sigma=0 passes the plane through")
    expect(not np.array_equal(out[2], yuv[3][2]), "a filtered chroma plane changes")

    # Short clips and small planes still filter (stages that do not fit are skipped).
    for frames, hh, ww in ((1, 16, 16), (3, 40, 40), (5, 4, 7)):
        _, nz = video(frames, hh, ww, sigma, 2)
        o = planes(core.nss.BM4D(clip_of(core, [[f] for f in nz], vs.GRAYS), sigma=sigma).get_frame(frames - 1))[0]
        expect(bool(np.isfinite(o).all()), f"a {frames}x{hh}x{ww} clip gives finite output")

    try:
        core.nss.BM4D(clip, sigma=sigma, memory_limit_mb=1).get_frame(0)
        expect(False, "memory_limit_mb=1 fails")
    except vs.Error as error:
        expect(any(k in str(error).lower() for k in ("resource", "memory", "budget")),
               f"memory_limit_mb=1 fails with a resource error ({error})")

    if failures:
        print(f"test_bm4d_plugin: {len(failures)} failure(s)")
        return 1
    print("test_bm4d_plugin: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
