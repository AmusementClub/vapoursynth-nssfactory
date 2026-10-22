#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""BM4D prototype against nss.BM3D (spatial and temporal) on video, one case per process.

A case is a segment of a video (luma, 8-bit units, centre crop) treated as
clean, plus seeded AWGN. Methods: nss.BM3D final=1 with radius 0/1/2
(rolling), each at sigma x 1 and x 4/3, and the BM4D prototype at sigma x 1
and x 4/3. Prints one JSON line with, per method, the PSNR over all frames,
over the interior frames (2 from each end) and over moving pixels (|clean
frame difference| > 4 to either neighbour), plus the segment's motion.

    NSS_SO=build/libnss.so python3 tests/bm4d_lab/compare.py VIDEO START FRAMES WIDTH HEIGHT SIGMA
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto  # noqa: E402


def load(path, start, frames, width, height):
    core = vs.core
    cache = "/tmp/bm4d-lab-" + hashlib.sha256(path.encode()).hexdigest()[:12] + ".lwi"
    src = core.lsmas.LWLibavSource(path, cachefile=cache)
    y = core.std.ShufflePlanes(src, 0, vs.GRAY)
    x0 = (y.width - width) // 2 & ~1
    y0 = (y.height - height) // 2 & ~1
    y = core.std.Crop(y, x0, y.width - width - x0, y0, y.height - height - y0)
    scale = 255.0 / ((1 << y.format.bits_per_sample) - 1)
    return np.stack([np.asarray(y.get_frame(start + t)[0], dtype=np.float64) * scale for t in range(frames)])


def psnr(a, b, mask=None):
    e = (a - b) ** 2
    e = e[mask] if mask is not None else e
    return float(10 * np.log10(255.0 ** 2 / np.mean(e)))


def run_bm3d(core, noisy, sigma, radius):
    T, H, W = noisy.shape
    blank = core.std.BlankClip(width=W, height=H, format=vs.GRAYS, length=T)

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:] = noisy[n] / 255.0
        return out
    clip = core.std.ModifyFrame(blank, blank, fill)
    kw = dict(sigma=sigma, final=1)
    if radius:
        kw.update(radius=radius, temporal_mode="rolling")
    out = core.nss.BM3D(clip, **kw)
    return np.stack([np.asarray(out.get_frame(t)[0], dtype=np.float64) * 255.0 for t in range(T)])


def main():
    path, start, frames, width, height, sigma = sys.argv[1], *map(int, sys.argv[2:6]), float(sys.argv[6])
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(os.environ["NSS_SO"])
    clean = load(path, start, frames, width, height)
    seed = int.from_bytes(hashlib.sha256(f"bm4d-lab:{path}:{start}:{sigma}".encode()).digest()[:8], "little")
    noisy = clean + np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape) * sigma
    d = np.abs(np.diff(clean, axis=0)) > 4
    moving = np.zeros(clean.shape, bool)
    moving[1:] |= d
    moving[:-1] |= d
    inner = slice(2, frames - 2)
    result = dict(video=os.path.basename(path), start=start, frames=frames, width=width, height=height, sigma=sigma,
                  motion=float(np.mean(np.abs(np.diff(clean, axis=0)))), moving_fraction=float(moving.mean()),
                  noisy=psnr(noisy, clean), methods={})
    outputs = {}
    for f in (1.0, 4 / 3):
        for r in (0, 1, 2):
            t0 = time.perf_counter()
            outputs[f"bm3d r{r} x{f:.2f}"] = (run_bm3d(core, noisy, sigma * f, r), time.perf_counter() - t0)
        t0 = time.perf_counter()
        final, basic = proto.bm4d(noisy, sigma * f)
        dt = time.perf_counter() - t0
        outputs[f"bm4d x{f:.2f}"] = (final, dt)
        outputs[f"bm4d-basic x{f:.2f}"] = (basic, dt)
    for name, (out, dt) in outputs.items():
        result["methods"][name] = dict(all=psnr(out, clean), inner=psnr(out[inner], clean[inner]),
                                       moving=psnr(out, clean, moving), seconds=dt)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
