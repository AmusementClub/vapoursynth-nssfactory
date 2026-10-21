#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.BM4D vs nss.BM4D: the device runs the CPU model operation for
operation (distances in the CPU's order, the same transforms, fixed-point
sums), so every frame must give the same bits: Gray over two chunks and two
tile columns, YUV420 with a bypassed plane, RGB, short clips and small
planes; plus frames asked for out of order. Exits 77 without VapourSynth or a
CUDA device.

usage: test_cuda_bm4d_live.py --cpu PATH --cuda PATH
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda bm4d live SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, length, sigma, seed):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.4 + 0.25 * ((((x + t) // 9) + y // 7) % 2) + 0.05 * np.sin((x + 2 * t) / (4 + p))
            if blank.format.color_family == vs.YUV and p:
                clean = clean - 0.5
            planes.append((clean + rng.normal(0, sigma / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--cuda", required=True)
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 8
    core.std.LoadPlugin(args.cpu)
    try:
        core.std.LoadPlugin(args.cuda)
        if core.nss_cuda.Backend().get("device_count", 0) == 0:
            raise RuntimeError("no device")
    except Exception as error:  # noqa: BLE001
        print(f"cuda bm4d live SKIP: {error}")
        return 77

    cases = [
        (vs.GRAYS, 400, 72, 40, 10, None),        # two chunks, two tile columns
        (vs.YUV420PS, 96, 64, 9, 12, [12, 0, 6]),
        (vs.RGBS, 64, 48, 6, 8, None),
        (vs.GRAYS, 20, 18, 3, 15, None),          # shorter than the Wiener cube
        (vs.GRAYS, 7, 9, 5, 15, None),
    ]
    failures = 0
    for i, (fmt, w, h, t, sigma, sig) in enumerate(cases):
        clip = make_clip(core, fmt, w, h, t, sigma, 200 + i)
        kw = dict(sigma=sig if sig is not None else sigma)
        cpu = [planes(f) for f in core.nss.BM4D(clip, **kw).frames()]
        gnode = core.nss_cuda.BM4D(clip, **kw)
        order = list(reversed(range(t)))  # out of order: the last chunk first
        gpu = {n: planes(gnode.get_frame(n)) for n in order}
        bad = [n for n in range(t) if not all(np.array_equal(a, b) for a, b in zip(cpu[n], gpu[n]))]
        failures += bool(bad)
        print(f"{vs.core.get_video_format(fmt).name} {w}x{h}x{t} sigma {kw['sigma']}: "
              f"{'same bits' if not bad else f'DIFFERENT in frames {bad[:5]}'}")
    print("test_cuda_bm4d_live:", "ok" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
