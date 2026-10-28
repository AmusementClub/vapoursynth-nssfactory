#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.PNLM vs nss.PNLM: the device runs the CPU model operation for
operation (include/nss/params/pnlm.hpp), so every frame must have the same
bits, with one stream and with three, frames requested in order and out of
order. Gray, YUV420 with a bypassed plane, RGB; odd and tiny sizes
(clamped grid edges); patches of 4, 6 and 8; windows on the fused and the key path; groups
of 2 to 32; d of 0 to 3 at the ends of the clip and inside it. Exits 77
without VapourSynth or a CUDA device.

usage: test_cuda_pnlm_live.py --cpu PATH --cuda PATH
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda pnlm live SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, seed, sigma, length):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.45 + 0.25 * (((x + 2 * t) // 11 + y // 9) % 2) + 0.1 * np.sin((x + t) / (5 + p)) * np.cos(y / 7)
            planes.append((clean + rng.normal(0, sigma / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def bits(frame):
    return [np.array(frame[p], dtype=np.float32).view(np.uint32) for p in range(frame.format.num_planes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--cuda", required=True)
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(args.cpu)
    try:
        core.std.LoadPlugin(args.cuda)
        if core.nss_cuda.Backend().get("device_count", 0) == 0:
            raise RuntimeError("no device")
    except Exception as error:  # noqa: BLE001
        print(f"cuda pnlm live SKIP: {error}")
        return 77

    clips = [
        (vs.GRAYS, 333, 201, 6.0, None),
        (vs.GRAYS, 256, 128, 2.0, None),
        (vs.YUV420PS, 262, 146, 8.0, [8.0, 0.0, 5.0]),
        (vs.RGBS, 141, 97, 12.0, None),
        (vs.GRAYS, 9, 8, 10.0, None),       # a grid of two references a row, the last clamped
        (vs.GRAYS, 8, 23, 10.0, None),      # one reference column at block_size 8
    ]
    settings = [
        dict(),                                            # defaults: block 6, a 2, group 16, d 1 (fused)
        dict(d=0, block_size=4),
        dict(d=0, block_size=8, a=1, group_size=2),
        dict(d=2, block_size=4, a=4, group_size=8),
        dict(d=1, block_size=6, a=5, group_size=16),       # key path
        dict(d=3, block_size=8, a=7, group_size=32),
        dict(d=0, block_size=6, a=3, group_size=24),
    ]
    failures = 0
    length = 6
    for i, (fmt, w, h, sigma, sig) in enumerate(clips):
        src = make_clip(core, fmt, w, h, 200 + i, sigma, length)
        for kw in settings:
            kw = dict(kw, sigma=sig if sig is not None else sigma)
            cpu = core.nss.PNLM(src, **kw)
            one = core.nss_cuda.PNLM(src, num_streams=1, **kw)
            three = core.nss_cuda.PNLM(src, num_streams=3, **kw)
            futures = [three.get_frame_async(n) for n in reversed(range(length))]
            out_of_order = [f.result() for f in reversed(futures)]
            same = True
            for n in range(length):
                want = bits(cpu.get_frame(n))
                for got in (bits(one.get_frame(n)), bits(out_of_order[n])):
                    same = same and all(np.array_equal(a, b) for a, b in zip(want, got))
            failures += not same
            print(f"{vs.core.get_video_format(fmt).name} {w}x{h} {kw}: "
                  f"{'same bits' if same else 'DIFFERENT BITS  FAIL'}", flush=True)

    # A limit too small for the clip is a creation error.
    src = make_clip(core, vs.GRAYS, 320, 240, 9, 5.0, 2)
    try:
        core.nss_cuda.PNLM(src, memory_limit_mb=1)
        print("memory_limit_mb=1 accepted  FAIL")
        failures += 1
    except vs.Error as error:
        print(f"memory_limit_mb=1: {error}")
    print("test_cuda_pnlm_live:", "ok" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
