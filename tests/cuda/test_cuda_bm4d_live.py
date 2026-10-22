#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.BM4D vs nss.BM4D: the device runs the CPU model's transforms and
fixed-point sums, but on sm_80+ matches by FP16 tensor-core distances
(2026-10-22), so near ties may take other cubes: every clip must be within
MIN_PSNR dB of the CPU's output, its PSNR to the clean clip within MAX_DPSNR
dB of the CPU's, and a second node must give the same bits. A clip FP16
cannot hold takes the exact matching (match_kernel) and must give the CPU's
bits. Gray over two chunks and two tile columns, YUV420 with a bypassed
plane, RGB, short clips and small planes, a flat clip; frames asked for out
of order. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_bm4d_live.py --cpu PATH --cuda PATH
"""
import argparse
import sys

MIN_PSNR = 50.0      # dB between the CPU's and the device's output
MAX_DPSNR = 0.05     # dB between their PSNRs to the clean clip

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda bm4d live SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, length, sigma, seed, kind="pattern"):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames, cleans = [], []
    for t in range(length):
        planes, cp = [], []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.4 + 0.25 * ((((x + t) // 9) + y // 7) % 2) + 0.05 * np.sin((x + 2 * t) / (4 + p))
            if kind == "flat":  # two flat halves far from the mean, faint noise: near-equal distances, wide bounds
                clean = np.where(x < w // 2, 0.0, 1.0)
            if blank.format.color_family == vs.YUV and p:
                clean = clean - 0.5
            cp.append(clean.astype(np.float32) * (2e5 if kind == "huge" else 1))
            noisy = clean + rng.normal(0, (0.3 if kind == "flat" else sigma) / 255, clean.shape)
            if kind == "huge":  # beyond FP16 after centring
                noisy = noisy * 2e5
            planes.append(noisy.astype(np.float32))
        frames.append(planes)
        cleans.append(cp)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill), cleans


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def psnr(frames_a, frames_b, peak=1.0):
    mse = np.mean([np.mean((x.astype(np.float64) - y) ** 2) for fa, fb in zip(frames_a, frames_b) for x, y in zip(fa, fb)])
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


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
        (vs.GRAYS, 400, 72, 40, 10, None, "pattern"),   # two chunks, two tile columns
        (vs.YUV420PS, 96, 64, 9, 12, [12, 0, 6], "pattern"),
        (vs.RGBS, 64, 48, 6, 8, None, "pattern"),
        (vs.GRAYS, 20, 18, 3, 15, None, "pattern"),     # shorter than the Wiener cube
        (vs.GRAYS, 7, 9, 5, 15, None, "pattern"),
        (vs.GRAYS, 64, 40, 12, 10, None, "flat"),       # equal distances: survivors overflow
        (vs.GRAYS, 48, 40, 10, 10, None, "huge"),       # beyond FP16
    ]
    failures = 0
    for i, (fmt, w, h, t, sigma, sig, kind) in enumerate(cases):
        clip, cleans = make_clip(core, fmt, w, h, t, sigma, 200 + i, kind)
        kw = dict(sigma=sig if sig is not None else sigma)
        cpu = [planes(f) for f in core.nss.BM4D(clip, **kw).frames()]
        gnode = core.nss_cuda.BM4D(clip, **kw)
        order = list(reversed(range(t)))  # out of order: the last chunk first
        gpu = {n: planes(gnode.get_frame(n)) for n in order}
        gpu = [gpu[n] for n in range(t)]
        again = [planes(f) for f in core.nss_cuda.BM4D(clip, **kw).frames()]
        repeatable = all(np.array_equal(a, b) for fa, fb in zip(gpu, again) for a, b in zip(fa, fb))
        same = all(np.array_equal(a, b) for fa, fb in zip(cpu, gpu) for a, b in zip(fa, fb))
        peak = 2e5 if kind == "huge" else 1.0
        between = psnr(cpu, gpu, peak)
        pg, pc = psnr(gpu, cleans, peak), psnr(cpu, cleans, peak)
        dpsnr = pg - pc  # the device's PSNR to the clean clip less the CPU's
        ok = repeatable and (same if kind == "huge" else between >= MIN_PSNR and dpsnr >= -MAX_DPSNR)
        failures += not ok
        print(f"{vs.core.get_video_format(fmt).name} {w}x{h}x{t} {kind} sigma {kw['sigma']}: "
              f"{'same bits' if same else f'{between:.1f} dB from the CPU'}, PSNR to clean {pg:.2f} ({dpsnr:+.4f} dB), "
              f"{'repeatable' if repeatable else 'NOT REPEATABLE'}{'' if ok else '  FAIL'}")
    print("test_cuda_bm4d_live:", "ok" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
