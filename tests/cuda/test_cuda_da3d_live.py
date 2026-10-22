#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.DA3D vs nss.DA3D: the device runs the CPU model with tree sums
over a patch instead of the CPU's 32-lane chains (2026-10-22), so its output
is close to the CPU's, not the same bits: every frame must be within
MIN_PSNR dB of the CPU's, its PSNR to the clean frame within MAX_DPSNR dB of
the CPU's, and its patch count within MAX_DPATCH of the CPU's; and a second
run with two streams must give the same bits. Gray, YUV420 with a bypassed
plane, YUV444, RGB joint and RGB plane by plane, small and odd shapes,
several noise levels. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_da3d_live.py --cpu PATH --cuda PATH
"""
import argparse
import sys

MIN_PSNR = 100.0     # dB between the CPU's and the device's output
MAX_DPSNR = 0.01     # dB between their PSNRs to the clean frame
MAX_DPATCH = 0.01    # relative difference of the patch counts

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda da3d live SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, seed, sigma, length=2):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    noisy, guide, cleans = [], [], []
    for t in range(length):
        np_, gp, cp = [], [], []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.45 + 0.25 * (((x + 3 * t) // 11 + y // 9) % 2) + 0.1 * np.sin((x + 2 * t) / (5 + p)) * np.cos(y / 7)
            if blank.format.color_family == vs.YUV and p:
                clean = clean - 0.5
            cp.append(clean.astype(np.float32))
            np_.append((clean + rng.normal(0, sigma / 255, clean.shape)).astype(np.float32))
            gp.append((clean + rng.normal(0, 0.4 * sigma / 255, clean.shape)).astype(np.float32))
        noisy.append(np_)
        guide.append(gp)
        cleans.append(cp)

    def clip_of(frames):
        def fill(n, f):
            out = f.copy()
            for p, plane in enumerate(frames[n]):
                np.asarray(out[p])[:] = plane
            return out
        return core.std.ModifyFrame(blank, blank, fill)
    return clip_of(noisy), clip_of(guide), cleans


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def psnr(a, b):
    mse = np.mean([np.mean((x.astype(np.float64) - y) ** 2) for x, y in zip(a, b)])
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


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
        print(f"cuda da3d live SKIP: {error}")
        return 77

    cases = [
        (vs.GRAYS, 150, 110, 10, None),
        (vs.GRAYS, 333, 201, 3, None),      # two tile rows and columns, low noise
        (vs.GRAYS, 7, 5, 20, None),         # smaller than a patch
        (vs.YUV420PS, 260, 196, 12, [12, 0, 8]),
        (vs.YUV444PS, 96, 80, 15, None),
        (vs.RGBS, 140, 130, 10, None),      # joint
        (vs.RGBS, 140, 130, 10, [10, 10, 9]),  # plane by plane
    ]
    failures = 0
    for i, (fmt, w, h, sigma, sig) in enumerate(cases):
        noisy, guide, cleans = make_clip(core, fmt, w, h, 100 + i, sigma)
        kw = dict(sigma=sig if sig is not None else sigma)
        cpu = core.nss.DA3D(noisy, guide, **kw)
        gpu = core.nss_cuda.DA3D(noisy, guide, **kw)
        gpu2 = core.nss_cuda.DA3D(noisy, guide, num_streams=2, **kw)
        for n in range(noisy.num_frames):
            fc, fg, fg2 = cpu.get_frame(n), gpu.get_frame(n), gpu2.get_frame(n)
            pc, pg = planes(fc), planes(fg)
            between = psnr(pc, pg)
            dpsnr = abs(psnr(pg, cleans[n]) - psnr(pc, cleans[n]))
            again = all(np.array_equal(a, b) for a, b in zip(pg, planes(fg2)))
            nc, ng = fc.props["_NSSDA3DPatches"], fg.props["_NSSDA3DPatches"]
            dpatch = abs(ng - nc) / max(nc, 1)
            ok = between >= MIN_PSNR and dpsnr <= MAX_DPSNR and dpatch <= MAX_DPATCH and again
            failures += not ok
            print(f"{vs.core.get_video_format(fmt).name} {w}x{h} sigma {kw['sigma']} frame {n}: "
                  f"{between:.1f} dB from the CPU, PSNR to clean {dpsnr:.5f} dB apart, "
                  f"patches {ng} / {nc}, {'repeatable' if again else 'NOT REPEATABLE'}{'' if ok else '  FAIL'}")
    print("test_cuda_da3d_live:", "ok" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
