#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.MCWNNM vs nss.MCWNNM on small random RGB / YUV444 clips across
block and group sizes (every group kernel capacity), residual centering,
adaptive aggregation, the ADMM schedule, outer rounds, per-channel sigma
(including bypassed channels), rclip and the temporal radius (legacy +
VAggregate, including clip boundary frames). Each case must reach the MCWNNM
tolerance (tests/data/cuda_tolerances_v1.json) and be run-to-run identical.
Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_mcwnnm_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda mcwnnm live SKIP: {error}")
    raise SystemExit(77)

REPO = Path(__file__).resolve().parents[2]


def make_clip(core, fmt, width, height, seed, length=4):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.5 + 0.2 * np.sin((x + 2 * t) / (5 + p)) * np.cos((y + t) / 7)
            if blank.format.color_family == vs.YUV and p:
                clean = clean - 0.5
            planes.append((clean + rng.normal(0, 5 / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def frame_planes(node, n):
    f = node.get_frame(n)
    return [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]


def psnr(a, b):
    mse = sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b)) / sum(x.size for x in a)
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.cpu).resolve()))
    core.std.LoadPlugin(path=str(Path(args.cuda).resolve()))
    if core.nss_cuda.Backend()["device_count"] == 0:
        print("cuda mcwnnm live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["MCWNNM"]["min_psnr"]

    rgb = make_clip(core, vs.RGBS, 41, 37, 4)
    yuv = make_clip(core, vs.YUV444PS, 44, 36, 5)
    guide = make_clip(core, vs.RGBS, 41, 37, 9)
    small = dict(bm_range=5)
    cases = [
        ("rgb default", rgb, dict(sigma=5, **small)),
        ("yuv444 default", yuv, dict(sigma=5, **small)),
        ("rgb direct", rgb, dict(sigma=5, residual=False, **small)),
        ("rgb adaptive", rgb, dict(sigma=5, adaptive_aggregation=True, **small)),
        ("rgb per-channel sigma", rgb, dict(sigma=[4, 8, 6], **small)),
        ("rgb bypassed channel", rgb, dict(sigma=[5, 0, 3], **small)),
        ("rgb all bypassed", rgb, dict(sigma=0, **small)),
        ("rgb iters1", rgb, dict(sigma=5, iters=1, **small)),
        ("rgb iters3 delta0.3", rgb, dict(sigma=5, iters=3, delta=0.3, **small)),
        ("rgb admm3 rho1 mu1.5", rgb, dict(sigma=5, admm_iter=3, rho=1.0, mu=1.5, **small)),
        ("rgb g1", rgb, dict(sigma=5, group_size=1, ps_num=1, **small)),
        ("rgb g3 b4", rgb, dict(sigma=5, group_size=3, block_size=4, block_step=4, **small)),
        ("rgb g16 step4", rgb, dict(sigma=8, group_size=16, block_step=4, **small)),
        ("rgb g32 b9", rgb, dict(sigma=8, group_size=32, block_size=9, block_step=5, **small)),
        ("rgb b2 g32", rgb, dict(sigma=8, group_size=32, block_size=2, block_step=2, **small)),
        ("rgb rclip", rgb, dict(sigma=5, rclip=guide, **small)),
    ]
    temporal = [
        ("rgb r1", rgb, dict(sigma=5, radius=1, **small)),
        ("yuv444 r2 g16", yuv, dict(sigma=5, radius=2, group_size=16, block_step=4, **small)),
    ]
    if args.quick:
        # Every kernel capacity with a short ADMM schedule (sanitizer runs).
        light = dict(admm_iter=2, **small)
        cases = [
            ("rgb quick", rgb, dict(sigma=5, **light)),
            ("rgb quick bypassed channel", rgb, dict(sigma=[5, 0, 3], **light)),
            ("rgb quick g16", rgb, dict(sigma=8, group_size=16, **light)),
            ("rgb quick g32 b4", rgb, dict(sigma=8, group_size=32, block_size=4, block_step=4, **light)),
        ]
        temporal = [("rgb quick r1", rgb, dict(sigma=5, radius=1, **light))]
    failures, worst, count = [], float("inf"), 0

    def check(label, clip, cpu, gpu, again):
        nonlocal worst, count
        for n in range(clip.num_frames):  # includes both clip boundaries
            a, b = frame_planes(cpu, n), frame_planes(gpu, n)
            value = psnr(a, b)
            worst = min(worst, value)
            count += 1
            if value < floor:
                failures.append(f"{label} frame {n}: psnr {value:.2f} < {floor}")
            if any(not np.array_equal(x, y) for x, y in zip(b, frame_planes(again, n))):
                failures.append(f"{label} frame {n}: not run-to-run identical")

    for label, clip, kw in cases:
        check(label, clip, core.nss.MCWNNM(clip, **kw), core.nss_cuda.MCWNNM(clip, **kw),
              core.nss_cuda.MCWNNM(clip, **kw))
    for label, clip, kw in temporal:
        r = kw["radius"]
        check(label, clip, core.nss.VAggregate(core.nss.MCWNNM(clip, **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.MCWNNM(clip, **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.MCWNNM(clip, **kw), clip, radius=r))
    # Repeated create/free must not leak device memory or fail.
    for _ in range(2 if args.quick else 20):
        core.nss_cuda.MCWNNM(rgb, sigma=5).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda mcwnnm live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
