#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.TWSC vs nss.TWSC on small random clips: given and blind sigma,
Gray / YUV420 / YUV444 / RGB (joint matching over the noisy planes of equal
geometry), equal and unequal row weights, small and large groups (the serial
and the block solver, Gram of either side), rounds with delta, the ADMM
schedule, bypassed planes, rclip and the temporal radius (legacy +
VAggregate, including clip boundary frames). Each case must reach the TWSC
tolerance (tests/data/cuda_tolerances_v1.json) and be run-to-run identical.
Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_twsc_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda twsc live SKIP: {error}")
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


PROPS = ("_NSSBlockSize", "_NSSGroupSize", "_NSSIterations", "_NSSSearchWindow", "_NSSBlockStep", "_NSSGroups")


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
        print("cuda twsc live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["TWSC"]["min_psnr"]

    gray = make_clip(core, vs.GRAYS, 37, 33, 1, length=3)
    yuv420 = make_clip(core, vs.YUV420PS, 40, 36, 2, length=3)
    yuv444 = make_clip(core, vs.YUV444PS, 33, 29, 3, length=3)
    rgb = make_clip(core, vs.RGBS, 33, 29, 4, length=3)
    guide = make_clip(core, vs.GRAYS, 37, 33, 9, length=3)
    light = dict(block_step=4, group_size=8, iters=2, search_window=13)
    cases = [
        ("gray light", gray, dict(sigma=5, **light)),
        ("gray blind", gray, dict(estimate_sigma=1, **light)),
        ("gray sigma25 delta", gray, dict(sigma=25, delta=0.2, **dict(light, iters=3))),
        ("gray admm", gray, dict(sigma=5, admm_iter=4, rho=1.0, mu=1.5, tol=1e-3, lambda2=0.7, **light)),
        ("gray g1", gray, dict(sigma=5, **dict(light, group_size=1))),
        ("gray b4 g24", gray, dict(sigma=5, block_size=4, **dict(light, group_size=24, block_step=3))),
        ("gray large group", gray, dict(sigma=5, block_step=5, group_size=90, iters=2, search_window=21)),
        ("gray b6 g40", gray, dict(sigma=5, block_size=6, block_step=5, group_size=40, iters=2, search_window=17)),
        ("gray sigma0", gray, dict(sigma=0, **light)),
        ("gray rclip", gray, dict(sigma=5, rclip=guide, **light)),
        ("yuv420", yuv420, dict(sigma=[5, 3, 3], **light)),
        ("yuv420 chroma off", yuv420, dict(sigma=[5, 0, 0], **light)),
        ("yuv444 unequal", yuv444, dict(sigma=[5, 3, 2], **light)),
        ("yuv444 one off", yuv444, dict(sigma=[5, 0, 3], **light)),
        ("rgb equal", rgb, dict(sigma=5, **light)),
        ("rgb unequal large group", rgb, dict(sigma=[5, 4, 3], block_size=4, block_step=4, group_size=60, iters=2,
                                              search_window=17)),
        ("rgb blind", rgb, dict(estimate_sigma=1, **light)),
    ]
    temporal = [
        ("gray r1", gray, dict(sigma=5, radius=1, **light)),
        ("rgb r1 unequal", rgb, dict(sigma=[5, 4, 3], radius=1, **light)),
        ("gray r2 large group", gray, dict(sigma=5, radius=2, ps_num=3, block_step=6, group_size=90, iters=2,
                                           search_window=13)),
    ]
    if args.quick:
        quick = dict(light, iters=1)
        cases = [("gray quick", gray, dict(sigma=5, **quick)),
                 ("yuv444 quick unequal", yuv444, dict(sigma=[5, 3, 2], **quick)),
                 ("gray quick large group", gray, dict(sigma=5, block_step=8, group_size=90, iters=2, search_window=13)),
                 ("rgb quick unequal large group", rgb, dict(sigma=[5, 4, 3], block_size=4, block_step=4, group_size=60,
                                                             iters=1, search_window=11))]
        temporal = [("gray quick r1", gray, dict(sigma=5, radius=1, **quick))]
    failures, worst, count = [], float("inf"), 0

    def check(label, clip, cpu, gpu, again, props_cpu=None, props_gpu=None):
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
            if props_cpu is not None:
                pa, pb = props_cpu.get_frame(n).props, props_gpu.get_frame(n).props
                for key in PROPS:
                    if pa.get(key) != pb.get(key):
                        failures.append(f"{label} frame {n}: {key} {pb.get(key)} != {pa.get(key)}")
                sa, sb = np.atleast_1d(pa["_NSSSigma"]), np.atleast_1d(pb["_NSSSigma"])
                if sa.shape != sb.shape or not np.allclose(sa, sb, rtol=1e-4, atol=1e-6):
                    failures.append(f"{label} frame {n}: _NSSSigma {sb} != {sa}")

    for label, clip, kw in cases:
        cpu, gpu = core.nss.TWSC(clip, **kw), core.nss_cuda.TWSC(clip, **kw)
        check(label, clip, cpu, gpu, core.nss_cuda.TWSC(clip, **kw), cpu, gpu)
    for label, clip, kw in temporal:
        r = kw["radius"]
        check(label, clip, core.nss.VAggregate(core.nss.TWSC(clip, **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.TWSC(clip, temporal_mode="legacy", **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.TWSC(clip, temporal_mode="legacy", **kw), clip, radius=r))
    # Repeated create/free must not leak device memory or fail.
    for _ in range(2 if args.quick else 10):
        core.nss_cuda.TWSC(gray, sigma=5, **light).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda twsc live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
