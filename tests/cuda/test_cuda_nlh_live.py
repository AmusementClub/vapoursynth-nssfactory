#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.NLH vs nss.NLH on small random clips: given and blind sigma,
Gray / YUV420 / YUV444 / RGB (shared luma-guided matching, area-averaged
guide for subsampled chroma), both noise models, stage shapes (block, step,
group, q, window), iteration counts, bypassed planes, rclip and the temporal
radius (legacy + VAggregate, including clip boundary frames). Each case must
reach the NLH tolerance (tests/data/cuda_tolerances_v1.json), be run-to-run
identical and carry the CPU's diagnostic frame properties. Exits 77 without
VapourSynth or a CUDA device.

usage: test_cuda_nlh_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda nlh live SKIP: {error}")
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


PROPS = ("_NSSBlockSize", "_NSSGroupSize", "_NSSIterations", "_NSSSearchWindow", "_NSSBlockStep", "_NSSQ",
         "_NSSGroups", "_NSSLambdaBasic", "_NSSHardStrength", "_NSSHardCoefficient", "_NSSWienerSigmaScale")


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
        print("cuda nlh live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["NLH"]["min_psnr"]

    gray = make_clip(core, vs.GRAYS, 45, 39, 1, length=3)
    yuv420 = make_clip(core, vs.YUV420PS, 48, 40, 2, length=3)
    yuv444 = make_clip(core, vs.YUV444PS, 41, 37, 3, length=3)
    rgb = make_clip(core, vs.RGBS, 41, 37, 4, length=3)
    guide = make_clip(core, vs.GRAYS, 45, 39, 9, length=3)
    rgb_guide = make_clip(core, vs.RGBS, 41, 37, 8, length=3)
    small = dict(search_window=11)
    cases = [
        ("gray sigma5", gray, dict(sigma=5, **small)),
        ("gray blind", gray, dict(**small)),
        ("gray sigma60", gray, dict(sigma=60, **small)),
        ("gray real", gray, dict(sigma=5, noise_model="real", **small)),
        ("gray default window", gray, dict(sigma=5)),
        ("gray shapes", gray, dict(sigma=5, block_size=[4, 6], block_step=[3, 5], group_size=[8, 4], q=[2, 8], **small)),
        ("gray g64 q16", gray, dict(sigma=5, group_size=64, q=16, block_size=[5, 4], **small)),
        ("gray iterations", gray, dict(sigma=5, basic_iters=2, wiener_iters=3, lambda_basic=0.5, hard_strength=0.5,
                                       wiener_sigma_scale=1.0, **small)),
        ("gray sigma0", gray, dict(sigma=0, **small)),
        ("gray rclip", gray, dict(sigma=5, rclip=guide, **small)),
        ("yuv420 sigma", yuv420, dict(sigma=[5, 3, 3], **small)),
        ("yuv420 chroma off", yuv420, dict(sigma=[5, 0, 0], **small)),
        ("yuv420 blind", yuv420, dict(**small)),
        ("yuv444 awgn", yuv444, dict(sigma=[5, 0, 3], noise_model="awgn", **small)),
        ("rgb sigma", rgb, dict(sigma=5, **small)),
        ("rgb plane off", rgb, dict(sigma=[5, 0, 3], **small)),
        ("rgb blind rclip", rgb, dict(rclip=rgb_guide, **small)),
    ]
    temporal = [
        ("gray r1", gray, dict(sigma=5, radius=1, **small)),
        ("rgb r1 blind", rgb, dict(radius=1, **small)),
        ("yuv420 r2 ps", yuv420, dict(sigma=[5, 3, 3], radius=2, ps_num=3, ps_range=3, **small)),
    ]
    if args.quick:
        light = dict(basic_iters=1, **small)
        cases = [("gray quick", gray, dict(sigma=5, **light)), ("yuv420 quick blind", yuv420, dict(**light)),
                 ("rgb quick", rgb, dict(sigma=5, group_size=4, **light))]
        temporal = [("gray quick r1", gray, dict(sigma=5, radius=1, **light))]
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
        cpu, gpu = core.nss.NLH(clip, **kw), core.nss_cuda.NLH(clip, **kw)
        check(label, clip, cpu, gpu, core.nss_cuda.NLH(clip, **kw), cpu, gpu)
    for label, clip, kw in temporal:
        r = kw["radius"]
        check(label, clip, core.nss.VAggregate(core.nss.NLH(clip, **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.NLH(clip, temporal_mode="legacy", **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.NLH(clip, temporal_mode="legacy", **kw), clip, radius=r))
    # Repeated create/free must not leak device memory or fail.
    for _ in range(2 if args.quick else 10):
        core.nss_cuda.NLH(gray, sigma=5, **small).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda nlh live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
