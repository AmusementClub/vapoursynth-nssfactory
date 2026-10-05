#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.NCSR vs nss.NCSR on small random clips across block and group
sizes (every group kernel capacity), outer rounds and delta, rclip and the
temporal radius (legacy + VAggregate, including clip boundary frames). Each
case must reach the NCSR tolerance (tests/data/cuda_tolerances_v1.json) and be
run-to-run identical. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_ncsr_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda ncsr live SKIP: {error}")
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
        print("cuda ncsr live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["NCSR"]["min_psnr"]

    gray = make_clip(core, vs.GRAYS, 53, 47, 1)
    yuv420 = make_clip(core, vs.YUV420PS, 56, 48, 2)
    rgb = make_clip(core, vs.RGBS, 41, 37, 4)
    guide = make_clip(core, vs.GRAYS, 53, 47, 9)
    small = dict(bm_range=5)
    cases = [
        ("gray default", gray, dict(sigma=5, **small)),
        ("gray iters1", gray, dict(sigma=5, iters=1, **small)),
        ("gray iters3 delta0.3", gray, dict(sigma=5, iters=3, delta=0.3, **small)),
        ("gray sigma20", gray, dict(sigma=20, **small)),
        ("gray g1", gray, dict(sigma=5, group_size=1, ps_num=1, **small)),
        ("gray g3 b4", gray, dict(sigma=5, group_size=3, block_size=4, block_step=4, **small)),
        ("gray g16 step4", gray, dict(sigma=8, group_size=16, block_step=4, **small)),
        ("gray g32 b6", gray, dict(sigma=8, group_size=32, block_size=6, block_step=3, **small)),
        ("gray g24 step6", gray, dict(sigma=8, group_size=24, block_step=6, **small)),
        ("gray b16 g32", gray, dict(sigma=8, group_size=32, block_size=16, block_step=8, **small)),
        ("gray b2 g32", gray, dict(sigma=8, group_size=32, block_size=2, block_step=2, **small)),
        ("gray rclip", gray, dict(sigma=5, rclip=guide, **small)),
        ("yuv420 per-plane", yuv420, dict(sigma=[5, 0, 3], **small)),
        ("rgb default", rgb, dict(sigma=5, **small)),
    ]
    temporal = [
        ("gray r1", gray, dict(sigma=5, radius=1, **small)),
        ("rgb r2 g16", rgb, dict(sigma=5, radius=2, group_size=16, block_step=4, **small)),
    ]
    if args.quick:
        cases = [cases[0], cases[6], cases[8], cases[12]]
        temporal = temporal[:1]
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
        check(label, clip, core.nss.NCSR(clip, **kw), core.nss_cuda.NCSR(clip, **kw), core.nss_cuda.NCSR(clip, **kw))
    for label, clip, kw in temporal:
        r = kw["radius"]
        check(label, clip, core.nss.VAggregate(core.nss.NCSR(clip, **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.NCSR(clip, temporal_mode="legacy", **kw), clip, radius=r),
              core.nss_cuda.VAggregate(core.nss_cuda.NCSR(clip, temporal_mode="legacy", **kw), clip, radius=r))
    # Repeated create/free must not leak device memory or fail.
    for _ in range(2 if args.quick else 20):
        core.nss_cuda.NCSR(gray, sigma=5).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda ncsr live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
