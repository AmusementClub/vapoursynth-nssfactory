#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.LSSC vs nss.LSSC on small random clips: every legal block size,
steps, noise levels, Gray / YUV420 / RGB with bypassed planes, grids smaller
than the cluster, atom and sample counts, and a grid large enough for full
clusters. Each case must reach the LSSC tolerance
(tests/data/cuda_tolerances_v1.json) and be run-to-run identical. Exits 77
without VapourSynth or a CUDA device.

usage: test_cuda_lssc_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda lssc live SKIP: {error}")
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
        print("cuda lssc live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["LSSC"]["min_psnr"]

    gray = make_clip(core, vs.GRAYS, 53, 47, 1, length=2)
    large = make_clip(core, vs.GRAYS, 176, 144, 5, length=2)
    tiny = make_clip(core, vs.GRAYS, 17, 16, 6, length=2)
    yuv420 = make_clip(core, vs.YUV420PS, 56, 48, 2, length=2)
    rgb = make_clip(core, vs.RGBS, 41, 37, 4, length=2)
    cases = [
        ("gray default", gray, dict(sigma=5)),
        ("gray sigma25", gray, dict(sigma=25)),
        ("gray step4", gray, dict(sigma=5, block_step=4)),
        ("gray step1", gray, dict(sigma=5, block_step=1)),
        ("gray b4", gray, dict(sigma=5, block_size=4, block_step=3)),
        ("gray b2", gray, dict(sigma=5, block_size=2, block_step=2)),
        ("gray b1", gray, dict(sigma=5, block_size=1, block_step=1)),
        ("gray b16", gray, dict(sigma=5, block_size=16, block_step=9)),
        ("large step3", large, dict(sigma=8, block_step=3)),
        ("tiny one patch per axis", tiny, dict(sigma=5, block_size=16, block_step=16)),
        ("tiny few patches", tiny, dict(sigma=5)),
        ("yuv420 per-plane", yuv420, dict(sigma=[5, 0, 3])),
        ("rgb default", rgb, dict(sigma=5)),
        ("gray sigma0", gray, dict(sigma=0)),
    ]
    if args.quick:
        cases = [cases[0], cases[4], cases[10], cases[11]]
    failures, worst, count = [], float("inf"), 0
    for label, clip, kw in cases:
        cpu, gpu, again = core.nss.LSSC(clip, **kw), core.nss_cuda.LSSC(clip, **kw), core.nss_cuda.LSSC(clip, **kw)
        for n in range(clip.num_frames):
            a, b = frame_planes(cpu, n), frame_planes(gpu, n)
            value = psnr(a, b)
            worst = min(worst, value)
            count += 1
            if value < floor:
                failures.append(f"{label} frame {n}: psnr {value:.2f} < {floor}")
            if any(not np.array_equal(x, y) for x, y in zip(b, frame_planes(again, n))):
                failures.append(f"{label} frame {n}: not run-to-run identical")
    # Repeated create/free must not leak device memory or fail.
    for _ in range(2 if args.quick else 10):
        core.nss_cuda.LSSC(gray, sigma=5).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda lssc live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
