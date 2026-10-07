#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.NLM vs nss.NLM on small random clips across channel modes,
temporal radius, search/patch radii, h, wref and rclip, including clip
boundary frames. Each case must reach the NLM tolerance
(tests/data/cuda_tolerances_v1.json) and be run-to-run identical. Exits 77
without VapourSynth or a CUDA device.

usage: test_cuda_nlm_live.py --cpu PATH --cuda PATH [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda nlm live SKIP: {error}")
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
            planes.append((clean + rng.normal(0, 8 / 255, clean.shape)).astype(np.float32))
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
        print("cuda nlm live SKIP: no CUDA device")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())["filters"]["NLM"]["min_psnr"]

    gray = make_clip(core, vs.GRAYS, 53, 47, 1)
    yuv420 = make_clip(core, vs.YUV420PS, 56, 48, 2)
    yuv444 = make_clip(core, vs.YUV444PS, 41, 37, 3)
    rgb = make_clip(core, vs.RGBS, 41, 37, 4)
    guide = make_clip(core, vs.GRAYS, 53, 47, 9)
    cases = [
        ("gray default", gray, {}),
        ("gray d0", gray, dict(d=0)),
        ("gray d2 a1 s1", gray, dict(d=2, a=1, s=1)),
        ("gray a3 s0 h3", gray, dict(a=3, s=0, h=3.0)),
        ("gray s6 wref0.5", gray, dict(s=6, wref=0.5)),
        ("gray rclip", gray, dict(rclip=guide)),
        ("yuv420 Y", yuv420, {}),
        ("yuv420 UV", yuv420, dict(channels="UV")),
        ("yuv444 YUV", yuv444, dict(channels="YUV")),
        ("rgb default", rgb, {}),
        ("rgb d0 a3", rgb, dict(d=0, a=3)),
        # Beyond the one-launch path: its shared memory (s) and its window (d).
        ("gray s30", gray, dict(s=30)),
        ("gray d9 a1 s1", gray, dict(d=9, a=1, s=1)),
        ("rgb s30 rclip", rgb, dict(s=30, d=2, rclip=make_clip(core, vs.RGBS, 41, 37, 11))),
        # Tiles of one pixel row or column at the right and bottom edges.
        ("gray 65x33", make_clip(core, vs.GRAYS, 65, 33, 5), dict(a=4)),
        ("gray 20x9", make_clip(core, vs.GRAYS, 20, 9, 6), dict(a=3, s=5)),
    ]
    if args.quick:
        cases = [cases[0], cases[5], cases[7], cases[9]]
    failures, worst, count = [], float("inf"), 0
    for label, clip, kw in cases:
        cpu = core.nss.NLM(clip, **kw)
        gpu = core.nss_cuda.NLM(clip, **kw)
        again = core.nss_cuda.NLM(clip, **kw)
        for n in range(clip.num_frames):  # includes both clip boundaries
            a, b = frame_planes(cpu, n), frame_planes(gpu, n)
            value = psnr(a, b)
            worst = min(worst, value)
            count += 1
            if value < floor:
                failures.append(f"{label} frame {n}: psnr {value:.2f} < {floor}")
            if any(not np.array_equal(x, y) for x, y in zip(b, frame_planes(again, n))):
                failures.append(f"{label} frame {n}: not run-to-run identical")
    # The window frames stay on the device between requests: any order of
    # requests, any number of threads and streams must give the frames that
    # requests in order give.
    long_gray = make_clip(core, vs.GRAYS, 53, 47, 21, length=24)
    long_rgb = make_clip(core, vs.RGBS, 41, 37, 22, length=24)
    orders = [("d1", long_gray, {}), ("d2 rclip", long_gray, dict(d=2, rclip=make_clip(core, vs.GRAYS, 53, 47, 23, length=24))),
              ("rgb d1", long_rgb, {}), ("d3 s30", long_gray, dict(d=3, a=1, s=30))]
    for label, clip, kw in ([orders[0]] if args.quick else orders):
        cpu = core.nss.NLM(clip, **kw)
        expected = [frame_planes(core.nss_cuda.NLM(clip, **kw), n) for n in range(clip.num_frames)]
        for n in range(clip.num_frames):
            value = psnr(frame_planes(cpu, n), expected[n])
            worst = min(worst, value)
            count += 1
            if value < floor:
                failures.append(f"order {label} frame {n}: psnr {value:.2f} < {floor}")
        rng = np.random.default_rng(5)
        for streams in (1, 2, 3):
            node = core.nss_cuda.NLM(clip, num_streams=streams, **kw)
            order = [int(n) for n in rng.permutation(clip.num_frames)] + list(range(clip.num_frames - 1, -1, -1))
            futures = [(n, node.get_frame_async(n)) for n in order]
            for n, future in futures:
                f = future.result()
                got = [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]
                count += 1
                if any(not np.array_equal(x, y) for x, y in zip(got, expected[n])):
                    failures.append(f"order {label} streams {streams} frame {n}: differs from the in-order frame")
    for line in failures:
        print("FAIL:", line)
    print(f"cuda nlm live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
