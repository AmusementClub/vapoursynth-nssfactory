#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.NLM vs nss.NLM on small random clips across channel modes,
temporal radius, search/patch radii, h, wref, wmode and rclip, including clip
boundary frames. Each case must reach the NLM tolerance
(tests/data/vk_tolerances_v1.json) and be run-to-run identical. Exits 77
without VapourSynth or a supported Vulkan device.

usage: test_vk_nlm_live.py --cpu PATH --vk PATH [--device N] [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"vk nlm live SKIP: {error}")
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
    parser.add_argument("--vk", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.cpu).resolve()))
    core.std.LoadPlugin(path=str(Path(args.vk).resolve()))
    backend = core.nss_vk.Backend()
    if backend["device_count"] == 0:
        print("vk nlm live SKIP: no Vulkan device")
        return 77
    support = core.nss_vk.Backend(device_id=args.device)["support"]
    if (support.decode() if isinstance(support, bytes) else support) == "unsupported":
        print(f"vk nlm live SKIP: device {args.device} is unsupported")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "vk_tolerances_v1.json").read_text())["filters"]["NLM"]["min_psnr"]
    vk = lambda clip, **kw: core.nss_vk.NLM(clip, device_id=args.device, **kw)

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
        # Tiles with nothing outside the image (they skip the clamping), with and without
        # the frames' windows in shared memory (one channel with the default radii has them).
        ("gray 200x150", make_clip(core, vs.GRAYS, 200, 150, 7, length=3), {}),
        ("gray 200x150 a4", make_clip(core, vs.GRAYS, 200, 150, 7, length=3), dict(a=4, s=2)),
        ("gray 200x150 rclip", make_clip(core, vs.GRAYS, 200, 150, 7, length=3),
         dict(rclip=make_clip(core, vs.GRAYS, 200, 150, 8, length=3))),
        ("rgb 150x110", make_clip(core, vs.RGBS, 150, 110, 9, length=3), {}),
        # Bisquare weights (zero past u = 1, so a larger h) on the tile path and beyond it.
        ("gray wmode1 h12", gray, dict(wmode=1, h=12.0)),
        ("gray wmode2 d2 h20", gray, dict(wmode=2, d=2, h=20.0)),
        ("gray 200x150 wmode3 h20", make_clip(core, vs.GRAYS, 200, 150, 7, length=3), dict(wmode=3, h=20.0)),
        ("rgb wmode2 h20", rgb, dict(wmode=2, h=20.0)),
        ("gray s30 wmode3 h20", gray, dict(s=30, wmode=3, h=20.0)),
    ]
    if args.quick:
        cases = [cases[0], cases[5], cases[7], cases[9]]
    failures, worst, count = [], float("inf"), 0
    for label, clip, kw in cases:
        cpu = core.nss.NLM(clip, **kw)
        gpu = vk(clip, **kw)
        again = vk(clip, **kw)
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
              ("rgb d1", long_rgb, {}), ("d3 s30", long_gray, dict(d=3, a=1, s=30)),
              ("d1 wmode2", long_gray, dict(wmode=2, h=20.0))]
    for label, clip, kw in ([orders[0]] if args.quick else orders):
        cpu = core.nss.NLM(clip, **kw)
        expected = [frame_planes(vk(clip, **kw), n) for n in range(clip.num_frames)]
        for n in range(clip.num_frames):
            value = psnr(frame_planes(cpu, n), expected[n])
            worst = min(worst, value)
            count += 1
            if value < floor:
                failures.append(f"order {label} frame {n}: psnr {value:.2f} < {floor}")
        # One node asked in order keeps the window frames on the device between
        # requests: the frames must be those of the fresh nodes.
        node = vk(clip, num_streams=1, **kw)
        for n in range(clip.num_frames):
            count += 1
            if any(not np.array_equal(x, y) for x, y in zip(frame_planes(node, n), expected[n])):
                failures.append(f"order {label} frame {n}: asked in order differs from a fresh node's frame")
        rng = np.random.default_rng(5)
        for streams in (1, 2, 3):
            node = vk(clip, num_streams=streams, **kw)
            order = [int(n) for n in rng.permutation(clip.num_frames)] + list(range(clip.num_frames - 1, -1, -1))
            futures = [(n, node.get_frame_async(n)) for n in order]
            for n, future in futures:
                f = future.result()
                got = [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]
                count += 1
                if any(not np.array_equal(x, y) for x, y in zip(got, expected[n])):
                    failures.append(f"order {label} streams {streams} frame {n}: differs from the in-order frame")
    # memory_limit_mb, downward until refused (where depends on BAR uploads):
    # smaller caches down to the bare window must give the same frames within
    # their limit; below the smallest plan, creation fails naming the limit.
    big = make_clip(core, vs.GRAYS, 1280, 720, 31, length=6)
    reference = [frame_planes(vk(big), n) for n in range(big.num_frames)]
    accepted = []
    for limit in (80, 60, 50, 40, 34, 30, 26, 23, 20, 4):
        try:
            node = vk(big, memory_limit_mb=limit)
            futures = [(n, node.get_frame_async(n)) for n in [5, 0, 3, 1, 4, 2] * 2]
            for n, future in futures:
                f = future.result()
                count += 1
                if any(not np.array_equal(np.array(f[p], dtype=np.float64), reference[n][p]) for p in range(f.format.num_planes)):
                    failures.append(f"memory_limit_mb={limit} frame {n}: differs from the unlimited frame")
                if f.props.get("_NSSResourcePeak", 0) > limit << 20:
                    failures.append(f"memory_limit_mb={limit} frame {n}: peak {f.props['_NSSResourcePeak']} over the limit")
            accepted.append(limit)
        except vs.Error as error:
            if "memory_limit_mb" not in str(error):
                failures.append(f"memory_limit_mb={limit}: unexpected error {error}")
            elif accepted and accepted[-1] < limit:
                failures.append(f"memory_limit_mb={limit} refused after {accepted[-1]} was accepted")
    if len(accepted) < 3 or 4 in accepted:
        failures.append(f"memory_limit_mb: accepted {accepted}")
    # More VapourSynth threads than slots: requests that wait for a frame
    # another request uploads must not hold the slots that one needs.
    core.num_threads = 16
    for label, clip, kw in ([orders[0]] if args.quick else [orders[0], orders[1], orders[3]]):
        expected = [frame_planes(vk(clip, **kw), n) for n in range(clip.num_frames)]
        node = vk(clip, num_streams=1, **kw)
        order = [int(n) for n in np.random.default_rng(9).permutation(clip.num_frames)] * 2
        futures = [(n, node.get_frame_async(n)) for n in order]
        for n, future in futures:
            f = future.result()
            count += 1
            if any(not np.array_equal(np.array(f[p], dtype=np.float64), expected[n][p]) for p in range(f.format.num_planes)):
                failures.append(f"16 threads {label} frame {n}: differs from the in-order frame")
    core.num_threads = 4
    for line in failures:
        print("FAIL:", line)
    print(f"vk nlm live: {count} frames, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
