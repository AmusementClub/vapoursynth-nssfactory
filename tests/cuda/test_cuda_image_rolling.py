#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.NLH and nss_cuda.TWSC rolling output against their legacy output
reduced as nss.VAggregate does it, in IEEE float32 (slices summed in ascending
center order, den == 1 -> num, den > 1e-12 -> num / den, else the source):
bit for bit, whatever the order the frames are asked for (chunks that carry
on, chunks that start afresh), the chunk size, the radius, the streams, the
format, bypassed planes, blind sigma and rclip. Exits 77 without VapourSynth
or a CUDA device.

usage: test_cuda_image_rolling.py --cuda PATH [--quick]   (--quick: a few cases, for sanitizer runs)
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda image rolling SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, seed, length):
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
            planes.append((clean + rng.normal(0, 5 / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def planes(frame):
    return [np.asarray(frame[p]).copy() for p in range(frame.format.num_planes)]


def reduced(fat, src, radius):
    """The legacy intermediate reduced in IEEE float32."""
    fats = [planes(f) for f in fat.frames()]
    out = []
    for n, s_planes in enumerate(planes(f) for f in src.frames()):
        frame = []
        for p, s in enumerate(s_planes):
            h = s.shape[0]
            num, den = np.zeros_like(s), np.zeros_like(s)
            for c in range(max(0, n - radius), min(len(fats) - 1, n + radius) + 1):
                k = 2 * (n - c + radius)
                num = num + fats[c][p][k * h:(k + 1) * h]
                den = den + fats[c][p][(k + 1) * h:(k + 2) * h]
            with np.errstate(divide="ignore", invalid="ignore"):
                frame.append(np.where(den == 1, num, np.where(den > np.float32(1e-12), num / den, s)))
        out.append(frame)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.cuda).resolve()))
    if core.nss_cuda.Backend()["device_count"] == 0:
        print("cuda image rolling SKIP: no CUDA device")
        return 77
    gray = make_clip(core, vs.GRAYS, 48, 40, 1, 7)
    rgb = make_clip(core, vs.RGBS, 48, 40, 2, 7)
    yuv = make_clip(core, vs.YUV420PS, 48, 40, 3, 7)
    yuv444 = make_clip(core, vs.YUV444PS, 48, 40, 4, 7)
    short = make_clip(core, vs.GRAYS, 48, 40, 5, 2)
    guide = make_clip(core, vs.GRAYS, 48, 40, 6, 7)
    guide_rgb = make_clip(core, vs.RGBS, 48, 40, 7, 7)
    light = dict(block_step=6, group_size=8, iters=2)
    cases = [
        ("NLH gray r1", "NLH", gray, dict(sigma=5, radius=1)),
        ("NLH gray r2", "NLH", gray, dict(sigma=5, radius=2)),
        ("NLH rgb r1", "NLH", rgb, dict(sigma=5, radius=1)),
        ("NLH yuv420 bypassed plane r1", "NLH", yuv, dict(sigma=[5, 0, 4], radius=1)),
        ("NLH gray blind r1", "NLH", gray, dict(radius=1)),
        ("NLH rgb rclip r1", "NLH", rgb, dict(sigma=5, radius=1, rclip=guide_rgb)),
        ("NLH short r2", "NLH", short, dict(sigma=5, radius=2)),
        ("TWSC gray r1", "TWSC", gray, dict(sigma=5, radius=1, **light)),
        ("TWSC gray r2", "TWSC", gray, dict(sigma=5, radius=2, **light)),
        ("TWSC rgb r1", "TWSC", rgb, dict(sigma=5, radius=1, **light)),
        ("TWSC yuv444 bypassed plane r1", "TWSC", yuv444, dict(sigma=[5, 0, 5], radius=1, **light)),
        ("TWSC gray blind r1", "TWSC", gray, dict(estimate_sigma=1, radius=1, **light)),
        ("TWSC gray rclip r1", "TWSC", gray, dict(sigma=5, radius=1, rclip=guide, **light)),
        ("TWSC all bypassed r1", "TWSC", gray, dict(sigma=0, radius=1, **light)),
    ]
    if args.quick:
        # Sanitizer runs (one NLH center takes about 20 s under memcheck): a
        # chunk afresh and one carried on, then both afresh.
        tiny = make_clip(core, vs.GRAYS, 32, 24, 9, 4)
        cases = [("NLH quick", "NLH", tiny, dict(sigma=5, radius=1)),
                 ("TWSC quick", "TWSC", tiny, dict(sigma=5, radius=1, **light))]
    orders = {
        "forward": None,
        "backward": [6, 5, 4, 3, 2, 1, 0],
        "hops": [0, 5, 1, 6, 3, 2, 4],
    }
    failures, count = [], 0
    for label, name, clip, kw in cases:
        make = getattr(core.nss_cuda, name)
        want = reduced(make(clip, temporal_mode="legacy", **kw), clip, kw["radius"])
        variants = [("chunk 4", {}), ("chunk 1", dict(rolling_chunk=1)), ("chunk 2", dict(rolling_chunk=2)),
                    ("chunk 3 two streams", dict(rolling_chunk=3, num_streams=2))]
        if args.quick:
            variants = [("chunk 2", dict(rolling_chunk=2))]
        for variant, extra in variants:
            for order_name, order in orders.items():
                if args.quick and order_name == "hops":
                    continue
                node = make(clip, **kw, **extra)
                if node.height != clip.height:
                    failures.append(f"{label} {variant}: output height {node.height}")
                    break
                frames = [n for n in order if n < clip.num_frames] if order else range(clip.num_frames)
                got = {n: planes(node.get_frame(n)) for n in frames}
                count += 1
                bad = [n for n in frames if any(not np.array_equal(a, b) for a, b in zip(got[n], want[n]))]
                if bad:
                    worst = max(float(np.max(np.abs(a - b))) for n in bad for a, b in zip(got[n], want[n]))
                    failures.append(f"{label} {variant} {order_name}: frames {bad} differ (max abs {worst:.3g})")
                props = node.get_frame(1).props
                if "_NSSFatRadius" in props or props.get("_NSSModel") is None:
                    failures.append(f"{label} {variant}: rolling frame properties {dict(props)}")
    # The rolling plan under a memory limit: at the smallest limit that creates, every frame renders (not with
    # --quick, which is for sanitizer runs).
    clip = make_clip(core, vs.GRAYS, 320, 240, 8, 6)
    for name, kw in () if args.quick else (("NLH", dict(sigma=5, radius=1)), ("TWSC", dict(sigma=5, radius=1, **light))):
        make = getattr(core.nss_cuda, name)
        lo, hi = 1, 4096
        while lo < hi:
            mid = (lo + hi) // 2
            try:
                make(clip, memory_limit_mb=mid, **kw)
                hi = mid
            except vs.Error:
                lo = mid + 1
        want = reduced(make(clip, temporal_mode="legacy", **kw), clip, 1)
        got = [planes(f) for f in make(clip, memory_limit_mb=lo, **kw).frames()]
        count += 1
        if any(not np.array_equal(a, b) for g, w in zip(got, want) for a, b in zip(g, w)):
            failures.append(f"{name} at the smallest limit ({lo} MiB) differs")
    for line in failures:
        print("FAIL:", line)
    print(f"cuda image rolling: {count} runs, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
