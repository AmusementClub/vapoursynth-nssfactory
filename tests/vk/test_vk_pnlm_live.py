#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.PNLM vs nss.PNLM, on the settings of test_cuda_pnlm_live.py: gray,
YUV420 with a bypassed plane, RGB; odd and tiny sizes (clamped grid edges);
patches of 4, 6 and 8; windows on the fused and the key path; groups of 2 to
32; d of 0 to 3 at the ends of the clip and inside it; one stream and three,
frames requested in order and out of order. Each case must reach the PNLM
tolerance (tests/data/vk_tolerances_v1.json) and be identical across the
runs. On NVIDIA every frame must also have the CPU's bits (the kernels
follow the CPU model operation for operation, with correctly rounded
quotients, and NVIDIA fuses fma); elsewhere the frames with the CPU's bits
are counted (D26: Mesa splits fma).
Exits 77 without VapourSynth or a supported Vulkan device.

usage: test_vk_pnlm_live.py --cpu PATH --vk PATH [--device N] [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"vk pnlm live SKIP: {error}")
    raise SystemExit(77)

REPO = Path(__file__).resolve().parents[2]


def make_clip(core, fmt, width, height, seed, sigma, length):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p else 0)
            h = height >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.45 + 0.25 * (((x + 2 * t) // 11 + y // 9) % 2) + 0.1 * np.sin((x + t) / (5 + p)) * np.cos(y / 7)
            planes.append((clean + rng.normal(0, sigma / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def psnr(a, b):
    if not all(np.all(np.isfinite(x)) and np.all(np.isfinite(y)) for x, y in zip(a, b)):
        return float("-inf")
    mse = sum(float(np.sum((x.astype(np.float64) - y) ** 2)) for x, y in zip(a, b)) / sum(x.size for x in a)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--vk", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="three clips, three settings")
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(args.cpu)
    try:
        core.std.LoadPlugin(args.vk)
        info = core.nss_vk.Backend(device_id=args.device)
        support = info.get("support", b"")
        support = support.decode() if isinstance(support, bytes) else support
        if support not in ("native", "emulated"):
            raise RuntimeError(f"device {args.device} is {support or 'missing'}")
        exact = info.get("vendor_id", 0) == 0x10DE
    except Exception as error:  # noqa: BLE001
        print(f"vk pnlm live SKIP: {error}")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "vk_tolerances_v1.json").read_text())["filters"]["PNLM"]["min_psnr"]

    clips = [
        (vs.GRAYS, 333, 201, 6.0, None),
        (vs.GRAYS, 256, 128, 2.0, None),
        (vs.YUV420PS, 262, 146, 8.0, [8.0, 0.0, 5.0]),
        (vs.RGBS, 141, 97, 12.0, None),
        (vs.GRAYS, 9, 8, 10.0, None),       # a grid of two references a row, the last clamped
        (vs.GRAYS, 8, 23, 10.0, None),      # one reference column at block_size 8
    ]
    settings = [
        dict(),                                            # defaults: block 6, a 2, group 16, d 1 (fused)
        dict(d=0, block_size=4),
        dict(d=0, block_size=8, a=1, group_size=2),
        dict(d=2, block_size=4, a=4, group_size=8),
        dict(d=1, block_size=6, a=5, group_size=16),       # key path
        dict(d=3, block_size=8, a=7, group_size=32),
        dict(d=0, block_size=6, a=3, group_size=24),
    ]
    if args.quick:
        clips, settings = clips[:3], [settings[0], settings[3], settings[4]]
    failures = 0
    length = 6
    frames_total = frames_same = 0
    worst = float("inf")
    for i, (fmt, w, h, sigma, sig) in enumerate(clips):
        src = make_clip(core, fmt, w, h, 200 + i, sigma, length)
        for kw in settings:
            kw = dict(kw, sigma=sig if sig is not None else sigma)
            cpu = core.nss.PNLM(src, **kw)
            one = core.nss_vk.PNLM(src, num_streams=1, device_id=args.device, **kw)
            three = core.nss_vk.PNLM(src, num_streams=3, device_id=args.device, **kw)
            futures = [three.get_frame_async(n) for n in reversed(range(length))]
            out_of_order = [f.result() for f in reversed(futures)]
            case_worst, same, stable = float("inf"), 0, True
            for n in range(length):
                want = planes(cpu.get_frame(n))
                a, b = planes(one.get_frame(n)), planes(out_of_order[n])
                stable = stable and all(np.array_equal(x.view(np.uint32), y.view(np.uint32)) for x, y in zip(a, b))
                case_worst = min(case_worst, psnr(want, a))
                same += all(np.array_equal(x.view(np.uint32), y.view(np.uint32)) for x, y in zip(want, a))
            frames_total += length
            frames_same += same
            worst = min(worst, case_worst)
            bad = case_worst < floor or not stable or (exact and same < length)
            failures += bad
            print(f"{vs.core.get_video_format(fmt).name} {w}x{h} {kw}: {case_worst:.2f} dB, "
                  f"{same}/{length} frames with the CPU's bits{'' if stable else ', NOT RUN-TO-RUN IDENTICAL'}"
                  f"{'  FAIL' if bad else ''}", flush=True)

    src = make_clip(core, vs.GRAYS, 320, 240, 9, 5.0, 2)
    try:
        core.nss_vk.PNLM(src, memory_limit_mb=1)
        print("memory_limit_mb=1 accepted  FAIL")
        failures += 1
    except vs.Error as error:
        print(f"memory_limit_mb=1: {error}")
    print(f"vk pnlm live: {frames_total} frames, worst psnr {worst:.2f} dB, {frames_same} with the CPU's bits, "
          f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
