#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Stability check for libnss_cuda: creates and frees every filter repeatedly,
fetching frames through each instance with several in flight, and watches the
process's device memory (nvidia-smi) for growth. ctest runs a short pass
(test_cuda_stress); run it with more --cycles for a long soak. Exits 77
without VapourSynth or a CUDA device.

usage: stress_cuda.py --cuda PATH [--cycles N] [--frames N]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda stress SKIP: {error}")
    raise SystemExit(77)


def device_memory_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return 0  # no nvidia-smi: the run still checks for failures
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) == os.getpid():
            return int(parts[1])
    return 0


def make_clip(core, fmt, width, height, length):
    rng = np.random.default_rng(1)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    planes = [(0.5 + rng.normal(0, 5 / 255, (height, width))).astype(np.float32) for _ in range(4)]

    def fill(n, f):
        out = f.copy()
        for p in range(out.format.num_planes):
            np.asarray(out[p])[:] = planes[(n + p) % 4]
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(path=str(Path(args.cuda).resolve()))
    if core.nss_cuda.Backend()["device_count"] == 0:
        print("cuda stress SKIP: no CUDA device")
        return 77
    gray = make_clip(core, vs.GRAYS, 320, 240, args.frames)
    rgb = make_clip(core, vs.RGBS, 320, 240, args.frames)
    n = core.nss_cuda
    light = dict(block_step=8, group_size=8, iters=2)
    builders = [
        ("BM3D", lambda: n.BM3D(gray, sigma=5)),
        ("BM3D rolling", lambda: n.BM3D(gray, sigma=5, radius=1, temporal_mode="rolling")),
        ("BM3D legacy", lambda: n.VAggregate(n.BM3D(gray, sigma=5, radius=1), gray, radius=1)),
        ("NLM", lambda: n.NLM(rgb)),
        ("WNNM", lambda: n.WNNM(gray, sigma=5)),
        ("MCWNNM", lambda: n.MCWNNM(rgb, sigma=5)),
        ("NCSR", lambda: n.NCSR(gray, sigma=5)),
        ("NLH", lambda: n.NLH(gray, sigma=5, basic_iters=1)),
        ("TWSC", lambda: n.TWSC(gray, sigma=5, **light)),
        ("LSSC", lambda: n.LSSC(gray, sigma=5)),
    ]
    readings, frames = [], 0
    for cycle in range(args.cycles):
        for name, build in builders:
            node = build()
            try:
                for f in [node.get_frame_async(i) for i in range(node.num_frames)]:
                    f.result()
                    frames += 1
            except vs.Error as error:
                print(f"FAIL: {name} in cycle {cycle}: {error}")
                return 1
            del node
        readings.append(device_memory_mb())
        if cycle % 10 == 0:
            print(f"cycle {cycle}: {frames} frames, device memory {readings[-1]} MiB", flush=True)
    # The context and caches settle in the first cycles; later growth is a leak.
    settled = readings[len(readings) // 4]
    growth = readings[-1] - settled
    print(f"cuda stress: {frames} frames over {args.cycles} cycles, device memory {settled} -> {readings[-1]} MiB")
    if growth > 64:
        print(f"FAIL: device memory grew by {growth} MiB")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
