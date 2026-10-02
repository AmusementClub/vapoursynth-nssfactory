#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.BM3D vs nss.BM3D on small random clips for every legal
block_size x group_size, basic and ref (Wiener) stages, Gray and RGB. Each
case must reach the BM3D tolerance (tests/data/cuda_tolerances_v1.json) and
be run-to-run identical. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_bm3d_live.py --cpu PATH --cuda PATH
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda bm3d live SKIP: {error}")
    raise SystemExit(77)

REPO = Path(__file__).resolve().parents[2]
BLOCKS = (1, 2, 4, 8, 12, 16, 32)
GROUPS = (1, 2, 4, 8, 16, 32, 64)


def make_clip(core, fmt, width, height, seed):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=2)
    planes = []
    for p in range(blank.format.num_planes):
        y, x = np.mgrid[:height, :width]
        clean = 0.5 + 0.2 * np.sin(x / (5 + p)) * np.cos(y / 7)
        planes.append((clean + rng.normal(0, 10 / 255, clean.shape)).astype(np.float32))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(planes):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def frame_planes(node):
    f = node.get_frame(0)
    return [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]


def psnr(a, b):
    mse = sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b)) / sum(x.size for x in a)
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--cuda", required=True)
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.cpu).resolve()))
    core.std.LoadPlugin(path=str(Path(args.cuda).resolve()))
    info = core.nss_cuda.Backend()
    if info["device_count"] == 0:
        print("cuda bm3d live SKIP: no CUDA device")
        return 77
    tolerances = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())
    floor = tolerances["filters"]["BM3D"]["min_psnr"]

    failures, worst, cases = [], float("inf"), 0
    gray = make_clip(core, vs.GRAYS, 72, 70, 1)
    rgb = make_clip(core, vs.RGBS, 40, 36, 2)
    for block in BLOCKS:
        for group in GROUPS:
            clip = gray if block == 32 or (block + group) % 3 else rgb
            if block > min(clip.width, clip.height):
                continue
            for stage in ("basic", "final"):
                kwargs = dict(sigma=10, block_size=block, group_size=group, bm_range=4)
                cpu_ref = core.nss.BM3D(clip, **kwargs) if stage == "final" else None
                gpu_ref = core.nss_cuda.BM3D(clip, **kwargs) if stage == "final" else None
                cpu = core.nss.BM3D(clip, ref=cpu_ref, **kwargs) if cpu_ref else core.nss.BM3D(clip, **kwargs)
                gpu = core.nss_cuda.BM3D(clip, ref=gpu_ref, **kwargs) if gpu_ref else core.nss_cuda.BM3D(clip, **kwargs)
                a, b = frame_planes(cpu), frame_planes(gpu)
                again = frame_planes(core.nss_cuda.BM3D(clip, ref=gpu_ref, **kwargs) if gpu_ref
                                     else core.nss_cuda.BM3D(clip, **kwargs))
                value = psnr(a, b)
                worst = min(worst, value)
                cases += 1
                label = f"b{block} g{group} {stage} {clip.format.name}"
                if value < floor:
                    failures.append(f"{label}: psnr {value:.2f} < {floor}")
                if any(not np.array_equal(x, y) for x, y in zip(b, again)):
                    failures.append(f"{label}: not run-to-run identical")
    # Repeated create/free must not leak device memory or fail.
    for _ in range(20):
        core.nss_cuda.BM3D(gray, sigma=5).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda bm3d live: {cases} cases, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
