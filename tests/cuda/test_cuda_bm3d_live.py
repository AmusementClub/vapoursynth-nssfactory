#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.BM3D vs nss.BM3D on small random clips for every legal
block_size x group_size, basic and ref (Wiener) stages, Gray and RGB, plus
the temporal paths (radius 1/2): legacy + VAggregate against the CPU,
rolling identical to legacy + VAggregate on the GPU, and VAggregate across
backends (the fat intermediate is a backend-neutral VS frame contract).
Each case must reach the BM3D tolerance (tests/data/cuda_tolerances_v1.json)
and be run-to-run identical. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_bm3d_live.py --cpu PATH --cuda PATH [--quick]

--quick runs a representative subset (for compute-sanitizer runs).
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


def make_clip(core, fmt, width, height, seed, length=2):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            y, x = np.mgrid[:height, :width]
            clean = 0.5 + 0.2 * np.sin((x + 2 * t) / (5 + p)) * np.cos((y + t) / 7)
            planes.append((clean + rng.normal(0, 10 / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def frame_planes(node, n=0):
    f = node.get_frame(n)
    return [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]


def within_ulp(a, b, ulps=4):
    return all(np.all(np.abs(x - y) <= ulps * np.spacing(np.maximum(np.abs(x), np.abs(y)).astype(np.float32)))
               for x, y in zip(a, b))


def temporal(core, quick, floor):
    """Legacy/rolling/cross-backend temporal checks; returns (cases, worst, failures)."""
    failures, worst, cases = [], float("inf"), 0
    clips = [make_clip(core, vs.GRAYS, 64, 56, 3, length=6)]
    if not quick:
        clips.append(make_clip(core, vs.RGBS, 48, 40, 4, length=6))
    for clip in clips:
        for radius in (1, 2):
            kw = dict(sigma=10, radius=radius, bm_range=5)
            cpu = core.nss.VAggregate(core.nss.BM3D(clip, **kw), clip, radius=radius)
            gpu = core.nss_cuda.VAggregate(core.nss_cuda.BM3D(clip, **kw), clip, radius=radius)
            rolling = core.nss_cuda.BM3D(clip, temporal_mode="rolling", rolling_chunk=4, **kw)
            mixed_a = core.nss.VAggregate(core.nss_cuda.BM3D(clip, **kw), clip, radius=radius)
            mixed_b = core.nss_cuda.VAggregate(core.nss.BM3D(clip, **kw), clip, radius=radius)
            for n in range(clip.num_frames):
                label = f"temporal r{radius} {clip.format.name} frame {n}"
                a, b = frame_planes(cpu, n), frame_planes(gpu, n)
                value = psnr(a, b)
                worst = min(worst, value)
                cases += 1
                if value < floor:
                    failures.append(f"{label}: legacy psnr {value:.2f} < {floor}")
                if any(not np.array_equal(x, y) for x, y in zip(b, frame_planes(rolling, n))):
                    failures.append(f"{label}: rolling differs from legacy + VAggregate")
                # Same fat input and summation order on both backends; the GPU
                # divides with IEEE rounding while the CPU fast-math TU was
                # measured up to 2 ulp off (2026-10-03), so allow a few ulp.
                if not within_ulp(frame_planes(mixed_a, n), b):
                    failures.append(f"{label}: nss.VAggregate(nss_cuda.BM3D) differs from nss_cuda.VAggregate by > 4 ulp")
                if not within_ulp(frame_planes(mixed_b, n), a):
                    failures.append(f"{label}: nss_cuda.VAggregate(nss.BM3D) differs from nss.VAggregate by > 4 ulp")
                again = frame_planes(core.nss_cuda.VAggregate(core.nss_cuda.BM3D(clip, **kw), clip, radius=radius), n)
                if any(not np.array_equal(x, y) for x, y in zip(b, again)):
                    failures.append(f"{label}: not run-to-run identical")
    return cases, worst, failures


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
    info = core.nss_cuda.Backend()
    if info["device_count"] == 0:
        print("cuda bm3d live SKIP: no CUDA device")
        return 77
    tolerances = json.loads((REPO / "tests" / "data" / "cuda_tolerances_v1.json").read_text())
    floor = tolerances["filters"]["BM3D"]["min_psnr"]

    failures, worst, cases = [], float("inf"), 0
    gray = make_clip(core, vs.GRAYS, 72, 70, 1)
    rgb = make_clip(core, vs.RGBS, 40, 36, 2)
    blocks, groups = ((4, 8, 16), (1, 8, 64)) if args.quick else (BLOCKS, GROUPS)
    for block in blocks:
        for group in groups:
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
    t_cases, t_worst, t_failures = temporal(core, args.quick, floor)
    cases += t_cases
    worst = min(worst, t_worst)
    failures += t_failures
    # Shapes that differ per plane share one slot: fused beside ordered planes, at a size where the
    # shared buffers matter, and compared with the CPU on a small clip.
    for fmt, w, h in ((vs.YUV420PS, 2560, 1440), (vs.YUV444PS, 1920, 1080)):
        big = core.std.BlankClip(width=w, height=h, format=fmt, length=2)
        for label, kw in (("4/32 + 16/16", dict(block_size=[4, 16, 16], group_size=[32, 16, 16])),
                          ("8/32 + 16/32 final step 2", dict(ref=big, block_size=[8, 16, 16], group_size=32, block_step=2)),
                          ("8/16 + 4/8", dict(block_size=[8, 4, 4], group_size=[16, 8, 8]))):
            cases += 1
            try:
                core.nss_cuda.BM3D(big, **kw).get_frame(0)
            except vs.Error as error:
                failures.append(f"mixed {big.format.name} {label}: {error}")
    yuv = make_clip(core, vs.YUV444PS, 64, 56, 5)
    for kw in (dict(block_size=[4, 16, 16], group_size=[32, 16, 16]), dict(block_size=[8, 4, 4], group_size=[16, 8, 8])):
        kw = dict(sigma=10, bm_range=4, **kw)
        value = psnr(frame_planes(core.nss.BM3D(yuv, **kw)), frame_planes(core.nss_cuda.BM3D(yuv, **kw)))
        worst = min(worst, value)
        cases += 1
        if value < floor:
            failures.append(f"mixed shapes {kw}: psnr {value:.2f} < {floor}")
    # Temporal filtering always stores and orders its patches, for every shape and both stages.
    seq = make_clip(core, vs.GRAYS, 64, 56, 6, length=4)
    for block, group in ((4, 8), (8, 16), (16, 16), (12, 16), (8, 32)):
        for stage in ("basic", "final"):
            kw = dict(sigma=10, radius=1, block_size=block, group_size=group, bm_range=4)
            cpu_ref = core.nss.VAggregate(core.nss.BM3D(seq, **kw), seq, radius=1) if stage == "final" else None
            gpu_ref = core.nss_cuda.BM3D(seq, temporal_mode="rolling", **kw) if stage == "final" else None
            cpu = core.nss.VAggregate(core.nss.BM3D(seq, **kw, **(dict(ref=cpu_ref) if cpu_ref else {})), seq, radius=1)
            gpu = core.nss_cuda.BM3D(seq, temporal_mode="rolling", **kw, **(dict(ref=gpu_ref) if gpu_ref else {}))
            value = psnr(frame_planes(cpu, 1), frame_planes(gpu, 1))
            worst = min(worst, value)
            cases += 1
            if value < floor:
                failures.append(f"temporal b{block} g{group} {stage}: psnr {value:.2f} < {floor}")
    # 4K must create at the default memory limit (no frames are requested). Rolling and YUV temporal
    # Wiener need more than the default and are not in this list. Each case takes up to about 1 GiB of
    # device memory while it exists.
    for fmt in (vs.GRAYS, vs.YUV420PS):
        uhd = core.std.BlankClip(width=3840, height=2160, format=fmt, length=4)
        modes = [("spatial", {}), ("final", dict(ref=uhd)), ("r1 legacy", dict(radius=1)),
                 ("block 4", dict(block_size=4)), ("16 / 16", dict(block_size=16, group_size=16))]
        if fmt == vs.GRAYS:
            modes.append(("r1 final", dict(radius=1, ref=uhd)))
        for label, kw in modes:
            cases += 1
            try:
                core.nss_cuda.BM3D(uhd, **kw)
            except vs.Error as error:
                failures.append(f"4K {uhd.format.name} {label}: {error}")
    for _ in range(2 if args.quick else 20):
        core.nss_cuda.BM3D(gray, sigma=5).get_frame(0)
    for line in failures:
        print("FAIL:", line)
    print(f"cuda bm3d live: {cases} cases, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
