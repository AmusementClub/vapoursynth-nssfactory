#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.BM3D vs nss.BM3D on small random clips (docs/vulkan-plan.md V5,
spatial output): every legal block_size x group_size, the basic stage and the
Wiener stage with ref, final=1 in one node, chroma=1 (CBM3D), planes left
alone, YUV420 planes of two sizes, padded widths, memory_limit_mb and many
threads. Each case must reach the BM3D tolerance (tests/data/
vk_tolerances_v1.json) and be run-to-run identical. Exits 77 without
VapourSynth or a supported Vulkan device.

usage: test_vk_bm3d_live.py --cpu PATH --vk PATH [--device N] [--quick]
"""
import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"vk bm3d live SKIP: {error}")
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
            h, w = (height, width) if p == 0 else (height >> blank.format.subsampling_h, width >> blank.format.subsampling_w)
            y, x = np.mgrid[:h, :w]
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


def psnr(a, b):
    # Non-finite planes fail outright: NaN compares false against the floor
    # and min(worst, nan) keeps worst.
    if not all(np.all(np.isfinite(x)) and np.all(np.isfinite(y)) for x, y in zip(a, b)):
        return float("-inf")
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
    if core.nss_vk.Backend()["device_count"] == 0:
        print("vk bm3d live SKIP: no Vulkan device")
        return 77
    support = core.nss_vk.Backend(device_id=args.device)["support"]
    if (support.decode() if isinstance(support, bytes) else support) == "unsupported":
        print(f"vk bm3d live SKIP: device {args.device} is unsupported")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "vk_tolerances_v1.json").read_text())["filters"]["BM3D"]["min_psnr"]
    vk = lambda clip, **kw: core.nss_vk.BM3D(clip, device_id=args.device, **kw)
    cpu = core.nss.BM3D

    failures, worst, cases = [], float("inf"), 0

    def check(label, a_node, b_node, again=None, frames=(0,)):
        nonlocal worst, cases
        for n in frames:
            a, b = frame_planes(a_node, n), frame_planes(b_node, n)
            value = psnr(a, b)
            worst = min(worst, value)
            cases += 1
            if value < floor:
                failures.append(f"{label} frame {n}: psnr {value:.2f} < {floor}")
            if again is not None and any(not np.array_equal(x, y) for x, y in zip(b, frame_planes(again, n))):
                failures.append(f"{label} frame {n}: not run-to-run identical")

    gray = make_clip(core, vs.GRAYS, 72, 70, 1)
    rgb = make_clip(core, vs.RGBS, 40, 36, 2)
    blocks, groups = ((4, 8, 16), (1, 8, 64)) if args.quick else (BLOCKS, GROUPS)
    for block in blocks:
        for group in groups:
            clip = gray if block == 32 or (block + group) % 3 else rgb
            if block > min(clip.width, clip.height):
                continue
            kw = dict(sigma=10, block_size=block, group_size=group, bm_range=4)
            label = f"b{block} g{group} {clip.format.name}"
            check(f"{label} basic", cpu(clip, **kw), vk(clip, **kw), vk(clip, **kw))
            cpu_ref, vk_ref = cpu(clip, **kw), vk(clip, **kw)
            check(f"{label} final", cpu(clip, ref=cpu_ref, **kw), vk(clip, ref=vk_ref, **kw), vk(clip, ref=vk_ref, **kw))

    yuv420 = make_clip(core, vs.YUV420PS, 66, 50, 5)
    yuv444 = make_clip(core, vs.YUV444PS, 64, 56, 9)
    odd = make_clip(core, vs.GRAYS, 61, 45, 6)  # rows that are not tight in VapourSynth frames
    others = [
        ("defaults gray", gray, dict(sigma=10)),
        ("final=1 gray", gray, dict(sigma=10, final=1)),
        ("final=1 rgb", rgb, dict(sigma=[10, 6, 3], final=1)),
        ("yuv420", yuv420, dict(sigma=[10, 7, 7])),
        ("yuv420 chroma only", yuv420, dict(sigma=[0, 8, 8])),
        ("yuv420 final=1", yuv420, dict(sigma=[10, 5, 0], final=1)),
        ("odd width", odd, dict(sigma=10, block_step=3)),
        ("odd width final=1", odd, dict(sigma=8, final=1)),
        ("bm_range 9 step 2", gray, dict(sigma=10, bm_range=9, block_step=2)),
        ("chroma", yuv444, dict(sigma=10, chroma=1)),
        ("chroma one plane alone", yuv444, dict(sigma=[10, 0, 7], chroma=1)),
        ("chroma final=1", yuv444, dict(sigma=10, chroma=1, final=1)),
        ("chroma block 16 group 32", yuv444, dict(sigma=10, chroma=1, block_size=16, group_size=32)),
        ("memory limit", gray, dict(sigma=10, memory_limit_mb=64)),
    ]
    for label, clip, kw in others:
        check(label, cpu(clip, **kw), vk(clip, **kw), vk(clip, **kw), frames=(0, 1))
    # Wiener with a guide clip of the other filter's estimate.
    ref = cpu(yuv420, sigma=[10, 7, 7])
    check("yuv420 ref", cpu(yuv420, ref=ref, sigma=[10, 7, 7]), vk(yuv420, ref=ref, sigma=[10, 7, 7]),
          vk(yuv420, ref=ref, sigma=[10, 7, 7]))

    # The driver's batches: a few references at a time (the per-batch buffers
    # reused, the barriers between batches), a device-memory shape, chroma.
    import os
    os.environ["NSS_VK_GROUP_BATCH"] = "7"
    batched = [("batches gray", gray, dict(sigma=10)),
               ("batches 32 / 64 ref", gray, dict(sigma=10, block_size=32, group_size=64, bm_range=4, final=1)),
               ("batches chroma final", yuv444, dict(sigma=10, chroma=1, final=1)),
               ("batches yuv420", yuv420, dict(sigma=[10, 7, 7]))]
    for label, clip, kw in batched:
        check(label, cpu(clip, **kw), vk(clip, **kw), vk(clip, **kw), frames=(0, 1))
    del os.environ["NSS_VK_GROUP_BATCH"]

    # Many threads asking for frames at once, against the same frames one at a time.
    long_clip = make_clip(core, vs.GRAYS, 64, 48, 7, length=24)
    node = vk(long_clip, sigma=10, num_streams=2)
    serial = [frame_planes(node, n) for n in range(long_clip.num_frames)]
    core.num_threads = 16
    parallel = vk(long_clip, sigma=10, num_streams=2)
    futures = [parallel.get_frame_async(n) for n in range(long_clip.num_frames)]
    for n, future in enumerate(futures):
        f = future.result()
        got = [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]
        if any(not np.array_equal(x, y) for x, y in zip(got, serial[n])):
            failures.append(f"16 threads frame {n}: differs from the serial render")
    core.num_threads = 4
    cases += long_clip.num_frames

    # An unmet memory limit is a creation error with the shared text.
    big = core.std.BlankClip(width=1280, height=720, format=vs.GRAYS, length=1)
    try:
        vk(big, sigma=10, memory_limit_mb=1).get_frame(0)
        failures.append("memory_limit_mb=1: no error")
    except vs.Error as error:
        if "memory_limit_mb" not in str(error):
            failures.append(f"memory_limit_mb=1: unexpected error {error}")

    for f in failures[:40]:
        print("FAIL:", f)
    print(f"vk bm3d live: {cases} cases, worst psnr {worst:.2f} dB, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
