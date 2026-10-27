#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.BM3D vs nss.BM3D on small random clips (docs/vulkan-plan.md V5,
V6): every legal block_size x group_size, the basic stage and the Wiener
stage with ref, final=1 in one node, chroma=1 (CBM3D), planes left alone,
YUV420 planes of two sizes, padded widths, memory_limit_mb and many threads;
then the temporal outputs (radius 1 / 2): legacy + nss.VAggregate against the
CPU, rolling against the legacy output reduced as nss.VAggregate does it,
chunks that carry on against chunks that start afresh (bit for bit: the sums
are fixed point), the chunk cache under jumps, final=1 and ref in rolling,
access orders, threads and memory limits. Each case must reach the BM3D
tolerance (tests/data/vk_tolerances_v1.json) and be run-to-run identical.
Exits 77 without VapourSynth or a supported Vulkan device.

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


class Reduced:
    """Legacy output reduced as nss.VAggregate does it (slices summed in
    ascending center order, identity rule), in IEEE float32 (as
    tests/cuda/test_cuda_bm3d_live.py)."""

    def __init__(self, fat, src, radius):
        self.fat, self.src, self.radius = fat, src, radius

    def planes(self, n):
        first, last = max(0, n - self.radius), min(self.fat.num_frames - 1, n + self.radius)
        fats = [self.fat.get_frame(c) for c in range(first, last + 1)]
        src = self.src.get_frame(n)
        out = []
        for p in range(src.format.num_planes):
            s = np.asarray(src[p])
            h = s.shape[0]
            num = np.zeros_like(s)
            den = np.zeros_like(s)
            for c, fat in zip(range(first, last + 1), fats):
                a = np.asarray(fat[p])
                k = 2 * (n - c + self.radius)
                num = num + a[k * h:(k + 1) * h]
                den = den + a[(k + 1) * h:(k + 2) * h]
            with np.errstate(divide="ignore", invalid="ignore"):
                done = np.where(den == 1, num, np.where(den > np.float32(1e-12), num / den, s))
            out.append(done.astype(np.float64))
        return out


def frame_planes(node, n=0):
    if isinstance(node, Reduced):
        return node.planes(n)
    f = node.get_frame(n)
    return [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]


def within_ulp(a, b, ulps):
    return all(np.all(np.abs(x - y) <= ulps * np.spacing(np.maximum(np.abs(x), np.abs(y)).astype(np.float32)))
               for x, y in zip(a, b))


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


# Rolling sums every center's patches in one set of fixed-point cells and
# divides once; the legacy slices are each center's cells as floats, summed in
# float by the reduction: the same terms, rounded differently.
ROLLING_ULPS = 64


def temporal(core, vk, cpu, floor, quick):
    """Legacy, rolling and their agreement; returns (cases, worst, failures)."""
    failures, worst, cases = [], float("inf"), 0
    clips = [make_clip(core, vs.GRAYS, 64, 56, 3, length=6)]
    if not quick:
        clips += [make_clip(core, vs.RGBS, 48, 40, 4, length=6), make_clip(core, vs.YUV420PS, 66, 50, 10, length=6)]
    for clip in clips:
        for radius in (1, 2):
            kw = dict(sigma=10, radius=radius, bm_range=5)
            want = core.nss.VAggregate(cpu(clip, **kw), clip, radius=radius)
            legacy = Reduced(vk(clip, temporal_mode="legacy", **kw), clip, radius)
            legacy_again = Reduced(vk(clip, temporal_mode="legacy", **kw), clip, radius)
            mixed = core.nss.VAggregate(vk(clip, temporal_mode="legacy", **kw), clip, radius=radius)
            rolling = vk(clip, rolling_chunk=4, **kw)  # rolling is the device default
            rolling_again = vk(clip, rolling_chunk=4, **kw)
            carried = {chunk: vk(clip, rolling_chunk=chunk, **kw) for chunk in (1, 2, 3)}
            fresh = vk(clip, rolling_chunk=2, **kw)
            backwards = {n: frame_planes(fresh, n) for n in reversed(range(clip.num_frames))}
            for n in range(clip.num_frames):
                label = f"temporal r{radius} {clip.format.name} frame {n}"
                a, b = frame_planes(want, n), frame_planes(legacy, n)
                value = psnr(a, b)
                worst = min(worst, value)
                cases += 1
                if value < floor:
                    failures.append(f"{label}: legacy psnr {value:.2f} < {floor}")
                if not same(b, frame_planes(legacy_again, n)):
                    failures.append(f"{label}: legacy not run-to-run identical")
                # The same summation order; the CPU's division is fast-math (2 ulp, 2026-10-03).
                if not within_ulp(frame_planes(mixed, n), b, 4):
                    failures.append(f"{label}: nss.VAggregate(nss_vk.BM3D) differs from the IEEE reduction by > 4 ulp")
                rolled = frame_planes(rolling, n)
                value = psnr(a, rolled)
                worst = min(worst, value)
                if value < floor:
                    failures.append(f"{label}: rolling psnr {value:.2f} < {floor}")
                if not within_ulp(rolled, b, ROLLING_ULPS):
                    failures.append(f"{label}: rolling differs from legacy reduced by > {ROLLING_ULPS} ulp")
                if not same(rolled, frame_planes(rolling_again, n)):
                    failures.append(f"{label}: rolling not run-to-run identical")
                # Fixed-point sums: chunks that carry on, chunks that start
                # afresh and any chunk size give the same bits.
                for chunk, node in carried.items():
                    if not same(frame_planes(node, n), backwards[n]):
                        failures.append(f"{label}: rolling chunks of {chunk} differ from chunks that start afresh")
                if not same(rolled, backwards[n]):
                    failures.append(f"{label}: rolling chunks of 4 differ from chunks that start afresh")

    # Both stages: final=1 in one rolling node against nss.BM3D's two nodes,
    # and the Wiener stage with a guide clip; chroma=1; planes left alone.
    seq = make_clip(core, vs.YUV444PS, 48, 40, 11, length=5)
    gray = clips[0]
    cases_two = [
        ("final=1 gray r1", gray, dict(sigma=10, radius=1, final=1, bm_range=4)),
        ("final=1 gray r2", gray, dict(sigma=10, radius=2, final=1, bm_range=4)),
        ("chroma r1", seq, dict(sigma=10, radius=1, chroma=1, bm_range=4)),
        ("chroma final=1 r1", seq, dict(sigma=10, radius=1, chroma=1, final=1, bm_range=4)),
        ("planes alone r1", seq, dict(sigma=[10, 0, 6], radius=1, bm_range=4)),
        ("chroma one plane alone r1", seq, dict(sigma=[10, 0, 6], radius=1, chroma=1, bm_range=4)),
    ]
    if not quick:
        cases_two += [
            ("final=1 yuv444 r2", seq, dict(sigma=[10, 6, 6], radius=2, final=1, bm_range=4)),
            ("chroma r2 16 / 32", seq, dict(sigma=10, radius=2, chroma=1, block_size=16, group_size=32, bm_range=4)),
        ]
    for label, clip, kw in cases_two:
        want = cpu(clip, temporal_mode="rolling", **kw)
        got = vk(clip, temporal_mode="rolling", **kw)
        fresh = vk(clip, temporal_mode="rolling", rolling_chunk=2, **kw)
        backwards = {n: frame_planes(fresh, n) for n in reversed(range(clip.num_frames))}
        carried = vk(clip, temporal_mode="rolling", rolling_chunk=2, **kw)
        for n in range(clip.num_frames):
            value = psnr(frame_planes(want, n), frame_planes(got, n))
            worst = min(worst, value)
            cases += 1
            if value < floor:
                failures.append(f"{label} frame {n}: rolling psnr {value:.2f} < {floor}")
            if not same(frame_planes(carried, n), backwards[n]):
                failures.append(f"{label} frame {n}: chunks that carry on differ from chunks that start afresh")
        if "final" not in kw:
            legacy_want = core.nss.VAggregate(cpu(clip, temporal_mode="legacy", **kw), clip, radius=kw["radius"])
            legacy_got = Reduced(vk(clip, temporal_mode="legacy", **kw), clip, kw["radius"])
            for n in (0, clip.num_frames // 2):
                value = psnr(frame_planes(legacy_want, n), frame_planes(legacy_got, n))
                worst = min(worst, value)
                cases += 1
                if value < floor:
                    failures.append(f"{label} frame {n}: legacy psnr {value:.2f} < {floor}")
        else:
            # The legacy intermediate of final=1 chains two nodes (a rolling basic stage).
            legacy_got = Reduced(vk(clip, temporal_mode="legacy", **kw), clip, kw["radius"])
            value = psnr(frame_planes(want, 1), frame_planes(legacy_got, 1))
            worst = min(worst, value)
            cases += 1
            if value < floor:
                failures.append(f"{label} frame 1: legacy (two nodes) psnr {value:.2f} < {floor}")
    for stage in ("basic", "final"):
        kw = dict(sigma=10, radius=1, bm_range=4)
        cpu_ref = core.nss.VAggregate(cpu(gray, **kw), gray, radius=1) if stage == "final" else None
        vk_ref = vk(gray, temporal_mode="rolling", **kw) if stage == "final" else None
        want = core.nss.VAggregate(cpu(gray, **kw, **(dict(ref=cpu_ref) if cpu_ref else {})), gray, radius=1)
        got = vk(gray, temporal_mode="rolling", **kw, **(dict(ref=vk_ref) if vk_ref else {}))
        for n in range(gray.num_frames):
            value = psnr(frame_planes(want, n), frame_planes(got, n))
            worst = min(worst, value)
            cases += 1
            if value < floor:
                failures.append(f"temporal ref {stage} frame {n}: psnr {value:.2f} < {floor}")

    # Shapes: warp and block kernels, scratch cubes, every one through the
    # predictive matcher.
    shapes = ((4, 8), (8, 16), (16, 16), (12, 16), (8, 32)) if not quick else ((8, 16),)
    for block, group in shapes:
        kw = dict(sigma=10, radius=1, block_size=block, group_size=group, bm_range=4)
        want = core.nss.VAggregate(cpu(gray, **kw), gray, radius=1)
        got = vk(gray, temporal_mode="rolling", **kw)
        value = psnr(frame_planes(want, 2), frame_planes(got, 2))
        worst = min(worst, value)
        cases += 1
        if value < floor:
            failures.append(f"temporal b{block} g{group}: psnr {value:.2f} < {floor}")

    # The chunk cache serves any request order: jumps between chunks give the
    # frames of a sequential pass, with the cache fixed at one chunk and when it grows.
    hop = make_clip(core, vs.GRAYS, 64, 56, 8, length=14)
    straight = vk(hop, sigma=10, radius=1, bm_range=4, rolling_chunk=4)
    expected = [frame_planes(straight, n) for n in range(hop.num_frames)]
    order = (0, 8, 1, 9, 4, 12, 0, 8, 13, 5, 1, 9, 2, 10, 3, 11, 6, 7)
    for label, kw in (("one chunk", dict(rolling_cache_chunks=1)), ("growing", {}), ("limit 2", dict(rolling_cache_limit=2))):
        cases += 1
        node = vk(hop, sigma=10, radius=1, bm_range=4, rolling_chunk=4, **kw)
        for n in order:
            if not same(frame_planes(node, n), expected[n]):
                failures.append(f"rolling cache {label}: frame {n} differs after jumping between chunks")
                break

    # Many threads and streams, frames asked for at once, against the serial pass.
    for streams in (1, 3):
        for label, kw in (("rolling", dict(temporal_mode="rolling", rolling_chunk=3)),
                          ("legacy", dict(temporal_mode="legacy"))):
            node = vk(hop, sigma=10, radius=1, bm_range=4, num_streams=streams, **kw)
            serial = [frame_planes(node, n) for n in range(hop.num_frames)]
            core.num_threads = 8
            parallel = vk(hop, sigma=10, radius=1, bm_range=4, num_streams=streams, **kw)
            futures = [parallel.get_frame_async(n) for n in range(hop.num_frames)]
            for n, future in enumerate(futures):
                f = future.result()
                got = [np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)]
                if not same(got, serial[n]):
                    failures.append(f"8 threads {label} {streams} stream(s) frame {n}: differs from the serial render")
            core.num_threads = 4
            cases += hop.num_frames

    # A limit changes the layout (shared state, per-frame copies, fewer
    # staging sets), never the output: the smallest limits that create.
    wide = make_clip(core, vs.YUV420PS, 640, 360, 12, length=5)
    for label, kw in (("rolling", dict(radius=1)), ("rolling final", dict(radius=1, final=1)),
                      ("legacy", dict(radius=1, temporal_mode="legacy"))):
        cases += 1
        free = [frame_planes(vk(wide, sigma=10, **kw), n) for n in (0, 3)]
        tried = 0
        for mb in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128):
            try:
                node = vk(wide, sigma=10, memory_limit_mb=mb, **kw)
                tight = [frame_planes(node, n) for n in (0, 3)]
            except vs.Error as error:
                if "memory_limit_mb" not in str(error):
                    failures.append(f"limit {label} {mb} MiB: unexpected error {error}")
                continue
            tried += 1
            if any(not same(x, y) for x, y in zip(free, tight)):
                failures.append(f"limit {label}: output at memory_limit_mb={mb} differs from the unlimited output")
            if tried == 3:
                break
        if tried == 0:
            failures.append(f"limit {label}: no limit up to 128 MiB creates")
    return cases, worst, failures


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

    t_cases, t_worst, t_failures = temporal(core, vk, cpu, floor, args.quick)
    cases += t_cases
    worst = min(worst, t_worst)
    failures += t_failures

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
