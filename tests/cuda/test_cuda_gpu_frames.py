#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Experimental R81 GPU-frame frontend for all nss_cuda filters.

The same plugin must keep plain vnode signatures before API 4.3. On R81, with
NSS_CUDA_GPU_FRAMES=1, every filter takes and returns GPU frames and matches its
CPU-frame path. Temporal, bypassed-plane, chained and concurrent-request paths
are included. Other cores explicitly download GPU inputs.

usage: test_cuda_gpu_frames.py --cuda PATH [--quick]
"""
import argparse
import hashlib
import os
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda gpu frames SKIP: {error}")
    raise SystemExit(77)


def make_clip(core, fmt, width, height, seed, length=8):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    yuv = blank.format.color_family == vs.YUV
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            w = width >> (blank.format.subsampling_w if p and yuv else 0)
            h = height >> (blank.format.subsampling_h if p and yuv else 0)
            y, x = np.mgrid[:h, :w]
            clean = 0.5 + 0.2 * np.sin((x + 2 * t) / (5 + p)) * np.cos((y + t) / 7)
            planes.append((clean + rng.normal(0, 8 / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def digest(node, frames):
    h = hashlib.sha256()
    for n in frames:
        f = node.get_frame(n)
        for p in range(f.format.num_planes):
            h.update(np.asarray(f[p]).tobytes())
        h.update(repr(sorted((k, v) for k, v in f.props.items() if not k.startswith("_NSSResource"))).encode())
    return h.hexdigest()


def numeric_match(a, b, frames):
    max_abs = 0.0
    squared = 0.0
    count = 0
    for n in frames:
        fa, fb = a.get_frame(n), b.get_frame(n)
        pa = sorted((k, v) for k, v in fa.props.items() if not k.startswith("_NSSResource"))
        pb = sorted((k, v) for k, v in fb.props.items() if not k.startswith("_NSSResource"))
        if pa != pb:
            return False, float("inf"), 0.0
        for p in range(fa.format.num_planes):
            aa, bb = np.asarray(fa[p]), np.asarray(fb[p])
            diff = np.abs(aa - bb)
            max_abs = max(max_abs, float(diff.max(initial=0)))
            squared += float(np.sum((aa - bb) ** 2, dtype=np.float64))
            count += diff.size
    psnr = float("inf") if squared == 0 else 10 * np.log10(count / squared)
    return max_abs <= 2e-6 and psnr >= 140, max_abs, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NSS_CUDA_GPU_FRAMES", "1")
    core = vs.core
    core.std.LoadPlugin(path=os.path.abspath(args.cuda))
    functions = {f.name: f.signature for f in core.nss_cuda.functions()}
    backend = core.nss_cuda.Backend()
    status = backend.get("gpu_frames", b"")
    status = status.decode() if isinstance(status, bytes) else status
    filters = ("BM3D", "NLM", "WNNM", "MCWNNM", "NCSR", "NLH", "TWSC", "LSSC", "DA3D", "BM4D", "PNLM")
    if core.api_version.api_minor < 3:
        bad = any(":all" in functions[name] or ":gpu" in functions[name] for name in filters) or not status.startswith("no")
        print(f"cuda gpu frames: API R{core.api_version.api_major}.{core.api_version.api_minor}, "
              f"plain signatures {'OK' if not bad else 'FAIL'}")
        return int(bad)
    wrong = [name for name in filters if not functions[name].startswith("clip:vnode:all;")]
    if wrong:
        print(f"FAIL: filters without GPU-node signatures: {wrong}")
        return 1
    if not backend.get("device_count", 0):
        print("cuda gpu frames SKIP: no CUDA device (signatures checked)")
        return 77
    try:
        usable = [d for d in core.vulkan_devices if d.get("usable")]
        reason = "VapourSynth has no usable Vulkan device"
    except (vs.Error, AttributeError) as error:
        usable, reason = [], str(error)
    if not usable:
        print(f"cuda gpu frames SKIP: {reason} (signatures checked)")
        return 77

    failures = []
    order = [0, 1, 2, 7, 4, 3]
    release = core.core_version.release_major
    if release != 81:
        src = make_clip(core, vs.GRAYS, 133, 111, 1)
        try:
            out = core.nss_cuda.NLM(src.std.GPUUpload(), d=1)
            same = digest(out, order) == digest(core.nss_cuda.NLM(src, d=1), order)
            ok = not out.gpu_resident and same
        except vs.Error as error:
            ok = False
            print(error)
        print(f"R{release} download fallback: {'OK' if ok else 'FAIL'}")
        return int(not ok)

    def compare(label, src, run):
        cpu = run(src)
        gpu = run(src.std.GPUUpload())
        if not gpu.gpu_resident:
            failures.append(f"{label}: output not GPU resident")
            return
        same, max_abs, psnr = numeric_match(cpu, gpu.std.GPUDownload(), order)
        if not same:
            failures.append(f"{label}: GPU frames differ from CPU frames (max {max_abs}, {psnr} dB)")
        print(f"{label}: {'OK' if same else 'MISMATCH'} (max {max_abs:.3g}, {psnr:.2f} dB)")

    cases = [
        (vs.GRAYS, 640, 360, dict(d=0, num_streams=2)),
        (vs.GRAYS, 333, 211, dict(d=1, num_streams=2)),
        (vs.YUV444PS, 160, 120, dict(d=1, channels="YUV")),
        (vs.YUV420PS, 160, 120, dict(d=0, channels="Y")),
        (vs.YUV420PS, 162, 122, dict(d=1, channels="UV")),
        (vs.RGBS, 96, 80, dict(d=2)),
        (vs.GRAYS, 133, 111, dict(d=1, a=20, s=12, h=6.0)),
    ]
    if args.quick:
        cases = cases[:3]
    for i, (fmt, w, h, kw) in enumerate(cases):
        compare(f"NLM {core.get_video_format(fmt).name} {w}x{h} {kw}",
                make_clip(core, fmt, w, h, i), lambda c, kw=kw: core.nss_cuda.NLM(c, **kw))

    bm3d = [
        (vs.GRAYS, 134, 112, dict(sigma=8)),
        (vs.GRAYS, 134, 112, dict(sigma=8, final=1)),
        (vs.YUV420PS, 134, 112, dict(sigma=[8, 0, 0])),
        (vs.YUV444PS, 134, 112, dict(sigma=8, chroma=1)),
        (vs.RGBS, 96, 80, dict(sigma=8, final=1)),
        (vs.GRAYS, 134, 112, dict(sigma=8, radius=1, temporal_mode="legacy")),
        (vs.YUV420PS, 134, 112, dict(sigma=[8, 0, 0], radius=1, temporal_mode="legacy")),
        (vs.GRAYS, 134, 112, dict(sigma=8, radius=1)),
        (vs.YUV420PS, 134, 112, dict(sigma=[8, 8, 0], radius=1)),
        (vs.GRAYS, 134, 112, dict(sigma=8, radius=1, final=1)),
        (vs.GRAYS, 128, 96, dict(sigma=8, radius=2, rolling_chunk=1)),
        (vs.GRAYS, 128, 96, dict(sigma=8, radius=1, final=1, temporal_mode="legacy")),
    ]
    if not args.quick:
        for i, (fmt, w, h, kw) in enumerate(bm3d):
            compare(f"BM3D {core.get_video_format(fmt).name} {w}x{h} {kw}",
                    make_clip(core, fmt, w, h, 100 + i), lambda c, kw=kw: core.nss_cuda.BM3D(c, **kw))

    group_cases = [
        ("WNNM", vs.GRAYS, dict(sigma=8, radius=0)),
        ("WNNM", vs.GRAYS, dict(sigma=8, radius=1)),
        ("MCWNNM", vs.RGBS, dict(sigma=8, radius=1, iters=2)),
        ("NCSR", vs.GRAYS, dict(sigma=8, radius=1, iters=2)),
        ("NLH", vs.GRAYS, dict(sigma=3, radius=0)),
        ("NLH", vs.GRAYS, dict(sigma=3, radius=1)),
        ("NLH", vs.YUV420PS, dict(sigma=[3, 0, 0], radius=1)),
        ("NLH", vs.YUV420PS, dict(sigma=[3, 0, 0], radius=1, temporal_mode="legacy")),
        ("TWSC", vs.GRAYS, dict(sigma=3, block_size=4, group_size=8, block_step=4, iters=2, radius=0)),
        ("TWSC", vs.GRAYS, dict(sigma=3, block_size=4, group_size=8, block_step=4, iters=2, radius=1)),
        ("LSSC", vs.GRAYS, dict(sigma=3)),
        ("LSSC", vs.YUV420PS, dict(sigma=[6, 0, 4])),
        ("LSSC", vs.RGBS, dict(sigma=8)),
        ("BM4D", vs.GRAYS, dict(sigma=8)),
        ("BM4D", vs.RGBS, dict(sigma=8)),
        ("BM4D", vs.YUV420PS, dict(sigma=[8, 0, 4])),
        ("PNLM", vs.GRAYS, dict(sigma=6)),
        ("PNLM", vs.YUV420PS, dict(sigma=[6, 0, 4])),
        ("PNLM", vs.RGBS, dict(sigma=8, d=1, a=5)),
    ]
    for i, (name, fmt, kw) in enumerate(group_cases):
        compare(f"{name} {core.get_video_format(fmt).name} {kw}", make_clip(core, fmt, 134, 112, 200 + i),
                lambda c, name=name, kw=kw: getattr(core.nss_cuda, name)(c, **kw))

    if not args.quick:
        stress_src = make_clip(core, vs.GRAYS, 64, 48, 250, length=72)
        stress_gpu = core.nss_cuda.BM4D(stress_src.std.GPUUpload(), sigma=8).std.GPUDownload()
        stress_order = [(i * 17) % stress_src.num_frames for i in range(2 * stress_src.num_frames)]
        expected = digest(stress_gpu, stress_order)
        requests = [stress_gpu.get_frame_async(n) for n in stress_order]
        actual = hashlib.sha256()
        try:
            for request in requests:
                f = request.result(timeout=120)
                for p in range(f.format.num_planes):
                    actual.update(np.asarray(f[p]).tobytes())
                actual.update(repr(sorted((k, v) for k, v in f.props.items()
                                          if not k.startswith("_NSSResource"))).encode())
        except Exception as error:
            failures.append(f"BM4D concurrent prefetch: {error}")
        else:
            same = actual.hexdigest() == expected
            print(f"BM4D concurrent prefetch: {'OK' if same else 'MISMATCH'}")
            if not same:
                failures.append("BM4D concurrent prefetch differs from sequential output")

    src = make_clip(core, vs.GRAYS, 133, 111, 20)
    ref = make_clip(core, vs.GRAYS, 133, 111, 21)
    same, max_abs, psnr = numeric_match(core.nss_cuda.NLM(src, rclip=ref, d=1),
                                       core.nss_cuda.NLM(src.std.GPUUpload(), rclip=ref.std.GPUUpload(), d=1).std.GPUDownload(), order)
    print(f"NLM rclip: {'OK' if same else 'MISMATCH'} (max {max_abs:.3g}, {psnr:.2f} dB)")
    if not same:
        failures.append("NLM rclip: GPU frames differ from CPU frames")

    same, max_abs, psnr = numeric_match(core.nss_cuda.BM3D(src, ref=ref, sigma=8),
                                       core.nss_cuda.BM3D(src.std.GPUUpload(), ref=ref.std.GPUUpload(), sigma=8).std.GPUDownload(), order)
    print(f"BM3D ref: {'OK' if same else 'MISMATCH'} (max {max_abs:.3g}, {psnr:.2f} dB)")
    if not same:
        failures.append("BM3D ref: GPU frames differ from CPU frames")

    for i, (fmt, kw) in enumerate([(vs.GRAYS, dict(sigma=3)), (vs.RGBS, dict(sigma=8)),
                                  (vs.YUV420PS, dict(sigma=[6, 0, 4]))]):
        guide = make_clip(core, fmt, 134, 112, 400 + i)
        compare(f"DA3D {core.get_video_format(fmt).name} {kw}", make_clip(core, fmt, 134, 112, 300 + i),
                lambda c, kw=kw, guide=guide: core.nss_cuda.DA3D(c, ref=guide.std.GPUUpload() if c.gpu_resident else guide, **kw))

    compare("chain BM3D final -> NLM", src,
            lambda c: core.nss_cuda.NLM(core.nss_cuda.BM3D(c, sigma=8, final=1), d=1))
    compare("chain NLM -> BM3D rolling", src,
            lambda c: core.nss_cuda.BM3D(core.nss_cuda.NLM(c, d=1), sigma=8, radius=1))
    compare("chain NLM -> NLM", src,
            lambda c: core.nss_cuda.NLM(core.nss_cuda.NLM(c, d=0), d=1, num_streams=2))

    cpu_output = core.nss_cuda.NLM(src, d=0)
    if cpu_output.gpu_resident:
        failures.append("CPU input produced a GPU frame")

    try:
        core.nss_cuda.NLM(src.std.GPUUpload(), rclip=ref).get_frame(0)
    except vs.Error as error:
        if "clip and rclip must both be CPU frames or both GPU frames" not in str(error):
            failures.append(f"mixed residency: unexpected error {error}")
        print("mixed residency: error OK")
    else:
        failures.append("mixed residency: no error")

    def poison(n, f):
        out = f.copy()
        np.asarray(out[0])[0, 0] = np.nan
        return out
    try:
        core.nss_cuda.NLH(core.std.ModifyFrame(src, src, poison).std.GPUUpload(), sigma=3).std.GPUDownload().get_frame(0)
    except vs.Error as error:
        if "nonfinite source/reference sample" not in str(error):
            failures.append(f"nonfinite input: unexpected error {error}")
        print("nonfinite input: error OK")
    else:
        failures.append("nonfinite input: no error")

    print(f"cuda gpu frames: R{release}, {len(failures)} failures")
    for failure in failures:
        print("  " + failure)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
