#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""D30 create/render/free soak for the three r3 Vulkan filters.

CPU frames run even on pre-4.3 cores. GPU frames are checked separately on
supported cores, including residency, downloaded pixels and repeat hashes.
Use --require-gpu for release admission: an unavailable GPU path then fails
instead of silently reducing coverage. CTest uses a short pass; use
--iterations 500 --require-gpu for a longer soak. This checks lifetimes and
output stability, not a quantitative device-memory leak bound.
"""
import argparse
import gc
import hashlib
import os
import sys


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def render(node, gpu, np):
    """Keep several requests in flight; release frames/futures before returning."""
    if gpu:
        if not node.gpu_resident:
            raise RuntimeError("GPU input produced CPU output")
        node = node.std.GPUDownload()
    hashes = []
    # Scrambled order includes both temporal boundaries and a repeated request.
    futures = [node.get_frame_async(i) for i in (0, 3, 1, 2, 0)]
    for future in futures:
        frame = future.result()
        h = hashlib.sha256()
        for p in range(frame.format.num_planes):
            plane = np.asarray(frame[p])
            if not np.isfinite(plane).all():
                raise RuntimeError("non-finite output")
            h.update(plane.tobytes())
        hashes.append(h.hexdigest())
    if hashes[0] != hashes[-1]:
        raise RuntimeError("repeat frame changed")
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--iterations", type=positive, default=25)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        import numpy as np
        import vapoursynth as vs
    except ImportError as error:
        print(f"vk stress {'FAIL' if args.require_gpu else 'SKIP'}: {error}")
        return 1 if args.require_gpu else 77

    core = vs.core
    core.std.LoadPlugin(path=os.path.abspath(args.plugin))
    # Backend errors/probe failures on a supported device must fail the test.
    info = core.nss_vk.Backend()
    if not info["device_count"] or text(info.get("support", "")) == "unsupported":
        print(f"vk stress {'FAIL' if args.require_gpu else 'SKIP'}: {info}")
        return 1 if args.require_gpu else 77
    if not info.get("probe_ok"):
        raise RuntimeError(f"Vulkan probe failed: {info}")

    gpu = text(info.get("gpu_frames", "")).startswith("yes")
    reason = text(info.get("gpu_frames", "unavailable"))
    if gpu:
        try:
            gpu = any(d.get("usable") for d in core.vulkan_devices)
            if not gpu:
                reason = "core has no usable Vulkan device"
        except (vs.Error, AttributeError) as error:
            gpu, reason = False, str(error)
    if args.require_gpu and not gpu:
        print(f"vk stress FAIL: GPU frames required: {reason}")
        return 1
    if not gpu:
        print(f"vk stress: CPU frames only ({reason})", flush=True)

    # No constant identity fixture: finite noisy data, padded rows, chroma and
    # temporal boundaries exercise transfers, retained windows and chunk stores.
    from test_vk_gpu_frames import make_clip
    sources = [make_clip(core, fmt, w, h, 104 + i, length=4)
               for i, (fmt, w, h) in enumerate(((vs.GRAYS, 133, 97),
                                               (vs.YUV420PS, 134, 98),
                                               (vs.RGBS, 133, 97)))]
    specs = (
        ("NLM", lambda clip: core.nss_vk.NLM(clip, d=1)),
        ("BM3D", lambda clip: core.nss_vk.BM3D(clip, sigma=8)),
        ("BM3D rolling", lambda clip: core.nss_vk.BM3D(clip, sigma=8, radius=1,
                                                    temporal_mode="rolling")),
        ("BM3D legacy", lambda clip: core.nss_vk.BM3D(clip, sigma=8, radius=1,
                                                   temporal_mode="legacy")),
        ("PNLM", lambda clip: core.nss_vk.PNLM(clip, sigma=6)),
    )
    baselines, lifetimes = {}, 0
    for iteration in range(args.iterations):
        for index, source in enumerate(sources):
            for name, make in specs:
                cpu_hashes = render(make(source), False, np)
                key = (index, name)
                if cpu_hashes != baselines.setdefault(key, cpu_hashes):
                    raise RuntimeError(f"{name}, format {index}: output changed in cycle {iteration}")
                lifetimes += 1
                if gpu:
                    gpu_hashes = render(make(source.std.GPUUpload()), True, np)
                    if gpu_hashes != cpu_hashes:
                        raise RuntimeError(f"{name}, format {index}: CPU/GPU-frame outputs differ")
                    lifetimes += 1
        gc.collect()
        if iteration % 10 == 0:
            print(f"vk stress cycle {iteration + 1}: {lifetimes} filter lifetimes", flush=True)
    print(f"vk stress OK: {lifetimes} filter lifetimes; gpu_frames={gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
