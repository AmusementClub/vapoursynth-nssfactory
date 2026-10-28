#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""VapourSynth R80+ GPU frames in nss_vk (D29, docs/vulkan-plan.md VG1-VG3).

On a core with API 4.3 (R80+): every nss_vk filter run on GPU-resident clips
(std.GPUUpload) must give the bits of the same filter on CPU frames and
return GPU-resident frames, across NLM's and BM3D's paths (tile and plane,
channel modes with shared planes, spatial / final / chroma / ref, legacy with
identity slices, rolling with chunks of one), rows padded or not, chains of
nss_vk filters that stay resident, and a legacy fat clip into the CPU
plugin's VAggregate (--cpu). Clips of mixed residency and a device_id that is
not VapourSynth's device are errors; NSS_VK_GPU_FRAMES=0 takes GPU clips as
CPU frames with the same output.

On an older core (R78, vapoursynth-classic) the same binary must load and
register plain `vnode` arguments (one binary for every core).

Exits 77 without VapourSynth or a Vulkan device VapourSynth can use.

usage: test_vk_gpu_frames.py --vk PATH [--cpu PATH] [--quick]
"""
import argparse
import hashlib
import os
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"vk gpu frames SKIP: {error}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vk", required=True)
    parser.add_argument("--cpu", help="libnss for the VAggregate case")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    core = vs.core
    log = []
    core.add_log_handler(lambda kind, message: log.append(message))
    core.std.LoadPlugin(path=os.path.abspath(args.vk))
    functions = {f.name: f.signature for f in core.nss_vk.functions()}
    filters = [name for name in ("NLM", "BM3D") if name in functions]
    status = core.nss_vk.Backend().get("gpu_frames", b"")
    status = status.decode() if isinstance(status, bytes) else status
    if core.api_version.api_minor < 3:
        # One binary for every core: plain clip arguments before API 4.3.
        bad = [name for name in filters if ":all" in functions[name] or ":gpu" in functions[name]]
        if not status.startswith("no"):
            bad.append(f"Backend gpu_frames {status!r}")
        print(f"vk gpu frames: API R{core.api_version.api_major}.{core.api_version.api_minor}, "
              f"plain signatures {'OK' if not bad else 'FAIL ' + str(bad)}")
        return 1 if bad else 0
    if core.core_version.release_major in (80, 81) and status != f"yes (R{core.core_version.release_major})":
        print(f"FAIL: Backend gpu_frames {status!r}")
        return 1
    missing = [name for name in filters if not functions[name].startswith("clip:vnode:all;")]
    if missing:
        print(f"FAIL: {missing} without vnode:all on API 4.3")
        return 1
    # Without a usable device (a loader below 1.4 raises here) there are no GPU frames.
    try:
        usable = [d for d in core.vulkan_devices if d.get("usable")]
        reason = "VapourSynth has no usable Vulkan device"
    except (vs.Error, AttributeError) as error:
        usable, reason = [], str(error)
    if not usable:
        print(f"vk gpu frames SKIP: {reason} (signatures checked)")
        return 77
    if args.cpu:
        core.std.LoadPlugin(path=os.path.abspath(args.cpu))

    failures = []
    order = [0, 1, 2, 7, 4, 3]

    def compare(label, src, run, frames=order, extra=None):
        cpu = run(src)
        gpu = run(src.std.GPUUpload())
        if not gpu.gpu_resident:
            failures.append(f"{label}: output not GPU resident")
            return
        a, b = digest(cpu, frames), digest(gpu.std.GPUDownload(), frames)
        if digest(src, frames) == a:
            failures.append(f"{label}: the case does not filter (identity)")
        if a != b:
            failures.append(f"{label}: GPU frames differ from CPU frames")
        print(f"{label}: {'OK' if a == b else 'MISMATCH'}")

    nlm = [
        (vs.GRAYS, 640, 360, dict(d=1)),
        (vs.GRAYS, 333, 211, dict(d=1)),  # padded rows: plane_copy.comp
        (vs.YUV444PS, 160, 120, dict(d=1, channels="YUV")),
        (vs.YUV420PS, 160, 120, dict(d=0, channels="Y")),  # chroma shared
        (vs.YUV420PS, 162, 122, dict(d=1, channels="UV")),  # luma shared
        (vs.RGBS, 96, 80, dict(d=2)),
        (vs.GRAYS, 133, 111, dict(d=1, a=6, s=6)),
        (vs.GRAYS, 133, 111, dict(d=1, a=20, s=12, h=6.0)),  # plane path (tile shared memory over 48 KiB)
    ]
    bm3d = [
        (vs.GRAYS, 333, 211, dict(sigma=8)),
        (vs.GRAYS, 333, 211, dict(sigma=8, final=1)),
        (vs.YUV420PS, 320, 240, dict(sigma=[8, 0, 0])),
        (vs.YUV444PS, 160, 120, dict(sigma=8, chroma=1)),
        (vs.RGBS, 160, 120, dict(sigma=8, final=1)),
        (vs.GRAYS, 133, 111, dict(sigma=8, radius=1, temporal_mode="legacy")),
        (vs.YUV420PS, 160, 120, dict(sigma=[8, 0, 0], radius=1, temporal_mode="legacy")),  # identity slices
        (vs.GRAYS, 133, 111, dict(sigma=8, radius=1)),
        (vs.YUV420PS, 162, 122, dict(sigma=[8, 8, 0], radius=1)),
        (vs.GRAYS, 133, 111, dict(sigma=8, radius=1, final=1)),
        (vs.GRAYS, 128, 96, dict(sigma=8, radius=2, rolling_chunk=1)),
        (vs.GRAYS, 128, 96, dict(sigma=8, radius=1, final=1, temporal_mode="legacy")),
    ]
    if args.quick:
        nlm, bm3d = nlm[:3], bm3d[:4]
    for i, (fmt, w, h, kw) in enumerate(nlm):
        compare(f"NLM {core.get_video_format(fmt).name} {w}x{h} {kw}", make_clip(core, fmt, w, h, i),
                lambda c, kw=kw: core.nss_vk.NLM(c, **kw))
    for i, (fmt, w, h, kw) in enumerate(bm3d):
        compare(f"BM3D {core.get_video_format(fmt).name} {w}x{h} {kw}", make_clip(core, fmt, w, h, 100 + i),
                lambda c, kw=kw: core.nss_vk.BM3D(c, **kw))

    src = make_clip(core, vs.GRAYS, 133, 111, 7)
    ref = make_clip(core, vs.GRAYS, 133, 111, 8)
    a = digest(core.nss_vk.BM3D(src, ref=ref, sigma=8), [0, 4])
    b = digest(core.nss_vk.BM3D(src.std.GPUUpload(), ref=ref.std.GPUUpload(), sigma=8).std.GPUDownload(), [0, 4])
    print(f"BM3D ref: {'OK' if a == b else 'MISMATCH'}")
    if a != b:
        failures.append("BM3D ref: GPU frames differ from CPU frames")
    rc = make_clip(core, vs.GRAYS, 133, 111, 9)
    a = digest(core.nss_vk.NLM(src, rclip=rc), [0, 4])
    b = digest(core.nss_vk.NLM(src.std.GPUUpload(), rclip=rc.std.GPUUpload()).std.GPUDownload(), [0, 4])
    print(f"NLM rclip: {'OK' if a == b else 'MISMATCH'}")
    if a != b:
        failures.append("NLM rclip: GPU frames differ from CPU frames")

    # Chains of nss_vk filters stay resident.
    compare("chain BM3D final -> NLM", src,
            lambda c: core.nss_vk.NLM(core.nss_vk.BM3D(c, sigma=8, final=1), d=1))
    compare("chain NLM -> BM3D rolling", src,
            lambda c: core.nss_vk.BM3D(core.nss_vk.NLM(c, d=1), sigma=8, radius=1))
    # A GPU fat clip into the CPU plugin's VAggregate (the core downloads it).
    if args.cpu and hasattr(core, "nss"):
        fat_cpu = core.nss_vk.BM3D(src, sigma=8, radius=1, temporal_mode="legacy")
        fat_gpu = core.nss_vk.BM3D(src.std.GPUUpload(), sigma=8, radius=1, temporal_mode="legacy")
        a = digest(core.nss.VAggregate(fat_cpu, src, radius=1), range(5))
        b = digest(core.nss.VAggregate(fat_gpu, src, radius=1), range(5))
        print(f"VAggregate of a GPU fat clip: {'OK' if a == b else 'MISMATCH'}")
        if a != b:
            failures.append("VAggregate of a GPU fat clip differs")

    # Errors.
    def expect_error(label, make, text):
        try:
            make().get_frame(0)
        except vs.Error as error:
            if text not in str(error):
                failures.append(f"{label}: unexpected error {error}")
            print(f"{label}: error OK")
            return
        failures.append(f"{label}: no error")
    expect_error("mixed residency", lambda: core.nss_vk.BM3D(src.std.GPUUpload(), ref=ref, sigma=8),
                 "clip and ref must both be CPU frames or both GPU frames")
    own_devices = core.nss_vk.Backend().get("device_count", 0)
    for device in range(own_devices):
        try:
            core.nss_vk.NLM(src.std.GPUUpload(), device_id=device).get_frame(0)
            print(f"device_id {device}: VapourSynth's device")
        except vs.Error as error:
            if "is not device_id" not in str(error):
                failures.append(f"device_id {device}: unexpected error {error}")
            print(f"device_id {device}: error OK")

    # NSS_VK_GPU_FRAMES=0: CPU frames, same output, one warning.
    os.environ["NSS_VK_GPU_FRAMES"] = "0"
    try:
        out = core.nss_vk.NLM(src.std.GPUUpload(), d=1)
        same = digest(out, [0, 3]) == digest(core.nss_vk.NLM(src, d=1), [0, 3])
        ok = not out.gpu_resident and same and any("NSS_VK_GPU_FRAMES=0" in m for m in log)
        print(f"NSS_VK_GPU_FRAMES=0: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append("NSS_VK_GPU_FRAMES=0: not CPU frames, output differs or no warning")
    finally:
        del os.environ["NSS_VK_GPU_FRAMES"]

    print(f"vk gpu frames: R{core.core_version.release_major}, {len(failures)} failures")
    for failure in failures:
        print("  " + failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
