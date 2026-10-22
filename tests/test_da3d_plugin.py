#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss.DA3D through VapourSynth: plane handling, the RGB joint path, sigma=0,
determinism under concurrent requests, frame properties, the memory limit and
a denoising sanity check.

usage: test_da3d_plugin.py --plugin PATH
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"DA3D plugin SKIP: {error}")
    raise SystemExit(77)

failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)
        print(f"FAIL: {what}")


def planes(frame):
    return [np.array(frame[p], dtype=np.float32) for p in range(frame.format.num_planes)]


def scene(w, h, n_planes, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = []
    for p in range(n_planes):
        v = 0.3 + 0.05 * p + 0.3 * (((x // 19) + (y // 13)) % 2) + 0.1 * np.sin(0.07 * x + 0.05 * y * (p + 1))
        v[(x - 50) ** 2 + (y - 40) ** 2 < 300] = 0.85 - 0.1 * p
        clean.append(v.astype(np.float32))
    return clean, rng


def clip_of(core, frames, fmt):
    """A clip whose frame n holds frames[n] (a list of planes)."""
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=len(frames))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 4
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))

    sigma = 15.0
    w, h, length = 150, 110, 4
    clean, rng = scene(w, h, 3, 1)
    noisy = [[(c + rng.standard_normal(c.shape) * sigma / 255).astype(np.float32) for c in clean] for _ in range(length)]
    guide = [[(c + rng.standard_normal(c.shape) * 0.4 * sigma / 255).astype(np.float32) for c in clean]
             for _ in range(length)]

    # Gray: improves on the guide, records its patch counts.
    g_noisy = clip_of(core, [[f[0]] for f in noisy], vs.GRAYS)
    g_guide = clip_of(core, [[f[0]] for f in guide], vs.GRAYS)
    out = core.nss.DA3D(g_noisy, g_guide, sigma=sigma)
    f = out.get_frame(0)
    mse_out = np.mean((planes(f)[0] - clean[0]) ** 2)
    mse_guide = np.mean((guide[0][0] - clean[0]) ** 2)
    print(f"gray: guide MSE {mse_guide:.3g}, DA3D {mse_out:.3g}, patches {f.props['_NSSDA3DPatches']}")
    expect(mse_out < 0.5 * mse_guide, "DA3D improves on the guide")
    expect(f.props["_NSSDA3DPatches"] > 0 and 0 <= f.props["_NSSDA3DGuideOnly"] <= f.props["_NSSDA3DPatches"],
           "patch count properties")

    # Determinism: frames requested concurrently and out of order equal a serial run.
    serial = [planes(out.get_frame(n)) for n in range(length)]
    again = core.nss.DA3D(g_noisy, g_guide, sigma=sigma)
    futures = [again.get_frame_async(n) for n in reversed(range(length))]
    concurrent = [planes(fut.result()) for fut in reversed(futures)]
    expect(all(np.array_equal(a[0], b[0]) for a, b in zip(serial, concurrent)), "concurrent frames equal serial ones")

    # YUV420: each plane on its own; luma equals the gray result; sigma=0 copies the guide's plane.
    yuv_noisy = [[f[0], f[1][::2, ::2] - 0.35, f[2][::2, ::2] - 0.4] for f in noisy]
    yuv_guide = [[f[0], f[1][::2, ::2] - 0.35, f[2][::2, ::2] - 0.4] for f in guide]
    yuv = core.nss.DA3D(clip_of(core, yuv_noisy, vs.YUV420PS), clip_of(core, yuv_guide, vs.YUV420PS), sigma=[sigma, 0, sigma])
    fy = planes(yuv.get_frame(0))
    expect(np.array_equal(fy[0], serial[0][0]), "YUV420 luma equals the gray result")
    expect(np.array_equal(fy[1], yuv_guide[0][1]), "sigma=0 passes the guide's plane through")
    expect(not np.array_equal(fy[2], yuv_guide[0][2]), "a filtered chroma plane changes")

    # RGB with one sigma is filtered jointly; with different sigmas plane by plane.
    rgb_noisy = clip_of(core, noisy, vs.RGBS)
    rgb_guide = clip_of(core, guide, vs.RGBS)
    joint = planes(core.nss.DA3D(rgb_noisy, rgb_guide, sigma=sigma).get_frame(0))
    split = planes(core.nss.DA3D(rgb_noisy, rgb_guide, sigma=[sigma, sigma, sigma * 1.0001]).get_frame(0))
    mse_joint = np.mean([(joint[p] - clean[p]) ** 2 for p in range(3)])
    mse_split = np.mean([(split[p] - clean[p]) ** 2 for p in range(3)])
    print(f"RGB: joint MSE {mse_joint:.3g}, plane by plane {mse_split:.3g}")
    expect(not np.array_equal(joint[0], split[0]), "RGB with one sigma takes the joint path")
    expect(mse_joint < np.mean([(guide[0][p] - clean[p]) ** 2 for p in range(3)]), "RGB joint improves on the guide")
    yuv444 = planes(core.nss.DA3D(clip_of(core, noisy, vs.YUV444PS), clip_of(core, guide, vs.YUV444PS),
                                  sigma=sigma).get_frame(0))
    expect(np.array_equal(yuv444[0], serial[0][0]), "YUV444 planes are filtered on their own")

    # The memory limit is a hard error, never a silent change.
    try:
        core.nss.DA3D(g_noisy, g_guide, sigma=sigma, memory_limit_mb=1).get_frame(0)
        expect(False, "memory_limit_mb=1 fails")
    except vs.Error as error:
        expect("resource" in str(error).lower() or "memory" in str(error).lower() or "budget" in str(error).lower(),
               f"memory_limit_mb=1 fails with a resource error ({error})")

    if failures:
        print(f"test_da3d_plugin: {len(failures)} failure(s)")
        return 1
    print("test_da3d_plugin: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
