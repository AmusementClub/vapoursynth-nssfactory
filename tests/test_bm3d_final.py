#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""BM3D final=1: both stages in one call.

final=1 must give, bit for bit, what the two calls it stands for give:
BM3D(clip, ref=BM3D(clip, <basic values>), ...), where the basic call takes
sigma_basic, block_size_basic and group_size_basic in place of sigma,
block_size and group_size and returns finished frames. Checked for Gray,
YUV 4:2:0 and YUV 4:4:4 with chroma, spatial and temporal (both modes), and
for the argument errors.

usage: test_bm3d_final.py --plugin PATH [--namespace NS]
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"bm3d final SKIP: {error}")
    raise SystemExit(77)

WIDTH, HEIGHT, FRAMES = 64, 48, 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--namespace", default="nss")
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(path=args.plugin)
    ns = args.namespace
    plugin = getattr(core, ns)
    if "BM3D" not in plugin.__dir__():
        print(f"bm3d final SKIP: {ns} has no BM3D")
        return 77
    if ns != "nss" and plugin.Backend().get("device_count", 1) == 0:
        print(f"bm3d final SKIP: {ns} has no device")
        return 77
    bm3d = plugin.BM3D
    rng = np.random.default_rng(23)

    def clip(fmt):
        blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=fmt, length=FRAMES)
        data = []
        for t in range(FRAMES):
            planes = []
            for p in range(blank.format.num_planes):
                h = HEIGHT >> (blank.format.subsampling_h if p else 0)
                w = WIDTH >> (blank.format.subsampling_w if p else 0)
                y, x = np.mgrid[:h, :w]
                planes.append((0.5 + 0.2 * np.sin((x + 2 * t) / (5 + p)) * np.cos((y + t) / 7) +
                               rng.normal(0, 10 / 255, (h, w))).astype(np.float32))
            data.append(planes)

        def fill(n, f):
            out = f.copy()
            for p, plane in enumerate(data[n]):
                np.asarray(out[p])[:] = plane
            return out
        return core.std.ModifyFrame(blank, blank, fill)

    def planes(node, n):
        f = node.get_frame(n)
        return [np.array(f[p]) for p in range(f.format.num_planes)]

    failures, cases = [], 0
    basics = ({}, dict(sigma_basic=14), dict(block_size_basic=4), dict(group_size_basic=16),
              dict(sigma_basic=[12, 0, 9], block_size_basic=16, group_size_basic=4))
    shapes = ({}, dict(block_size=4, block_step=3), dict(block_size=16, group_size=32, block_step=8, bm_range=5))
    for fmt, extras in ((vs.GRAYS, ({},)), (vs.YUV420PS, ({}, dict(sigma=[10, 0, 6]))),
                        (vs.YUV444PS, (dict(chroma=1), dict(chroma=1, sigma=[0, 10, 10])))):
        src = clip(fmt)
        for extra in extras:
            for shape in shapes:
                for basic in basics:
                    if "block_step" in shape and basic.get("block_size_basic", 99) < shape["block_step"]:
                        continue
                    for temporal in ({}, dict(radius=1), dict(radius=2, rolling_chunk=3, temporal_mode="rolling"),
                                     dict(radius=1, temporal_mode="legacy")):
                        if temporal and (shape or len(basic) > 1) and fmt != vs.GRAYS:
                            continue  # the temporal paths once per format is enough
                        kw = dict(dict(sigma=10), **extra, **shape, **temporal)
                        first = dict(kw)
                        for name in ("sigma", "block_size", "group_size"):
                            if f"{name}_basic" in basic:
                                first[name] = basic[f"{name}_basic"]
                        if temporal:
                            first["temporal_mode"] = "rolling"
                        label = f"{src.format.name} {kw} {basic}"
                        try:
                            one = bm3d(src, final=1, **kw, **basic)
                            two = bm3d(src, ref=bm3d(src, **first), **kw)
                        except vs.Error as error:
                            failures.append(f"{label}: {error}")
                            continue
                        cases += 1
                        if (one.width, one.height, one.num_frames) != (two.width, two.height, two.num_frames):
                            failures.append(f"{label}: output shape differs")
                            continue
                        order = (0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 7) if temporal else (0, 4)
                        for n in order:
                            if any(not np.array_equal(a, b) for a, b in zip(planes(one, n), planes(two, n))):
                                failures.append(f"{label} frame {n}: final=1 differs from the two calls")
                                break
    gray = clip(vs.GRAYS)
    errors = ((dict(final=2), "final must be 0 or 1"),
              (dict(final=1, ref=gray), "use only one of final and ref"),
              (dict(sigma_basic=5), "sigma_basic, block_size_basic and group_size_basic require final=1"),
              (dict(block_size_basic=4), "sigma_basic, block_size_basic and group_size_basic require final=1"),
              (dict(group_size_basic=4, ref=gray), "sigma_basic, block_size_basic and group_size_basic require final=1"),
              (dict(final=1, sigma_basic=-1), "sigma_basic must be finite and non-negative"),
              (dict(final=1, block_size_basic=3), "block_size_basic must be one of 1, 2, 4, 8, 12, 16, 32"),
              (dict(final=1, group_size_basic=3), "group_size_basic must be one of 1, 2, 4, 8, 16, 32, 64"),
              (dict(final=1, block_size_basic=4, block_step=8), "block_step must be in [1, block_size_basic]"),
              (dict(final=1, group_size_basic=2, ps_num=4), "ps_num must be in [1, group_size_basic]"))
    for kw, message in errors:
        cases += 1
        try:
            bm3d(gray, **kw)
            failures.append(f"{kw}: no error")
        except vs.Error as error:
            if str(error) != f"{ns}.BM3D: {message}":
                failures.append(f"{kw}: {error}")
    for failure in failures:
        print("FAIL:", failure)
    print(f"bm3d final ({ns}): {cases} cases, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
