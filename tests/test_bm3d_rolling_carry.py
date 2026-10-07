#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""BM3D temporal_mode="rolling": chunks in order carry on from one another.

A chunk that follows the last one computed starts from the sums that chunk
left for it instead of running its first centers again. Whatever the order
of the requests, the frames must equal, bit for bit, legacy + VAggregate:
checked in order (every chunk but the first carries on), backwards (every
chunk starts afresh) and in a random order, for chunks shorter and longer
than the window, a clip that ends inside a chunk, a reference clip, YUV
4:2:0 and YUV 4:4:4 with chroma.

usage: test_bm3d_rolling_carry.py --plugin PATH
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"bm3d rolling carry SKIP: {error}")
    raise SystemExit(77)

WIDTH, HEIGHT = 48, 40


def make_clip(core, fmt, frames, seed):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=fmt, length=frames)
    data = []
    for t in range(frames):
        planes = []
        for p in range(blank.format.num_planes):
            w = WIDTH >> (blank.format.subsampling_w if p else 0)
            h = HEIGHT >> (blank.format.subsampling_h if p else 0)
            y, x = np.mgrid[:h, :w]
            clean = (0.5 if p == 0 else 0.0) + 0.2 * np.sin((x + 2 * t) / 5) * np.cos((y + t) / 7)
            planes.append((clean + rng.normal(0, 6 / 255, clean.shape)).astype(np.float32))
        data.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(data[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def planes_of(frame):
    return [np.array(frame[p]) for p in range(frame.format.num_planes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(path=args.plugin)
    light = dict(sigma=5.0, block_step=4, bm_range=4, ps_num=2, ps_range=2)
    gray = make_clip(core, vs.GRAYS, 11, 1)
    cases = [
        ("gray r1 chunk4", gray, dict(radius=1)),
        ("gray r1 chunk1", gray, dict(radius=1, rolling_chunk=1)),
        ("gray r2 chunk2", gray, dict(radius=2, rolling_chunk=2)),
        ("gray r2 chunk3", gray, dict(radius=2, rolling_chunk=3)),
        ("gray r3 chunk8", gray, dict(radius=3, rolling_chunk=8)),
        ("gray r1 ref", gray, dict(radius=1, rolling_chunk=2, ref=make_clip(core, vs.GRAYS, 11, 2))),
        ("gray r5 short clip", make_clip(core, vs.GRAYS, 4, 3), dict(radius=5, rolling_chunk=1)),
        ("yuv420 r1", make_clip(core, vs.YUV420PS, 10, 4), dict(radius=1, rolling_chunk=3)),
        ("yuv420 r1 luma only", make_clip(core, vs.YUV420PS, 10, 4), dict(radius=1, rolling_chunk=3, sigma=[5.0, 0.0, 0.0])),
        ("yuv444 r1 chroma", make_clip(core, vs.YUV444PS, 10, 5), dict(radius=1, rolling_chunk=3, chroma=1)),
        ("yuv444 r2 chroma", make_clip(core, vs.YUV444PS, 10, 5), dict(radius=2, rolling_chunk=1, chroma=1, sigma=[5.0, 0.0, 4.0])),
    ]
    failures = 0
    for label, clip, kw in cases:
        kw = {**light, **kw}
        chained = {k: v for k, v in kw.items() if k != "rolling_chunk"}
        expected_node = core.nss.VAggregate(core.nss.BM3D(clip, temporal_mode="legacy", **chained), clip, radius=kw["radius"])
        expected = [planes_of(expected_node.get_frame(n)) for n in range(clip.num_frames)]
        rng = np.random.default_rng(7)
        orders = {"in order": list(range(clip.num_frames)), "backwards": list(range(clip.num_frames - 1, -1, -1)),
                  "random": [int(n) for n in rng.integers(0, clip.num_frames, 3 * clip.num_frames)]}
        for name, order in orders.items():
            node = core.nss.BM3D(clip, temporal_mode="rolling", rolling_cache_chunks=1, **kw)
            bad = [n for n in order
                   if any(not np.array_equal(a, b) for a, b in zip(planes_of(node.get_frame(n)), expected[n]))]
            if bad:
                failures += 1
                print(f"FAIL {label}, {name}: frames {sorted(set(bad))} differ from legacy + VAggregate")
    print(f"bm3d rolling carry: {len(cases)} cases, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
