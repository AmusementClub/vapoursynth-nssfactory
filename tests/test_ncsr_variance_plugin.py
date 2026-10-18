#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""NCSR v3: low-noise continuity, actual-count padding, temporal model identity.

Synthetic fixtures require no external images. Runs independently on either
backend; CPU/GPU agreement alone cannot detect the shared v2 variance bug.
"""
import argparse
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"NCSR variance SKIP: {error}")
    raise SystemExit(77)


def pixels(clip, n=0):
    frame = clip.get_frame(n)
    return np.stack([np.asarray(frame[p]).copy() for p in range(frame.format.num_planes)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--namespace", default="nss", choices=("nss", "nss_cuda"))
    parser.add_argument("--cpu", help="CPU plugin for VAggregate when testing CUDA")
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 2
    if args.cpu:
        core.std.LoadPlugin(path=str(Path(args.cpu).resolve()))
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    ns = getattr(core, args.namespace)
    if args.namespace == "nss_cuda" and ns.Backend()["device_count"] == 0:
        print("NCSR variance SKIP: no CUDA device")
        return 77

    count = 0
    for fmt in (vs.GRAYS, vs.RGBS):
        blank = core.std.BlankClip(width=32, height=32, length=3, format=fmt)
        y, x = np.mgrid[:32, :32]
        # Narrow bright strokes over a nonrepeating background, with no noise.
        base = (0.12 + ((x * 17 + y * 31) % 97) / 400).astype(np.float32)
        base[((x % 11 == 3) & (y > 3)) | ((y == 9) & (x > 5))] = 0.9
        source = np.stack([base * (1 - 0.1 * p) for p in range(blank.format.num_planes)])

        def fill(n, f):
            out = f.copy()
            for p, plane in enumerate(source):
                np.asarray(out[p])[:] = plane
            return out

        clip = core.std.ModifyFrame(blank, blank, fill)
        for sigma in (0, 0.1):
            output = pixels(ns.NCSR(clip, sigma=sigma))
            error = float(np.max(np.abs(output - source)))
            assert np.isfinite(output).all()
            if sigma == 0:
                assert np.array_equal(output, source), "sigma zero must be exact identity"
            else:
                # 0.1 gray-level noise cannot justify copying unrelated strokes.
                assert error < 0.5 / 255, f"low-sigma stroke damage: max error {error * 255:.6f}"
            assert np.array_equal(output, pixels(ns.NCSR(clip, sigma=sigma))), "nondeterministic output"
            count += 1

        # bm_range=1 supplies at most nine actual matches. Increasing capacity
        # from 9 to 16/32 must not add virtual observations to the second moment.
        for block in (1, 4, 8):
            kw = dict(sigma=10, block_size=block, block_step=block, bm_range=1, iters=1)
            reference = pixels(ns.NCSR(clip, group_size=9, **kw))
            for capacity in (16, 32):
                output = pixels(ns.NCSR(clip, group_size=capacity, **kw))
                assert np.isfinite(output).all() and np.isfinite(reference).all()
                np.testing.assert_allclose(output, reference, rtol=2e-4, atol=2e-5,
                                           err_msg=f"padded group changed output: block={block} capacity={capacity}")
                count += 1

        for sigma in (0, 0.1):
            raw = ns.NCSR(clip, sigma=sigma, radius=1, temporal_mode="legacy")
            for n in (0, 2, 1):
                f = raw.get_frame(n)
                assert f.props['_NSSModel'] == 5
                assert f.props['_NSSModelVersion'] == 3
                assert f.props['_NSSFatVersion'] == 2
            aggregate = core.nss.VAggregate(raw, clip, radius=1)
            for n in (0, 2, 1):
                output = pixels(aggregate, n)
                assert np.isfinite(output).all()
                if sigma == 0:
                    assert np.array_equal(output, source)
            stale = core.std.SetFrameProps(raw, _NSSModelVersion=2)
            try:
                core.nss.VAggregate(stale, clip, radius=1).get_frame(1)
            except vs.Error as error:
                assert "incompatible contribution identity" in str(error), str(error)
            else:
                raise AssertionError("VAggregate accepted NCSR v2 contributions")
            count += 1
    print(f"NCSR v3 {args.namespace}: {count} checks passed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
