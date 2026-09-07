#!/usr/bin/env python3
"""Real VapourSynth zero-rank WNNM oracle; requires NSS_SO.

All valid spatial patches fit in each search window and in group_size. With
sigma=255 and residual=1, the centered spectrum is entirely removed. The
oracle independently averages every patch row and scatters that mean; it does
not call the production matcher, SVD, shrink or aggregation implementation.
"""
import json
import os

import numpy as np
import vapoursynth as vs


def main():
    core = vs.core
    core.num_threads = 2
    core.std.LoadPlugin(path=os.environ["NSS_SO"])
    results = []
    for block, width, group in ((1, 2, 2), (4, 7, 4), (8, 15, 8)):
        height = block
        values = np.tile(0.25 + (np.arange(width, dtype=np.float32) - (width - 1) / 2) / 1024,
                         (height, 1)).astype(np.float32)
        blank = core.std.BlankClip(width=width, height=height, length=3, format=vs.GRAYS)

        def fill(n, f, values=values):
            result = f.copy()
            np.asarray(result[0])[:] = values
            return result

        src = core.std.ModifyFrame(blank, blank, fill)
        patches = [values[:, x:x + block].astype(np.float64) for x in range(width - block + 1)]
        assert len(patches) == group
        mean = np.mean(patches, axis=0)
        num = np.zeros_like(values, dtype=np.float64)
        den = np.zeros_like(num)
        for x in range(len(patches)):
            num[:, x:x + block] += mean
            den[:, x:x + block] += 1
        expected = num / den
        for adaptive in (0, 1):
            output = core.nss.WNNM(src, sigma=255, residual=1, adaptive_aggregation=adaptive,
                                   block_size=block, block_step=block, group_size=group,
                                   bm_range=16, radius=0)
            for n in (2, 0, 1, 0):
                actual = np.array(output.get_frame(n)[0])
                error = float(np.max(np.abs(actual - expected)))
                if not np.isfinite(actual).all() or error > 2e-6:
                    raise AssertionError(f"b{block} g{group} adaptive={adaptive} n={n}: max_abs={error}")
                results.append(dict(block=block, group=group, adaptive=adaptive, frame=n, max_abs=error))
    print(json.dumps(dict(passed=True, cases=results), indent=2))


if __name__ == "__main__":
    main()
