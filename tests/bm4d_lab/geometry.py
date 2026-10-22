#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss.BM4D (chunks and tiles) against the prototype on the whole volume.

    NSS_SO=build/libnss.so python3 tests/bm4d_lab/geometry.py VIDEO START FRAMES WIDTH HEIGHT SIGMA
"""
import hashlib
import json
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto  # noqa: E402
from compare import load, psnr  # noqa: E402


def main():
    path, start, frames, width, height, sigma = sys.argv[1], *map(int, sys.argv[2:6]), float(sys.argv[6])
    core = vs.core
    core.num_threads = 8
    core.std.LoadPlugin(os.environ["NSS_SO"])
    clean = load(path, start, frames, width, height)
    seed = int.from_bytes(hashlib.sha256(f"bm4d-geometry:{path}:{start}:{sigma}".encode()).digest()[:8], "little")
    noisy = clean + np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape) * sigma
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=frames)

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:] = noisy[n] / 255.0
        return out
    clip = core.std.ModifyFrame(blank, blank, fill)
    plugin = np.stack([np.asarray(f[0], dtype=np.float64) * 255.0
                       for f in (core.nss.BM4D(clip, sigma=sigma).get_frame(t) for t in range(frames))])
    whole, _ = proto.bm4d(noisy, sigma)
    per_frame = [dict(frame=t, plugin=psnr(plugin[t], clean[t]), whole=psnr(whole[t], clean[t])) for t in range(frames)]
    cols = np.abs(np.arange(width) - width // 2) < 8
    print(json.dumps(dict(start=start, frames=frames, width=width, height=height, sigma=sigma,
                          plugin=psnr(plugin, clean), whole=psnr(whole, clean), agree=psnr(plugin, whole),
                          tile_seam=dict(plugin=psnr(plugin[:, :, cols], clean[:, :, cols]),
                                         whole=psnr(whole[:, :, cols], clean[:, :, cols])),
                          per_frame=per_frame)), flush=True)


if __name__ == "__main__":
    main()
