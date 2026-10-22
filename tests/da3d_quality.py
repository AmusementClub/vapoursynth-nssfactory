#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""DA3D gain over its guide on the ds/ images: one case per process.

For one image, colour kind (gray: BT.601 luma; rgb) and sigma (8-bit), adds
seeded AWGN, makes the guide with nss.BM3D (both stages, sigma x 4/3, the
factor that suits BM3D at its defaults), runs nss.DA3D with the true sigma on
one thread and prints one JSON line: PSNR of the noisy input, the guide and
DA3D against the clean image, DA3D's time on one thread and its patch counts.

    NSS_SO=build/libnss.so python3 tests/da3d_quality.py MAPPA.png gray 10
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs
from PIL import Image

DS = Path(__file__).resolve().parents[1] / "ds"


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))


def main():
    image, kind, sigma = sys.argv[1], sys.argv[2], float(sys.argv[3])
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(os.environ["NSS_SO"])
    with Image.open(DS / image) as pic:
        rgb = np.asarray(pic.convert("RGB"), dtype=np.float64)
    clean = rgb if kind == "rgb" else (rgb @ np.array([0.299, 0.587, 0.114]))[:, :, None]
    seed = int.from_bytes(hashlib.sha256(f"da3d-quality:{image}:{kind}:{sigma}".encode()).digest()[:8], "little")
    noisy = (clean + np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape) * sigma).astype(np.float32)
    h, w, c = noisy.shape
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGBS if c == 3 else vs.GRAYS, length=1)

    def fill(n, f):
        out = f.copy()
        for p in range(c):
            np.asarray(out[p])[:] = noisy[:, :, p] / 255.0
        return out
    clip = core.std.ModifyFrame(blank, blank, fill)
    g = core.nss.BM3D(clip, sigma=[sigma * 4 / 3] * c, final=1).get_frame(0)
    guide = core.std.ModifyFrame(blank, blank, lambda n, f: g)
    t0 = time.perf_counter()
    f = core.nss.DA3D(clip, guide, sigma=[sigma] * c).get_frame(0)
    seconds = time.perf_counter() - t0
    as_img = lambda fr: np.stack([np.asarray(fr[p]) * 255.0 for p in range(c)], axis=2)
    print(json.dumps(dict(image=image, kind=kind, sigma=sigma, noisy=psnr(noisy, clean), guide=psnr(as_img(g), clean),
                          da3d=psnr(as_img(f), clean), seconds=seconds, patches=f.props["_NSSDA3DPatches"],
                          guide_only=f.props["_NSSDA3DGuideOnly"], pixels=w * h)), flush=True)


if __name__ == "__main__":
    main()
