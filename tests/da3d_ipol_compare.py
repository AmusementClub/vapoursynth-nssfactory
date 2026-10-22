#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss.DA3D against the IPOL reference implementation, same inputs and tile grid.

The reference is https://github.com/gfacciol/da3d at commit 642f592
(BSD-2-Clause) with tests/da3d_ref/ipol-tiles.patch applied: the patch adds the
fixed tile grid of nss.DA3D (environment DA3D_TILE=128) and prints the patch
counts. Build it with FFTW 3 (single precision) and OpenMP, e.g.

    git clone https://github.com/gfacciol/da3d && cd da3d && git checkout 642f592
    patch -p1 < .../tests/da3d_ref/ipol-tiles.patch
    g++ -O3 -fopenmp -I. DA3D.cpp WeightMap.cpp Utils.cpp DemoUtils.cpp main.cpp iio.c -lfftw3f -o da3d

(iio.h enables libpng/libjpeg/libtiff; comment those defines out when the
headers are missing: this script only uses PFM input and text output).

Inputs are ds/ images with seeded AWGN; the guide is nss.BM3D with both
stages at 4/3 sigma. Reported per case: PSNR of the guide, the reference and
nss.DA3D against the clean image, the PSNR between the two outputs, their
largest and mean absolute difference (8-bit units), and the patch counts.
Gate: the two PSNRs against the clean image agree within --psnr-tol dB and the
outputs agree to --agree-db dB or more.

    NSS_SO=build/libnss.so python3 tests/da3d_ipol_compare.py --reference /path/to/da3d
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "ds"
CASES = [  # image, kind, sigma (8-bit)
    ("MAPPA.png", "gray", 10.0),
    ("Ufotable.png", "gray", 25.0),
    ("David Production.png", "gray", 5.0),
    ("Ufotable.png", "rgb", 10.0),
    ("MAPPA.png", "rgb", 5.0),
]


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))


def write_pfm(path, arr):  # HxWxC, top row first, little endian (as iio reads it)
    h, w, c = arr.shape
    with open(path, "wb") as f:
        f.write(f"{'PF' if c == 3 else 'Pf'}\n{w} {h}\n-1.0\n".encode())
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def read_text_image(path, h, w, c):  # iio's fallback writer: P2/P3 with %a or %g samples
    with open(path) as f:
        for _ in range(3):
            f.readline()
        vals = [float.fromhex(t) if "x" in t else float(t) for t in f.read().split()]
    return np.array(vals, dtype=np.float32).reshape(h, w, c)


def load(image, kind, sigma):
    from PIL import Image
    with Image.open(DS / image) as pic:
        rgb = np.asarray(pic.convert("RGB"), dtype=np.float64)
    clean = rgb if kind == "rgb" else (rgb @ np.array([0.299, 0.587, 0.114]))[:, :, None]
    seed = int.from_bytes(hashlib.sha256(f"da3d-ipol:{image}:{kind}:{sigma}".encode()).digest()[:8], "little")
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape) * sigma
    return clean, (clean + noise).astype(np.float32)


def run_nss(core, vs, noisy, sigma):
    h, w, c = noisy.shape
    fmt = vs.RGBS if c == 3 else vs.GRAYS
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=1)

    def fill(n, f):
        out = f.copy()
        for p in range(c):
            np.asarray(out[p])[:] = noisy[:, :, p] / 255.0
        return out
    clip = core.std.ModifyFrame(blank, blank, fill)
    guide = core.nss.BM3D(clip, sigma=[sigma * 4 / 3] * c, final=1)
    da3d = core.nss.DA3D(clip, guide, sigma=[sigma] * c)
    g = guide.get_frame(0)
    gimg = np.stack([np.asarray(g[p]) * 255.0 for p in range(c)], axis=2).astype(np.float32)
    t0 = time.perf_counter()
    f = da3d.get_frame(0)
    dt = time.perf_counter() - t0
    out = np.stack([np.asarray(f[p]) * 255.0 for p in range(c)], axis=2)
    return gimg, out, dt, f.props["_NSSDA3DPatches"], f.props["_NSSDA3DGuideOnly"]


def run_reference(binary, noisy, guide, sigma, tmp, threads):
    h, w, c = noisy.shape
    write_pfm(tmp / "noisy.pfm", noisy)
    write_pfm(tmp / "guide.pfm", guide)
    env = dict(os.environ, DA3D_TILE="128", OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    r = subprocess.run([binary, str(tmp / "noisy.pfm"), str(tmp / "guide.pfm"), repr(sigma), str(tmp / "out.txt")],
                       env=env, capture_output=True, text=True, check=True)
    dt = time.perf_counter() - t0
    patches = sum(int(m) for m in re.findall(r"iter (\d+)", r.stderr))
    guide_only = sum(int(m) for m in re.findall(r"guide_only (\d+)", r.stderr))
    return read_text_image(tmp / "out.txt", h, w, c), dt, patches, guide_only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="the patched IPOL da3d binary")
    ap.add_argument("--plugin", default=os.environ.get("NSS_SO"), help="libnss (default: $NSS_SO)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--psnr-tol", type=float, default=0.02)
    ap.add_argument("--agree-db", type=float, default=50.0)
    args = ap.parse_args()
    import vapoursynth as vs
    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for image, kind, sigma in CASES:
            clean, noisy = load(image, kind, sigma)
            guide, ours, t_ours, p_ours, go_ours = run_nss(core, vs, noisy, sigma)
            ref, t_ref, p_ref, go_ref = run_reference(args.reference, noisy, guide, sigma, tmp, args.threads)
            diff = np.abs(ours.astype(np.float64) - ref)
            p_g, p_r, p_o, agree = psnr(guide, clean), psnr(ref, clean), psnr(ours, clean), psnr(ours, ref)
            good = abs(p_o - p_r) <= args.psnr_tol and agree >= args.agree_db
            ok &= good
            print(f"{image} {kind} sigma {sigma:g}: guide {p_g:.3f} reference {p_r:.3f} nss {p_o:.3f} dB "
                  f"(diff {p_o - p_r:+.3f}); outputs agree {agree:.1f} dB, max {diff.max():.3f} mean {diff.mean():.4f}; "
                  f"patches ref {p_ref} ({go_ref} guide only) nss {p_ours} ({go_ours}); "
                  f"time ref {t_ref:.2f} s ({args.threads} threads, with file IO) nss {t_ours:.2f} s (1 thread) "
                  f"{'ok' if good else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
