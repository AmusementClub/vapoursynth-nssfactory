#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Default-parameter smoke: every public filter must produce a well-formed
finite frame. Plugin path comes from NSS_SO (or argv[1])."""
import os
import sys

import numpy as np
import vapoursynth as vs

PLUGIN = os.environ.get("NSS_SO") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not PLUGIN:
    raise SystemExit("usage: NSS_SO=path/to/libnss smoke_defaults.py")
core = vs.core
core.std.LoadPlugin(path=PLUGIN)

W, H, N = 96, 64, 3
rng = np.random.default_rng(7)

def noisy(fmt, planes):
    clip = core.std.BlankClip(width=W, height=H, format=fmt, length=N, fpsnum=24)
    def fill(n, f):
        fout = f.copy()
        for p in range(planes):
            plane = np.asarray(fout[p])
            h, w = fout.height, fout.width
            if plane.ndim == 1:
                plane = plane.reshape(h, plane.shape[0] // h)
            plane[:, :w] = rng.normal(0.5, 0.05, (h, w)).astype(np.float32)
        return fout
    return core.std.ModifyFrame(clip, clip, fill)

gray = noisy(vs.GRAYS, 1)
rgb = noisy(vs.RGBS, 3)

failures = []

def check(name, node, ref):
    try:
        f = node.get_frame(0)
        assert (f.width, f.height, f.format.id) == (ref.width, ref.height, ref.format.id), \
            f"shape {f.width}x{f.height}/{f.format.name} != ref"
        for p in range(f.format.num_planes):
            plane = np.asarray(f[p])
            h, w = f.height, f.width
            if plane.ndim == 1:
                plane = plane.reshape(h, plane.shape[0] // h)
            assert np.isfinite(plane[:, :w]).all(), f"non-finite values in plane {p}"
        print(f"OK   {name:16s} {f.width}x{f.height} {f.format.name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL {name:16s} {type(e).__name__}: {e}")

check("NLM", core.nss.NLM(gray), gray)
check("BM3D", core.nss.BM3D(gray), gray)
check("BM3D-RGB", core.nss.BM3D(rgb), rgb)
check("WNNM", core.nss.WNNM(gray), gray)
check("MCWNNM-RGB", core.nss.MCWNNM(rgb), rgb)
check("TWSC", core.nss.TWSC(gray), gray)
check("TWSC-RGB", core.nss.TWSC(rgb), rgb)
check("NLH", core.nss.NLH(gray), gray)
check("NLH-RGB", core.nss.NLH(rgb), rgb)
check("NCSR", core.nss.NCSR(gray), gray)
check("LSSC", core.nss.LSSC(gray), gray)
fat = core.nss.BM3D(gray, radius=1)
check("VAggregate", core.nss.VAggregate(fat, gray, radius=1), gray)
print("Backend:", core.nss.Backend())

if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("SMOKE PASS")
