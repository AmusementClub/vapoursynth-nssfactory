#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""temporal_mode="rolling" against legacy + VAggregate for every temporal filter.

Rolling returns normalized frames with the source's size. Filters whose rolling
mode is legacy + VAggregate inside the plugin must match the explicit chain bit
for bit; a filter with its own rolling path (--native-rolling, a comma list) is held to
--min-psnr instead. radius 0 ignores the mode.

usage: test_temporal_rolling.py --plugin PATH [--namespace NS] [--native-rolling F1,F2] [--min-psnr DB]
"""
import argparse
import sys

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"temporal rolling SKIP: {error}")
    raise SystemExit(77)

WIDTH, HEIGHT, FRAMES = 48, 40, 5
LIGHT = dict(block_step=4, bm_range=4, ps_range=2)
CASES = (
    ("BM3D", vs.GRAYS, dict(sigma=5.0, radius=1)),
    ("WNNM", vs.GRAYS, dict(sigma=5.0, radius=1, **LIGHT)),
    ("WNNM", vs.GRAYS, dict(sigma=5.0, radius=2, rolling_chunk=2, **LIGHT)),
    ("MCWNNM", vs.RGBS, dict(sigma=5.0, radius=1, iters=1, **LIGHT)),
    ("NCSR", vs.GRAYS, dict(sigma=5.0, radius=1, iters=1, **LIGHT)),
    ("NCSR", vs.GRAYS, dict(sigma=5.0, radius=1, iters=2, **LIGHT)),
    ("NLH", vs.GRAYS, dict(sigma=5.0, radius=1)),
    ("TWSC", vs.GRAYS, dict(sigma=5.0, radius=1, block_step=6, group_size=8, iters=1)),
)


def source(core, fmt):
    rng = np.random.default_rng(42)
    blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=fmt, length=FRAMES)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    planes = [[(0.5 + 0.25 * np.sin((x + 2 * t + 5 * p) / 7) * np.cos((y + t) / 5)
                + rng.normal(0, 5 / 255, x.shape)).astype(np.float32) for t in range(FRAMES)]
              for p in range(blank.format.num_planes)]

    def fill(n, f):
        out = f.copy()
        for p in range(len(planes)):
            np.asarray(out[p])[:] = planes[p][n]
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def frames(node):
    return [[np.asarray(f[p]).copy() for p in range(f.format.num_planes)] for f in node.frames()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--namespace", default="nss")
    parser.add_argument("--native-rolling", default="")
    parser.add_argument("--min-psnr", type=float, default=80.0)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(path=args.plugin)
    ns = getattr(core, args.namespace)
    if args.namespace != "nss" and ns.Backend()["device_count"] == 0:
        print("temporal rolling SKIP: no CUDA device")
        return 77
    device = set(filter(None, args.native_rolling.split(",")))
    failures = 0
    for name, fmt, kw in CASES:
        src = source(core, fmt)
        make = getattr(ns, name)
        rolling = make(src, temporal_mode="rolling", **kw)
        chain = ns.VAggregate(make(src, **{k: v for k, v in kw.items() if k != "rolling_chunk"}), src, radius=kw["radius"])
        label = f"{args.namespace}.{name} {', '.join(f'{k}={v}' for k, v in kw.items())}"
        if (rolling.width, rolling.height, rolling.num_frames) != (WIDTH, HEIGHT, FRAMES):
            print(f"FAIL {label}: rolling output is {rolling.width}x{rolling.height}x{rolling.num_frames}")
            failures += 1
            continue
        a, b = frames(rolling), frames(chain)
        worst = max(float(np.max(np.abs(pa - pb))) for fa, fb in zip(a, b) for pa, pb in zip(fa, fb))
        single_round = kw.get("iters", 1) == 1
        if name in device and (name != "NCSR" or single_round):
            mse = max(float(np.mean((pa - pb) ** 2)) for fa, fb in zip(a, b) for pa, pb in zip(fa, fb))
            psnr = 200.0 if mse == 0 else -10 * np.log10(mse)
            ok, note = psnr >= args.min_psnr, f"psnr {psnr:.2f} dB"
        else:
            ok, note = worst == 0, f"max abs {worst:.3g}"
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {note}")
        failures += not ok
        spatial = {**kw, "radius": 0}
        spatial.pop("rolling_chunk", None)
        plain, moded = frames(make(src, **spatial)[0]), frames(make(src, temporal_mode="rolling", **spatial)[0])
        if any(not np.array_equal(pa, pb) for pa, pb in zip(plain[0], moded[0])):
            print(f"FAIL {label}: radius 0 changed with temporal_mode")
            failures += 1
    for bad, text in (("bogus", "temporal_mode must be rolling or legacy"), ("fused", "temporal_mode=fused is not supported")):
        for name in ("WNNM", "MCWNNM", "NCSR", "NLH", "TWSC"):
            fmt = vs.RGBS if name == "MCWNNM" else vs.GRAYS
            try:
                getattr(ns, name)(source(core, fmt), temporal_mode=bad)
            except vs.Error as error:
                if f"{args.namespace}.{name}: {text}" in str(error):
                    continue
                print(f"FAIL {name} temporal_mode={bad}: {error}")
            else:
                print(f"FAIL {name} temporal_mode={bad}: accepted")
            failures += 1
    print(f"temporal rolling ({args.namespace}): {len(CASES)} cases, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
