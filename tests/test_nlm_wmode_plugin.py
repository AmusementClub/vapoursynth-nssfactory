#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""NLM wmode 0..3 against a direct float64 NLM of the KNLMeansCL model.

The weight of u = box sum of 3*delta^2 * 255^2 / (3 h^2 (2s+1)^2) is exp(-u)
(Welsch) or max(1 - u, 0) to the power 1, 2, 8 (Bisquare A, B, C). Pixels
within a + s of the border are left out: there the plugin takes the mirrored
offset's weight at clamp(p - o). Bisquare weights past u = 1 are exactly zero,
so a small h must return the input up to the finish's rounding. Runs on
either backend.
"""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"NLM wmode SKIP: {error}")
    raise SystemExit(77)

TOLERANCE = 1e-5
ZERO_WEIGHT_TOLERANCE = 2.5e-7  # about 2 ulp near 1: one candidate weight of 1e-6 would move it far more


def make_clip(core, width, height, seed, length=3):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=length)
    y, x = np.mgrid[:height, :width]
    frames = [(0.5 + 0.2 * np.sin((x + 2 * t) / 5) * np.cos((y + t) / 7) +
               rng.normal(0, 8 / 255, (height, width))).astype(np.float32) for t in range(length)]

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:] = frames[n]
        return out
    return core.std.ModifyFrame(blank, blank, fill), frames


def shifted(a, ox, oy):  # a(clamp(x + ox), clamp(y + oy))
    h, w = a.shape
    return a[np.clip(np.arange(h) + oy, 0, h - 1)][:, np.clip(np.arange(w) + ox, 0, w - 1)]


def box(a, s):
    out = sum(shifted(a, j, 0) for j in range(-s, s + 1))
    return sum(shifted(out, 0, j) for j in range(-s, s + 1))


def reference(frames, n, d, a, s, h, wref, wmode):
    centre = frames[n].astype(np.float64)
    h2_inv_norm = 255.0 ** 2 / (3 * h * h * (2 * s + 1) ** 2)
    num, den = np.zeros_like(centre), np.zeros_like(centre)
    maxw = np.full_like(centre, np.finfo(np.float32).eps)
    for dt in range(-d, d + 1):
        frame = frames[min(max(n + dt, 0), len(frames) - 1)].astype(np.float64)
        for oy in range(-a, a + 1):
            for ox in range(-a, a + 1):
                if dt == 0 and ox == 0 and oy == 0:
                    continue
                neighbour = shifted(frame, ox, oy)
                u = box(3 * (centre - neighbour) ** 2, s) * h2_inv_norm
                weight = np.exp(-u) if wmode == 0 else np.maximum(1 - u, 0) ** (1, 2, 8)[wmode - 1]
                num += weight * neighbour
                den += weight
                maxw = np.maximum(maxw, weight)
    m = wref * maxw
    return (m * centre + num) / (m + den)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--namespace", default="nss", choices=("nss", "nss_cuda"))
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 2
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    ns = getattr(core, args.namespace)
    if args.namespace == "nss_cuda" and ns.Backend()["device_count"] == 0:
        print("NLM wmode SKIP: no CUDA device")
        return 77

    clip, frames = make_clip(core, 61, 45, 1)
    failures, worst, count = [], 0.0, 0
    cases = [dict(h=3.0), dict(h=12.0), dict(h=30.0), dict(d=0, a=3, s=1, h=20.0),
             dict(d=2, s=2, wref=0.5, h=20.0)]
    for wmode in range(4):
        outputs = []
        for kw in cases:
            node = ns.NLM(clip, wmode=wmode, **kw)
            margin = kw.get("a", 2) + kw.get("s", 4)
            for n in range(clip.num_frames):
                got = np.asarray(node.get_frame(n)[0]).astype(np.float64)
                want = reference(frames, n, kw.get("d", 1), kw.get("a", 2), kw.get("s", 4), kw["h"],
                                 kw.get("wref", 1.0), wmode)
                error = float(np.max(np.abs(got - want)[margin:-margin, margin:-margin]))
                worst = max(worst, error)
                count += 1
                if error > TOLERANCE:
                    failures.append(f"wmode={wmode} {kw} frame {n}: max error {error:.3g}")
                if n == 1:
                    outputs.append(got[margin:-margin, margin:-margin])
        if wmode:  # h = 3: u > 1 for every candidate inside, so the input back to the finish's rounding
            count += 1
            moved = float(np.max(np.abs(outputs[0] - frames[1][6:-6, 6:-6])))
            if moved > ZERO_WEIGHT_TOLERANCE:
                failures.append(f"wmode={wmode} h=3: Bisquare weights past u = 1 are not zero (moved {moved:.3g})")
    for line in failures:
        print("FAIL:", line)
    print(f"NLM wmode: {count} checks, worst error {worst:.3g}, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
