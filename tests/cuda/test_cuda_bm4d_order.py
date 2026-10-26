#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.BM4D under requests in any order: chunk seeks (one frame per chunk,
shuffled, all at once), whole-clip shuffles and random repeats, also at the
smallest memory_limit_mb the filter accepts. Every request must finish (no
wait cycle among compute permits, slots, the chunk order and the cache) within
a bound, without exceeding the limit, and every frame must equal the
sequential render bit for bit. Exits 77 without VapourSynth or a CUDA device.

usage: test_cuda_bm4d_order.py --cuda PATH
"""
import argparse
import hashlib
import os
import random
import sys
from concurrent.futures import wait
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda bm4d order SKIP: {error}")
    raise SystemExit(77)

TIMEOUT = 120  # seconds for one pattern; a wait cycle never finishes


def make_clip(core, fmt, width, height, length):
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)

    def fill(n, f):
        out = f.copy()
        rng = np.random.default_rng(n)
        for p in range(out.format.num_planes):
            plane = np.asarray(out[p])
            plane[...] = (0.5 + 0.1 * rng.standard_normal(plane.shape)).astype(np.float32)
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def digest(frame):
    h = hashlib.md5()
    for p in range(frame.format.num_planes):
        h.update(bytes(frame[p]))
    return h.hexdigest()


def reference(core, clip, kw):
    """Each frame asked for alone, one after another."""
    out = core.nss_cuda.BM4D(clip, sigma=8, **kw)
    return [digest(out.get_frame(n)) for n in range(clip.num_frames)]


def render(core, clip, order, kw, label):
    """The digest of every request of `order`, asked for all at once (repeats included)."""
    out = core.nss_cuda.BM4D(clip, sigma=8, **kw)
    futures = [out.get_frame_async(n) for n in order]
    _, pending = wait(futures, timeout=TIMEOUT)
    if pending:
        print(f"cuda bm4d order FAIL: {label}: {len(pending)} of {len(order)} requests still waiting after {TIMEOUT} s")
        sys.stdout.flush()
        os._exit(1)  # the stuck requests would block the core's teardown
    got = []
    for n, f in zip(order, futures):
        if f.exception() is not None:
            return None, f"frame {n}: {f.exception()}"
        got.append((n, digest(f.result())))
    return got, None


def smallest_limit(core, clip):
    lo, hi = 1, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            core.nss_cuda.BM4D(clip, sigma=8, memory_limit_mb=mid).get_frame(0)
            hi = mid
        except vs.Error:
            lo = mid + 1
    return lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True)
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 32
    core.std.LoadPlugin(path=str(Path(args.cuda).resolve()))
    if core.nss_cuda.Backend()["device_count"] == 0:
        print("cuda bm4d order SKIP: no CUDA device")
        return 77
    failures = 0
    cases = [("gray 32x32 x512", make_clip(core, vs.GRAYS, 32, 32, 512), {}),
             ("yuv444 48x40 x96", make_clip(core, vs.YUV444PS, 48, 40, 96), {})]
    limited = make_clip(core, vs.GRAYS, 64, 48, 320)
    cases.append(("gray 64x48 x320 smallest limit", limited, {"memory_limit_mb": smallest_limit(core, limited)}))
    for name, clip, kw in cases:
        n = clip.num_frames
        try:
            ref = reference(core, clip, kw)
        except vs.Error as error:
            print(f"cuda bm4d order FAIL: {name} reference: {error}")
            failures += 1
            continue
        chunk_frames = [c * 32 + 8 for c in range((n + 31) // 32) if c * 32 + 8 < n]
        patterns = [(f"chunk seek {s}", random.Random(s).sample(chunk_frames, len(chunk_frames))) for s in range(8)]
        patterns += [(f"shuffle {s}", random.Random(100 + s).sample(range(n), n)) for s in range(2)]
        for s in range(4):
            rng = random.Random(200 + s)  # one generator per pattern: random frames, each likely asked for again
            patterns.append((f"repeats {s}", [rng.randrange(n) for _ in range(3 * n)]))
        patterns.append(("reverse", list(range(n - 1, -1, -1))))
        for pname, order in patterns:
            got, error = render(core, clip, order, kw, f"{name} {pname}")
            wrong = [] if error else sorted({k for k, v in got if v != ref[k]})
            if error or wrong:
                failures += 1
                print(f"cuda bm4d order FAIL: {name} {pname}: {error or f'{len(wrong)} frames differ, e.g. {wrong[:4]}'}")
        print(f"{name} {kw or ''}: {len(patterns)} request orders")
    print(f"cuda bm4d order {'FAIL' if failures else 'OK'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
