#!/usr/bin/env python3
"""Real VS shape, identity, stride, budget and concurrent-request contracts.

The constant/zero-noise and synthetic contribution oracles are independent of
production kernels. Nonconstant shape tests assert finite output and report
clean-reference errors; they do not invent a universal quality oracle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource

import numpy as np
import vapoursynth as vs

MODELS = {"BM3D": 1, "WNNM": 2, "MCWNNM": 3, "TWSC": 4, "NCSR": 5, "NLH": 6}


def clip(core, width=35, height=33, frames=5, fmt=vs.GRAYS, parent_extra=0, constant=False):
    parent = core.std.BlankClip(width=width + parent_extra, height=height, length=frames, format=fmt)

    def fill(n, f):
        out = f.copy()
        for plane in range(out.format.num_planes):
            a = np.asarray(out[plane])
            y, x = np.indices(a.shape)
            # Dyadic values give an exact, spatially nonconstant stride oracle.
            a[:] = .25 if constant else ((x * 3 + y * 5 + n * 7 + plane * 11) % 97) / 128.
        return out

    result = core.std.ModifyFrame(parent, parent, fill)
    return result.std.Crop(right=parent_extra) if parent_extra else result


def options(name, block=4, group=8, radius=0, sigma=3):
    if name == "NLM":
        return dict(d=radius, a=1, s=1, h=3)
    if name == "LSSC":
        return dict(sigma=sigma, block_size=block, block_step=block)
    kw = dict(sigma=sigma, block_size=block, block_step=block, group_size=group,
              bm_range=2, radius=radius, ps_num=min(group, 2), ps_range=1)
    if name in ("MCWNNM", "TWSC", "NCSR"):
        kw["iters"] = 1
    if name == "MCWNNM":
        kw["admm_iter"] = 2
    if name == "NLH":
        kw["q"] = 2
    return kw


def evaluate(node, n=0):
    frame = node.get_frame(n)
    arrays = [np.array(frame[p]) for p in range(frame.format.num_planes)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise AssertionError("nonfinite real-plugin output")
    stats = frame.props.get("_NSSResourceBytes")
    if stats is not None:
        owned = sum(int(v) for v in stats[:5])
        if owned > frame.props["_NSSResourceLimit"]:
            raise AssertionError("reported resource limit exceeded")
    return arrays, dict(frame.props)


def reject(call, contains):
    try:
        call()
    except vs.Error as error:
        if contains not in str(error):
            raise AssertionError(f"wrong failure ({contains}): {error}") from error
    else:
        raise AssertionError(f"expected observable error: {contains}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 8
    core.max_cache_size = 32
    core.std.LoadPlugin(path=os.environ["NSS_SO"])
    rows = []
    gray = clip(core)
    rgb = clip(core, fmt=vs.RGBS)
    names = [*MODELS, "LSSC", "NLM"]
    # Full BM public shape product, including short actual groups.
    for block, group, final in itertools.product((1, 2, 4, 8, 12, 16, 32), (1, 2, 4, 8, 16, 32, 64), (False, True)):
        kw = options("BM3D", block, group)
        if final:
            kw["ref"] = gray
        values, _ = evaluate(core.nss.BM3D(gray, **kw))
        rows.append(dict(test="shape", algorithm="BM3D", block=block, group=group, final=final,
                         output_sha256=hashlib.sha256(values[0].tobytes()).hexdigest()))
    for name in ("WNNM", "TWSC", "NCSR", "MCWNNM", "NLH", "LSSC"):
        blocks = (4, 8) if name == "NLH" else (1, 2, 4, 8, 9) if name == "MCWNNM" else (1, 2, 4, 8, 16)
        groups = (2, 4, 8, 16) if name == "NLH" else (1,) if name == "LSSC" else (1, 3, 8, 32)
        for block, group in itertools.product(blocks, groups):
            source = rgb if name == "MCWNNM" else gray
            kw = options(name, block, group)
            values, _ = evaluate(getattr(core.nss, name)(source, **kw))
            rows.append(dict(test="shape", algorithm=name, block=block, group=group,
                             output_sha256=hashlib.sha256(values[0].tobytes()).hexdigest()))
    # Source and reference views have the same pixels but different parent strides.
    for name in names:
        fmt = vs.RGBS if name in ("MCWNNM", "TWSC") else vs.GRAYS
        source = clip(core, fmt=fmt, parent_extra=61)
        reference = clip(core, fmt=fmt, parent_extra=125)
        ordinary = clip(core, fmt=fmt)
        kw = options(name, radius=1 if name in MODELS else 0)
        key = "ref" if name == "BM3D" else "rclip"
        if name not in ("LSSC",):
            kw[key] = reference
        unusual = getattr(core.nss, name)(source, **kw)
        if key in kw:
            kw[key] = ordinary
        packed = getattr(core.nss, name)(ordinary, **kw)
        max_abs = 0.
        for n in (4, 0, 2):
            a, _ = evaluate(unusual, n)
            b, _ = evaluate(packed, n)
            for x, y in zip(a, b):
                max_abs = max(max_abs, float(np.max(np.abs(x.astype(np.float64) - y))))
        if max_abs > 2e-5:
            raise AssertionError(f"independent stride changed {name}: {max_abs}")
        rows.append(dict(test="independent_stride", algorithm=name, max_abs=max_abs,
                         source_stride=source.get_frame(0).get_stride(0),
                         ref_stride=reference.get_frame(0).get_stride(0)))
    # Every fat producer must carry identity even on exact sigma-zero bypass.
    for fmt in (vs.GRAYS, vs.YUV420PS, vs.RGBS):
        source = clip(core, width=36, height=34, fmt=fmt)
        for name, model in MODELS.items():
            if name == "MCWNNM" and fmt != vs.RGBS:
                continue
            raw = getattr(core.nss, name)(source, **options(name, radius=2, sigma=0))
            output = core.nss.VAggregate(raw, source, radius=2)
            for n in (4, 0, 2, 1):
                fat_frame = raw.get_frame(n)
                for key, value in dict(_NSSFatVersion=2, _NSSFatCenter=n, _NSSFatRadius=2,
                                       _NSSFatLayout=1, _NSSModel=model, _NSSModelVersion=2).items():
                    if fat_frame.props[key] != value:
                        raise AssertionError((name, key, value, fat_frame.props[key]))
                actual, _ = evaluate(output, n)
                expected = source.get_frame(n)
                if any(not np.array_equal(a, np.asarray(expected[p])) for p, a in enumerate(actual)):
                    raise AssertionError(f"zero sigma changed {name}, format={fmt}, n={n}")
            rows.append(dict(test="zero_sigma_identity", algorithm=name, format=fmt))
    # Independent synthetic target-contribution oracle (not fat-vs-rolling).
    for length, radius in itertools.product((1, 2, 5), (0, 1, 2)):
        source = clip(core, width=8, height=8, frames=length)
        blank = core.std.BlankClip(width=8, height=8 * 2 * (2 * radius + 1), length=length, format=vs.GRAYS)
        def impulse(n, f, radius=radius):
            out = f.copy()
            a = np.asarray(out[0])
            for sl in range(2 * radius + 1):
                a[sl * 16:sl * 16 + 8] = (n + 1) * (sl + 1)
                a[sl * 16 + 8:(sl + 1) * 16] = sl + 2
            out.props.update(_NSSFatVersion=2, _NSSFatRadius=radius, _NSSFatCenter=n,
                             _NSSFatLayout=1, _NSSModel=1, _NSSModelVersion=2, _NSSNoiseProfile=1)
            return out
        fat = core.std.ModifyFrame(blank, blank, impulse)
        output = core.nss.VAggregate(fat, source, radius=radius)
        for target in range(length):
            num = den = 0
            for center in range(max(0, target-radius), min(length, target+radius+1)):
                sl = target-center+radius
                num += (center+1)*(sl+1)
                den += sl+2
            actual, _ = evaluate(output, target)
            if np.max(np.abs(actual[0]-num/den)) > 2e-6:
                raise AssertionError("destination contribution oracle mismatch")
        reject(lambda: core.nss.VAggregate(blank, source, radius=radius).get_frame(0), "identity")
        evaluate(core.nss.VAggregate(blank, source, radius=radius, allow_legacy=1))
        for key, value in (("_NSSFatVersion", 1), ("_NSSFatCenter", 99), ("_NSSFatRadius", radius+1),
                           ("_NSSNoiseProfile", 7), ("_NSSFatLayout", 0)):
            bad = core.std.SetFrameProp(fat, prop=key, intval=value)
            reject(lambda: core.nss.VAggregate(bad, source, radius=radius, allow_legacy=1).get_frame(0), "identity")
        rows.append(dict(test="contribution_oracle", length=length, radius=radius))
    # Rolling has its own exact oracle when sigma is zero, across motion/cuts.
    source = clip(core, width=36, height=34, frames=37)
    rolling = core.nss.BM3D(source, **options("BM3D", radius=3, sigma=0), temporal_mode="rolling",
                           rolling_chunk=2, rolling_cache_limit=1, memory_limit_mb=16)
    core.std.SetVideoCache(rolling, mode=0)
    requests = list(np.random.default_rng(3917).integers(0, 37, 128))
    def check_request(n):
        actual, props = evaluate(rolling, int(n))
        if not np.array_equal(actual[0], np.asarray(source.get_frame(int(n))[0])):
            raise AssertionError("rolling independent identity oracle failed")
        return int(props["_NSSResourcePeak"])
    with ThreadPoolExecutor(8) as pool:
        peaks = list(pool.map(check_request, requests))
    rows.append(dict(test="rolling_concurrent_identity", requests=len(requests), workers=8, peak=max(peaks)))
    # Nonzero rolling cache handoff/recompute must be deterministic as well.
    rolling = core.nss.BM3D(source, **options("BM3D", radius=2), temporal_mode="rolling",
                           rolling_chunk=2, rolling_cache_limit=1, memory_limit_mb=16)
    core.std.SetVideoCache(rolling, mode=0)
    golden = {n: evaluate(rolling, n)[0][0] for n in (0, 1, 12, 24, 36)}
    def check_nonzero(n):
        actual, props = evaluate(rolling, n)
        if not np.array_equal(actual[0], golden[n]):
            raise AssertionError("rolling concurrent recomputation changed pixels")
        return list(props["_NSSResourceBytes"])
    with ThreadPoolExecutor(8) as pool:
        stats = list(pool.map(check_nonzero, list(golden) * 12))
    workspace_bytes = {int(s[0]) for s in stats}
    if len(workspace_bytes) != 1:
        raise AssertionError("rolling scratch grew with worker handoff")
    rows.append(dict(test="rolling_concurrent_nonzero", requests=len(stats), workers=8,
                     workspace_bytes=list(workspace_bytes), peak_cached=max(int(s[1]) for s in stats),
                     peak_pinned=max(int(s[2]) for s in stats)))
    # Runtime budget errors must be visible; subsequent healthy nodes still work.
    large = clip(core, width=384, height=256, frames=5)
    for repeat in range(3):
        node = core.nss.BM3D(large, **options("BM3D", radius=2), memory_limit_mb=1)
        reject(lambda: node.get_frame(2), "memory_limit_mb")
        del node
        node = core.nss.BM3D(large, **options("BM3D", radius=2), temporal_mode="rolling",
                             rolling_chunk=2, rolling_cache_limit=1, memory_limit_mb=1)
        reject(lambda: node.get_frame(2), "memory_limit_mb")
        del node
        gc.collect()
    for name in MODELS:
        small = clip(core, width=2, height=2, fmt=vs.RGBS if name == "MCWNNM" else vs.GRAYS)
        reject(lambda: getattr(core.nss, name)(small, **options(name)), "block_size")
    reject(lambda: core.nss.WNNM(gray, ps_num=2**31-1), "ps_num")
    reject(lambda: core.nss.NLM(gray, d=2**31-1), "invalid d/a/s/h")
    rows.append(dict(test="failure_and_recovery", passed=True))
    report = dict(passed=True, checks=len(rows), cases=rows, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  plugin_sha256=hashlib.sha256(Path(os.environ["NSS_SO"]).read_bytes()).hexdigest(),
                  numpy=np.__version__, vapoursynth=str(vs.__version__))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
