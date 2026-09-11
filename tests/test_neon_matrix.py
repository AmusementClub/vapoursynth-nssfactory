#!/usr/bin/env python3
"""Public shape/options and scheduling matrix, with reusable pixel captures.

Same-binary scheduling comparisons are exact. Cross-build captures are evidence
inputs, not an automatic numeric/performance gate. Independent mathematical
oracles remain in test_bm3d_semantics, test_svd_scale and the primitive tests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np
import vapoursynth as vs

from test_plan01_plugin import clip, evaluate, options


def cases():
    blocks = (1, 2, 4, 8, 12, 16, 32)
    groups = (1, 2, 4, 8, 16, 32, 64)
    for block, group, stage, mode in itertools.product(
            blocks, groups, ("basic", "final", "two-stage"), ("spatial", "legacy", "rolling")):
        yield dict(algorithm="BM3D", block=block, group=group, stage=stage, mode=mode)
    for name in ("WNNM", "MCWNNM", "TWSC", "NCSR", "NLH", "LSSC"):
        bs = (4, 8) if name == "NLH" else (1, 2, 4, 8, 9) if name == "MCWNNM" else (1, 2, 4, 8, 16)
        gs = (2, 4, 8, 16) if name == "NLH" else (1,) if name == "LSSC" else (1, 3, 8, 16, 32)
        for block, group, radius in itertools.product(bs, gs, (0,) if name == "LSSC" else (0, 1)):
            yield dict(algorithm=name, block=block, group=group, radius=radius,
                       format="RGBS" if name == "MCWNNM" else "GRAYS")
    for block, group, radius in itertools.product((8, 9), (1, 3, 8, 16, 32), (0, 1)):
        yield dict(algorithm="TWSC", block=block, group=group, radius=radius, format="RGBS")
    for fmt, channels in (("GRAYS", "Y"), ("RGBS", "RGB"), ("RGBS", "AUTO"),
                          ("YUV444PS", "YUV"), ("YUV420PS", "UV"), ("YUV420PS", "Y")):
        for radius, wref in itertools.product((0, 1, 2), (0., 1., .5)):
            yield dict(algorithm="NLM", radius=radius, format=fmt,
                       extra=dict(channels=channels, wmode=0, wref=wref, h=7., a=2, s=2), reference=True)
    for name in ("WNNM", "MCWNNM"):
        for residual, adaptive in itertools.product((0, 1), (0, 1)):
            yield dict(algorithm=name, radius=1, reference=True,
                       format="RGBS" if name == "MCWNNM" else "GRAYS",
                       extra=dict(residual=residual, adaptive_aggregation=adaptive))
    for name in ("MCWNNM", "TWSC", "NCSR"):
        for fmt, radius in itertools.product(("RGBS", "YUV444PS"), (0, 1)):
            extra = dict(iters=2, delta=.2, sigma=[3., 4., 5.])
            if name == "MCWNNM":
                extra.update(admm_iter=3, rho=2., mu=1.05)
            if name == "TWSC":
                extra.update(lambda2=.6, admm_iter=10)
            yield dict(algorithm=name, radius=radius, format=fmt, extra=extra, reference=True)
    for q, radius in itertools.product((2, 4, 8), (0, 1)):
        yield dict(algorithm="NLH", radius=radius, reference=True, extra=dict(q=q))


def graph(core, case):
    name = case["algorithm"]
    fmt = getattr(vs, case.get("format", "GRAYS"))
    # Both chroma planes fit the default block. Crop nodes exercise real host
    # views, while C++ guard fixtures prove independently unequal row strides.
    source = clip(core, width=36, height=34, frames=3, fmt=fmt, parent_extra=64)
    reference = clip(core, width=36, height=34, frames=3, fmt=fmt, parent_extra=128)
    radius = (0 if case["mode"] == "spatial" else 1) if name == "BM3D" else case.get("radius", 0)
    kw = options(name, case.get("block", 4), case.get("group", 8), radius=radius)
    kw.update(case.get("extra", {}))
    if name == "BM3D":
        mode = case["mode"]
        if mode == "rolling":
            kw.update(temporal_mode="rolling", rolling_chunk=1, rolling_cache_limit=1)

        def bm(ref=None):
            node = core.nss.BM3D(source, **kw, **({"ref": ref} if ref is not None else {}))
            return core.nss.VAggregate(node, source, radius=radius) if mode == "legacy" else node

        node = bm() if case["stage"] == "basic" else bm(reference) if case["stage"] == "final" else bm(bm())
    else:
        if case.get("reference"):
            kw["rclip"] = reference
        node = getattr(core.nss, name)(source, **kw)
        if radius and name != "NLM":
            node = core.nss.VAggregate(node, source, radius=radius)
    core.std.SetVideoCache(node, mode=0)
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    core = vs.core
    core.max_cache_size = 32
    core.std.LoadPlugin(path=os.environ["NSS_SO"])
    caps = {k: v.decode() if isinstance(v, bytes) else v for k, v in dict(core.nss.Backend()).items()}
    report = dict(passed=False, platform=platform.platform(), architecture=platform.machine(),
                  backend=caps, plugin_sha256=hashlib.sha256(Path(os.environ["NSS_SO"]).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(Path(__file__).with_name("test_plan01_plugin.py").read_bytes()).hexdigest(),
                  numpy=np.__version__, vapoursynth=str(vs.__version__), threads=[1, 2, 4], cases=[],
                  numeric_cross_build_gate=False, performance_gate=False)
    pixels = {}
    try:
        for index, case in enumerate(cases()):
            row = dict(index=index, parameters=case, passed=False)
            report["cases"].append(row)
            core.num_threads = 1
            baseline = graph(core, case)
            golden = {n: evaluate(baseline, n)[0] for n in (0, 1, 2)}
            del baseline
            for n, planes in golden.items():
                for p, array in enumerate(planes):
                    pixels[f"c{index}_n{n}_p{p}"] = array
            for threads in (2, 4):
                core.num_threads = threads
                candidate = graph(core, case)
                order = (2, 0, 1, 2)
                with ThreadPoolExecutor(threads) as pool:
                    actual = list(pool.map(lambda n: evaluate(candidate, n)[0], order))
                for n, planes in zip(order, actual):
                    for p, array in enumerate(planes):
                        if not np.array_equal(array, golden[n][p]):
                            np.savez(args.out / "failure.npz", expected=golden[n][p], actual=array)
                            raise AssertionError(f"case {index}, threads={threads}, n={n}, p={p}: scheduling changed pixels")
                del candidate
            row["passed"] = True
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        np.savez_compressed(args.out / "pixels.npz", **pixels)
        report["pixels_sha256"] = hashlib.sha256((args.out / "pixels.npz").read_bytes()).hexdigest()
        (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"{sum(row['passed'] for row in report['cases'])}/{len(report['cases'])} cases passed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
