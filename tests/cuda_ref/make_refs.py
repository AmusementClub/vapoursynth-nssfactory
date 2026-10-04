#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Freeze CPU reference outputs for the cross-backend cases (CUDA plan C1).

Writes one <case>.npz per case into --out (keys f<frame>_p<plane>) and a
manifest recording the generator plugin, host, source state, input hashes and
output hashes. Only the manifest is committed (tests/data/cuda_refs_v1.json);
the arrays live under artifacts/.

usage: make_refs.py --plugin build/libnss.so --out DIR [--manifest FILE] [--jobs N] [--only REGEX]

NSS_HOST_LABEL overrides the recorded host name (keep personal host names out of
committed manifests).
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

_core = None


def _init(plugin):
    global _core
    import vapoursynth as vs
    _core = vs.core
    _core.num_threads = 2
    _core.std.LoadPlugin(path=plugin)


def _run(c, out_dir):
    import vapoursynth as vs
    _, kind = C.INPUTS[c["input"]][0], C.INPUTS[c["input"]][1]
    _, noisy = C.make_input(c["input"])
    clip = C.to_clip(_core, vs, noisy, kind)
    start = time.perf_counter()
    result = C.fetch(C.build(_core, "nss", clip, c), c["frames"])
    seconds = time.perf_counter() - start
    np.savez(Path(out_dir) / f"{c['id']}.npz",
             **{f"f{i}_p{p}": plane for i, planes in enumerate(result) for p, plane in enumerate(planes)})
    finite = all(np.isfinite(p).all() for planes in result for p in planes)
    return c["id"], dict(output_sha256=C.arrays_sha256(result), seconds=round(seconds, 3), finite=finite)


def _cost(c):
    size = C.INPUTS[c["input"]][2]
    return (size is None, c["id"].startswith(("twsc-g128", "nlh")), len(c["frames"]))


def source_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=C.REPO,
                               capture_output=True, text=True).stdout.strip() != ""
        return dict(commit=head, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        # Mirrored trees (e.g. the GPU dev host) carry the state written at sync time.
        recorded = C.REPO / ".nss_source.json"
        return json.loads(recorded.read_text()) if recorded.exists() else dict(commit=None, dirty=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=str(C.REPO / "tests" / "data" / "cuda_refs_v1.json"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    plugin = str(Path(args.plugin).resolve())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = [c for c in C.CASES if re.search(args.only, c["id"])]
    selected.sort(key=_cost, reverse=True)

    inputs = {}
    for name in sorted({c["input"] for c in selected}):
        image = C.INPUTS[name][0]
        clean, noisy = C.make_input(name)
        inputs[name] = dict(zip(("image", "kind", "size", "sigma8", "frames", "pan"), C.INPUTS[name]),
                            image_sha256=C.file_sha256(C.DS / image),
                            clean_sha256=C.arrays_sha256(clean), noisy_sha256=C.arrays_sha256(noisy))

    import vapoursynth as vs
    _init(plugin)
    backend = dict(_core.nss.Backend())
    backend = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in backend.items()}

    results = {}
    with cf.ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(plugin,)) as pool:
        futures = {pool.submit(_run, c, str(out)): c["id"] for c in selected}
        errors = {}
        for future in cf.as_completed(futures):
            try:
                cid, info = future.result()
            except Exception as error:  # report every failing case, then stop before writing a manifest
                errors[futures[future]] = repr(error)
                print(f"{futures[future]:34s} ERROR {error!r}", flush=True)
                continue
            results[cid] = info
            print(f"{cid:34s} {info['seconds']:9.2f}s  {info['output_sha256'][:12]}", flush=True)
    if errors:
        print(f"{len(errors)} cases failed; manifest not written")
        return 1

    manifest = dict(
        schema=C.SCHEMA,
        generated=dict(date=dt.date.today().isoformat(), host=os.environ.get("NSS_HOST_LABEL", platform.node()),
                       machine=platform.machine(),
                       plugin_sha256=C.file_sha256(plugin), backend=backend, source=source_state(),
                       vapoursynth=vs.__version__, numpy=np.__version__, artifacts=out.name),
        inputs=inputs,
        cases={c["id"]: dict(c, **results[c["id"]]) for c in sorted(selected, key=lambda c: c["id"])},
    )
    Path(args.manifest).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    bad = [cid for cid, info in results.items() if not info["finite"]]
    print(f"wrote {args.manifest}: {len(results)} cases" + (f", NON-FINITE: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
