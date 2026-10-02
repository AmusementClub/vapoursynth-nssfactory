#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Compare a backend against the frozen CPU reference outputs (CUDA plan C1).

Rebuilds each case's input from tests/cuda_ref/cases.py, checks the input hash
against the manifest, runs the case on --namespace, and reports PSNR / max-abs
/ fraction of samples off by more than 1e-3 against the stored reference.
With --tolerances, cases are gated per filter (case overrides win). --repeat 2
also checks run-to-run determinism. --namespace nss with another CPU build
measures the cross-ISA spread used to calibrate the tolerances (D16).

usage: cuda_compare.py --plugin PATH --refs DIR [--namespace nss_cuda] [--tolerances FILE]
                       [--only REGEX] [--expect-missing F1,F2] [--repeat N] [--jobs N]
                       [--bm3dcuda] [--report FILE]
"""
import argparse
import concurrent.futures as cf
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "cuda_ref"))
import cases as C  # noqa: E402

DEFAULT_MANIFEST = C.REPO / "tests" / "data" / "cuda_refs_v1.json"
_core = None
_input_cache = {}


def _init(plugin, namespace):
    global _core
    import vapoursynth as vs
    _core = vs.core
    _core.num_threads = 2
    if not hasattr(_core, namespace):
        _core.std.LoadPlugin(path=plugin)


def _inputs(name):
    if name not in _input_cache:
        _input_cache[name] = C.make_input(name)
    return _input_cache[name]


def load_ref(refs, cid, frames):
    data = np.load(Path(refs) / f"{cid}.npz")
    out = []
    for i in range(len(frames)):
        planes, p = [], 0
        while f"f{i}_p{p}" in data:
            planes.append(data[f"f{i}_p{p}"])
            p += 1
        out.append(planes)
    return out


def _bm3dcuda_row(c, clip, clean, result):
    """Noisy->clean PSNR of the tested output and of bm3dcuda at matching settings."""
    args = c["args"]
    if c["filter"] != "BM3D" or c["aggregate"] or args.get("radius") or "ref" in args:
        return None
    if args.get("block_size", 8) != 8 or args.get("group_size", 8) != 8:
        return None
    sigma = args.get("sigma", 3.0)
    node = _core.bm3dcuda.BM3D(clip, sigma=sigma, block_step=args.get("block_step", 8),
                              bm_range=args.get("bm_range", 7))
    other = C.fetch(node, c["frames"])
    flat = lambda frames: [p for planes in frames for p in planes]
    return dict(clean_psnr=C.psnr(flat(result), flat(clean)), bm3dcuda_clean_psnr=C.psnr(flat(other), flat(clean)))


def _run(c, refs, namespace, repeat, expected_noisy, with_bm3dcuda):
    import vapoursynth as vs
    kind = C.INPUTS[c["input"]][1]
    clean, noisy = _inputs(c["input"])
    if C.arrays_sha256(noisy) != expected_noisy:
        return c["id"], dict(status="input-mismatch")
    clip = C.to_clip(_core, vs, noisy, kind)
    try:
        runs = [C.fetch(C.build(_core, namespace, clip, c), c["frames"]) for _ in range(max(1, repeat))]
    except AttributeError as error:
        return c["id"], dict(status="missing", error=str(error))
    except vs.Error as error:
        return c["id"], dict(status="error", error=str(error))
    row = C.compare(runs[0], load_ref(refs, c["id"], c["frames"]))
    hashes = {C.arrays_sha256(r) for r in runs}
    row.update(status="ok", deterministic=len(hashes) == 1, output_sha256=C.arrays_sha256(runs[0]))
    if with_bm3dcuda:
        extra = _bm3dcuda_row(c, clip, [clean[n] for n in c["frames"]], runs[0])
        if extra:
            row.update(extra)
    return c["id"], row


def gate(c, row, tolerances):
    if tolerances is None:
        return None
    rule = dict(tolerances.get("default", {}))
    rule.update(tolerances.get("filters", {}).get(c["filter"], {}))
    rule.update(tolerances.get("cases", {}).get(c["id"], {}))
    problems = []
    if not row["finite"]:
        problems.append("non-finite output")
    if "min_psnr" in rule and row["psnr"] < rule["min_psnr"]:
        problems.append(f"psnr {row['psnr']:.2f} < {rule['min_psnr']}")
    if "max_abs" in rule and row["max_abs"] > rule["max_abs"]:
        problems.append(f"max_abs {row['max_abs']:.3g} > {rule['max_abs']}")
    if "max_frac_gt_1e3" in rule and row["frac_gt_1e3"] > rule["max_frac_gt_1e3"]:
        problems.append(f"frac>1e-3 {row['frac_gt_1e3']:.3g} > {rule['max_frac_gt_1e3']}")
    if rule.get("deterministic", False) and not row["deterministic"]:
        problems.append("nondeterministic")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--namespace", default="nss_cuda")
    parser.add_argument("--tolerances")
    parser.add_argument("--only", default="")
    parser.add_argument("--expect-missing", default="")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--bm3dcuda", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("schema") != C.SCHEMA:
        raise SystemExit(f"manifest schema {manifest.get('schema')!r} != {C.SCHEMA!r}")
    tolerances = json.loads(Path(args.tolerances).read_text()) if args.tolerances else None
    expect_missing = {f for f in args.expect_missing.split(",") if f}
    plugin = str(Path(args.plugin).resolve())
    selected = [c for c in C.CASES if re.search(args.only, c["id"]) and c["id"] in manifest["cases"]]
    stale = [c["id"] for c in selected if {k: c[k] for k in ("filter", "input", "args", "frames", "aggregate")}
             != {k: manifest["cases"][c["id"]][k] for k in ("filter", "input", "args", "frames", "aggregate")}]
    if stale:
        raise SystemExit(f"case definitions differ from the manifest (regenerate refs): {stale}")

    rows = {}
    with cf.ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(plugin, args.namespace)) as pool:
        futures = [pool.submit(_run, c, args.refs, args.namespace, args.repeat,
                               manifest["inputs"][c["input"]]["noisy_sha256"], args.bm3dcuda) for c in selected]
        for future in cf.as_completed(futures):
            cid, row = future.result()
            rows[cid] = row

    failures = 0
    print(f"{'case':34s} {'status':8s} {'psnr':>8s} {'max_abs':>10s} {'frac>1e-3':>10s} det  verdict")
    for c in selected:
        row = rows[c["id"]]
        if row["status"] == "missing":
            verdict = "expected-missing" if c["filter"] in expect_missing else "FAIL missing"
        elif row["status"] != "ok":
            verdict = f"FAIL {row['status']}: {row.get('error', '')}"
        else:
            problems = gate(c, row, tolerances)
            verdict = "report" if problems is None else ("pass" if not problems else "FAIL " + "; ".join(problems))
        row["verdict"] = verdict
        failures += verdict.startswith("FAIL")
        if row["status"] == "ok":
            psnr = "inf" if math.isinf(row["psnr"]) else f"{row['psnr']:.2f}"
            line = (f"{c['id']:34s} {'ok':8s} {psnr:>8s} {row['max_abs']:10.3g} {row['frac_gt_1e3']:10.3g} "
                    f"{'y' if row['deterministic'] else 'N'}    {verdict}")
            if "bm3dcuda_clean_psnr" in row:
                line += f"  [clean {row['clean_psnr']:.3f} dB vs bm3dcuda {row['bm3dcuda_clean_psnr']:.3f} dB]"
        else:
            line = f"{c['id']:34s} {row['status']:8s} {'':>8s} {'':>10s} {'':>10s}      {verdict}"
        print(line)
    print(f"{len(selected)} cases, {failures} failures")
    if args.report:
        report = dict(namespace=args.namespace, plugin=plugin, plugin_sha256=C.file_sha256(plugin),
                      manifest=str(args.manifest), reference_plugin_sha256=manifest["generated"]["plugin_sha256"],
                      rows=rows)
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
