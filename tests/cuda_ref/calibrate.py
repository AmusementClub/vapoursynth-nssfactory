#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Derive per-filter GPU tolerances from measured CPU cross-build spread (D16).

Inputs are cuda_compare.py --report files from other CPU builds (other ISA,
compiler or the NSS_BM_EXPERIMENT=0 reference build) compared against the
frozen references. For each filter the floor is the worst PSNR any CPU build
already shows; the GPU threshold is that floor minus a fixed margin, clamped
to [FLOOR_DB, CAP_DB]. Thresholds are a starting point: a phase may tighten or
relax a filter only by bumping the tolerances schema and recording why.

usage: calibrate.py --out FILE REPORT [REPORT ...]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

SCHEMA = "nssfactory.cuda_tolerances.v1"
MARGIN_DB = 10.0
CAP_DB = 60.0  # RMS 1e-3, ~0.26 of an 8-bit code value
FLOOR_DB = 40.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    spread = {}
    sources = []
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        sources.append(dict(file=Path(path).name, plugin_sha256=report["plugin_sha256"],
                            reference_plugin_sha256=report["reference_plugin_sha256"]))
        for cid, row in report["rows"].items():
            if row.get("status") != "ok":
                raise SystemExit(f"{path}: {cid} status {row.get('status')}")
            filt = C.CASE_BY_ID[cid]["filter"]
            entry = spread.setdefault(filt, dict(min_psnr=math.inf, max_abs=0.0, worst_case=None))
            if row["psnr"] < entry["min_psnr"]:
                entry.update(min_psnr=row["psnr"], worst_case=cid)
            entry["max_abs"] = max(entry["max_abs"], row["max_abs"])

    filters = {}
    for filt, entry in sorted(spread.items()):
        observed = min(entry["min_psnr"], CAP_DB + MARGIN_DB)
        filters[filt] = dict(min_psnr=round(max(FLOOR_DB, observed - MARGIN_DB), 1))
        entry["min_psnr"] = None if math.isinf(entry["min_psnr"]) else round(entry["min_psnr"], 3)
    tolerances = dict(
        schema=SCHEMA,
        references="cuda_refs_v1.json",
        rule=f"min_psnr = clamp(min CPU cross-build PSNR - {MARGIN_DB:g} dB, {FLOOR_DB:g}, {CAP_DB:g}); "
             "identical builds count as the cap; every case must be finite and run-to-run deterministic",
        default=dict(deterministic=True),
        filters=filters,
        cases={},
        calibration=dict(sources=sources, spread=spread),
    )
    Path(args.out).write_text(json.dumps(tolerances, indent=1, sort_keys=True) + "\n")
    for filt, rule in filters.items():
        print(f"{filt:11s} spread min {spread[filt]['min_psnr']} dB (max_abs {spread[filt]['max_abs']:.3g}, "
              f"{spread[filt]['worst_case']}) -> min_psnr {rule['min_psnr']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
