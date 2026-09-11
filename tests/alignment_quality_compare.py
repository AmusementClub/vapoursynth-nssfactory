#!/usr/bin/env python3
"""Summarize saved native/reference/baseline pixels without tuning tolerances."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from alignment_campaign import metrics, sha


def compare(root, scope, suffix):
    reports = {}
    for variant in ("current", "baseline", "reference"):
        folder = root / f"quality-{variant}-{suffix}"
        report = json.loads((folder / "summary.json").read_text())
        if not report.get("passed"):
            raise ValueError(f"incomplete quality run: {folder}")
        reports[variant] = {
            (row["case"]["id"], row["model"]): (folder, row)
            for row in report["rows"]
        }
    if not reports["current"] or any(
        reports[v].keys() != reports["current"].keys() for v in reports
    ):
        raise ValueError("inconsistent quality case coverage")
    result = []
    for key, (_, current) in reports["current"].items():
        arrays = {}
        row = dict(case=key[0], model=key[1], scope=scope, crop=current["crop"])
        for variant, cases in reports.items():
            folder, item = cases[key]
            if any(item["case"][f] != current["case"][f]
                   for f in ("noisy_sha256", "clean_sha256", "sigma")):
                raise ValueError("input identity differs across implementations")
            if item["crop"] != current["crop"] or item["shape"] != current["shape"]:
                raise ValueError("input extent differs across implementations")
            output = folder / item["output"]
            if sha(output) != item["output_sha256"]:
                raise ValueError("saved output hash mismatch")
            arrays[variant] = np.load(output)
            q = item["quality"]
            row[variant] = dict(
                psnr_rgb_db=-20 * math.log10(q["rms"]) if q["rms"] else math.inf,
                ssim_channel_mean=float(np.mean([p["ssim"] for p in q["psnr_ssim"]])),
                quality=q, output_sha256=item["output_sha256"],
                seconds=item["seconds"],
            )
        row["reference_difference"] = metrics(arrays["reference"], arrays["current"])
        row["migration_difference"] = metrics(arrays["baseline"], arrays["current"])
        row["psnr_change_db"] = row["current"]["psnr_rgb_db"] - row["baseline"]["psnr_rgb_db"]
        result.append(row)
    return result


def main(args):
    root = Path(args.root).resolve()
    rows = compare(root, "central 16x16 of saved RGB512 samples", "crop16")
    rows += compare(root, "saved RGB512 samples", "nlh512")
    report = dict(
        scope="Three fixed DIV2K sigma25 inputs; no full-image TWSC result",
        rows=rows,
        interpretation="Quality and numerical differences are measurements, not automatic fidelity passes",
        script_sha256=sha(__file__),
    )
    (root / "quality-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["case"], row["model"], row["scope"],
              "reference max", row["reference_difference"]["max_abs"],
              "PSNR change", row["psnr_change_db"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
