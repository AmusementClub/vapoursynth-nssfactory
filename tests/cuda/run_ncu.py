#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Profile a benchmark with Nsight Compute using the real Python executable."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--impl", default="nss_cuda")
    parser.add_argument("--mode", choices=["cpu", "resident"], default="resident")
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch-count", type=int, default=3)
    parser.add_argument("--ncu", default="ncu")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compare = Path(__file__).resolve().parents[1] / "compare_gpu_filters.py"
    env = dict(os.environ)
    if args.mode == "resident":
        env["NSS_CUDA_GPU_FRAMES"] = "1"
    command = [args.ncu, "--target-processes", "all", "--launch-count", str(args.launch_count),
               "--section", "SpeedOfLight", "--section", "Occupancy", "--section", "SchedulerStats",
               "--section", "WarpStateStats", "--metrics", "launch__registers_per_thread", "--force-overwrite",
               "-o", str(output), sys.executable, str(compare), "bench", "--case", args.case, "--impl", args.impl,
               "--mode", args.mode, "--runs", "1", "--seconds", "0.5", "--warm-seconds", "0.2"]
    print(" ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    details = output.with_name(output.name + "-details.txt")
    with details.open("w") as handle:
        subprocess.run([args.ncu, "-i", str(output.with_suffix(".ncu-rep")), "--page", "details"],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(details, flush=True)


if __name__ == "__main__":
    main()
