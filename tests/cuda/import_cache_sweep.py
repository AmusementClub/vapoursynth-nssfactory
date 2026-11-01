#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Run the fixed-input import-cache matrix and write a manifest."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from pathlib import Path


def concurrent_instance(compare, case, streams, instance, seconds, env, target):
    command = [sys.executable, str(compare), "bench", "--case", case, "--impl", "nss_cuda", "--mode", "resident",
               "--runs", "1", "--seconds", str(seconds), "--warm-seconds", "0.2"]
    proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    if proc.returncode == 0 and proc.stdout.strip():
        record = json.loads(proc.stdout.strip().splitlines()[-1])
    else:
        record = dict(case=case, impl="nss_cuda", mode="resident", returncode=proc.returncode,
                      error=(proc.stderr.strip().splitlines() or ["worker failed"])[-1])
    record.update(streams=streams, instance=instance)
    target.write_text(json.dumps(record) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--nsys")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--streams", default="1,2,3")
    parser.add_argument("--instances", type=int, default=2)
    args = parser.parse_args()
    streams = [int(value) for value in args.streams.split(",")]
    if not streams or min(streams) < 1 or args.instances < 1:
        parser.error("--streams and --instances must be positive")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    compare = Path(__file__).resolve().parents[1] / "compare_gpu_filters.py"
    env = dict(os.environ)
    env.pop("NSS_BENCH_NUM_STREAMS", None)
    records = []
    concurrency = []
    for cache in (4, 8, 16):
        env["NSS_CUDA_GPU_IMPORT_CACHE"] = str(cache)
        env.pop("NSS_BENCH_NUM_STREAMS", None)
        for family in ("nlm", "bm3d", "pnlm"):
            target = out / f"cache-{cache}-{family}.jsonl"
            subprocess.run([sys.executable, str(compare), "campaign", "--family", family, "--impls", "nss_cuda",
                            "--modes", "resident", "--out", str(target), "--rounds", str(args.rounds),
                            "--runs", str(args.runs), "--seconds", str(args.seconds)], env=env, check=True)
            records.append(str(target))
        for stream_count in streams:
            env["NSS_BENCH_NUM_STREAMS"] = str(stream_count)
            target = out / f"cache-{cache}-streams-{stream_count}-nlm-default.jsonl"
            command = [sys.executable, str(compare), "bench", "--case", "nlm-default", "--impl", "nss_cuda",
                       "--mode", "resident", "--runs", str(args.runs), "--seconds", str(args.seconds)]
            proc = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
            target.write_text(proc.stdout.strip().splitlines()[-1] + "\n")
            records.append(str(target))
            if args.nsys:
                report = out / f"import-cache-{cache}-streams-{stream_count}"
                log = report.with_suffix(".log")
                with log.open("w") as handle:
                    subprocess.run([args.nsys, "profile", "--trace=cuda", "--stats=true", "--force-overwrite=true",
                                    "-o", str(report), sys.executable, str(compare), "bench", "--case", "nlm-default",
                                    "--impl", "nss_cuda", "--mode", "resident", "--runs", "1", "--seconds", "1",
                                    "--warm-seconds", "0.2"], env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   check=True)
                records.extend([str(report.with_suffix(".nsys-rep")), str(report.with_suffix(".sqlite")), str(log)])
            targets = [out / f"cache-{cache}-streams-{stream_count}-instance-{instance}.json"
                       for instance in range(args.instances)]
            with ThreadPoolExecutor(max_workers=args.instances) as pool:
                results = list(pool.map(lambda item: concurrent_instance(compare, "nlm-default", stream_count,
                                                                          item[0], args.seconds, env.copy(), item[1]),
                                        enumerate(targets)))
            records.extend(str(target) for target in targets)
            concurrency.append(dict(cache=cache, streams=stream_count, instances=args.instances,
                                    failures=sum("error" in result for result in results),
                                    records=[str(target) for target in targets]))
    manifest = dict(schema="nssfactory.import_cache_sweep.v1", caches=[4, 8, 16],
                    families=["nlm", "bm3d", "pnlm"], modes=["resident"], rounds=args.rounds, runs=args.runs,
                    seconds=args.seconds, streams=streams, instances=args.instances, concurrency=concurrency,
                    records=records)
    (out / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
