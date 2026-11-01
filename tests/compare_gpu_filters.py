#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Matched BM3D/NLM/PNLM speed, memory and output comparison on one GPU host."""

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from compare_third_party import INPUTS, arrays_sha256, make_input, psnr, to_clip

SCHEMA = "nssfactory.gpu_filter_compare.v2"
REPO = Path(__file__).resolve().parents[1]
PLUGIN_ENV = {
    "nss": "NSS_SO",
    "nss_cuda": "NSS_CUDA_SO",
    "nss_vk": "NSS_VK_SO",
    "bm3dcuda": "BM3DCUDA_SO",
    "bm3dcuda_rtc": "BM3DCUDA_RTC_SO",
    "nlm_cuda": "NLM_CUDA_SO",
    "vsfeel": "VSFEEL_SO",
    "bm3dvk2": "BM3DVK2_SO",
    "vszipcl": "VSZIPCL_SO",
    "vszip": "VSZIP_SO",
}
BM3D_IMPLS = ["nss_cuda", "nss_vk", "bm3dcuda", "bm3dcuda_rtc", "vsfeel", "bm3dvk2", "vszip"]
NLM_IMPLS = ["nss_cuda", "nss_vk", "nlm_cuda", "vsfeel", "vszipcl", "vszip"]
PNLM_IMPLS = ["nss_cuda", "nss_vk"]
RESIDENT_IMPLS = {"nss_cuda", "nss_vk", "vsfeel", "bm3dvk2"}
CASES = {
    "bm3d-light": dict(family="bm3d", sigma=3, block_step=8, bm_range=4, radius=0, final=False),
    "bm3d-default": dict(family="bm3d", sigma=3, block_step=8, bm_range=9, radius=0, final=False),
    "bm3d-heavy": dict(family="bm3d", sigma=10, block_step=4, bm_range=16, radius=1, final=True),
    "nlm-light": dict(family="nlm", d=0, a=2, s=2),
    "nlm-default": dict(family="nlm", d=1, a=2, s=4),
    "nlm-heavy": dict(family="nlm", d=2, a=4, s=6),
    "pnlm-light": dict(family="pnlm", sigma=3, d=0, a=2, block_size=6, group_size=16),
    "pnlm-default": dict(family="pnlm", sigma=3, d=1, a=2, block_size=6, group_size=16),
    "pnlm-heavy": dict(family="pnlm", sigma=3, d=2, a=8, block_size=6, group_size=16),
}


def load_plugin(core, impl):
    import vapoursynth as vs
    path = os.environ.get(PLUGIN_ENV[impl])
    if not path:
        raise RuntimeError(f"{PLUGIN_ENV[impl]} is not set")
    try:
        core.std.LoadPlugin(path=str(Path(path).resolve()))
    except vs.Error as error:
        if "already loaded" not in str(error):
            raise


def bm3d_node(core, impl, clip, case, mode):
    kw = dict(sigma=case["sigma"], block_step=case["block_step"], bm_range=case["bm_range"],
              radius=case["radius"], ps_num=2, ps_range=4)
    final = case["final"]
    if impl in {"nss_cuda", "nss_vk"}:
        if os.environ.get("NSS_BENCH_NUM_STREAMS"):
            kw["num_streams"] = int(os.environ["NSS_BENCH_NUM_STREAMS"])
        return getattr(core, impl).BM3D(clip, group_size=8, final=int(final), **kw)
    if impl in {"bm3dcuda", "bm3dcuda_rtc", "vszip"}:
        ns = getattr(core, impl)
        if not final:
            return ns.BM3D(clip, **kw)
        basic = ns.BM3Dv2(clip, **kw)
        return ns.BM3Dv2(clip, ref=basic, **kw)
    if impl == "vsfeel":
        basic = core.vsfeel.BM3D(clip, **kw) if final else None
        return core.vsfeel.BM3D(clip, ref=basic, **kw) if final else core.vsfeel.BM3D(clip, **kw)
    if impl == "bm3dvk2":
        kw.update(profile="fast", block_size=8, group_size=8)
        basic = core.bm3dvk2.BM3D(clip, **kw) if final else None
        return core.bm3dvk2.BM3D(clip, ref=basic, **kw) if final else core.bm3dvk2.BM3D(clip, **kw)
    raise ValueError(impl)


def nlm_node(core, impl, clip, case):
    kw = dict(d=case["d"], a=case["a"], s=case["s"], h=1.2, channels="Y")
    if impl in {"nss_cuda", "nss_vk"}:
        if os.environ.get("NSS_BENCH_NUM_STREAMS"):
            kw["num_streams"] = int(os.environ["NSS_BENCH_NUM_STREAMS"])
        return getattr(core, impl).NLM(clip, **kw)
    if impl == "nlm_cuda":
        return core.nlm_cuda.NLMeans(clip, **kw)
    if impl in {"vsfeel", "vszipcl", "vszip"}:
        return getattr(core, impl).NLMeans(clip, **kw)
    raise ValueError(impl)


def pnlm_node(core, impl, clip, case):
    kw = dict(sigma=case["sigma"], d=case["d"], a=case["a"],
              block_size=case["block_size"], group_size=case["group_size"])
    if os.environ.get("NSS_BENCH_NUM_STREAMS"):
        kw["num_streams"] = int(os.environ["NSS_BENCH_NUM_STREAMS"])
    return getattr(core, impl).PNLM(clip, **kw)


def make_node(core, impl, clip, case, mode):
    if mode == "resident" and impl not in RESIDENT_IMPLS:
        raise ValueError(f"{impl} has no GPU-frame path")
    gpu_only = impl in {"vsfeel", "bm3dvk2"}
    source = clip.std.GPUUpload() if mode == "resident" or gpu_only else clip
    if case["family"] == "bm3d":
        node = bm3d_node(core, impl, source, case, mode)
    elif case["family"] == "pnlm":
        node = pnlm_node(core, impl, source, case)
    else:
        node = nlm_node(core, impl, source, case)
    if gpu_only and mode == "cpu":
        node = node.std.GPUDownload()
    if mode == "resident" and not is_resident(node):
        raise RuntimeError(f"{impl} returned a CPU frame in resident mode")
    return node


def setup(impl, case_id, mode, threads):
    import vapoursynth as vs
    core = vs.core
    core.num_threads = threads
    if mode == "resident" or impl in {"vsfeel", "bm3dvk2"}:
        core.set_vulkan_device(0)
    if impl == "nss_cuda" and mode == "resident":
        os.environ["NSS_CUDA_GPU_FRAMES"] = "1"
    else:
        os.environ.pop("NSS_CUDA_GPU_FRAMES", None)
    load_plugin(core, impl)
    case = CASES[case_id]
    _, noisy = make_input("gseq-s10")
    clip = to_clip(core, vs, noisy, INPUTS["gseq-s10"][1], length=1 << 20)
    return core, clip, make_node(core, impl, clip, case, mode)


class MemoryMonitor:
    def __init__(self):
        self.baseline = 0
        self.peak = 0
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def read(self):
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == str(os.getpid()) and fields[1].isdigit():
                return int(fields[1])
        return 0

    def run(self):
        while not self.stop_event.is_set():
            current = self.read()
            self.peak = max(self.peak, current)
            self.samples += 1
            self.stop_event.wait(0.05)

    def start(self):
        self.baseline = self.read()
        self.peak = self.baseline
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.read())


def is_resident(node):
    return bool(getattr(node, "gpu_resident", False))


def output_arrays(node, frame_number):
    if is_resident(node):
        node = node.std.GPUDownload()
    frame = node.get_frame(frame_number)
    return [np.array(frame[p], dtype=np.float32, copy=True) for p in range(frame.format.num_planes)]


def quality(args):
    import vapoursynth as vs
    core, clip, node = setup(args.impl, args.case, args.mode, args.threads)
    clean, noisy = make_input("gseq-s10")
    index = 2
    out = output_arrays(node, index)
    flat = np.concatenate([plane.ravel() for plane in out])
    record = dict(schema=SCHEMA, operation="quality", impl=args.impl, case=args.case, mode=args.mode,
                  core=vs.__version__, input_sha256=arrays_sha256(noisy),
                  output_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                  clean_psnr=psnr([out], [clean[index]]), finite=bool(np.isfinite(flat).all()),
                  min=float(flat.min()), max=float(flat.max()), resident=bool(is_resident(node)))
    print(json.dumps(record), flush=True)


def bench(args):
    import vapoursynth as vs
    monitor = MemoryMonitor()
    monitor.start()
    core, clip, node = setup(args.impl, args.case, args.mode, args.threads)
    clip = None

    def run(start, count):
        begin = time.perf_counter()
        for _ in node[start:start + count].frames(close=True):
            pass
        return count / (time.perf_counter() - begin)

    warm_count = max(8, args.frames // 4)
    rate = run(0, warm_count)
    warm_count = max(warm_count, int(rate * args.warm_seconds) + 1)
    rate = run(warm_count, warm_count)
    count = max(args.frames, int(rate * args.seconds) + 1)
    start = 2 * warm_count
    fps = []
    for index in range(args.runs):
        fps.append(run(start + index * count, count))
    frame = node.get_frame(0)
    props = frame.props
    resource_peak = int(props.get("_NSSResourcePeak", 0))
    stats = props.get("_NSSResourceBytes")
    resource_bytes = sum(int(value) for value in stats[:5]) if stats is not None else 0
    del frame
    monitor.stop()
    record = dict(schema=SCHEMA, operation="bench", impl=args.impl, case=args.case, mode=args.mode,
                  core=vs.__version__, threads=args.threads, median_fps=statistics.median(fps), fps=fps,
                  frames_per_run=count,
                  gpu_mib_baseline=monitor.baseline, gpu_mib_peak=monitor.peak, gpu_samples=monitor.samples,
                  rss_mib_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  resource_mib_peak=resource_peak / (1 << 20), resource_mib_current=resource_bytes / (1 << 20),
                  resident=bool(is_resident(node)))
    print(json.dumps(record), flush=True)


def selected_jobs(args):
    cases = [case_id for case_id in CASES if case_id.startswith(args.family + "-")]
    implementations = {"bm3d": BM3D_IMPLS, "nlm": NLM_IMPLS, "pnlm": PNLM_IMPLS}[args.family]
    if args.impls:
        implementations = [impl for impl in implementations if impl in args.impls.split(",")]
    jobs = []
    selected_modes = args.modes.split(",") if hasattr(args, "modes") else ["cpu", "resident"]
    for case_id in cases:
        for impl in implementations:
            modes = [mode for mode in selected_modes if mode == "cpu" or (impl in RESIDENT_IMPLS and args.resident)]
            for mode in modes:
                jobs.append((case_id, impl, mode))
    return jobs


def campaign(args):
    jobs = selected_jobs(args)
    out = Path(args.out)
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                rec = json.loads(line)
                completed.add((rec["case"], rec["impl"], rec["mode"], rec.get("round", 0)))
            except (json.JSONDecodeError, KeyError):
                pass
    for round_index in range(args.rounds):
        ordered = jobs if round_index % 2 == 0 else list(reversed(jobs))
        for case_id, impl, mode in ordered:
            if (case_id, impl, mode, round_index) in completed:
                continue
            cmd = [sys.executable, __file__, "bench", "--case", case_id, "--impl", impl, "--mode", mode,
                   "--threads", str(args.threads), "--runs", str(args.runs), "--seconds", str(args.seconds),
                   "--warm-seconds", str(args.warm_seconds), "--frames", str(args.frames)]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
            begin = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
            if proc.returncode == 0 and proc.stdout.strip():
                rec = json.loads(proc.stdout.strip().splitlines()[-1])
            else:
                rec = dict(schema=SCHEMA, operation="bench", case=case_id, impl=impl, mode=mode,
                           error=(proc.stderr.strip().splitlines() or ["worker failed"])[-1], returncode=proc.returncode)
            rec["round"] = round_index
            rec["wall_seconds"] = time.perf_counter() - begin
            with out.open("a") as handle:
                handle.write(json.dumps(rec) + "\n")
            print(f"r{round_index} {case_id:14s} {impl:14s} {mode:8s} "
                  f"{rec.get('median_fps', rec.get('error'))}", flush=True)


def quality_campaign(args):
    out = Path(args.out)
    for case_id, impl, mode in selected_jobs(args):
        cmd = [sys.executable, __file__, "quality", "--case", case_id, "--impl", impl, "--mode", mode,
               "--threads", str(args.threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ, timeout=args.timeout)
        if proc.returncode == 0 and proc.stdout.strip():
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
        else:
            rec = dict(schema=SCHEMA, operation="quality", case=case_id, impl=impl, mode=mode,
                       error=(proc.stderr.strip().splitlines() or ["worker failed"])[-1], returncode=proc.returncode)
        with out.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(f"{case_id:14s} {impl:14s} {mode:8s} {rec.get('clean_psnr', rec.get('error'))}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ["bench", "quality"]:
        child = sub.add_parser(name)
        child.add_argument("--case", choices=CASES, required=True)
        child.add_argument("--impl", required=True)
        child.add_argument("--mode", choices=["cpu", "resident"], default="cpu")
        child.add_argument("--threads", type=int, default=8)
        child.add_argument("--runs", type=int, default=3)
        child.add_argument("--seconds", type=float, default=2.0)
        child.add_argument("--warm-seconds", type=float, default=1.0)
        child.add_argument("--frames", type=int, default=32)
    for name in ["campaign", "quality-campaign"]:
        child = sub.add_parser(name)
        child.add_argument("--family", choices=["bm3d", "nlm", "pnlm"], required=True)
        child.add_argument("--out", required=True)
        child.add_argument("--impls")
        child.add_argument("--modes", default="cpu,resident")
        child.add_argument("--threads", type=int, default=8)
        child.add_argument("--rounds", type=int, default=2)
        child.add_argument("--runs", type=int, default=3)
        child.add_argument("--seconds", type=float, default=2.0)
        child.add_argument("--warm-seconds", type=float, default=1.0)
        child.add_argument("--frames", type=int, default=32)
        child.add_argument("--timeout", type=float, default=300)
        child.add_argument("--resident", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.command == "list":
        for case_id, case in CASES.items():
            print(case_id, json.dumps(case, sort_keys=True))
    else:
        globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
