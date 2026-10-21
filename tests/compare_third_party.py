#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Output and speed comparison of nss / nss_cuda against third-party plugins.

Same input clip, same arguments: BM3D against bm3dcuda / bm3dcuda_rtc / bm3dcpu
(VapourSynth-BM3DCUDA), WNNM against wnnm (VapourSynth-WNNM), NLM against
nlm_ispc (vs-nlm-ispc) and nlm_cuda (vs-nlm-cuda).

  quality: for every case, PSNR of each output against the clean image, and
           agreement (PSNR / max abs / share of samples off by > 1e-3) against
           the case's anchor output (nss CPU).
  bench:   one implementation x one case per process: warm up, then --runs
           timed passes of at least --frames frames and --min-seconds seconds
           each; prints the median and every run. An implementation named
           `nss_cuda@r1` is nss_cuda loaded from NSS_CUDA_R1_SO (an earlier
           release), with the same arguments.

Inputs are the ds/ snapshot images (1920x1080) with seeded AWGN, built the same
way for every implementation. Temporal inputs are a 5-frame pan (shift t rows,
2t columns, edge-padded).

usage: compare_third_party.py quality [--only REGEX] [--report FILE]
       compare_third_party.py bench --impl NAME --case ID [--threads N] [--frames N] [--runs N]
       compare_third_party.py bench-all --out FILE [--single] [--rounds N] [--only REGEX]
       compare_third_party.py summarize [--quality FILE...] [--bench FILE]
       compare_third_party.py list

Plugin paths: NSS_SO, NSS_CUDA_SO, NSS_CUDA_R1_SO, NLM_CUDA_SO (defaults: build-cuda/,
~/src/vs-nlm-cuda/build/); THIRD_PARTY_SO, a colon-separated list loaded by path when autoloading is off. Published results: docs/api/cuda/compare.md.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "ds"
SCHEMA = "nssfactory.third_party_compare.v1"
PLUGINS = {
    "nss": os.environ.get("NSS_SO", str(REPO / "build-cuda" / "libnss.so")),
    "nss_cuda": os.environ.get("NSS_CUDA_SO", str(REPO / "build-cuda" / "src" / "cuda" / "libnss_cuda.so")),
    "nlm_cuda": os.environ.get("NLM_CUDA_SO", str(Path.home() / "src/vs-nlm-cuda/build/libvsnlm_cuda.so")),
}

# name: (image, kind, sigma8, frames)
INPUTS = {
    "g-s5": ("MAPPA.png", "gray", 5.0, 1),
    "g-s25": ("MAPPA.png", "gray", 25.0, 1),
    "yuv444-s10": ("Ufotable.png", "yuv444", 10.0, 1),
    "rgb-s5": ("Ufotable.png", "rgb", 5.0, 1),
    "gseq-s10": ("MAPPA.png", "gray", 10.0, 5),
}


def _seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def _convert(rgb, kind):
    if kind == "rgb":
        return list(rgb)
    if kind == "gray":
        return [np.einsum("c,chw->hw", np.array([0.299, 0.587, 0.114]), rgb)]
    if kind == "yuv444":  # full-range BT.709, centred chroma
        r, g, b = rgb
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return [y, (b - y) / 1.8556, (r - y) / 1.5748]
    raise ValueError(kind)


def make_input(name):
    """(clean, noisy): lists over frames of lists of float32 planes."""
    from PIL import Image
    image, kind, sigma8, frames = INPUTS[name]
    with Image.open(DS / image) as pic:
        rgb = np.asarray(pic.convert("RGB"), dtype=np.float64).transpose(2, 0, 1) / 255.0
    _, h, w = rgb.shape
    pad = 2 * frames
    padded = np.pad(rgb, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    clean_frames, noisy_frames = [], []
    for t in range(frames):
        crop = padded[:, pad + t:pad + t + h, pad + 2 * t:pad + 2 * t + w]
        clean, noisy = [], []
        for p, plane in enumerate(_convert(crop, kind)):
            rng = np.random.Generator(np.random.PCG64(_seed(f"{SCHEMA}:{name}:{t}:{p}")))
            clean.append(np.ascontiguousarray(plane, dtype=np.float32))
            noisy.append((plane + rng.standard_normal(plane.shape) * (sigma8 / 255.0)).astype(np.float32))
        clean_frames.append(clean)
        noisy_frames.append(noisy)
    return clean_frames, noisy_frames


def arrays_sha256(frames):
    h = hashlib.sha256()
    for planes in frames:
        for plane in planes:
            h.update(np.ascontiguousarray(plane, dtype="<f4").tobytes())
    return h.hexdigest()


def to_clip(core, vs, frames, kind, length=None):
    """A clip of `length` frames cycling through `frames`. The frames are built
    once and handed out as-is, so the source costs one Python call per frame."""
    fmt = {"gray": vs.GRAYS, "rgb": vs.RGBS, "yuv444": vs.YUV444PS}[kind]
    height, width = frames[0][0].shape
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=len(frames))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    filled = core.std.ModifyFrame(blank, blank, fill)
    store = [filled.get_frame(n) for n in range(len(frames))]
    src = core.std.BlankClip(filled, length=length or len(frames))
    return core.std.ModifyFrame(src, src, lambda n, f: store[n % len(store)])


# ---- implementations -------------------------------------------------------
# Each case maps implementation name -> builder(core, clip) with the same
# arguments spelled in that plugin's vocabulary. The first entry is the anchor.

def _bm3d_basic(**kw):
    return {
        "nss": lambda c, x: c.nss.BM3D(x, **kw),
        "nss_cuda": lambda c, x: c.nss_cuda.BM3D(x, **kw),
        "bm3dcuda": lambda c, x: c.bm3dcuda.BM3D(x, **kw),
        "bm3dcuda_rtc": lambda c, x: c.bm3dcuda_rtc.BM3D(x, **kw),
        "bm3dcpu": lambda c, x: c.bm3dcpu.BM3D(x, **kw),
    }


def _bm3d_final(**kw):
    def third(ns):
        return lambda c, x: getattr(c, ns).BM3D(x, ref=getattr(c, ns).BM3D(x, **kw), **kw)
    return {
        "nss": lambda c, x: c.nss.BM3D(x, final=1, **kw),
        "nss_cuda": lambda c, x: c.nss_cuda.BM3D(x, final=1, **kw),
        "bm3dcuda": third("bm3dcuda"),
        "bm3dcuda_rtc": third("bm3dcuda_rtc"),
        "bm3dcpu": third("bm3dcpu"),
    }


def _bm3d_temporal(radius, **kw):
    return {
        "nss": lambda c, x: c.nss.VAggregate(c.nss.BM3D(x, radius=radius, temporal_mode="legacy", **kw), x, radius=radius),
        "nss_rolling": lambda c, x: c.nss.BM3D(x, radius=radius, temporal_mode="rolling", **kw),
        "nss_cuda": lambda c, x: c.nss_cuda.BM3D(x, radius=radius, temporal_mode="rolling", **kw),
        "bm3dcuda": lambda c, x: c.bm3dcuda.BM3Dv2(x, radius=radius, **kw),
        "bm3dcuda_rolling": lambda c, x: c.bm3dcuda.BM3Dv2(x, radius=radius, temporal_mode="rolling", **kw),
        "bm3dcuda_rtc": lambda c, x: c.bm3dcuda_rtc.BM3Dv2(x, radius=radius, **kw),
        "bm3dcpu": lambda c, x: c.bm3dcpu.BM3Dv2(x, radius=radius, **kw),
    }


def _wnnm(**kw):
    return {
        "nss": lambda c, x: c.nss.WNNM(x, **kw),
        "nss_cuda": lambda c, x: c.nss_cuda.WNNM(x, **kw),
        "wnnm": lambda c, x: c.wnnm.WNNM(x, **kw),
    }


def _wnnm_temporal(radius, **kw):
    return {
        "nss": lambda c, x: c.nss.VAggregate(c.nss.WNNM(x, radius=radius, temporal_mode="legacy", **kw), x, radius=radius),
        "nss_cuda": lambda c, x: c.nss_cuda.WNNM(x, radius=radius, temporal_mode="rolling", **kw),
        "wnnm": lambda c, x: c.wnnm.WNNM(x, radius=radius, **kw),  # aggregates internally
    }


def _nlm(**kw):
    return {
        "nss": lambda c, x: c.nss.NLM(x, **kw),
        "nss_cuda": lambda c, x: c.nss_cuda.NLM(x, **kw),
        "nlm_ispc": lambda c, x: c.nlm_ispc.NLMeans(x, **kw),
        "nlm_cuda": lambda c, x: c.nlm_cuda.NLMeans(x, **kw),
    }


GEO = dict(block_step=8, bm_range=7)  # BM3DCUDA's bm_range default is 9; pin both sides
CASES = {
    "bm3d-g-s5": ("g-s5", _bm3d_basic(sigma=5, **GEO)),
    "bm3d-g-s25": ("g-s25", _bm3d_basic(sigma=25, **GEO)),
    "bm3d-g-s25-step4": ("g-s25", _bm3d_basic(sigma=25, block_step=4, bm_range=12)),
    "bm3d-final-g-s5": ("g-s5", _bm3d_final(sigma=5, **GEO)),
    "bm3d-final-g-s25": ("g-s25", _bm3d_final(sigma=25, **GEO)),
    "bm3d-yuv444-s10-chroma": ("yuv444-s10", _bm3d_basic(sigma=10, chroma=1, **GEO)),
    "bm3d-yuv444-s10-planes": ("yuv444-s10", _bm3d_basic(sigma=10, **GEO)),
    "bm3d-gseq-s10-r1": ("gseq-s10", _bm3d_temporal(1, sigma=10, **GEO)),
    "wnnm-g-s5": ("g-s5", _wnnm(sigma=5)),
    "wnnm-g-s25": ("g-s25", _wnnm(sigma=25)),
    "wnnm-g-s25-step4g16": ("g-s25", _wnnm(sigma=25, block_step=4, group_size=16, bm_range=10)),
    "wnnm-gseq-s10-r1": ("gseq-s10", _wnnm_temporal(1, sigma=10)),
    "nlm-g-s5-d0": ("g-s5", _nlm(d=0, a=2, s=4, h=4.0)),
    "nlm-g-s25-d0-a3": ("g-s25", _nlm(d=0, a=3, s=4, h=18.0)),
    "nlm-rgb-s5-d0": ("rgb-s5", _nlm(d=0, a=2, s=4, h=4.0, channels="RGB")),
    "nlm-gseq-s10-d1": ("gseq-s10", _nlm(d=1, a=2, s=4, h=8.0)),
}


def core_with_plugins(threads):
    import vapoursynth as vs
    core = vs.core
    core.num_threads = threads
    for ns, path in PLUGINS.items():
        if not hasattr(core, ns) and Path(path).exists():
            core.std.LoadPlugin(path=path)
    # Third-party plugins by path, for cores without autoloading (colon-separated list).
    for path in filter(None, os.environ.get("THIRD_PARTY_SO", "").split(":")):
        try:
            core.std.LoadPlugin(path=path)
        except vs.Error as e:
            if "already loaded" not in str(e):
                raise
    return vs, core


def fetch(node, frames):
    out = []
    for n in frames:
        f = node.get_frame(n)
        out.append([np.array(f[p], dtype=np.float32, copy=True) for p in range(f.format.num_planes)])
    return out


def _flat(frames):
    return [p for planes in frames for p in planes]


def psnr(a, b):
    num = sum(float(np.sum((x.astype(np.float64) - y) ** 2)) for x, y in zip(_flat(a), _flat(b)))
    mse = num / sum(x.size for x in _flat(a))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def agreement(test, ref):
    diff = np.concatenate([np.abs(t.astype(np.float64) - r).ravel() for t, r in zip(_flat(test), _flat(ref))])
    return dict(psnr=psnr(test, ref), max_abs=float(diff.max()), frac_gt_1e3=float(np.mean(diff > 1e-3)))


def quality(args):
    vs, core = core_with_plugins(args.threads)
    rows, inputs = [], {}
    for cid, (inp, impls) in CASES.items():
        if args.only and not re.search(args.only, cid):
            continue
        if inp not in inputs:
            inputs[inp] = make_input(inp)
        clean, noisy = inputs[inp]
        kind = INPUTS[inp][1]
        clip = to_clip(core, vs, noisy, kind)
        frames = list(range(len(noisy)))
        anchor = None
        row = dict(case=cid, input=inp, input_sha256=arrays_sha256(noisy),
                   noisy_psnr=psnr(noisy, clean), impls={})
        for name, make in impls.items():
            try:
                t0 = time.perf_counter()
                out = fetch(make(core, clip), frames)
                rec = dict(clean_psnr=psnr(out, clean), seconds=time.perf_counter() - t0,
                           finite=bool(all(np.isfinite(p).all() for p in _flat(out))))
                if kind != "gray":
                    rec["plane_clean_psnr"] = [psnr([[f[p]] for f in out], [[f[p]] for f in clean])
                                               for p in range(len(out[0]))]
                if anchor is None:
                    anchor = (name, out)
                else:
                    rec["vs_" + anchor[0]] = agreement(out, anchor[1])
            except (vs.Error, AttributeError) as e:
                rec = dict(error=(str(e).splitlines() or [type(e).__name__])[0])
            row["impls"][name] = rec
            line = (f"{rec['clean_psnr']:8.3f} dB" if "clean_psnr" in rec else "   error   ")
            if anchor and "vs_" + anchor[0] in rec:
                a = rec["vs_" + anchor[0]]
                line += f"  vs {anchor[0]}: {a['psnr']:7.2f} dB  max {a['max_abs']:.2e}  >1e-3 {100 * a['frac_gt_1e3']:.3f}%"
            if "error" in rec:
                line += "  " + rec["error"]
            print(f"{cid:26s} {name:16s} {line}", flush=True)
        rows.append(row)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(schema=SCHEMA, cases=rows), indent=1))


def bench(args):
    impl = args.impl
    if impl.endswith("@r1"):  # the same builder on an earlier release of the plugin
        PLUGINS[impl[:-3]] = os.environ["NSS_CUDA_R1_SO"]
        impl = impl[:-3]
    vs, core = core_with_plugins(args.threads)
    inp, impls = CASES[args.case]
    _, noisy = make_input(inp)
    clip = to_clip(core, vs, noisy, INPUTS[inp][1], length=1 << 20)
    node = clip if impl == "source" else impls[impl](core, clip)
    serial = args.threads == 1

    def run(start, count):
        t = time.perf_counter()
        if serial:
            for n in range(start, start + count):
                node.get_frame(n)
        else:
            for _ in node[start:start + count].frames(close=True):
                pass
        return count / (time.perf_counter() - t)
    warm = max(4, args.frames // 4)
    rate = run(0, warm)
    count = max(args.frames, int(rate * args.min_seconds) + 1)  # each timed pass at least min_seconds
    fps = [run(warm + i * count, count) for i in range(args.runs)]
    print(json.dumps(dict(impl=args.impl, case=args.case, threads=args.threads, frames=args.frames,
                          median_fps=statistics.median(fps), fps=fps)), flush=True)


GPU_IMPLS = {"nss_cuda", "bm3dcuda", "bm3dcuda_rolling", "bm3dcuda_rtc", "nlm_cuda"}
BENCH_CASES = ["bm3d-g-s5", "bm3d-final-g-s5", "bm3d-yuv444-s10-chroma", "bm3d-gseq-s10-r1",
               "wnnm-g-s5", "wnnm-gseq-s10-r1", "nlm-g-s5-d0", "nlm-rgb-s5-d0", "nlm-gseq-s10-d1"]


def bench_all(args):
    """Every (case, impl, threads) in its own process; --rounds rounds in
    alternating order. Appends one JSON line per process to --out."""
    import subprocess
    jobs = []
    for cid in BENCH_CASES:
        if args.only and not re.search(args.only, cid):
            continue
        versions = ["nss_cuda@r1"] if args.r1 and "nss_cuda" in CASES[cid][1] else []
        for impl in ["source", *CASES[cid][1], *versions]:
            gpu = impl in GPU_IMPLS or impl.endswith("@r1")
            for threads in ([args.gpu_threads] if gpu or impl == "source" else [1, args.cpu_threads]):
                if threads == 1 and not args.single:
                    continue
                frames = args.frames if threads > 1 else max(8, args.frames // 4)
                if impl == "source":
                    frames = 512
                jobs.append((cid, impl, threads, frames))
    for r in range(args.rounds):
        for cid, impl, threads, frames in (jobs if r % 2 == 0 else jobs[::-1]):
            cmd = [sys.executable, __file__, "bench", "--case", cid, "--impl", impl, "--threads", str(threads),
                   "--frames", str(frames), "--runs", str(args.runs), "--min-seconds", str(args.min_seconds)]
            env = dict(os.environ)
            env.setdefault("MKL_NUM_THREADS", "1")  # wnnm: VS threads only, as for every other CPU plugin
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else \
                json.dumps(dict(impl=impl, case=cid, threads=threads, error=(p.stderr.strip().splitlines() or ["?"])[-1]))
            rec = json.loads(line)
            rec["round"] = r
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"r{r} {cid:24s} {impl:16s} t{threads:<3d} {rec.get('median_fps', rec.get('error'))}", flush=True)


def summarize(args):
    """Markdown tables from quality reports and bench JSON lines."""
    for path in args.quality or []:
        for row in json.loads(Path(path).read_text())["cases"]:
            print(f"\n**{row['case']}** (noisy {row['noisy_psnr']:.2f} dB)\n")
            print("| impl | clean PSNR (dB) | vs anchor PSNR (dB) | max abs | > 1e-3 |")
            print("|---|---:|---:|---:|---:|")
            for name, rec in row["impls"].items():
                if "error" in rec:
                    print(f"| {name} | error: {rec['error']} | | | |")
                    continue
                vs_key = next((k for k in rec if k.startswith("vs_")), None)
                a = rec.get(vs_key) if vs_key else None
                tail = (f"{a['psnr']:.2f} | {a['max_abs']:.2e} | {100 * a['frac_gt_1e3']:.3f}%" if a
                        else "anchor | | ")
                print(f"| {name} | {rec['clean_psnr']:.3f} | {tail} |")
    if args.bench:
        pooled = {}
        for line in Path(args.bench).read_text().splitlines():
            rec = json.loads(line)
            pooled.setdefault((rec["case"], rec["impl"], rec["threads"]), []).extend(rec.get("fps", []))
        print("\n| case | impl | threads | median fps | min | max | n |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for (cid, impl, threads), fps in pooled.items():
            if fps:
                print(f"| {cid} | {impl} | {threads} | {statistics.median(fps):.1f} | {min(fps):.1f} | {max(fps):.1f} | {len(fps)} |")
            else:
                print(f"| {cid} | {impl} | {threads} | error | | | |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--only")
    q.add_argument("--report")
    q.add_argument("--threads", type=int, default=os.cpu_count())
    b = sub.add_parser("bench")
    b.add_argument("--impl", required=True)
    b.add_argument("--case", required=True)
    b.add_argument("--threads", type=int, default=os.cpu_count())
    b.add_argument("--frames", type=int, default=64)
    b.add_argument("--runs", type=int, default=5)
    b.add_argument("--min-seconds", type=float, default=2.0)
    a = sub.add_parser("bench-all")
    a.add_argument("--out", required=True)
    a.add_argument("--only")
    a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--frames", type=int, default=64)
    a.add_argument("--cpu-threads", type=int, default=os.cpu_count())
    a.add_argument("--gpu-threads", type=int, default=8)
    a.add_argument("--single", action="store_true", help="also time CPU impls single-threaded")
    a.add_argument("--min-seconds", type=float, default=2.0)
    a.add_argument("--r1", action="store_true", help="also time nss_cuda@r1 (NSS_CUDA_R1_SO)")
    m = sub.add_parser("summarize")
    m.add_argument("--quality", nargs="*")
    m.add_argument("--bench")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "list":
        for cid, (inp, impls) in CASES.items():
            print(cid, inp, " ".join(impls))
    else:
        {"quality": quality, "bench": bench, "bench-all": bench_all, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
