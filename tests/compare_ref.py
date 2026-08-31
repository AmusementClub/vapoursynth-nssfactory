#!/usr/bin/env python3
"""Compare nss vs nlm_ispc / bm3dcpu on PSNR and wall time."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import vapoursynth as vs

core = vs.core

NSS = os.environ.get("NSS_SO", "/home/owen/nss/build/libnss.so")
BM3DCPU = os.environ.get("BM3DCPU_SO", "/home/owen/VapourSynth-BM3DCUDA/build/cpu_source/libbm3dcpu.so")
NLM_ISPC = os.environ.get("NLM_ISPC_SO", "/home/owen/vs-nlm-ispc/build/libvsnlm_ispc.so")
WNNM_SO = os.environ.get("WNNM_SO", "")

core.std.LoadPlugin(NSS)
core.std.LoadPlugin(BM3DCPU)
core.std.LoadPlugin(NLM_ISPC)
has_wnnm = False
if WNNM_SO and os.path.isfile(WNNM_SO):
    core.std.LoadPlugin(WNNM_SO)
    has_wnnm = hasattr(core, "wnnm")


def plane_clip(arr: np.ndarray, nframes: int) -> vs.VideoNode:
    h, w = arr.shape
    base = core.std.BlankClip(
        width=w, height=h, format=vs.GRAYS, length=nframes, fpsnum=24, fpsden=1, color=[0.0]
    )
    buf = np.ascontiguousarray(arr, dtype=np.float32)

    def sel(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        np.asarray(fout[0])[:, :] = buf
        return fout

    return core.std.ModifyFrame(clip=base, clips=base, selector=sel)


def frame_np(clip: vs.VideoNode, n: int = 0) -> np.ndarray:
    f = clip.get_frame(n)
    return np.array(f[0], dtype=np.float32, copy=True)


def psnr_db(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def time_ms(clip: vs.VideoNode, frames: int) -> float:
    clip.get_frame(0)
    t0 = time.perf_counter()
    for i in range(frames):
        clip.get_frame(i)
    return (time.perf_counter() - t0) * 1000.0 / float(frames)


def make_pair(w: int, h: int, seed: int, sigma_8bit: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    clean = (0.35 + 0.25 * np.sin(xx / 18.0) * np.cos(yy / 14.0) + 0.05 * (xx + yy) / (w + h)).astype(
        np.float32
    )
    noisy = clean + rng.randn(h, w).astype(np.float32) * np.float32(sigma_8bit / 255.0)
    return clean, noisy


def report(name: str, clean: np.ndarray, noisy: np.ndarray, den: vs.VideoNode, frames: int) -> dict:
    out = frame_np(den, 0)
    row = {
        "name": name,
        "psnr_clean": psnr_db(out, clean),
        "psnr_noisy": psnr_db(out, noisy),
        "finite": bool(np.isfinite(out).all()),
        "ms": time_ms(den, frames),
        "arr": out,
    }
    print(
        f"{name:22s}  PSNR vs clean {row['psnr_clean']:7.3f} dB  vs noisy {row['psnr_noisy']:7.3f} dB  "
        f"{row['ms']:8.2f} ms/frame  finite={row['finite']}"
    )
    return row


def run_case(tag: str, w: int, h: int, frames: int, threads: int) -> None:
    core.num_threads = threads
    sigma = 3.0
    clean, noisy = make_pair(w, h, seed=42, sigma_8bit=sigma)
    src = plane_clip(noisy, frames)
    print(f"\n=== {tag}  {w}x{h}  frames={frames}  threads={threads}  AWGN sigma={sigma} ===")
    print(f"{'noisy vs clean':22s}  PSNR vs clean {psnr_db(noisy, clean):7.3f} dB")

    nlm_nss = core.nss.NLM(src, d=1, a=2, s=4, h=1.2, channels="Y")
    nlm_ref = core.nlm_ispc.NLMeans(src, d=1, a=2, s=4, h=1.2, channels="Y")
    a = report("nss.NLM", clean, noisy, nlm_nss, frames)
    b = report("nlm_ispc.NLMeans", clean, noisy, nlm_ref, frames)
    print(f"{'NLM nss vs ref':22s}  PSNR {psnr_db(a['arr'], b['arr']):7.3f} dB  "
          f"delta_clean {a['psnr_clean'] - b['psnr_clean']:+7.3f} dB  "
          f"time {b['ms'] / a['ms'] if a['ms'] else 0:.2f}x (ref/nss)")

    bm_nss = core.nss.BM3D(src, sigma=sigma, radius=0)
    bm_ref = core.bm3dcpu.BM3D(src, sigma=sigma, radius=0)
    a = report("nss.BM3D", clean, noisy, bm_nss, frames)
    b = report("bm3dcpu.BM3D", clean, noisy, bm_ref, frames)
    print(f"{'BM3D nss vs ref':22s}  PSNR {psnr_db(a['arr'], b['arr']):7.3f} dB  "
          f"delta_clean {a['psnr_clean'] - b['psnr_clean']:+7.3f} dB  "
          f"time {b['ms'] / a['ms'] if a['ms'] else 0:.2f}x (ref/nss)")

    if has_wnnm:
        w_nss = core.nss.WNNM(src, sigma=sigma, residual=0, radius=0)
        w_ref = core.wnnm.WNNM(src, sigma=sigma, residual=0, radius=0)
        a = report("nss.WNNM", clean, noisy, w_nss, frames)
        b = report("wnnm.WNNM", clean, noisy, w_ref, frames)
        print(f"{'WNNM nss vs ref':22s}  PSNR {psnr_db(a['arr'], b['arr']):7.3f} dB  "
              f"delta_clean {a['psnr_clean'] - b['psnr_clean']:+7.3f} dB  "
              f"time {b['ms'] / a['ms'] if a['ms'] else 0:.2f}x (ref/nss)")
    else:
        w_nss = core.nss.WNNM(src, sigma=sigma, residual=0, radius=0)
        report("nss.WNNM (no ref plugin)", clean, noisy, w_nss, frames)


def main() -> None:
    print("plugins: nss, bm3dcpu, nlm_ispc", "+ wnnm" if has_wnnm else "(wnnm ref missing)")
    run_case("psnr", 256, 256, frames=1, threads=1)
    run_case("speed", 1280, 720, frames=2, threads=1)
    run_case("speed-2t", 1280, 720, frames=2, threads=2)


if __name__ == "__main__":
    main()
