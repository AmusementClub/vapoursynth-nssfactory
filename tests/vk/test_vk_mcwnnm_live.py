#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.MCWNNM vs nss.MCWNNM on small random RGB clips (docs/vulkan-plan.md V8):
block sizes by group sizes 1 to 32 (the three kernels up to 8, the block
kernel's parallel Jacobi to 32), residual, unequal and zero sigmas (several
channel classes), admm_iter / rho / mu, outer rounds (iters, delta),
adaptive aggregation off, rclip, memory_limit_mb and many threads; then the
temporal outputs (radius 1 / 2, one and two rounds): legacy + nss.VAggregate
against the CPU, rolling against the legacy output reduced as nss.VAggregate
does it. Each case must reach the MCWNNM tolerance
(tests/data/vk_tolerances_v1.json) and be run-to-run identical; with
NSS_VK_FORCE_EMULATED_WARP=1 the emulated warp layer's kernels run, which
give the native layer's bits (checked against a native node of the same
clip). Exits 77 without VapourSynth or a supported Vulkan device.

usage: test_vk_mcwnnm_live.py --cpu PATH --vk PATH [--device N] [--quick]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import vapoursynth as vs
except ImportError as error:
    print(f"vk mcwnnm live SKIP: {error}")
    raise SystemExit(77)

REPO = Path(__file__).resolve().parents[2]
BLOCKS = (2, 3, 5, 8, 9)  # 3 block^2 <= nss::kSvdMaxM
GROUPS = (1, 3, 8, 9, 13, 16, 17, 24, 32)


def make_clip(core, fmt, width, height, seed, length=2):
    rng = np.random.default_rng(seed)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=length)
    frames = []
    for t in range(length):
        planes = []
        for p in range(blank.format.num_planes):
            h, w = (height, width) if p == 0 else (height >> blank.format.subsampling_h, width >> blank.format.subsampling_w)
            y, x = np.mgrid[:h, :w]
            clean = 0.5 + 0.2 * np.sin((x + 2 * t) / (5 + p)) * np.cos((y + t) / 7) + 0.1 * (((x // 6) + (y // 5)) % 2)
            planes.append((clean + rng.normal(0, 10 / 255, clean.shape)).astype(np.float32))
        frames.append(planes)

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


class Reduced:
    """Legacy output reduced as nss.VAggregate does it (slices summed in
    ascending center order, identity rule), in IEEE float32."""

    def __init__(self, fat, src, radius):
        self.fat, self.src, self.radius = fat, src, radius

    def planes(self, n):
        first, last = max(0, n - self.radius), min(self.fat.num_frames - 1, n + self.radius)
        fats = [self.fat.get_frame(c) for c in range(first, last + 1)]
        src = self.src.get_frame(n)
        out = []
        for p in range(src.format.num_planes):
            s = np.asarray(src[p])
            h = s.shape[0]
            num = np.zeros_like(s)
            den = np.zeros_like(s)
            for c, fat in zip(range(first, last + 1), fats):
                a = np.asarray(fat[p])
                k = 2 * (n - c + self.radius)
                num = num + a[k * h:(k + 1) * h]
                den = den + a[(k + 1) * h:(k + 2) * h]
            with np.errstate(divide="ignore", invalid="ignore"):
                done = np.where(den == 1, num, np.where(den > np.float32(1e-12), num / den, s))
            out.append(done.astype(np.float64))
        return out


def frames(node, count=None):
    if count is None:
        count = node.src.num_frames if isinstance(node, Reduced) else node.num_frames
    if isinstance(node, Reduced):
        return [node.planes(n) for n in range(count)]
    out = []
    for n in range(count):
        f = node.get_frame(n)
        out.append([np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)])
    return out


def psnr(a, b):
    mse = np.mean([np.mean((x - y) ** 2) for fa, fb in zip(a, b) for x, y in zip(fa, fb)])
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def same(a, b):
    return all(np.array_equal(x, y) for fa, fb in zip(a, b) for x, y in zip(fa, fb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--vk", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    core = vs.core
    core.num_threads = 8
    core.std.LoadPlugin(args.cpu)
    try:
        core.std.LoadPlugin(args.vk)
        info = core.nss_vk.Backend(device_id=args.device)
        support = info.get("support", b"")
        support = support.decode() if isinstance(support, bytes) else support
        if support not in ("native", "emulated"):
            raise RuntimeError(f"device {args.device} is {support or 'missing'}")
    except Exception as error:  # noqa: BLE001
        print(f"vk mcwnnm live SKIP: {error}")
        return 77
    floor = json.loads((REPO / "tests" / "data" / "vk_tolerances_v1.json").read_text())["filters"]["MCWNNM"]["min_psnr"]
    emulated = os.environ.get("NSS_VK_FORCE_EMULATED_WARP") == "1"

    def vk(clip, **kw):
        return core.nss_vk.MCWNNM(clip, device_id=args.device, **kw)

    failures, worst, cases = [], float("inf"), 0

    def check(name, want, got, again):
        nonlocal worst, cases
        cases += 1
        p = psnr(want, got)
        worst = min(worst, p)
        if not same(got, again):
            failures.append(f"{name}: not run-to-run identical")
        if not p >= floor:
            failures.append(f"{name}: {p:.1f} dB from the CPU (< {floor})")

    # Spatial shapes (one round: the shapes' kernels; two below).
    clip = make_clip(core, vs.RGBS, 48, 40, 1)
    blocks = (2, 5, 8) if args.quick else BLOCKS
    groups = (1, 8, 9, 16, 17, 32) if args.quick else GROUPS
    for block in blocks:
        for group in groups:
            for residual in (0, 1):
                kw = dict(sigma=20, block_size=block, group_size=group, block_step=max(1, block // 2),
                          bm_range=4, ps_num=min(2, group), residual=residual, iters=1)
                name = f"b{block} g{group} r{residual}"
                check(name, frames(core.nss.MCWNNM(clip, **kw)), frames(vk(clip, **kw)), frames(vk(clip, **kw)))
    # Options, channel classes, rounds, guides and resources.
    guide = make_clip(core, vs.RGBS, 48, 40, 4)
    extra = [
        ("defaults", dict(sigma=20)),
        ("sigma [20, 10, 5]", dict(sigma=[20, 10, 5], iters=1)),
        ("sigma [20, 10, 5] g16", dict(sigma=[20, 10, 5], group_size=16, iters=1)),
        ("sigma [20, 10, 20] g32", dict(sigma=[20, 10, 20], group_size=32, iters=1)),
        ("sigma [20, 0, 20]", dict(sigma=[20, 0, 20], iters=1)),
        ("sigma [0, 15, 0] g16", dict(sigma=[0, 15, 0], group_size=16, iters=1)),
        ("admm_iter 1", dict(sigma=20, admm_iter=1, iters=1)),
        ("admm_iter 3 g16", dict(sigma=20, admm_iter=3, group_size=16, iters=1)),
        ("rho 1 mu 1.1", dict(sigma=20, rho=1.0, mu=1.1, iters=1)),
        ("rho 6 mu 1 g32", dict(sigma=20, rho=6.0, mu=1.0, group_size=32, iters=1)),
        ("iters 2", dict(sigma=20, iters=2)),
        ("iters 3 delta 0.3", dict(sigma=20, iters=3, delta=0.3)),
        ("iters 2 g16 delta 0", dict(sigma=20, iters=2, delta=0.0, group_size=16)),
        ("iters 2 g32 sigmas", dict(sigma=[20, 10, 5], iters=2, group_size=32)),
        ("adaptive 0", dict(sigma=20, adaptive_aggregation=0, iters=1)),
        ("adaptive 0 g24 residual", dict(sigma=20, adaptive_aggregation=0, group_size=24, residual=1, iters=1)),
        ("step 3 g16", dict(sigma=20, block_step=3, group_size=16, iters=1)),
        ("rclip", dict(sigma=20, rclip=guide, iters=1)),
        ("rclip iters 2 g16", dict(sigma=20, rclip=guide, iters=2, group_size=16)),
        ("memory 64", dict(sigma=20, group_size=32, memory_limit_mb=64)),
    ]
    for name, kw in extra:
        check(name, frames(core.nss.MCWNNM(clip, **kw)), frames(vk(clip, **kw)), frames(vk(clip, **kw)))
    core.num_threads = 24
    kw = dict(sigma=20, group_size=17)
    check("24 threads", frames(core.nss.MCWNNM(clip, **kw)), frames(vk(clip, **kw)), frames(vk(clip, **kw)))
    core.num_threads = 8
    # Temporal: legacy against the CPU, rolling against the reduced legacy.
    seq = make_clip(core, vs.RGBS, 48, 40, 5, length=6)
    for radius in (1,) if args.quick else (1, 2):
        for group, iters in ((8, 1), (8, 2), (16, 2), (32, 1)):
            kw = dict(sigma=20, radius=radius, bm_range=4, group_size=group, iters=iters)
            want = frames(core.nss.VAggregate(core.nss.MCWNNM(seq, **kw), seq, radius=radius))
            legacy = frames(core.nss.VAggregate(vk(seq, temporal_mode="legacy", **kw), seq, radius=radius))
            legacy_again = frames(core.nss.VAggregate(vk(seq, temporal_mode="legacy", **kw), seq, radius=radius))
            check(f"legacy r{radius} g{group} iters {iters}", want, legacy, legacy_again)
            rolling = frames(vk(seq, **kw))  # rolling is the device default
            check(f"rolling r{radius} g{group} iters {iters}", want, rolling, frames(vk(seq, **kw)))
            if iters == 1:
                # rolling sums every center's patches in one set of fixed-point
                # cells; the reduced legacy slices sum the same terms in float
                reduced = frames(Reduced(vk(seq, temporal_mode="legacy", **kw), seq, radius))
                if not psnr(reduced, rolling) >= 120:
                    failures.append(f"rolling r{radius} g{group}: {psnr(reduced, rolling):.1f} dB from the reduced legacy output")
    # The emulated layer gives the native layer's bits: a native run of one
    # case in a subprocess without the switch.
    if emulated:
        script = (
            "import sys, hashlib, numpy as np, vapoursynth as vs\n"
            "core = vs.core\n"
            f"core.std.LoadPlugin({args.vk!r})\n"
            "b = core.std.BlankClip(width=48, height=40, format=vs.RGBS, length=2)\n"
            "rng = np.random.default_rng(9)\n"
            "data = [rng.random((3, 40, 48), dtype=np.float32) for _ in range(2)]\n"
            "def fill(n, f):\n"
            "    o = f.copy()\n    for p in range(3): np.asarray(o[p])[:] = data[n][p]\n    return o\n"
            "c = core.std.ModifyFrame(b, b, fill)\n"
            "h = hashlib.sha256()\n"
            "for g in (8, 16, 32):\n"
            f"    for f in core.nss_vk.MCWNNM(c, sigma=20, group_size=g, device_id={args.device}).frames():\n"
            "        for p in range(3): h.update(np.asarray(f[p]).tobytes())\n"
            "print(h.hexdigest())\n")
        env = dict(os.environ)
        digests = []
        for force in ("1", "0"):
            env["NSS_VK_FORCE_EMULATED_WARP"] = force
            digests.append(subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                                          check=True).stdout.strip())
        cases += 1
        if digests[0] != digests[1]:
            failures.append(f"emulated layer: digest {digests[0][:16]} against native {digests[1][:16]}")
    layer = "emulated" if emulated else "native"
    print(f"test_vk_mcwnnm_live ({layer}): {cases} cases, worst {worst:.1f} dB from the CPU (floor {floor})")
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
