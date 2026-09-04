#!/usr/bin/env python3
"""Run one deterministic 1080p NSS plugin workload for external profilers."""

from __future__ import annotations

import json
import hashlib
import os
import time

import numpy as np
import vapoursynth as vs


def make_source(
    core: vs.Core,
    algorithm: str,
    frames: int,
    width: int,
    height: int,
    seed: int = 42,
    gray8_path: str | None = None,
) -> vs.VideoNode:
    rng = np.random.RandomState(seed)
    if gray8_path:
        raw = np.fromfile(gray8_path, dtype=np.uint8)
        expected = width * height
        if raw.size != expected:
            raise ValueError(
                f"gray8 sample has {raw.size} bytes, expected {expected}: {gray8_path}"
            )
        base = raw.reshape(height, width).astype(np.float32) * np.float32(1.0 / 255.0)
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        base = (
            0.35
            + 0.2 * np.sin(xx / 37.0) * np.cos(yy / 29.0)
            + 0.05 * (xx + yy) / (width + height)
        ).astype(np.float32)
    planes = [
        base + rng.randn(height, width).astype(np.float32) * np.float32(3.0 / 255.0)
        for _ in range(3 if algorithm == "mcwnnm" else 1)
    ]
    fmt = vs.RGBS if algorithm == "mcwnnm" else vs.GRAYS
    blank = core.std.BlankClip(
        width=width,
        height=height,
        format=fmt,
        length=frames + 1,
        color=[0.0] * len(planes),
    )

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        output = f.copy()
        for plane, values in enumerate(planes):
            np.asarray(output[plane])[:, :] = values
        return output

    return core.std.ModifyFrame(clip=blank, clips=blank, selector=fill)


def make_filter(core: vs.Core, algorithm: str, source: vs.VideoNode) -> vs.VideoNode:
    if algorithm == "nlm":
        return core.nss.NLM(
            source,
            d=int(os.environ.get("NSS_PROFILE_NLM_D", "1")),
            a=int(os.environ.get("NSS_PROFILE_NLM_A", "2")),
            s=int(os.environ.get("NSS_PROFILE_NLM_S", "4")),
            h=1.2,
            channels="Y",
        )
    if algorithm == "bm3d":
        block_size = int(os.environ.get("NSS_PROFILE_BM3D_BLOCK", "8"))
        kwargs = dict(
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_BM3D_RADIUS", "0")),
            block_size=block_size,
            block_step=int(os.environ.get("NSS_PROFILE_BM3D_STEP", str(min(block_size, 8)))),
            group_size=int(os.environ.get("NSS_PROFILE_BM3D_GROUP", "8")),
            bm_range=int(os.environ.get("NSS_PROFILE_BM3D_RANGE", "7")),
        )
        if os.environ.get("NSS_PROFILE_BM3D_V2", "0") == "1":
            return core.nss.BM3Dv2(
                source,
                temporal_mode=os.environ.get("NSS_PROFILE_BM3D_TEMPORAL_MODE", "rolling"),
                rolling_chunk=int(os.environ.get("NSS_PROFILE_BM3D_ROLLING_CHUNK", "4")),
                rolling_cache_limit=int(os.environ.get("NSS_PROFILE_BM3D_ROLLING_CACHE", "16")),
                **kwargs,
            )
        return core.nss.BM3D(source, **kwargs)
    if algorithm == "wnnm":
        block_size = int(os.environ.get("NSS_PROFILE_WNNM_BLOCK", "8"))
        return core.nss.WNNM(
            source,
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_WNNM_RADIUS", "0")),
            block_size=block_size,
            block_step=int(os.environ.get("NSS_PROFILE_WNNM_STEP", str(min(block_size, 8)))),
            group_size=int(os.environ.get("NSS_PROFILE_WNNM_GROUP", "8")),
            bm_range=int(os.environ.get("NSS_PROFILE_WNNM_RANGE", "7")),
            residual=int(os.environ.get("NSS_PROFILE_WNNM_RESIDUAL", "0")),
        )
    if algorithm == "twsc":
        block_size = int(os.environ.get("NSS_PROFILE_TWSC_BLOCK", "8"))
        return core.nss.TWSC(
            source,
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_TWSC_RADIUS", "0")),
            block_size=block_size,
            block_step=int(os.environ.get("NSS_PROFILE_TWSC_STEP", str(min(block_size, 8)))),
            group_size=int(os.environ.get("NSS_PROFILE_TWSC_GROUP", "8")),
            bm_range=int(os.environ.get("NSS_PROFILE_TWSC_RANGE", "7")),
            iters=int(os.environ.get("NSS_PROFILE_TWSC_ITERS", "2")),
        )
    if algorithm == "ncsr":
        block_size = int(os.environ.get("NSS_PROFILE_NCSR_BLOCK", "8"))
        return core.nss.NCSR(
            source,
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_NCSR_RADIUS", "0")),
            block_size=block_size,
            block_step=int(os.environ.get("NSS_PROFILE_NCSR_STEP", str(min(block_size, 8)))),
            group_size=int(os.environ.get("NSS_PROFILE_NCSR_GROUP", "8")),
            bm_range=int(os.environ.get("NSS_PROFILE_NCSR_RANGE", "7")),
            iters=int(os.environ.get("NSS_PROFILE_NCSR_ITERS", "2")),
        )
    if algorithm == "lssc":
        return core.nss.LSSC(
            source,
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_LSSC_RADIUS", "0")),
            block_size=int(os.environ.get("NSS_PROFILE_LSSC_BLOCK", "8")),
            block_step=int(os.environ.get("NSS_PROFILE_LSSC_STEP", "8")),
        )
    if algorithm == "nlh":
        block_size = int(os.environ.get("NSS_PROFILE_NLH_BLOCK", "8"))
        block_step = int(os.environ.get("NSS_PROFILE_NLH_STEP", str(min(block_size, 8))))
        return core.nss.NLH(
            source,
            sigma=3,
            radius=int(os.environ.get("NSS_PROFILE_NLH_RADIUS", "0")),
            block_size=block_size,
            block_step=block_step,
            group_size=int(os.environ.get("NSS_PROFILE_NLH_GROUP", "16")),
            bm_range=int(os.environ.get("NSS_PROFILE_NLH_RANGE", "20")),
            q=int(os.environ.get("NSS_PROFILE_NLH_Q", "4")),
        )
    if algorithm == "mcwnnm":
        block_size = int(os.environ.get("NSS_PROFILE_MCWNNM_BLOCK", "8"))
        sigma = float(os.environ.get("NSS_PROFILE_MCWNNM_SIGMA", "3"))
        return core.nss.MCWNNM(
            source,
            sigma=[sigma, sigma, sigma],
            radius=int(os.environ.get("NSS_PROFILE_MCWNNM_RADIUS", "0")),
            block_size=block_size,
            block_step=int(os.environ.get("NSS_PROFILE_MCWNNM_STEP", str(min(block_size, 8)))),
            group_size=int(os.environ.get("NSS_PROFILE_MCWNNM_GROUP", "8")),
            bm_range=int(os.environ.get("NSS_PROFILE_MCWNNM_RANGE", "7")),
            residual=int(os.environ.get("NSS_PROFILE_MCWNNM_RESIDUAL", "1")),
            admm_iter=int(os.environ.get("NSS_PROFILE_MCWNNM_ADMM_ITER", "10")),
            iters=int(os.environ.get("NSS_PROFILE_MCWNNM_ITERS", "2")),
        )
    raise ValueError(f"unknown algorithm: {algorithm}")


def main() -> None:
    algorithm = os.environ["NSS_PROFILE_ALGORITHM"].lower()
    frames = int(os.environ["NSS_PROFILE_FRAMES"])
    width = int(os.environ.get("NSS_PROFILE_WIDTH", "1920"))
    height = int(os.environ.get("NSS_PROFILE_HEIGHT", "1080"))
    seed = int(os.environ.get("NSS_PROFILE_SEED", "42"))
    gray8_path = os.environ.get("NSS_PROFILE_GRAY8")
    plugin = os.environ["NSS_SO"]
    if frames < 1 or width < 1 or height < 1:
        raise ValueError("frames and dimensions must be positive")

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=plugin)
    source = make_source(core, algorithm, frames, width, height, seed, gray8_path)
    output = make_filter(core, algorithm, source)
    last_output = output.get_frame(0)
    start = time.perf_counter()
    for frame in range(1, frames + 1):
        last_output = output.get_frame(frame)
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    output_path = os.environ.get("NSS_PROFILE_OUTPUT")
    output_file = open(output_path, "wb") if output_path else None
    try:
        for plane in range(last_output.format.num_planes):
            values = np.ascontiguousarray(np.asarray(last_output[plane])[:, :width], dtype=np.float32)
            payload = values.tobytes()
            digest.update(payload)
            if output_file:
                output_file.write(payload)
    finally:
        if output_file:
            output_file.close()
    print(
        json.dumps(
            {
                "schema": "nssfactory.profile.cpu.v1",
                "algorithm": algorithm,
                "width": width,
                "height": height,
                "frames": frames,
                "threads": 1,
                "seed": seed,
                "sample": os.path.basename(gray8_path) if gray8_path else "synthetic",
                "parameters": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith("NSS_PROFILE_") and key not in {"NSS_PROFILE_OUTPUT"}
                },
                "wall_time_s": elapsed,
                "milliseconds_per_frame": elapsed * 1000.0 / frames,
                "output_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
