#!/usr/bin/env python3
"""Single-process VAggregate timing payload for paired x86/C4 runs."""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import vapoursynth as vs


def main() -> int:
    width = int(os.environ.get("NSS_PROFILE_WIDTH", "1920"))
    height = int(os.environ.get("NSS_PROFILE_HEIGHT", "1080"))
    frames = int(os.environ.get("NSS_PROFILE_FRAMES", "40"))
    radius = int(os.environ.get("NSS_PROFILE_RADIUS", "1"))
    plugin = os.environ["NSS_SO"]

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=plugin)
    src = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=frames + 1, color=[0.25])
    fat = core.std.BlankClip(
        width=width,
        height=height * (2 * radius + 1) * 2,
        format=vs.GRAYS,
        length=frames + 1,
        color=[1.0],
    )
    output = core.nss.VAggregate(fat, src, radius=radius)

    output.get_frame(0)
    start = time.perf_counter()
    last = None
    for n in range(1, frames + 1):
        last = output.get_frame(n)
    elapsed = time.perf_counter() - start
    assert last is not None
    values = np.ascontiguousarray(np.asarray(last[0])[:, :width], dtype=np.float32)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    print(
        json.dumps(
            {
                "schema": "nssfactory.profile.vaggregate.v1",
                "radius": radius,
                "frames": frames,
                "width": width,
                "height": height,
                "milliseconds_per_frame": elapsed * 1000.0 / frames,
                "output_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
