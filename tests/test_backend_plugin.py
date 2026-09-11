#!/usr/bin/env python3
"""Native host admission and truthful target diagnostics (no performance claim)."""
import json
import os
import platform

import numpy as np
import vapoursynth as vs

from test_plan01_plugin import clip, evaluate, options


def main():
    core = vs.core
    core.num_threads = 2
    core.std.LoadPlugin(path=os.environ["NSS_SO"])
    caps = dict(core.nss.Backend())
    for key in ("target_name", "build_mode"):
        if isinstance(caps[key], bytes):
            caps[key] = caps[key].decode()
    assert caps["executable"] == 1, caps
    assert caps["lssc_sme_available"] <= caps["lssc_sme_compiled"], caps
    if caps["portable_test"]:
        assert caps["lssc_sme_available"] == 0, caps
    selected = caps["selected_target"]
    assert selected and selected & caps["compiled_targets"] & caps["runtime_targets"], caps
    expected = os.environ.get("NSS_EXPECT_BACKEND")
    if expected:
        assert caps["target_name"] == expected, caps
    rows = []
    for name in ("NLM", "BM3D", "WNNM", "MCWNNM", "TWSC", "NCSR", "NLH", "LSSC"):
        source = clip(core, width=24, height=24, frames=3,
                      fmt=vs.RGBS if name == "MCWNNM" else vs.GRAYS)
        radii = (0,) if name == "LSSC" else (0, 1)
        for radius in radii:
            kwargs = options(name, block=8, group=8, radius=radius, sigma=10)
            node = getattr(core.nss, name)(source, **kwargs)
            if radius and name != "NLM":
                node = core.nss.VAggregate(node, source, radius=radius)
            pixels, _ = evaluate(node, 1)
            original = source.get_frame(1)
            difference = max(float(np.max(np.abs(a - np.asarray(original[p]))))
                             for p, a in enumerate(pixels))
            if name == "BM3D" and not radius:
                assert difference > 1e-6, "default b8/g8 BM3D returned the source unchanged"
                # Exercise explicit Final independently of the temporal test.
                final, _ = evaluate(core.nss.BM3D(source, ref=node, **kwargs), 1)
                assert all(np.isfinite(a).all() for a in final)
            rows.append(dict(algorithm=name, radius=radius, source_max_abs=difference))
    print(json.dumps(dict(passed=True, architecture=platform.machine(), backend=caps, cases=rows)))


if __name__ == "__main__":
    main()
