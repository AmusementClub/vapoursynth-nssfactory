#!/usr/bin/env python3
"""Reject numeric narrowing at real public VSMap boundaries; requires NSS_SO."""
import json
import os

import vapoursynth as vs


INTEGER_ARGS = {
    "BM3D": "block_size group_size block_step bm_range radius ps_num ps_range rolling_chunk rolling_cache_chunks rolling_cache_limit",
    "WNNM": "block_size block_step group_size bm_range radius ps_num ps_range residual adaptive_aggregation",
    "MCWNNM": "block_size block_step group_size bm_range radius ps_num ps_range residual adaptive_aggregation admm_iter iters",
    "TWSC": "block_size block_step group_size bm_range radius ps_num ps_range iters",
    "NCSR": "block_size block_step group_size bm_range radius ps_num ps_range iters",
    "NLH": "block_size block_step group_size bm_range radius ps_num ps_range q",
    "LSSC": "block_size block_step radius",
    "NLM": "d a s wmode",
    "VAggregate": "radius planes",
}
INTEGER_ARGS = {name: keys + " memory_limit_mb" for name, keys in INTEGER_ARGS.items()}
INTEGER_ARGS["VAggregate"] += " allow_legacy"
FLOAT_ARGS = {
    "BM3D": "sigma", "WNNM": "sigma", "MCWNNM": "sigma rho mu delta",
    "TWSC": "sigma lambda1 lambda2 delta", "NCSR": "sigma delta",
    "NLH": "sigma", "LSSC": "sigma", "NLM": "h wref",
}


def main():
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=os.environ["NSS_SO"])
    src = core.std.BlankClip(width=32, height=32, length=3, format=vs.RGBS)
    fat = core.std.BlankClip(width=32, height=192, length=3, format=vs.RGBS)

    def create(name, kwargs):
        if name == "VAggregate":
            return core.nss.VAggregate(fat, src, **dict(dict(radius=1), **kwargs))
        return getattr(core.nss, name)(src, **kwargs)

    rejected = 0
    for table, values in (
        (INTEGER_ARGS, (2**32 + 1, 2**32 + 8, -(2**32) + 1, 2**31, -(2**31) - 1, 2**63 - 1, -(2**63))),
        (FLOAT_ARGS, (float("nan"), float("inf"), -float("inf"), 1e300, -1e300)),
    ):
        for name, keys in table.items():
            for key in keys.split():
                for value in values:
                    try:
                        create(name, {key: value})
                    except vs.Error as error:
                        if "not representable" not in str(error) or key not in str(error):
                            raise AssertionError(f"{name}.{key} did not reach the common numeric check: {error}")
                        rejected += 1
                    else:
                        raise AssertionError(f"{name}.{key} accepted {value}")
    # Cover non-first array entries and BM3D's short-array inheritance boundary.
    for name, key, value in (
        ("BM3D", "block_size", [4, 2**32 + 8]),
        ("BM3D", "bm_range", [4, 2**32 + 1]),
        ("BM3D", "sigma", [3, 1e300]),
        ("VAggregate", "planes", [0, 2**32 + 1]),
    ):
        try:
            create(name, {key: value})
        except vs.Error as error:
            assert "not representable" in str(error) and f"{key}[1]" in str(error), str(error)
            rejected += 1
        else:
            raise AssertionError(f"{name}.{key} accepted invalid second element")
    # Integer limits belong to each filter's business validation, not narrowing.
    for value in (-(2**31), 2**31 - 1):
        try:
            create("BM3D", dict(radius=value))
        except vs.Error as error:
            assert "not representable" not in str(error), str(error)
        else:
            raise AssertionError("invalid radius passed business validation")
    for name in INTEGER_ARGS:
        create(name, {})
    create("BM3D", dict(block_size=[4, 8], group_size=[4, 8], sigma=[3, 0]))
    create("VAggregate", dict(planes=[0, 2]))
    print(json.dumps(dict(passed=True, rejected=rejected, filters=len(INTEGER_ARGS))))


if __name__ == "__main__":
    main()
