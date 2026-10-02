#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Freeze the public creation-time interface of the nss plugin.

Records every registered signature plus, for each filter argument and a fixed
set of probe values across three formats, whether creation succeeds (with the
output format/size/length) or the exact error text. Creation only: no frames
are requested. Use --write to regenerate the golden file after an intended
interface change; otherwise the run must match it exactly.

usage: test_plugin_interface.py --plugin PATH [--golden FILE] [--write]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import vapoursynth as vs
except ImportError as error:  # ctest treats 77 as an explicit skip
    print(f"plugin interface SKIP: {error}")
    raise SystemExit(77)

GOLDEN = Path(__file__).resolve().parent / "data" / "plugin_interface_v1.json"
INT_PROBES = (-1, 0, 1, 2, 3, 4, 7, 8, 9, 16, 17, 33, 65, 1000)
FLOAT_PROBES = (-1.0, 0.0, 0.5, 1.0, 3.0, 1000.0)
DATA_PROBES = ("", "bogus", "auto", "AUTO", "Y", "UV", "YUV", "RGB", "legacy", "rolling", "fused",
               "awgn", "real", "gaussian")
SKIP_ARGS = {"clip", "src", "ref", "rclip"}


def parse_signature(signature):
    args = []
    for item in filter(None, signature.split(";")):
        name, kind, *flags = item.split(":")
        args.append((name, kind.rstrip("[]"), kind.endswith("[]"), "opt" in flags))
    return args


def describe(node):
    vi = node
    fmt = vi.format
    return dict(format=None if fmt is None else fmt.name, width=vi.width, height=vi.height, frames=vi.num_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    functions = {f.name: f.signature for f in core.nss.functions()}

    clips = {
        "GRAYS": core.std.BlankClip(width=48, height=40, length=4, format=vs.GRAYS, color=0.5),
        "RGBS": core.std.BlankClip(width=48, height=40, length=4, format=vs.RGBS, color=[0.5, 0.4, 0.3]),
        "YUV420PS": core.std.BlankClip(width=48, height=40, length=4, format=vs.YUV420PS, color=[0.5, 0, 0]),
    }

    def create(name, clip, kwargs):
        if name == "VAggregate":
            radius = kwargs.get("radius", 1)
            radius = radius if isinstance(radius, int) and 0 <= radius <= 16 else 1
            base = core.nss.BM3D(clip, radius=radius) if radius else core.nss.BM3D(clip)
            return core.nss.VAggregate(base, clip, **kwargs)
        return getattr(core.nss, name)(clip, **kwargs)

    def outcome(name, clip, kwargs):
        try:
            return dict(ok=describe(create(name, clip, kwargs)))
        except vs.Error as error:
            return dict(error=str(error))

    cases = {}
    for name in sorted(functions):
        if name in ("Version", "Backend"):
            continue
        for fmt, clip in clips.items():
            cases[f"{name}|{fmt}|defaults"] = outcome(name, clip, {})
            for arg, kind, is_array, optional in parse_signature(functions[name]):
                if arg in SKIP_ARGS or kind in ("vnode", "anode", "vframe", "func"):
                    continue
                probes = INT_PROBES if kind == "int" else FLOAT_PROBES if kind == "float" else DATA_PROBES
                for value in probes:
                    cases[f"{name}|{fmt}|{arg}={value!r}"] = outcome(name, clip, {arg: value})
                if is_array and kind in ("int", "float"):
                    cases[f"{name}|{fmt}|{arg}=pair"] = outcome(name, clip, {arg: [probes[4], probes[3]]})

    # Combined probes for paths a single argument cannot reach.
    combos = {
        "BM3D": [dict(radius=r, temporal_mode="rolling", **extra)
                 for r in (1, 2)
                 for extra in ({}, *({k: v} for k in ("rolling_chunk", "rolling_cache_chunks", "rolling_cache_limit")
                                     for v in (0, 1, 4, 64, 65)),
                               dict(rolling_cache_chunks=2, rolling_cache_limit=2))],
        "TWSC": [dict(sigma=3, estimate_sigma=1), dict(bm_range=3, search_window=9), dict(block_size=4),
                 dict(block_size=4, block_step=8), dict(group_size=1), dict(group_size=1, ps_num=2),
                 dict(group_size=4, ps_num=8), dict(estimate_sigma=1, radius=1), dict(sigma=[3, 0, 3]),
                 dict(sigma=[3, 0]), dict(sigma=[3, 2, 1, 0]), dict(sigma=1e-45)],
        "NLH": [dict(sigma=[3, 0, 3]), dict(sigma=60), dict(sigma=60, noise_model="awgn"),
                dict(sigma=3, noise_model="real"), dict(sigma=3, block_size=[4, 8], q=[2, 4]),
                dict(sigma=3, block_size=[16, 16]), dict(sigma=3, block_step=[8, 2]),
                dict(sigma=3, bm_range=3, search_window=[9, 9]), dict(sigma=3, group_size=[4, 8], ps_num=8),
                dict(sigma=3, radius=1, ps_num=3), dict(sigma=0), dict(noise_model="awgn"),
                dict(sigma=3, basic_iters=2, wiener_iters=3, lambda_basic=0.5)],
    }
    for name in ("TWSC", "NLH"):
        for fmt, clip in clips.items():
            for kwargs in ({}, dict(sigma=3)):
                cases[f"{name}|{fmt}|rclip-self:{sorted(kwargs.items())}"] = outcome(name, clip, dict(kwargs, rclip=clip))
            other = core.std.BlankClip(clip, width=clip.width - 8)
            cases[f"{name}|{fmt}|rclip-mismatch"] = outcome(name, clip, dict(rclip=other))
    for name, variants in combos.items():
        for fmt, clip in clips.items():
            for kwargs in variants:
                key = ",".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
                cases[f"{name}|{fmt}|combo:{key}"] = outcome(name, clip, kwargs)

    record = dict(schema="nssfactory.plugin_interface.v1", functions=functions, cases=cases)
    if args.write:
        Path(args.golden).write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"wrote {args.golden}: {len(functions)} functions, {len(cases)} cases")
        return 0
    golden = json.loads(Path(args.golden).read_text())
    problems = []
    if golden["functions"] != functions:
        for key in sorted(set(golden["functions"]) | set(functions)):
            if golden["functions"].get(key) != functions.get(key):
                problems.append(f"signature {key}: {golden['functions'].get(key)!r} -> {functions.get(key)!r}")
    for key in sorted(set(golden["cases"]) | set(cases)):
        if golden["cases"].get(key) != cases.get(key):
            problems.append(f"{key}: {golden['cases'].get(key)} -> {cases.get(key)}")
    for line in problems[:40]:
        print(line)
    print(f"plugin interface: {len(cases)} cases, {len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
