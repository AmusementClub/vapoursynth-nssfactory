#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Freeze the public creation-time interface of the nss plugin.

Records every registered signature plus, for each filter argument and a fixed
set of probe values across three formats, whether creation succeeds (with the
output format/size/length) or the exact error text. Creation only: no frames
are requested. Use --write to regenerate the golden file after an intended
interface change; otherwise the run must match it exactly.

Another backend (D14) is checked against the same golden with --namespace:
every signature must be the CPU signature followed by exactly the
--backend-args tail, error text must match with the "nss." prefix replaced,
and filters listed in --expect-missing may be absent or partial: their
creation probes (and VAggregate's while BM3D is listed) are skipped, but a
present listed filter's signature is still checked.

usage: test_plugin_interface.py --plugin PATH [--golden FILE] [--write]
                                [--namespace NS --backend-args A,B --expect-missing F1,F2]
"""
import argparse
import json
import re
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
    parser.add_argument("--namespace", default="nss")
    parser.add_argument("--backend-args", default="")
    parser.add_argument("--expect-missing", default="")
    parser.add_argument("--default-temporal", choices=("legacy", "rolling"), default="legacy")
    parser.add_argument("--rolling-cache", choices=("fixed", "adaptive"), default="fixed")
    args = parser.parse_args()
    ns = args.namespace
    backend_args = [a for a in args.backend_args.split(",") if a]
    expect_missing = {f for f in args.expect_missing.split(",") if f}
    if args.write and ns != "nss":
        raise SystemExit("--write only regenerates the CPU golden")

    core = vs.core
    core.num_threads = 1
    if not hasattr(core, ns):
        core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    plugin = getattr(core, ns)
    functions = {f.name: f.signature for f in plugin.functions()}

    clips = {
        "GRAYS": core.std.BlankClip(width=48, height=40, length=4, format=vs.GRAYS, color=0.5),
        "RGBS": core.std.BlankClip(width=48, height=40, length=4, format=vs.RGBS, color=[0.5, 0.4, 0.3]),
        "YUV420PS": core.std.BlankClip(width=48, height=40, length=4, format=vs.YUV420PS, color=[0.5, 0, 0]),
    }

    def create(name, clip, kwargs):
        if name == "VAggregate":
            radius = kwargs.get("radius", 1)
            radius = radius if isinstance(radius, int) and 0 <= radius <= 16 else 1
            base = plugin.BM3D(clip, radius=radius, temporal_mode="legacy") if radius else plugin.BM3D(clip)
            return plugin.VAggregate(base, clip, **kwargs)
        # The golden records the CPU, whose default is legacy. A backend whose
        # default is rolling is probed in legacy mode wherever the probe leaves
        # the mode unset, and its own default is checked separately below.
        if args.default_temporal == "rolling" and "temporal_mode:" in functions[name] and not kwargs.get("temporal_mode"):
            kwargs = dict(kwargs, temporal_mode="legacy")
        return getattr(plugin, name)(clip, **kwargs)

    def cpu_signature(name):
        """Strip and check the backend-only tail so probes cover the shared arguments only."""
        signature = functions[name]
        if ns == "nss" or name in ("Version", "Backend"):
            return signature
        items = [item for item in signature.split(";") if item]
        tail = [item.split(":")[0] for item in items[len(items) - len(backend_args):]] if backend_args else []
        if tail != backend_args:
            raise SystemExit(f"{ns}.{name}: signature tail {tail} != --backend-args {backend_args}")
        return "".join(item + ";" for item in items[:len(items) - len(backend_args)])

    shared = {name: cpu_signature(name) for name in functions}
    skipped = set(expect_missing)
    if "BM3D" in skipped:
        skipped.add("VAggregate")
    # A backend without a usable device cannot create filters: check the
    # signatures only (compile-only CI lanes) and skip the creation probes.
    if ns != "nss" and "Backend" in functions and plugin.Backend().get("device_count", 1) == 0:
        print(f"note: {ns} has no device; signatures checked, creation probes skipped")
        skipped |= set(shared)

    def outcome(name, clip, kwargs):
        try:
            return dict(ok=describe(create(name, clip, kwargs)))
        except vs.Error as error:
            return dict(error=str(error))

    cases = {}
    for name in sorted(shared):
        if name in ("Version", "Backend") or name in skipped:
            continue
        for fmt, clip in clips.items():
            cases[f"{name}|{fmt}|defaults"] = outcome(name, clip, {})
            for arg, kind, is_array, optional in parse_signature(shared[name]):
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
                               dict(rolling_cache_chunks=2, rolling_cache_limit=2))] +
                [dict(chroma=1, **extra)
                 for extra in ({}, dict(radius=1, temporal_mode="legacy"), dict(radius=1, temporal_mode="rolling"),
                               dict(sigma=[0, 3, 3]), dict(block_size=[8, 4, 4]), dict(block_size=[8, 3, 3]))],
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
        if name in skipped:
            continue
        for fmt, clip in clips.items():
            for kwargs in ({}, dict(sigma=3)):
                cases[f"{name}|{fmt}|rclip-self:{sorted(kwargs.items())}"] = outcome(name, clip, dict(kwargs, rclip=clip))
            other = core.std.BlankClip(clip, width=clip.width - 8)
            cases[f"{name}|{fmt}|rclip-mismatch"] = outcome(name, clip, dict(rclip=other))
    for name, variants in combos.items():
        if name in skipped:
            continue
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
    if ns != "nss":
        # Version/Backend describe the backend itself and are not part of the
        # shared filter interface (D14).
        prefix = re.compile(r"\bnss\.")
        shared = {k: v for k, v in shared.items() if k not in ("Version", "Backend")}
        # A listed filter may be absent; when it is already present (partially
        # implemented) its signature is still checked, only its probes skip.
        golden["functions"] = {k: v for k, v in golden["functions"].items()
                               if (k not in expect_missing or k in shared) and k not in ("Version", "Backend")}
        golden["cases"] = {key: ({"error": prefix.sub(ns + ".", value["error"])} if "error" in value else value)
                           for key, value in golden["cases"].items() if key.split("|")[0] not in skipped}
        partial = sorted(expect_missing & set(functions))
        if partial:
            print(f"note: {partial} present but listed in --expect-missing: signatures checked, probes skipped")
    problems = []
    if golden["functions"] != shared:
        for key in sorted(set(golden["functions"]) | set(shared)):
            if golden["functions"].get(key) != shared.get(key):
                problems.append(f"signature {key}: {golden['functions'].get(key)!r} -> {shared.get(key)!r}")
    def same(expected, actual):
        # Errors that name only the plugin ("nss: ...") come from code shared by
        # the backends; a backend may report them under either name.
        if expected == actual:
            return True
        error = (expected or {}).get("error", "")
        return ns != "nss" and error.startswith("nss: ") and actual == {"error": ns + error[3:]}

    if args.default_temporal == "rolling":
        clip = clips["GRAYS"]
        for name in sorted(shared):
            if name in skipped or "temporal_mode:" not in functions[name]:
                continue
            try:
                node = getattr(plugin, name)(core.std.BlankClip(clip, format=vs.RGBS) if name == "MCWNNM" else clip, radius=1)
            except vs.Error as error:
                problems.append(f"{name} default temporal mode: {error}")
                continue
            if node.height != clip.height:
                problems.append(f"{name} default temporal mode: output height {node.height}, expected {clip.height}")
    if args.rolling_cache == "adaptive":
        # The golden records the CPU, where the two cache arguments name one
        # value. An adaptive backend takes both (start and growth limit).
        both = "use only one of rolling_cache_limit and rolling_cache_chunks"
        for key in sorted(cases):
            if both in (golden["cases"].get(key) or {}).get("error", ""):
                if "ok" not in cases[key]:
                    problems.append(f"{key}: adaptive cache arguments rejected: {cases[key]}")
                cases[key] = golden["cases"][key]  # checked above; not a difference
        try:
            plugin.BM3D(clips["GRAYS"], radius=1, temporal_mode="rolling", rolling_cache_chunks=4, rolling_cache_limit=2)
            problems.append("rolling_cache_limit below rolling_cache_chunks accepted")
        except vs.Error as error:
            if "rolling_cache_limit must be at least rolling_cache_chunks" not in str(error):
                problems.append(f"rolling_cache_limit below rolling_cache_chunks: {error}")
    for key in sorted(set(golden["cases"]) | set(cases)):
        if not same(golden["cases"].get(key), cases.get(key)):
            problems.append(f"{key}: {golden['cases'].get(key)} -> {cases.get(key)}")
    for line in problems[:40]:
        print(line)
    print(f"plugin interface ({ns}): {len(cases)} cases, {len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
