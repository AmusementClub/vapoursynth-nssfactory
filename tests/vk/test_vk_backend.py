#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_vk.Backend smoke: the plugin loads with or without a Vulkan loader,
reports the D22/D23 support level, and on a supported device its load probe
runs once at a required subgroup size. Exits 77 (ctest skip) without
VapourSynth, without a Vulkan device, or on an unsupported device (after
checking that the missing features are named).

usage: test_vk_backend.py --plugin PATH [--device N]
"""
import argparse
import sys
from pathlib import Path

try:
    import vapoursynth as vs
except ImportError as error:
    print(f"vk backend SKIP: {error}")
    raise SystemExit(77)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    version = core.nss_vk.Version()
    assert (version if isinstance(version, (str, bytes)) else version["version"]), "empty version"
    kwargs = {} if args.device is None else {"device_id": args.device}

    def backend(**kw):
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in core.nss_vk.Backend(**kw).items()}

    info = backend(**kwargs)
    print(info)
    assert info["shader_tools"], "no shader toolchain recorded"
    if info["device_count"] == 0:
        assert info["error"], "no device and no reason"
        print(f"vk backend SKIP: no Vulkan device ({info['error']})")
        return 77
    try:
        backend(device_id=info["device_count"])
    except vs.Error as error:
        assert "out of range" in str(error), error
    else:
        raise AssertionError("out-of-range device_id was accepted")
    assert info["support"] in ("native", "emulated", "unsupported"), info
    if info["support"] == "unsupported":
        assert info["missing"], f"unsupported without a reason: {info}"
        print(f"vk backend SKIP: {info['name']} unsupported: {info['missing']} "
              f"(probe {'ok' if info['probe_ok'] else info['probe_error']})")
        return 77
    assert not info["missing"], info
    assert info["native_lanes"] == info["max_subgroup_size"] >= 1, info
    assert info["shared_memory_kb"] >= 48, info
    assert info["probe_ok"] == 1, f"probe failed: {info['probe_error']}"
    assert info["min_subgroup_size"] <= info["probe_subgroup_size"] <= info["max_subgroup_size"], info
    print(f"vk backend OK: {info['name']} ({info['driver_name']} {info['driver_info']}, Vulkan {info['api_version']}) "
          f"{info['support']}, subgroup {info['min_subgroup_size']}-{info['max_subgroup_size']}, "
          f"probe {info['probe_ms']:.1f} ms at {info['probe_subgroup_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
