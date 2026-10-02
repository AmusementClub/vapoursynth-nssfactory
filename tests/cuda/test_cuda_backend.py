#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""nss_cuda.Backend smoke: the plugin loads, reports the D13 support level and
its load probe runs on the first CUDA device. Exits 77 (ctest skip) when no
CUDA device or VapourSynth is available.

--force-jit expects CUDA_FORCE_PTX_JIT=1 in the environment (set by ctest):
the driver then ignores every cubin and must JIT the embedded compute PTX, so
a passing probe proves the D13 best-effort fallback actually runs. It is
skipped when the driver is older than the build toolkit, because such a driver
cannot JIT newer PTX at all (cubins still run through minor-version
compatibility).

usage: test_cuda_backend.py --plugin PATH [--force-jit]
"""
import argparse
import os
import sys
from pathlib import Path

try:
    import vapoursynth as vs
except ImportError as error:
    print(f"cuda backend SKIP: {error}")
    raise SystemExit(77)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--force-jit", action="store_true")
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    info = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in core.nss_cuda.Backend().items()}
    print(info)
    version = core.nss_cuda.Version()
    assert (version if isinstance(version, (str, bytes)) else version["version"]), "empty version"
    assert info["compiled_architectures"], "no compiled architectures recorded"
    if info["device_count"] == 0:
        print(f"cuda backend SKIP: no CUDA device ({info['error']})")
        return 77
    if args.force_jit:
        if os.environ.get("CUDA_FORCE_PTX_JIT") != "1":
            raise SystemExit("--force-jit needs CUDA_FORCE_PTX_JIT=1")
        if "virtual" not in info["compiled_architectures"]:
            print("cuda backend SKIP: no PTX embedded in this build")
            return 77
        if not info["ptx_jit_available"]:
            print(f"cuda backend SKIP: driver CUDA {info['driver_version']} is older than the build toolkit "
                  f"{info['runtime_version']}; the driver cannot JIT this build's PTX")
            return 77
    assert info["support"] in ("native", "jit"), f"device not supported: {info}"
    assert info["probe_ok"] == 1, f"probe failed: {info['probe_error']}"
    major, minor = (int(x) for x in info["compute_capability"].split("."))
    assert major >= 7, info
    try:
        core.nss_cuda.Backend(device_id=info["device_count"])
    except vs.Error as error:
        assert "out of range" in str(error), error
    else:
        raise AssertionError("out-of-range device_id was accepted")
    mode = "forced PTX JIT" if args.force_jit else info["support"]
    print(f"cuda backend OK: {info['name']} sm_{major}{minor} {mode}, probe {info['probe_ms']:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
