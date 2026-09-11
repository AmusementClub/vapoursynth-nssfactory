#!/usr/bin/env python3
"""Native ARM build/host gate. This is not a performance or N-RELEASE gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HIGHWAY_COMMIT = "2607d3b5b0113992fe84d3848859eae13b3b52c1"
PLUGIN_TESTS = (
    "test_backend_plugin", "test_plan01_plugin", "test_plan01_formats",
    "test_parameter_bounds_plugin", "test_wnnm_zero_rank_plugin",
    "test_vaggregate_plugin", "test_bm3d_rolling", "test_bm3d_semantics",
    "test_neon_matrix", "test_full_image_plugin",
)
REQUIRED_TESTS = {"test_backend", "test_host_failure", "test_resources",
                  "test_workspace", "test_svd_scale", "test_wnnm_zero_rank",
                  "test_bm3d_contract", "test_temporal_matcher", "test_guarded_primitives", "test_lssc_omp",
                  "test_twsc", "test_nlh", "test_batch"}


def capture(*args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stdout[-2000:]}")
    return result.stdout.strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("dynamic", "neon", "portable-test"), default="neon")
    parser.add_argument("--cxx", default=os.environ.get("NSS_C4A_CXX", "g++-15"))
    parser.add_argument("--cc", default=os.environ.get("NSS_C4A_CC", "gcc-15"))
    parser.add_argument("--highway", type=Path, default=os.environ.get("NSS_C4A_HIGHWAY_SOURCE"))
    parser.add_argument("--vs-include", type=Path, default=os.environ.get("VapourSynth_INCLUDE_DIR"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sanitize", action="store_true", help="ASan/UBSan lane (Linux GCC only)")
    parser.add_argument("--disable-lssc-sme", action="store_true", help="Build the Highway-only LSSC fallback control")
    args = parser.parse_args()
    source, build, out = (p.resolve() for p in (args.source, args.build, args.out))
    # Preserve prior gate results and compiler caches; each lane gets its own paths.
    if build.exists() or out.exists():
        parser.error("--build and --out must be new paths")
    if args.jobs < 1 or not args.highway or not args.vs_include:
        parser.error("positive --jobs, --highway and --vs-include are required")
    out.mkdir(parents=True)
    report = dict(schema="nssfactory.arm-native-gate.v1", passed=False,
                  source=str(source), build=str(build), mode=args.mode,
                  architecture=platform.machine(), os=platform.platform(),
                  python=sys.executable, commands=[], performance_gate=False,
                  n_functional=False, n_accelerated=False, n_release=False,
                  sanitizers=args.sanitize)

    def run(name, command, env=None):
        started = time.monotonic()
        with (out / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report["commands"].append(dict(name=name, argv=command, returncode=result.returncode,
                                       elapsed_seconds=time.monotonic() - started))
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {out / (name + '.log')}")

    try:
        if platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("requires a native ARM64 Python process")
        if platform.system() == "Darwin":
            translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
            if translated.stdout.strip() == "1":
                raise RuntimeError("translated processes cannot certify native ARM")
        cxx, cc = shutil.which(args.cxx), shutil.which(args.cc)
        if not cxx or not cc:
            raise RuntimeError("requested native compilers are unavailable")
        triple = capture(cxx, "-dumpmachine")
        if not triple.startswith(("aarch64", "arm64")):
            raise RuntimeError(f"non-ARM compiler: {triple}")
        report.update(cxx=capture(cxx, "--version"), cc=capture(cc, "--version"), triple=triple)
        sanitizer_flags = ["-DNSS_ENABLE_LSSC_SME=OFF"] if args.disable_lssc_sme else []
        test_env = dict(os.environ)
        if args.sanitize:
            if platform.system() != "Linux" or "clang" in report["cxx"].lower():
                raise RuntimeError("--sanitize currently requires native Linux GCC")
            sanitizer_flags += ["-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"]
            test_env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        highway = args.highway.resolve()
        # Snapshot dependencies are root-owned. Trust only this explicit path
        # for these read-only queries, without changing global Git settings.
        git = ("git", "-c", f"safe.directory={highway}", "-C", str(highway))
        if capture(*git, "rev-parse", "HEAD") != HIGHWAY_COMMIT:
            raise RuntimeError("Highway does not match the pinned 1.4.0 commit")
        if capture(*git, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Highway tracked sources are modified")
        import vapoursynth as vs
        import numpy as np
        if not hasattr(vs, "core"):
            raise RuntimeError("VapourSynth headers/namespace alone are not a native runtime")
        report.update(highway_commit=HIGHWAY_COMMIT, vapoursynth=str(vs.__version__), numpy=np.__version__)
        if str(vs.__version__) != "R75":
            raise RuntimeError("this reference gate requires the pinned VapourSynth R75 runtime")
        report["vs_headers"] = {name: digest(args.vs_include / name) for name in ("VapourSynth4.h", "VSHelper4.h")}
        files = [source / "CMakeLists.txt"]
        for folder in ("src", "include", "cmake", "tests", "contracts"):
            files.extend(p for p in (source / folder).rglob("*") if p.is_file()
                         and p.suffix in (".cpp", ".hpp", ".h", ".cmake", ".txt", ".py", ".sh", ".json", ".md"))
        inputs = {str(p.relative_to(source)): digest(p) for p in sorted(set(files))}
        (out / "source-inputs.json").write_text(json.dumps(inputs, indent=2))
        report["source_inputs_sha256"] = digest(out / "source-inputs.json")
        run("configure", ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                          "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={cxx}",
                          f"-DCMAKE_C_COMPILER={cc}", f"-DNSS_HWY_TARGET_MODE={args.mode}",
                          "-DNSS_AVX2_DEFAULTS=OFF", "-DNSS_AVX2_EXPERIMENT=0",
                          "-DNSS_GIT_DESCRIBE=source-" + report["source_inputs_sha256"][:12],
                          "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DFETCHCONTENT_SOURCE_DIR_HIGHWAY={highway}",
                          f"-DVapourSynth_INCLUDE_DIR={args.vs_include.resolve()}", *sanitizer_flags],
            dict(os.environ, NSS_HWY_TARGET_MODE=args.mode))
        run("build", ["cmake", "--build", str(build), "-j", str(args.jobs)])
        plugin = build / ("libnss.dylib" if platform.system() == "Darwin" else "libnss.so")
        if not plugin.is_file():
            raise RuntimeError("full plugin target was not built")
        report["plugin_sha256"] = digest(plugin)
        junit = out / "ctest.xml"
        run("ctest", ["ctest", "--test-dir", str(build), "--output-on-failure", "--output-junit", str(junit), "-j1"], test_env)
        cases = ET.parse(junit).findall(".//testcase")
        skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None or c.attrib.get("status") == "notrun"]
        passed = {c.attrib["name"] for c in cases if c.attrib["name"] not in skipped
                  and c.find("failure") is None and c.find("error") is None}
        if not REQUIRED_TESTS <= passed or set(skipped) - {"test_avx2_matcher"}:
            raise RuntimeError("required native tests were absent/skipped")
        report["ctest"] = dict(registered=len(cases), passed=len(passed), skipped=skipped)
        env = dict(test_env, NSS_SO=str(plugin))
        if args.sanitize:
            asan = Path(capture(cxx, "-print-file-name=libasan.so")).resolve()
            cxx_runtime = Path(capture(cxx, "-print-file-name=libstdc++.so")).resolve()
            if not asan.is_file() or not cxx_runtime.is_file():
                raise RuntimeError("GCC ASan/C++ runtime unavailable for plugin preload")
            # Python itself does not link libstdc++. Load it at startup so ASan
            # can resolve __cxa_throw before VS is imported through dlopen.
            env["LD_PRELOAD"] = f"{asan}:{cxx_runtime}" + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            # The interpreter/VS runtime is external to this plugin. Leak
            # detection remains enabled for all standalone C++ fixtures.
            env["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
            report["sanitizer_runtime"] = dict(path=str(asan), sha256=digest(asan),
                                                cxx_path=str(cxx_runtime), cxx_sha256=digest(cxx_runtime),
                                                plugin_leak_detection=False)
        for test in PLUGIN_TESTS:
            extra = ["--out", str(out / test)] if test in ("test_neon_matrix", "test_full_image_plugin") else []
            run(test, [sys.executable, str(source / "tests" / f"{test}.py"), *extra], env)
        run("alignment-math", [sys.executable, str(source / "tests/test_alignment_math.py"),
                              "--probe", str(build / "tests/alignment_probe"),
                              "--out", str(out / "alignment-math")], test_env)
        smoke = json.loads((out / "test_backend_plugin.log").read_text())
        report["backend"] = smoke["backend"]
        expected = {"EMU128"} if args.mode == "portable-test" else {"NEON", "NEON_WITHOUT_AES"}
        if (smoke["backend"]["target_name"] not in expected or smoke["backend"]["float_lanes"] != 4
                or smoke["backend"]["build_mode"] != args.mode):
            raise RuntimeError("actual backend does not match the requested ARM lane")
        if any(not (source / p).is_file() or digest(source / p) != h for p, h in inputs.items()):
            raise RuntimeError("source inputs changed during the gate")
        report.update(passed=True, scope="Native build, guarded primitives, numeric fixtures and public shape/scheduling matrix; exhaustive functional, performance and release gates remain open")
    except Exception as error:
        report["error"] = str(error)
    finally:
        # This is the maximum individual child RSS, not aggregate VM memory.
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report["max_child_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
