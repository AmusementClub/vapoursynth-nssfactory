#!/usr/bin/env python3
"""Build a private instrumented OMP object against an actual native library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--cxx', default='clang++')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
source = args.source.resolve()
here = Path(__file__).resolve().parent
original = (source / 'src/cpu/lssc/omp.cpp').read_text()
before, workspace = original.split('int lssc_omp_workspace(', 1)
anchor = '        if (best < 0 || best_abs < 1e-12f) {'
if workspace.count(anchor) != 1:
    raise RuntimeError('source anchor mismatch')
workspace = workspace.replace(anchor,
    '        nss_trace_omp(y, m, D, atoms, ldd, r, corr, used, best, best_abs, t);\n' + anchor)
generated = args.out / 'omp_traced.cpp'
generated.write_text('#include "lssc_trace.hpp"\n' + before + 'int lssc_omp_workspace(' + workspace)
libs = [args.build / name for name in ('libnss_cpu.a', 'libnss_svd_qreplay.a', '_deps/highway-build/libhwy.a')]
command = [args.cxx, '-std=c++20', '-O3', '-DNSS_AVX2_DEFAULTS=0', '-DNSS_AVX2_EXPERIMENT=0',
           '-I' + str(source / 'include'), '-I' + str(source / 'src'), '-I' + str(here),
           str(generated), str(here / 'lssc_trace.cpp'), *map(str, libs), '-lpthread', '-o', str(args.out / 'trace')]
proc = subprocess.run(command, text=True, capture_output=True)
(args.out / 'build.log').write_text(proc.stdout + proc.stderr)
proc.check_returncode()
files = [source / 'src/cpu/lssc/omp.cpp', generated, here / 'lssc_trace.cpp', here / 'lssc_trace.hpp', *libs, args.out / 'trace']
(args.out / 'manifest.json').write_text(json.dumps(dict(command=command, hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}), indent=2))
