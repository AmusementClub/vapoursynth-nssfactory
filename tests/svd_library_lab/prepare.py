#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Patch a disposable source-lab tree for the GCP SVD comparison."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
if root.name != 'source-lab' or not str(root).startswith('/tmp/nss-svd-'):
    raise SystemExit('requires /tmp/nss-svd-*/source-lab')
changes = {}
def edit(name, old, new):
    p = root / name
    s = p.read_text()
    assert s.count(old) == 1, (name, old, s.count(old))
    changes.setdefault(name, {})['before'] = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text(s.replace(old, new))
    changes[name]['after'] = hashlib.sha256(p.read_bytes()).hexdigest()

for name in ('linalg.cpp', 'svd_batch.cpp', 'full.cpp'):
    edit('src/cpu/twsc/' + name, '#include "nss/cpu_twsc_full.hpp"',
         '#include "nss/cpu_twsc_full.hpp"\n#include "cpu/twsc/svd_lab.hpp"')
edit('src/cpu/twsc/linalg.cpp', 'constexpr double tolerance = 1e-6;',
     'const double tolerance = svd_lab_mode()==1 && std::is_same_v<T,double> ? 1e-13 : 1e-6;')
edit('src/cpu/twsc/linalg.cpp', '    double_fallback = false;\n    if (orthogonal_double) {',
     '''    double_fallback = false;
    if (svd_lab_mode() >= 2 && n >= 24) {
        double_fallback = true;
        if (svd_lab_decompose(a,m,n,lda,work) && valid_svd(a,m,n,lda,work)) return true;
        svd_lab_fallback();
        return qr_svd(a,m,n,lda,work,work.qr64,work.aux64) && valid_svd(a,m,n,lda,work);
    }
    if (orthogonal_double) {''')
edit('src/cpu/twsc/svd_batch.cpp', 'const auto tolerance=hn::Set(d,1e-6);',
     'const auto tolerance=hn::Set(d,svd_lab_mode()==1?1e-13:1e-6);')
edit('src/cpu/twsc/svd_batch.cpp', '    HWY_DYNAMIC_DISPATCH(TwscSvd64Batch)(work,m,n,count);',
     '''    if (svd_lab_mode() >= 2) {
        for(int i=0;i<count;++i) {
            if(!svd_lab_decompose(work[i]->input.data(),m,n,m,*work[i])) {
                svd_lab_fallback();
                HWY_DYNAMIC_DISPATCH(TwscSvd64Batch)(work+i,m,n,1);
            }
        }
        return;
    }
    HWY_DYNAMIC_DISPATCH(TwscSvd64Batch)(work,m,n,count);''')
edit('src/cpu/twsc/full.cpp', '\n}\n\nTwscSolverStats twsc_filter_full(',
     '\n    svd_lab_capture(w.input.data(),m,n);\n}\n\nTwscSolverStats twsc_filter_full(')
edit('src/cpu/twsc/full.cpp', '    const bool diagonal = !NSS_ALIGNMENT_GENERIC && off_norm <= 1e-12 * gram_norm;',
     '    const bool diagonal = !NSS_ALIGNMENT_GENERIC && off_norm <= 1e-12 * gram_norm;\n    svd_lab_diagonal(diagonal);')
for name in ('svd_lab.hpp', 'svd_lab.cpp'):
    dst = root / 'src/cpu/twsc' / name
    shutil.copyfile(Path(__file__).parent / name, dst)
    changes[str(dst.relative_to(root))] = {'after': hashlib.sha256(dst.read_bytes()).hexdigest()}
with (root/'CMakeLists.txt').open('a') as f:
    f.write('''
# Isolated experiment only: stable system OpenBLAS ABI, no production dependency.
find_library(SVD_LAB_OPENBLAS NAMES openblas libopenblas.so.0 REQUIRED)
target_sources(nss_cpu PRIVATE src/cpu/twsc/svd_lab.cpp)
set_source_files_properties(src/cpu/twsc/svd_lab.cpp PROPERTIES COMPILE_OPTIONS "-O3;-fno-fast-math;-ffp-contract=off")
target_link_libraries(nss_cpu PUBLIC ${SVD_LAB_OPENBLAS})
add_executable(svd_lab_bench tests/svd_library_lab/bench.cpp)
target_link_libraries(svd_lab_bench PRIVATE nss_cpu)
target_compile_options(svd_lab_bench PRIVATE -O3 -fno-fast-math -ffp-contract=off)
target_include_directories(svd_lab_bench PRIVATE src)
''')
changes['CMakeLists.txt'] = {'after': hashlib.sha256((root/'CMakeLists.txt').read_bytes()).hexdigest()}
(root.parent/'patch-manifest.json').write_text(json.dumps(changes, indent=2)+'\n')
