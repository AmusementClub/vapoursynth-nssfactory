#!/usr/bin/env python3
"""Materialize one isolated mixed-precision OMP hypothesis from frozen r3."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser(description=__doc__)
for name in ['source','out']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();shutil.copytree(a.source,a.out,ignore=shutil.ignore_patterns('.git','build*','artifacts','__pycache__'))
here=Path(__file__).resolve().parent;path=a.out/'src/cpu/lssc/omp.cpp';original=path.read_text();text='#include "cpu/lssc/mixed_omp.hpp"\n'+original
anchor='    int count = 0;';assert text.count(anchor)==1;text=text.replace(anchor,'    detail::MixedOmpSearch mixed_search(D, m, atoms, ldd);\n'+anchor)
start=text.index('        int atom = 0;',text.index('int lssc_omp_workspace('));end=text.index('        if (best < 0 || largest < double(1e-12f)) break;',start)
text=text[:start]+'        if (!mixed_search.select(work, used, best, largest)) {\n'+text[start:end]+'        }\n'+text[end:];path.write_text(text)
shutil.copy2(here/'mixed_omp.hpp',a.out/'src/cpu/lssc/mixed_omp.hpp');shutil.copy2(here/'test_mixed_omp.cpp',a.out/'tests/test_mixed_omp.cpp')
reference=original
for name in ['lssc_omp_workspace','lssc_omp','lssc_denoise_plane']:
 import re
 reference=re.sub(r'\b'+name+r'\b',name+'_reference',reference)
(a.out/'tests/omp_reference.cpp').write_text(reference)
with (a.out/'tests/CMakeLists.txt').open('a') as f:f.write('\nadd_executable(test_lssc_mixed test_mixed_omp.cpp omp_reference.cpp)\ntarget_include_directories(test_lssc_mixed PRIVATE "${PROJECT_SOURCE_DIR}/src")\ntarget_link_libraries(test_lssc_mixed PRIVATE nss_cpu)\nadd_test(NAME test_lssc_mixed COMMAND test_lssc_mixed)\n')
changes={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in ['src/cpu/lssc/omp.cpp','src/cpu/lssc/mixed_omp.hpp','tests/test_mixed_omp.cpp','tests/omp_reference.cpp','tests/CMakeLists.txt']}
(a.out/'mixed-candidate.json').write_text(json.dumps(dict(isolated=True,production_default_changed=False,baseline_omp_sha256=hashlib.sha256(original.encode()).hexdigest(),changes=changes),indent=2))
