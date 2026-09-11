#!/usr/bin/env python3
"""Non-timed full-plane diagnostics linked to the actual candidate libraries."""
from pathlib import Path
import argparse,json,shlex,subprocess,hashlib
p=argparse.ArgumentParser(description=__doc__)
for name in ['source','build','out']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
original=a.source/'src/cpu/lssc/omp.cpp';text=original.read_text().replace('#include "cpu/lssc/mixed_omp.hpp"','#include "cpu/lssc/mixed_omp.hpp"\nnss::detail::MixedOmpStats mixed_stats;').replace('mixed_search.select(work, used, best, largest)','mixed_search.select(work, used, best, largest, &mixed_stats)');generated=a.out/'omp_diagnostic.cpp';generated.write_text(text)
entry=next(v for v in json.loads((a.build/'compile_commands.json').read_text()) if v['file'].endswith('/lssc/omp.cpp'));cmd=shlex.split(entry['command']);argv=[];skip=False
for value in cmd:
 if skip:skip=False;continue
 if value=='-o':skip=True;continue
 if value=='-c' or value==entry['file']:continue
 argv.append(value)
obj=a.out/'omp.o';argv+=['-c',str(generated.resolve()),'-o',str(obj.resolve())];r=subprocess.run(argv,cwd=entry['directory'],capture_output=True,text=True);(a.out/'compile.log').write_text(r.stdout+r.stderr);r.check_returncode()
libs=[a.build/n for n in ['libnss_cpu.a','libnss_svd_qreplay.a','_deps/highway-build/libhwy.a']];link=[cmd[0],'-std=c++20','-O3','-I'+str(a.source.resolve()/'include'),'-I'+str(a.source.resolve()/'src'),str(here/'mixed_diagnostics.cpp'),str(obj),*map(str,libs),'-lpthread','-o',str(a.out/'diagnostic')];r=subprocess.run(link,capture_output=True,text=True);(a.out/'link.log').write_text(r.stdout+r.stderr);r.check_returncode()
files=[original,generated,here/'mixed_diagnostics.cpp',Path(__file__),*libs,a.out/'diagnostic'];(a.out/'manifest.json').write_text(json.dumps(dict(commands=[argv,link],files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
