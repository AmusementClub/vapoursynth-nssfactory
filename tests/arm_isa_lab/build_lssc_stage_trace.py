#!/usr/bin/env python3
"""Private current-source LSSC stage capture and dictionary replay."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['source','build','out']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
prototypes='void lssc_step(const float*,int,const float*,int,int,const float*,const unsigned char*,int,double,int);\nvoid lssc_coefficients(const float*,int,const int*,int);\nvoid lssc_dictionary(float*,int,int,int,int);\n'
commands=json.loads((a.build/'compile_commands.json').read_text());objects=[];records=[]
for name in ['omp','dict']:
 original=a.source/('src/cpu/lssc/'+name+'.cpp');text=original.read_text()
 if name=='omp':
  anchor='        if (best < 0 || largest < double(1e-12f)) break;';assert text.count(anchor)==1
  text=text.replace(anchor,'        lssc_step(y,m,D,atoms,ldd,work,used,best,largest,iteration);\n'+anchor)
  anchor='    return count;';assert text.count(anchor)==1
  text=text.replace(anchor,'    lssc_coefficients(a,atoms,support,count);\n'+anchor)
 else:
  anchor='    if (patches && n > 0 && lda >= m && iters > 0) {';assert text.count(anchor)==1
  text=text.replace(anchor,'    lssc_dictionary(D,m,atoms,ldd,0);\n'+anchor)
  anchor='    }\n}\n\nvoid lssc_dict_init(';assert text.count(anchor)==1
  text=text.replace(anchor,'    }\n    lssc_dictionary(D,m,atoms,ldd,1);\n}\n\nvoid lssc_dict_init(')
 generated=a.out/(name+'.cpp');generated.write_text(prototypes+text);obj=a.out/(name+'.o');objects.append(obj)
 entry=next(x for x in commands if x['file'].endswith('/lssc/'+name+'.cpp'));argv=shlex.split(entry['command']);new=[];skip=False
 for value in argv:
  if skip:skip=False;continue
  if value=='-o':skip=True;continue
  if value=='-c' or value==entry['file']:continue
  new.append(value)
 new+=['-c',str(generated.resolve()),'-o',str(obj.resolve())];proc=subprocess.run(new,cwd=entry['directory'],capture_output=True,text=True);(a.out/(name+'-build.log')).write_text(proc.stdout+proc.stderr);proc.check_returncode();records.append(new)
libs=[a.build/name for name in ['libnss_cpu.a','libnss_svd_qreplay.a','_deps/highway-build/libhwy.a']]
cmd=[new[0],'-std=c++20','-O3','-I'+str(a.source/'include'),'-DNSS_AVX2_DEFAULTS=0',str(here/'lssc_stage_trace.cpp'),*map(str,objects),*map(str,libs),'-lpthread','-o',str(a.out/'trace')]
proc=subprocess.run(cmd,capture_output=True,text=True);(a.out/'link.log').write_text(proc.stdout+proc.stderr);proc.check_returncode();records.append(cmd)
files=[a.source/('src/cpu/lssc/'+n+'.cpp') for n in ['omp','dict']]+[a.out/(n+'.cpp') for n in ['omp','dict']]+libs+[here/'lssc_stage_trace.cpp',Path(__file__),a.out/'trace']
(a.out/'manifest.json').write_text(json.dumps(dict(commands=records,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
