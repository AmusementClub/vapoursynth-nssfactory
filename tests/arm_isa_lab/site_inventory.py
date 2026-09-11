#!/usr/bin/env python3
"""Inventory SIMD-sensitive sites and prove active preprocessing branches.

Pragma markers are inserted into a private source copy. The actual native
build flags are reused with -E, so production binaries never contain markers.
Runtime evidence is attached at the tested kernel-family level, not invented
as per-line execution coverage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
for n in ('source','build','out'):parser.add_argument('--'+n,type=Path,required=True)
parser.add_argument('--jobs',type=int,default=2)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
source=args.source.resolve();private=args.out.resolve()/'source'
shutil.copytree(source,private,ignore=shutil.ignore_patterns('.git','build*','artifacts','ds','__pycache__'))
pattern=re.compile(r'HWY_MAX_BYTES|HWY_TARGET\b|(?:FixedTag|CappedTag|ScalableTag)<|Reduce(?:Sum|Min|Max)\(|LowerHalf\(|UpperHalf\(|TableLookupLanes\(|Reverse(?:2|4|8)?\(|LoadN\(|StoreN\(|(?:InterleaveLower|InterleaveUpper|Combine|Concat\w*|OddEven|ExtractLane|GatherIndexN?|ScatterIndexN?|LoadInterleaved[24]|StoreInterleaved[24]|FirstN|CountTrue|AllFalse|AllTrue|FindFirstTrue)\(|\b(?:V\d*|VW|VF|M)\s+\w+\[[^]]+\]')
sites=[]
for path in sorted((source/'src/cpu').rglob('*')):
    if path.suffix not in ('.cpp','.hpp') or 'codelet_' in path.name:continue
    lines=path.read_text().splitlines();new=[];stack=[]
    for number,line in enumerate(lines,1):
        stripped=line.strip();kind=None
        if stripped.startswith('//'):
            new.append(line)
            continue
        if re.match(r'#\s*(?:if|ifdef|ifndef)\b',stripped):
            stack.append(bool(re.search(r'HWY_MAX_BYTES|HWY_TARGET\b',stripped)))
        width_branch=bool(stack and stack[-1] and re.match(r'#\s*(?:else|elif)\b',stripped))
        if pattern.search(line) or width_branch:
            kind='width_branch' if stripped.startswith('#') else 'vector_array' if re.search(r'\b(?:V\d*|VW|VF|M)\s+\w+\[',line) else 'simd_expression'
        if re.match(r'#\s*endif\b',stripped) and stack:stack.pop()
        first=number-1
        while first>0 and lines[first-1].rstrip().endswith('\\'):first-=1
        in_macro=lines[first].lstrip().startswith('#define')
        if kind and not in_macro and ((number>1 and lines[number-2].rstrip().endswith('\\')) or line.rstrip().endswith('\\')):
            raise RuntimeError(f'continued directive needs explicit review: {path}:{number}')
        if kind:
            ident=len(sites)
            sites.append(dict(id=ident,source=str(path.relative_to(source)),line=number,kind=kind,
                              code=stripped,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            marker=f'#pragma message("NSSINV:{ident}:" NSS_INV_STRING(HWY_TARGET))'
            if in_macro:
                # _Pragma is expanded only when the macro is used, rather
                # than claiming that an unused macro body was compiled.
                sites[-1]['kind']='macro_expansion'
                if first==number-1:
                    match=re.match(r'^(#\s*define\s+\w+(?:\([^)]*\))?)(\s+)(.*)$',line)
                    if not match:raise RuntimeError('unsupported macro definition')
                    new.append(match[1]+match[2]+f'NSS_INV_MARK({ident}) '+match[3])
                else:new.extend((f'NSS_INV_MARK({ident}) \\',line))
            elif stripped.startswith('#'):
                new.extend((line,marker))
            else:new.extend((marker,line))
        else:new.append(line)
    (private/path.relative_to(source)).write_text('\n'.join(new)+'\n')
mark=args.out.resolve()/'markers.hpp'
mark.write_text('#define NSS_INV_STRING_I(x) #x\n#define NSS_INV_STRING(x) NSS_INV_STRING_I(x)\n'
                '#define NSS_INV_PRAGMA_I(x) _Pragma(#x)\n#define NSS_INV_PRAGMA(x) NSS_INV_PRAGMA_I(x)\n'
                '#define NSS_INV_MARK(id) NSS_INV_PRAGMA(message("NSSINV:" #id ":" NSS_INV_STRING(HWY_TARGET)))\n')
commands=json.loads((args.build/'compile_commands.json').read_text())
selected=[c for c in commands if '/src/cpu/' in c['file']]
if not selected:raise RuntimeError('no CPU translation units in compile database')

def preprocess(pair):
    number,entry=pair
    original=Path(entry['file']).resolve()
    # Derive the original configured source root from the CPU path. It can
    # differ from --source when auditing an identical integrated source copy.
    configured=Path(str(original).split('/src/cpu/',1)[0])
    argv=entry.get('arguments') or shlex.split(entry['command'])
    converted=[];skip=False
    for value in argv:
        if skip:skip=False;continue
        if value=='-o':skip=True;continue
        if value=='-c':continue
        converted.append(value.replace(str(configured),str(private)))
    converted+=['-E','-include',str(mark)]
    proc=subprocess.run(converted,cwd=entry['directory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    # Clang reports pragma messages during preprocessing; GCC retains them in
    # preprocessed output and diagnoses them during compilation. Inspect both.
    pragmas=[line for line in proc.stdout.splitlines() if '#pragma' in line and 'NSSINV:' in line]
    normalized='\n'.join(re.sub(r'"\s*"','',line) for line in pragmas)
    log=args.out/f'tu-{number}.log';log.write_text(proc.stderr+'\n'+normalized+'\n')
    if proc.returncode:raise RuntimeError(f'preprocess failed: {entry["file"]}: {log}')
    active=sorted(set(int(x) for x in re.findall(r'NSSINV:(\d+):',proc.stderr+'\n'+normalized)))
    return dict(source=str(original.relative_to(configured)),command=converted,active=active)

with ThreadPoolExecutor(max_workers=args.jobs) as pool:executions=list(pool.map(preprocess,enumerate(selected)))
if not any(run['active'] for run in executions):raise RuntimeError('no active markers; compiler marker collection failed')
active={site['id']:[] for site in sites}
for run in executions:
    for ident in run['active']:active[ident].append(run['source'])
evidence={
 'backend':'test_backend;test_backend_plugin',
 'bm':'test_guarded_primitives;test_matcher;test_bm3d_contract;test_temporal_matcher;test_bm3d_semantics;test_neon_matrix',
 'wnnm':'test_svd;test_svd_scale;test_batch;test_guarded_primitives;test_wnnm_zero_rank;test_neon_matrix',
 'nlh':'test_nlh;test_guarded_primitives;test_neon_matrix',
 'nlm':'test_nlm_cpu;test_nlm_stripe;test_nlm_accum_range;test_neon_matrix',
 'mcwnnm':'test_mcwnnm;test_guarded_primitives;test_neon_matrix',
 'ncsr':'test_ncsr;test_batch;test_svd_scale;test_neon_matrix',
 'twsc':'test_twsc;test_batch;test_neon_matrix',
 'lssc':'test_lssc;test_lssc_omp;test_neon_matrix',
 'common':'test_common;test_guarded_primitives;test_batch;test_neon_matrix',
}
for site in sites:
    path=Path(site['source']);family=path.parts[2] if len(path.parts)>3 else path.stem
    site['compiled_in']=active[site['id']]
    site['status']='active_native_branch' if site['compiled_in'] else 'excluded_by_native_configuration'
    site['runtime_evidence_scope']='kernel_family' if site['compiled_in'] else 'not_required_for_excluded_branch'
    site['tests']=evidence[family] if site['compiled_in'] else ''
    if site['compiled_in'] and ('lab' in path.name):raise RuntimeError('lab site unexpectedly active in release flags')
    for test in site['tests'].split(';'):
        if test and not any((source/'tests'/(test+suffix)).exists() for suffix in ('.cpp','.py','.sh')):
            raise RuntimeError('test evidence path missing: '+test)
report=dict(schema='nssfactory.arm-site-inventory.v1',sites=sites,translation_units=executions,
            generated_codelets='Included through width-gated adapters; source marker inventory excludes generated arithmetic.',
            runtime_scope='Compilation branches proven per site; native execution/math/guard evidence is per kernel family.')
(args.out/'inventory.json').write_text(json.dumps(report,indent=2))
fields=['id','source','line','kind','status','runtime_evidence_scope','tests','code']
with (args.out/'inventory.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in sites:writer.writerow({k:row[k] for k in fields})
print(json.dumps(dict(sites=len(sites),active=sum(bool(s['compiled_in']) for s in sites),translation_units=len(executions))))
