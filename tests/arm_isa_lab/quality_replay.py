#!/usr/bin/env python3
"""Causal reference/matcher replay on retained against-clean outliers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare
from match_replay import trace_delta


def worker(args):
    import vapoursynth as vs
    from quality_fixture import graph
    from test_plan01_plugin import evaluate
    core=vs.core;core.num_threads=1;core.max_cache_size=64
    core.std.LoadPlugin(path=str(args.plugin.resolve()))
    case=json.loads(args.case.read_text());data=np.load(args.pack/'fixtures.npz')
    reference=np.load(args.reference)['pixels'] if args.reference else None
    node=graph(core,data,case,pilot=args.pilot,reference=reference)
    requested=(0,1,2,3,4) if args.pilot else (0,2,4)
    actual=np.stack([np.stack(evaluate(node,n)[0]) for n in requested])
    np.savez_compressed(args.out,pixels=actual)


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    quality=json.loads(args.comparison.read_text())
    pixels=[np.load(p/'pixels.npz') for p in (args.left_capture,args.right_capture)]
    report=dict(numerical_admission=False,cases=[])
    for row in quality['cases']:
        case=row['parameters'];index=row['index'];name=case['algorithm']
        if name in ('NLM','LSSC') or row['pixel_comparison']['passed']:continue
        directory=args.out/f'c{index}';directory.mkdir()
        casepath=directory/'case.json';casepath.write_text(json.dumps(case,indent=2))
        def invoke(plugin,label,*,pilot=False,reference=None,matching=None):
            output=directory/(label+'.npz');record=directory/(label+'.bin')
            env=dict(os.environ);env.pop('NSS_MATCH_RECORD',None);env.pop('NSS_MATCH_REPLAY',None)
            if name!='BM3D':env['NSS_MATCH_RECORD']=str(record)
            if matching:env['NSS_MATCH_REPLAY']=str(matching)
            command=[sys.executable,__file__,'worker','--plugin',str(plugin.resolve()),'--pack',str(args.pack.resolve()),
                     '--case',str(casepath),'--out',str(output)]
            if pilot:command.append('--pilot')
            if reference:command+=['--reference',str(reference)]
            proc=subprocess.run(command,env=env,text=True,capture_output=True)
            output.with_suffix('.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
            if name!='BM3D':
                meta=json.loads(Path(str(record)+'.meta.json').read_text())
                if not meta['replay_consumed'] or not meta['calls']:raise RuntimeError('unconsumed/empty matching trace')
            return np.load(output)['pixels'],record
        actual={};logs={};reproduced={}
        for side,plugin,capture in zip(('baseline','candidate'),(args.baseline,args.candidate),pixels):
            actual[side],logs[side]=invoke(plugin,side)
            reproduced[side]=bool(np.array_equal(actual[side],capture[f'c{index}']))
        fixed={};extra={}
        if name=='BM3D':
            pilots={side:invoke(plugin,side+'-pilot',pilot=True)[0] for side,plugin in (('baseline',args.baseline),('candidate',args.candidate))}
            extra['pilot']=compare(pilots['baseline'],pilots['candidate'])
            for reference in ('baseline','candidate'):
                outputs=[invoke(plugin,side+'-using-'+reference,reference=directory/(reference+'-pilot.npz'))[0]
                         for side,plugin in (('baseline',args.baseline),('candidate',args.candidate))]
                fixed[reference]=compare(*outputs)
        else:
            extra['trace']=trace_delta(logs['baseline'],logs['candidate'])
            for reference in ('baseline','candidate'):
                outputs=[invoke(plugin,side+'-using-'+reference,matching=logs[reference])[0]
                         for side,plugin in (('baseline',args.baseline),('candidate',args.candidate))]
                fixed[reference]=compare(*outputs)
        item=dict(index=index,parameters=case,reproduced=reproduced,original=row['pixel_comparison'],fixed=fixed,**extra)
        report['cases'].append(item)
        print(json.dumps(dict(index=index,algorithm=name,reproduced=reproduced,original=item['original']['max_abs'],
                              fixed=[v['max_abs'] for v in fixed.values()],pilot=extra.get('pilot',{}).get('max_abs'))),flush=True)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    report['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [args.baseline,args.candidate,Path(__file__),Path(__file__).with_name('quality_fixture.py'),*(args.out.glob('c*/*.npz')),*(args.out.glob('c*/*.bin'))]}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    w=sub.add_parser('worker')
    for n in ('plugin','pack','case','out'):w.add_argument('--'+n,type=Path,required=True)
    w.add_argument('--pilot',action='store_true');w.add_argument('--reference',type=Path)
    r=sub.add_parser('run')
    for n in ('baseline','candidate','pack','comparison','left-capture','right-capture','out'):r.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();worker(args) if args.command=='worker' else run(args)
