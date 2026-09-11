#!/usr/bin/env python3
"""Paired complete-pipeline TWSC/NLH controls and frozen-v2 migration evidence.

Each worker preloads real VS source frames before timing. Startup, decoding and
output serialization are excluded; estimation/matching/all iterations/aggregation
are included. No quality or speed threshold is inferred from a changed algorithm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from paper_compare import quality, cpu_activity


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_topology():
    if not hasattr(os,'sched_getaffinity'): return None,[]
    allowed=sorted(os.sched_getaffinity(0));cpu=allowed[0]
    path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
    siblings=[]
    if path.exists():
        for part in path.read_text().strip().split(','):
            endpoints=list(map(int,part.split('-')))
            siblings.extend(range(endpoints[0],endpoints[-1]+1))
    return cpu,[x for x in siblings if x!=cpu]


def cpu_ticks():
    cpu,siblings=cpu_topology();wanted={f'cpu{x}' for x in [cpu,*siblings] if x is not None}
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0] in wanted: result[fields[0]]=list(map(int,fields[1:9]))
    return result


def metrics(a,b):
    e=np.abs(a.astype(float)-b.astype(float))
    return dict(max_abs=float(e.max()),rms=float(np.sqrt(np.mean(e*e))),
                p99=float(np.quantile(e,.99)),exact=bool(np.array_equal(a,b)),
                psnr_ssim=[quality(x,y) for x,y in zip(a,b)])


def params(model,profile,old=False):
    if profile=='defaults': return dict(sigma=10)
    if profile=='blind': return {} if model=='NLH' or old else dict(estimate_sigma=1)
    common=dict(sigma=10,block_size=4,block_step=4,group_size=8,bm_range=3)
    if model=='TWSC':
        common.update(iters=2,lambda2=1,delta=0)
        if not old: common.update(admm_iter=10)
    elif not old: common.update(basic_iters=2,q=4)
    return common


def worker(args):
    import vapoursynth as vs
    core=vs.core;core.num_threads=1;core.max_cache_size=128
    plugin=Path(args.plugin).resolve();core.std.LoadPlugin(path=str(plugin))
    images=np.load(args.input).astype(np.float32)
    if images.ndim==3: images=images[None]
    t,c,h,w=images.shape
    blank=core.std.BlankClip(width=w,height=h,length=t,format=vs.RGBS if c==3 else vs.GRAYS)
    source_fills=[0]
    def fill(n,f):
        source_fills[0]+=1;out=f.copy()
        for p in range(c): np.copyto(np.asarray(out[p]),images[n,p])
        return out
    source=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(source,mode=1,fixedsize=t)
    held=[source.get_frame(n) for n in range(t)]
    kwargs=json.loads(args.params)
    def build():
        node=getattr(core.nss,args.model)(source,**kwargs)
        if kwargs.get('radius',0): node=core.nss.VAggregate(node,source,radius=kwargs['radius'])
        core.std.SetVideoCache(node,mode=0)
        return node
    warm=build()
    with warm.get_frame(t//2): pass
    del warm
    node=build();before_fills=source_fills[0]
    before=cpu_ticks() if Path('/proc/stat').exists() else {}
    started=time.perf_counter()
    frame=node.get_frame(t//2)
    elapsed=time.perf_counter()-started
    activity=cpu_activity(before,cpu_ticks()) if before else {}
    pixels=np.array([np.asarray(frame[p]) for p in range(c)])
    props={k:v for k,v in dict(frame.props).items() if k.startswith('_NSS') and isinstance(v,(float,int,list))}
    np.save(args.output,pixels)
    timed_fills=source_fills[0]-before_fills
    result=dict(seconds=elapsed,timed_source_fills=timed_fills,parameters=kwargs,resolved=props,
                plugin_sha256=sha(plugin),input_sha256=sha(args.input),pixels_sha256=sha(args.output),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                smt_siblings=cpu_topology()[1],
                backend=dict(core.nss.Backend()),cpu_activity=activity)
    result['backend']={k:(v.decode() if isinstance(v,bytes) else v) for k,v in result['backend'].items()}
    if timed_fills: raise AssertionError('source was evaluated inside the timing boundary')
    print(json.dumps(result))


def campaign(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    if hasattr(os,'sched_getaffinity') and os.sched_getaffinity(0)!={args.cpu}: raise RuntimeError(f'run Linux campaign under taskset -c {args.cpu}')
    rng=np.random.default_rng(20260908)
    fixtures={}
    for channels in (1,3):
        y,x=np.mgrid[:24,:24]
        # Quantized deterministic input keeps matching ties interpretable.
        a=np.array([.2+(x+2*y+p*5)/256+rng.integers(-12,13,(24,24))/256 for p in range(channels)],np.float32)
        path=out/f'input-{channels}.npy';np.save(path,a);fixtures[channels]=path
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    rows=[]; comparisons=[]
    scripts={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('paper_compare.py'))}
    for model in args.models:
        for profile in args.profiles:
            for channels in args.channels:
                variants=[('generic',args.generic),('optimized',args.optimized)]
                if args.baseline: variants.append(('baseline',args.baseline))
                for repeat in range(args.repeats):
                    for variant,plugin in (variants if repeat%2==0 else variants[::-1]):
                        label=f'{model}-{profile}-c{channels}-r{repeat}-{variant}'
                        output=out/f'{label}.npy'
                        command=[sys.executable,str(Path(__file__).resolve()),'worker','--plugin',str(Path(plugin).resolve()),
                                 '--model',model,'--input',str(fixtures[channels]),'--output',str(output),
                                 '--params',json.dumps(params(model,profile,variant=='baseline'))]
                        print(label,flush=True)
                        run=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                        (out/f'{label}.log').write_bytes(run.stdout)
                        row=dict(model=model,profile=profile,channels=channels,repeat=repeat,variant=variant,output=output.name,returncode=run.returncode)
                        if run.returncode==0: row.update(json.loads(run.stdout.decode().splitlines()[-1]))
                        rows.append(row)
                        (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                selected=[r for r in rows if (r['model'],r['profile'],r['channels'])==(model,profile,channels)]
                comparison=dict(model=model,profile=profile,channels=channels,shape=[channels,24,24])
                for variant,_ in variants:
                    v=[r for r in selected if r['variant']==variant and r['returncode']==0]
                    if len(v)!=args.repeats: continue
                    comparison[variant]=dict(median_seconds=statistics.median(r['seconds'] for r in v),
                        deterministic=len({r['pixels_sha256'] for r in v})==1)
                    if variant!='optimized':
                        opt=[r for r in selected if r['variant']=='optimized' and r['returncode']==0]
                        if len(opt)==args.repeats:
                            comparison[variant]['difference_to_optimized']=metrics(np.load(out/v[0]['output']),np.load(out/opt[0]['output']))
                            comparison[variant]['paired_time_ratio']=statistics.median(a['seconds']/b['seconds'] for a,b in zip(v,opt))
                comparisons.append(comparison)
    report=dict(scope='24x24 complete-pipeline controls; bounded/default/estimation profiles are separate; not original-size paper reproduction',
                platform=platform.platform(),machine=platform.machine(),scripts=scripts,rows=rows,comparisons=comparisons,
                workers_passed=all(r['returncode']==0 for r in rows))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report['workers_passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    w=sub.add_parser('worker')
    for name in ('plugin','model','input','output','params'): w.add_argument('--'+name,required=True)
    c=sub.add_parser('run');c.add_argument('--generic',required=True);c.add_argument('--optimized',required=True);c.add_argument('--baseline');c.add_argument('--out',required=True)
    c.add_argument('--repeats',type=int,default=3);c.add_argument('--channels',type=int,nargs='+',default=[1,3])
    c.add_argument('--cpu',type=int,default=0)
    c.add_argument('--models',nargs='+',default=['TWSC','NLH']);c.add_argument('--profiles',nargs='+',default=['bounded','defaults','blind'])
    a=p.parse_args()
    if a.mode=='worker': worker(a)
    else: raise SystemExit(0 if campaign(a) else 1)
