#!/usr/bin/env python3
"""Same-input TWSC A/B workers, bounded real fixtures, and gated PMU requests.

Persistent workers warm once, then execute only when the paired controller asks.
GCP profiling is separate from LAN wall-time acceptance. No full-size TWSC run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from alignment_campaign import cpu_ticks, cpu_topology
from paper_compare import cpu_activity


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_ticks():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():result[fields[0]]=list(map(int,fields[1:9]))
    return result


def prepare(args):
    root,out=Path(args.fixtures),Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((root/'fixtures.json').read_text())['cases'];cases=[]
    for row in rows:
        if row['sigma']!=25:continue
        noisy=np.fromfile(root/row['noisy'],dtype='<f4').reshape(3,512,512)
        clean=np.fromfile(root/row['clean'],dtype='<f4').reshape(3,512,512)
        if sha(root/row['noisy'])!=row['noisy_sha256'] or sha(root/row['clean'])!=row['clean_sha256']:raise ValueError('fixture identity')
        name=row['id'].split('-')[1]
        for size in (24,32):
            start=(512-size)//2
            for channels in (1,3):
                label=f'{name}-{size}-c{channels}'
                a=noisy[:channels,start:start+size,start:start+size].copy()
                z=clean[:channels,start:start+size,start:start+size].copy()
                np.save(out/(label+'.npy'),a);np.save(out/(label+'-clean.npy'),z)
                cases.append(dict(name=label,input=label+'.npy',clean=label+'-clean.npy',parameters=dict(sigma=25),
                                  crop=[start,start,size,size],original=row,channels='R plane as Gray' if channels==1 else 'RGB'))
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')


class Worker:
    def __init__(self,args):
        import vapoursynth as vs
        self.args=args;self.core=vs.core;self.core.num_threads=1;self.core.max_cache_size=128
        self.core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
        self.images=np.load(args.input).astype(np.float32)
        if self.images.ndim==3:self.images=self.images[None]
        self.t,self.c,h,w=self.images.shape;self.fills=0
        self.kwargs=json.loads(args.params)
        blank=self.core.std.BlankClip(width=w,height=h,length=self.t,format=vs.RGBS if self.c==3 else vs.GRAYS)
        def fill(n,f):
            self.fills+=1;out=f.copy()
            for p in range(self.c):np.copyto(np.asarray(out[p]),self.images[n,p])
            return out
        self.source=self.core.std.ModifyFrame(blank,blank,fill)
        self.core.std.SetVideoCache(self.source,mode=1,fixedsize=self.t)
        self.held=[self.source.get_frame(n) for n in range(self.t)]
        self.ctl=self.ack=None
        if args.control:
            self.ctl=os.open(args.control,os.O_RDWR);self.ack=os.open(args.ack,os.O_RDWR)
        warm=self.build();warm.get_frame(self.t//2)
        self.identity=dict(plugin_sha256=sha(args.plugin),input_sha256=sha(args.input),parameters=self.kwargs,
                           backend={k:(v.decode() if isinstance(v,bytes) else v) for k,v in dict(self.core.nss.Backend()).items()})
    def command(self,value):
        if self.ctl is None:return
        os.write(self.ctl,(value+'\n').encode())
        if os.read(self.ack,128).rstrip(b'\x00\n\r')!=b'ack':raise RuntimeError('perf acknowledgement')
    def build(self):
        node=self.core.nss.TWSC(self.source,**self.kwargs)
        if self.kwargs.get('radius',0):node=self.core.nss.VAggregate(node,self.source,radius=self.kwargs['radius'])
        self.core.std.SetVideoCache(node,mode=0);return node
    def run(self,output):
        nodes=[self.build() for _ in range(self.args.inner)]
        before_fills=self.fills;before=all_ticks();self.command('enable');started=time.perf_counter()
        frames=[node.get_frame(self.t//2) for node in nodes]
        elapsed=time.perf_counter()-started;self.command('disable');after=all_ticks()
        arrays=[np.array([np.asarray(f[p]) for p in range(self.c)]) for f in frames]
        if not all(np.array_equal(a,arrays[0]) for a in arrays):raise AssertionError('inner repeat changed pixels')
        if self.fills!=before_fills:raise AssertionError('source fill inside timed boundary')
        np.save(output,arrays[0])
        props={k:v for k,v in dict(frames[0].props).items() if k.startswith('_NSS') and isinstance(v,(int,float,list))}
        return dict(self.identity,seconds=elapsed/self.args.inner,elapsed_seconds=elapsed,inner=self.args.inner,
                    output_sha256=sha(output),resolved=props,timed_source_fills=0,
                    affinity=sorted(os.sched_getaffinity(0)),smt_siblings=cpu_topology()[1],cpu_activity=cpu_activity(before,after))


def server(args):
    worker=Worker(args);print(json.dumps(dict(ready=True,identity=worker.identity)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        print(json.dumps(worker.run(request['output'])),flush=True)


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    if os.sched_getaffinity(0)!={args.cpu}:raise RuntimeError('pin controller and workers to selected CPU')
    cases=json.loads(Path(args.cases).read_text());selected=set(args.case or [c['name'] for c in cases])
    cases=[c for c in cases if c['name'] in selected]
    if len(cases)!=len(selected):raise ValueError('unknown case')
    rows=[];summaries=[]
    for case in cases:
        children={};logs={};params=case['parameters']
        inner=case.get('inner',1)
        try:
            for label,plugin in [('base',args.base),('candidate',args.candidate)]:
                log=open(out/f"{case['name']}-{label}-stderr.log",'w');logs[label]=log
                parameters=case.get('base_parameters',params) if label=='base' else params
                cmd=[sys.executable,str(Path(__file__).resolve()),'server','--plugin',str(Path(plugin).resolve()),
                     '--input',str((Path(args.cases).resolve().parent/case['input']).resolve()),'--params',json.dumps(parameters),'--inner',str(inner)]
                child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                                       env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
                children[label]=child
                ready=json.loads(child.stdout.readline());assert ready['ready']
            for repeat in range(args.repeats):
                for label in (['base','candidate'] if repeat%2==0 else ['candidate','base']):
                    child=children[label];file=out/f"{case['name']}-{label}-{repeat}.npy"
                    child.stdin.write(json.dumps(dict(output=str(file)))+'\n');child.stdin.flush()
                    data=json.loads(child.stdout.readline());data.update(case=case['name'],variant=label,repeat=repeat,output=file.name)
                    rows.append(data);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(case['name'],repeat,label,data['seconds'],flush=True)
            a=[r for r in rows if r['case']==case['name'] and r['variant']=='base']
            b=[r for r in rows if r['case']==case['name'] and r['variant']=='candidate']
            hashes={r['output_sha256'] for r in a+b};ratios=[x['seconds']/y['seconds'] for x,y in zip(a,b)]
            rng=np.random.default_rng(20260908)
            medians=np.median(rng.choice(ratios,(10000,len(ratios)),replace=True),axis=1)
            first=np.load(out/a[0]['output']);second=np.load(out/b[0]['output'])
            summaries.append(dict(case=case,base_seconds=statistics.median(r['seconds'] for r in a),
                     candidate_seconds=statistics.median(r['seconds'] for r in b),paired_ratio=statistics.median(ratios),
                     paired_ratio_ci95=np.quantile(medians,[.025,.975]).tolist(),exact_all=len(hashes)==1,
                     max_abs=float(np.max(np.abs(first.astype(float)-second.astype(float)))),
                     base_deterministic=len({r['output_sha256'] for r in a})==1,candidate_deterministic=len({r['output_sha256'] for r in b})==1))
        finally:
            for child in children.values():
                if child.poll() is None:
                    try:child.stdin.write('{"stop":true}\n');child.stdin.flush();child.wait(timeout=10)
                    except (BrokenPipeError,subprocess.TimeoutExpired):child.terminate();child.wait()
            for log in logs.values():log.close()
    report=dict(scope='Bounded complete-pipeline paired LAN A/B; defaults unchanged unless explicitly listed',
                source_sha256=sha(__file__),repeats=args.repeats,rows=rows,comparisons=summaries,
                exact_all=all(s['exact_all'] for s in summaries))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('prepare');s.add_argument('--fixtures',required=True);s.add_argument('--out',required=True)
    for mode in ('server','profile'):
        s=sub.add_parser(mode)
        for key in ('plugin','input','params'):s.add_argument('--'+key,required=True)
        s.add_argument('--inner',type=int,default=1);s.add_argument('--control');s.add_argument('--ack');s.add_argument('--output')
    s=sub.add_parser('run')
    for key in ('base','candidate','cases','out'):s.add_argument('--'+key,required=True)
    s.add_argument('--case',nargs='+');s.add_argument('--repeats',type=int,default=7);s.add_argument('--cpu',type=int,default=0)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='run':run(args)
    elif args.mode=='server':server(args)
    else:print(json.dumps(Worker(args).run(args.output)))
