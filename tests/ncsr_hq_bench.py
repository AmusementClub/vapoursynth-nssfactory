#!/usr/bin/env python3
"""Paired same-model NCSR HQ optimization campaign; isolated plugin adapters only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
from paper_compare import cpu_activity, cpu_ticks, fixture, quality, save_json, sha


def worker(args):
    import vapoursynth as vs
    if args.samples<1:raise ValueError('At least one measured frame is required')
    root,case=fixture(args.fixtures,args.case)
    os.environ['NSS_NCSR_ALIGNMENT_FLAGS']='31'  # Frozen reference adapter only.
    values=np.fromfile(root/case['noisy'],dtype='<f4').reshape(case['height'],case['width'])
    core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    count=args.samples+args.warmup
    blank=core.std.BlankClip(width=case['width'],height=case['height'],length=count,format=vs.GRAYS)
    fills=0
    def fill(n,f):
        nonlocal fills
        fills+=1;out=f.copy();np.asarray(out[0])[:]=values;return out
    source=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(source,mode=1,fixedsize=count,maxsize=count)
    for n in range(count):source.get_frame(n)
    options=dict(sigma=case['sigma'],block_size=7,block_step=1,group_size=16,bm_range=30,iters=9,delta=.02,radius=0)
    options.update(json.loads(args.params))
    node=core.nss.NCSR(source,**options)
    warm_hash=None
    if args.warmup:
        warm=node.get_frame(0)
        warm_hash=hashlib.sha256(np.asarray(warm[0]).astype('<f4').tobytes()).hexdigest()
    before=fills;seconds=[];hashes=[];traces=[]
    for n in range(args.warmup,count):
        start=time.perf_counter();frame=node.get_frame(n);seconds.append(time.perf_counter()-start)
        pixels=np.asarray(frame[0]).astype('<f4')
        hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        trace=frame.props.get('_NcsrAlignmentTrace')
        if not trace:raise RuntimeError('HQ adapter trace missing; refusing to benchmark a different algorithm')
        trace=json.loads(trace)
        if trace.get('model')!='clustered-pca-hq' and trace.get('flags')!=31:raise RuntimeError('Unexpected model trace')
        if len(trace['iterations'])!=options['iters']:raise RuntimeError('Unexpected iteration budget')
        traces.append(trace)
    if fills!=before:raise RuntimeError('Timed source fill')
    if len(set(hashes+([warm_hash] if warm_hash else [])))!=1:raise RuntimeError('Repeated output differs')
    pixels.tofile(args.output)
    save_json(str(args.output)+'.json',dict(seconds=statistics.median(seconds),timings=seconds,
        output_sha256=sha(args.output),plugin_sha256=sha(args.plugin),parameters=options,flags=31,
        samples=args.samples,warm_output_sha256=warm_hash,repeat_hashes_equal=True,timed_source_fills=0,
        full_shape_warmup=bool(args.warmup),evaluated_frames=count,traces=traces,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing_boundary=('warm get_frame' if args.warmup else 'cold first full-shape get_frame')+
            '; entire training/matching/filtering/aggregation included; source preloaded; output serialization excluded'))


def campaign(args):
    if os.sched_getaffinity(0)!={0}:raise RuntimeError('CPU0 affinity required')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    fixtures=Path(args.fixtures).resolve()
    cases=json.loads((fixtures/'fixtures.json').read_text())['cases']
    if args.cases:cases=[c for c in cases if c['id'] in args.cases]
    if not cases:raise ValueError('No matching cases')
    plugins=dict(item.split('=',1) for item in args.plugins)
    if len(plugins)!=len(args.plugins):raise ValueError('Duplicate plugin labels')
    plugins={key:str(Path(value).resolve()) for key,value in plugins.items()}
    manifest=dict(schema='nss.ncsr-hq-paired.v1',plugins=plugins,
        plugin_sha256={key:sha(value) for key,value in plugins.items()},
        parameters=json.loads(args.params),fixtures=str(fixtures),cases=[c['id'] for c in cases],
        affinity=sorted(os.sched_getaffinity(0)),harness_sha256=sha(__file__),
        cpuinfo=Path('/proc/cpuinfo').read_text(),platform=list(os.uname()),samples=args.samples,warmup=args.warmup)
    manifest_path=out/'environment.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('Refusing to mix different campaign inputs')
    save_json(manifest_path,manifest)
    result_path=out/'results.jsonl'
    rows=[json.loads(x) for x in result_path.read_text().splitlines()] if result_path.exists() else []
    done={(r['case'],r['variant'],r['repeat']) for r in rows if r.get('ok')}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in cases:
        clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(case['height'],case['width'])
        for repeat in range(args.repeat_start,args.repeat_start+args.repeats):
            order=list(plugins)
            if repeat%2:order.reverse()
            for variant in order:
                if (case['id'],variant,repeat) in done:continue
                output=out/f"{case['id']}-{variant}-r{repeat}.f32"
                row=dict(case=case['id'],variant=variant,repeat=repeat,sigma=case['sigma'],image=case['image'],
                    width=case['width'],height=case['height'],noisy_sha256=case['noisy_sha256'],
                    clean_sha256=case['clean_sha256'],output=output.name)
                cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--fixtures',str(fixtures),
                    '--case',case['id'],'--plugin',plugins[variant],'--params',args.params,
                    '--samples',str(args.samples),'--warmup',str(args.warmup),'--output',str(output)]
                print(f"START {case['id']} {variant} r{repeat}",flush=True)
                try:
                    before=cpu_ticks();start=time.monotonic()
                    run=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=args.timeout)
                    row['process_seconds']=time.monotonic()-start;row['cpu_activity']=cpu_activity(before,cpu_ticks())
                    output.with_suffix('.log').write_bytes(run.stdout)
                    if run.returncode:raise RuntimeError(run.stdout[-2500:].decode(errors='replace'))
                    pixels=np.fromfile(output,dtype='<f4').reshape(clean.shape)
                    if not np.isfinite(pixels).all():raise RuntimeError('Nonfinite output')
                    row.update(ok=True,quality=quality(clean,pixels),**json.loads(Path(str(output)+'.json').read_text()))
                except Exception as error:row.update(ok=False,error=str(error))
                with result_path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({k:row.get(k) for k in ('case','variant','repeat','ok','seconds','quality','peak_rss_kib','error')},allow_nan=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('worker','campaign'):
        p=sub.add_parser(mode);p.add_argument('--fixtures',required=True)
        p.add_argument('--params',default='{"block_size":7,"block_step":1,"group_size":16,"bm_range":30,"iters":9,"delta":0.02}')
        p.add_argument('--samples',type=int,default=1)
        p.add_argument('--warmup',type=int,choices=(0,1),default=1)
        if mode=='worker':
            p.add_argument('--case',required=True);p.add_argument('--plugin',required=True);p.add_argument('--output',required=True)
        else:
            p.add_argument('--plugins',nargs='+',required=True);p.add_argument('--cases',nargs='*');p.add_argument('--out',required=True)
            p.add_argument('--repeats',type=int,default=3);p.add_argument('--repeat-start',type=int,default=0)
            p.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    (worker if args.mode=='worker' else campaign)(args)


if __name__=='__main__':main()
