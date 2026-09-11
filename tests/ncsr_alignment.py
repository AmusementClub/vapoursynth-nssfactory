#!/usr/bin/env python3
"""Bounded NCSR model-ablation campaign. All experimental modes are opt-in."""
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
from paper_compare import cpu_activity, cpu_ticks, fixture, quality, save_json, sha


VARIANTS = {
    'baseline': dict(flags=0, baseline=True),
    'hook_control': dict(flags=0),
    'A': dict(flags=1),
    'W': dict(flags=2),
    'AW': dict(flags=3),
    'B': dict(flags=4),
    'AWB': dict(flags=7),
    'AWB_reuse': dict(flags=23),
    'C': dict(flags=31),
    'D_step4': dict(flags=31, block_step=4),
    'D_step2': dict(flags=31, block_step=2),
    'D_step1': dict(flags=31, block_step=1),
    'D_range15': dict(flags=31, block_step=2, bm_range=15),
    'D_range30': dict(flags=31, block_step=2, bm_range=30),
    'D_iters3': dict(flags=31, block_step=2, bm_range=15, iters=3),
    'D_iters6': dict(flags=31, block_step=2, bm_range=15, iters=6),
    'D_iters9': dict(flags=31, block_step=2, bm_range=15, iters=9),
    'D_group16': dict(flags=31, block_step=2, bm_range=15, group_size=16),
    'D_author_budget': dict(flags=31, block_step=1, bm_range=30, group_size=16,
                            block_size=7, iters=9, delta=.02),
    'legacy_step1': dict(flags=0,baseline=True,block_step=1),
    'W_step1': dict(flags=2,block_step=1),
    'B_step1': dict(flags=4,block_step=1),
    'WB_step1': dict(flags=6,block_step=1),
    'AWB_step1': dict(flags=7,block_step=1),
    'legacy_author_budget': dict(flags=0,baseline=True,block_step=1,bm_range=30,group_size=16,
                                 block_size=7,iters=9,delta=.02),
}


def worker(args):
    import vapoursynth as vs
    root,case=fixture(args.fixtures,args.case)
    config=dict(VARIANTS[args.variant])
    flags=config.pop('flags'); baseline=config.pop('baseline',False)
    os.environ['NSS_NCSR_ALIGNMENT_FLAGS']=str(flags)
    plugin=Path(args.baseline if baseline else args.candidate).resolve()
    values=np.fromfile(root/case['noisy'],dtype='<f4').reshape(case['height'],case['width'])
    core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=str(plugin))
    count=args.samples+1
    blank=core.std.BlankClip(width=case['width'],height=case['height'],length=count,format=vs.GRAYS)
    fills=0
    def fill(n,f):
        nonlocal fills
        fills+=1;out=f.copy();np.asarray(out[0])[:]=values;return out
    source=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(source,mode=1,fixedsize=count,maxsize=count)
    for n in range(count):source.get_frame(n)
    options=dict(sigma=case['sigma'],block_size=8,block_step=8,group_size=8,bm_range=7,iters=2,delta=.1,radius=0)
    options.update(config)
    node=core.nss.NCSR(source,**options)
    warm=node.get_frame(0);warm_hash=hashlib.sha256(np.asarray(warm[0]).astype('<f4').tobytes()).hexdigest()
    before=fills;seconds=[];hashes=[];traces=[]
    for n in range(1,count):
        start=time.perf_counter();frame=node.get_frame(n);seconds.append(time.perf_counter()-start)
        pixels=np.asarray(frame[0]).astype('<f4')
        hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        trace=frame.props.get('_NcsrAlignmentTrace')
        traces.append(json.loads(trace) if trace else None)
    if fills!=before:raise RuntimeError('Timed source fill')
    if len(set(hashes+[warm_hash]))!=1:raise RuntimeError('Repeated output differs')
    pixels.tofile(args.output)
    save_json(str(args.output)+'.json',dict(seconds=statistics.median(seconds),timings=seconds,
        output_sha256=sha(args.output),plugin_sha256=sha(plugin),parameters=options,flags=flags,
        samples=args.samples,warm_output_sha256=warm_hash,repeat_hashes_equal=True,timed_source_fills=0,
        traces=traces,timing_boundary='warm get_frame incl entire training/matching/filtering/aggregation; source preloaded; output serialization excluded'))


def campaign(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    fixtures=Path(args.fixtures).resolve()
    cases=json.loads((fixtures/'fixtures.json').read_text())['cases']
    if args.cases:cases=[c for c in cases if c['id'] in args.cases]
    rows=[];path=root/'results.jsonl'
    if path.exists():rows=[json.loads(line) for line in path.read_text().splitlines()]
    completed={(r['case'],r['variant'],r['repeat']) for r in rows if r.get('ok')}
    if os.sched_getaffinity(0)!={0}:raise RuntimeError('CPU0 affinity required')
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    save_json(root/'environment.json',dict(platform=list(os.uname()),affinity=sorted(os.sched_getaffinity(0)),
        cpuinfo=Path('/proc/cpuinfo').read_text(),baseline_sha256=sha(args.baseline),candidate_sha256=sha(args.candidate),
        harness_sha256=sha(__file__),variants=VARIANTS))
    for case in cases:
        clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(case['height'],case['width'])
        noisy=np.fromfile(fixtures/case['noisy'],dtype='<f4').reshape(clean.shape)
        for repeat in range(args.repeat_start,args.repeat_start+args.repeats):
            variants=list(args.variants)
            if repeat%2:variants.reverse()
            for variant in variants:
                if (case['id'],variant,repeat) in completed:continue
                output=root/f"{case['id']}-{variant}-r{repeat}.f32"
                row=dict(case=case['id'],variant=variant,repeat=repeat,sigma=case['sigma'],image=case['image'],
                    width=case['width'],height=case['height'],noisy_sha256=case['noisy_sha256'],clean_sha256=case['clean_sha256'],
                    noisy_quality=quality(clean,noisy),output=output.name)
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--fixtures',str(fixtures),'--case',case['id'],
                    '--variant',variant,'--baseline',str(Path(args.baseline).resolve()),'--candidate',str(Path(args.candidate).resolve()),
                    '--samples',str(args.samples),'--output',str(output)]
                print(f"START {case['id']} {variant} r{repeat}",flush=True)
                try:
                    before=cpu_ticks();started=time.time()
                    run=subprocess.run(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=args.timeout)
                    row['cpu_activity']=cpu_activity(before,cpu_ticks());row['process_seconds']=time.time()-started
                    output.with_suffix('.log').write_bytes(run.stdout)
                    if run.returncode:raise RuntimeError(run.stdout[-2500:].decode(errors='replace'))
                    pixels=np.fromfile(output,dtype='<f4').reshape(clean.shape)
                    if not np.isfinite(pixels).all():raise RuntimeError('Nonfinite output')
                    row.update(ok=True,quality=quality(clean,pixels),**json.loads(Path(str(output)+'.json').read_text()))
                except Exception as error:
                    row.update(ok=False,error=str(error))
                with path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row)
                print(json.dumps({k:row.get(k) for k in ('case','variant','repeat','ok','seconds','quality','error')},allow_nan=False),flush=True)
    save_json(root/'results.json',dict(schema='nss.ncsr-alignment.v1',rows=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('worker','campaign'):
        p=sub.add_parser(mode)
        for name in ('fixtures','baseline','candidate'):p.add_argument('--'+name,required=True)
        p.add_argument('--samples',type=int,default=3)
        if mode=='worker':
            p.add_argument('--case',required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
            p.add_argument('--output',required=True)
        else:
            p.add_argument('--out',required=True);p.add_argument('--cases',nargs='+')
            p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline','hook_control','A','W','AW','B','AWB','AWB_reuse','C'])
            p.add_argument('--repeats',type=int,default=3);p.add_argument('--repeat-start',type=int,default=0)
            p.add_argument('--timeout',type=int,default=180)
    args=parser.parse_args();globals()[args.mode](args)


if __name__=='__main__':main()
