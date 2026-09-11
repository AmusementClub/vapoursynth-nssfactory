#!/usr/bin/env python3
"""Warm alternating P1 versus P2 diagnostic timing; no default promotion."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time

from scipy.io import loadmat

from benchmark import cpu_delta, cpu_ticks, digest, eligible
from p2_reference import denoise_diagnostic
from reference import denoise
from run import read_image, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','dictionary','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--width',type=int)
    parser.add_argument('--height',type=int)
    parser.add_argument('--sigma',type=float,required=True)
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--require-cpu0',action='store_true')
    args=parser.parse_args()
    if args.pairs<1 or os.getenv('OPENBLAS_NUM_THREADS')!='1' or os.getenv('OMP_NUM_THREADS')!='1':
        raise ValueError('positive pairs and single-thread BLAS environment required')
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if args.require_cpu0 and affinity!=[0]:
        raise ValueError('verified CPU0 affinity required')
    image=read_image(args.input,args.width,args.height)
    dictionary=loadmat(args.dictionary,variable_names=['D'])['D']
    sigma=args.sigma/255
    functions={'p1':lambda:denoise(image,dictionary,sigma),
               'p2_energy_overlap':lambda:denoise_diagnostic(image,dictionary,sigma,solver='energy_gain',grouping='overlap')}
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,function in functions.items():
        result=function()
        hashes[name]=digest(result.output)
        result.output.astype('<f4').tofile(out/(name+'.f32'))
        del result
    rows=[]
    for pair in range(args.pairs):
        order=list(functions) if pair%2==0 else list(reversed(functions))
        for name in order:
            gc.collect()
            before=cpu_ticks()
            start=time.perf_counter()
            result=functions[name]()
            seconds=time.perf_counter()-start
            activity=cpu_delta(before,cpu_ticks())
            assert digest(result.output)==hashes[name]
            row={'pair':pair,'variant':name,'seconds':seconds,'output_sha256':hashes[name],
                 'cpu_activity':activity,'idle_sibling_verified':eligible(activity)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del result
    valid=[i for i in range(args.pairs) if all(eligible(r['cpu_activity']) for r in rows if r['pair']==i)]
    selected=[r for r in rows if r['pair'] in valid] if args.require_cpu0 else rows
    medians={name:statistics.median(r['seconds'] for r in selected if r['variant']==name) for name in functions} if selected else None
    report={'schema':'nss.paper-p2-timing.v1','rows':rows,'median_seconds':medians,'valid_idle_pairs':len(valid),
            'same_input_repeat_hashes_equal':True,'formal_c4_gate':False,'auto_promoted_policy':None,
            'scope':'Python reference implementations including diagnostic trace allocations; no dictionary learning; not C++ port prediction',
            'input_sha256':sha(args.input),'dictionary_sha256':sha(args.dictionary),'platform':platform.platform(),
            'affinity':affinity,'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in
                ('reference.py','p2_reference.py','p2_benchmark.py')}}
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'median_seconds':medians,'valid_idle_pairs':len(valid)},indent=2))


if __name__=='__main__':
    main()
