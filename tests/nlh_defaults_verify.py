#!/usr/bin/env python3
"""Compare public NLH stage controls with an independent NumPy implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from alignment_reference_nlh_fast import denoise
from nlh_defaults_inputs import file_sha,save_json


def cases():
    rng=np.random.default_rng(20260909)
    for block in range(2,17):
        other=18-block;channels=3 if block%3==0 else 1
        image=rng.integers(64,448,(channels,24,25)).astype(np.float32)/1024
        q=(2,4,8,16)[block%4]
        q=min(q,1 << ((block*block).bit_length()-1))
        params=dict(block_size=[block,other],block_step=[max(1,block//2),other],
                    group_size=[(2,4,8,16,32,64)[block%6],(2,8,32)[block%3]],
                    q=[q,2],search_window=[20,3 if block%2 else 40],basic_iters=1,
                    lambda_basic=.4,wiener_iters=1+block%3,hard_strength=.75,wiener_sigma_scale=.16)
        yield f'block-{block}',image,15.,params
    yield 'ties',np.full((1,24,25),.375,np.float32),25.,dict(
        block_size=[6,9],block_step=[4,7],q=[2,8],group_size=[8,32],search_window=[40,40],
        basic_iters=1,wiener_iters=2,lambda_basic=.6,hard_strength=1.,wiener_sigma_scale=.08)
    yield 'zero',rng.random((3,24,25),dtype=np.float32),0.,dict(
        block_size=[16,2],block_step=[1,1],q=[16,2],group_size=[64,2],search_window=[40,40],
        basic_iters=2,wiener_iters=2,lambda_basic=.6,hard_strength=1.,wiener_sigma_scale=.08)


def main():
    import vapoursynth as vs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--exact-baseline');parser.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{args.cpu})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    rows=[];saved={}
    previous=np.load(args.exact_baseline) if args.exact_baseline else None
    for name,image,sigma,parameters in cases():
        c,h,w=image.shape
        blank=core.std.BlankClip(width=w,height=h,length=1,format=vs.RGBS if c==3 else vs.GRAYS)
        def fill(n,f):
            result=f.copy()
            for p in range(c):np.asarray(result[p])[:]=image[p]
            return result
        source=core.std.ModifyFrame(blank,blank,fill)
        node=core.nss.NLH(source,sigma=sigma,**parameters)
        frame=node.get_frame(0);pixels=np.stack([np.asarray(frame[p]).copy() for p in range(c)])
        started=time.perf_counter();reference,logs=denoise(image,sigma,parameters=parameters)
        delta=np.abs(reference.astype(float)-pixels.astype(float))
        equal=np.array_equal(pixels,previous[name]) if previous is not None else None
        row=dict(case=name,parameters=parameters,sigma=sigma,shape=list(image.shape),
                 max_abs=float(delta.max()),rms=float(np.sqrt(np.mean(delta*delta))),
                 reference_seconds=time.perf_counter()-started,exact_baseline=equal,
                 passed=bool(np.allclose(pixels,reference,atol=2e-5,rtol=2e-4)) and equal is not False,
                 output_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),reference_stages=logs)
        rows.append(row);saved[name]=pixels
        print(name,row['max_abs'],row['passed'],flush=True)
        save_json(out/'verification.json',dict(passed=all(r['passed'] for r in rows),rows=rows,
                  plugin_sha256=file_sha(args.plugin),script_sha256=file_sha(__file__),
                  reference_sha256=file_sha(Path(__file__).with_name('alignment_reference_nlh_fast.py'))))
        del frame,node,source
    np.savez_compressed(out/'outputs.npz',**saved)
    if not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':
    main()
