#!/usr/bin/env python3
"""Saved DIV2K/other fixture evaluation for the TWSC/NLH replacement.

Native defaults run at the requested original fixture size. --crop is explicit,
preserves saved noise samples, and is recorded independently of full-image runs.
The independent NumPy/SciPy implementation can run the same selected crop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from alignment_campaign import metrics
from alignment_reference import spatial_reference


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root=Path(args.fixtures).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((root/'fixtures.json').read_text())['cases']
    cases=[c for c in cases if c['sigma']==args.sigma and (not args.case or c['id']==args.case)]
    if not cases: raise ValueError('no selected fixtures')
    core=None
    if args.plugin:
        import vapoursynth as vs
        core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    rows=[]
    for case in cases:
        for key in ('clean','noisy'):
            if sha(root/case[key])!=case[key+'_sha256']: raise ValueError('fixture identity mismatch')
        shape=(case.get('channels',1),case['height'],case['width'])
        clean=np.fromfile(root/case['clean'],dtype='<f4').reshape(shape)
        noisy=np.fromfile(root/case['noisy'],dtype='<f4').reshape(shape)
        crop=None
        if args.crop:
            h,w=shape[1:];size=args.crop
            if size>min(h,w): raise ValueError('crop exceeds input')
            x,y=(w-size)//2,(h-size)//2;crop=[x,y,size,size]
            clean=clean[:,y:y+size,x:x+size].copy();noisy=noisy[:,y:y+size,x:x+size].copy()
        for model in args.models:
            label=case['id']+'-'+model
            np.save(out/(label+'-input.npy'),noisy);np.save(out/(label+'-clean.npy'),clean)
            print(label,'START',flush=True)
            if core is not None:
                c,h,w=noisy.shape
                blank=core.std.BlankClip(width=w,height=h,length=1,format=vs.RGBS if c==3 else vs.GRAYS)
                def fill(n,f):
                    copy=f.copy()
                    for p in range(c): np.copyto(np.asarray(copy[p]),noisy[p])
                    return copy
                source=core.std.ModifyFrame(blank,blank,fill)
                held=source.get_frame(0)
                node=getattr(core.nss,model)(source,sigma=case['sigma'])
                started=time.perf_counter();frame=node.get_frame(0);elapsed=time.perf_counter()-started
                pixels=np.array([np.asarray(frame[p]) for p in range(c)])
                props={k:v for k,v in dict(frame.props).items() if k.startswith('_NSS') and isinstance(v,(int,float,list))}
            else:
                started=time.perf_counter();pixels=spatial_reference(noisy,case['sigma'],model);elapsed=time.perf_counter()-started;props={}
            if not np.isfinite(pixels).all(): raise ValueError('nonfinite output')
            output=out/(label+'.npy');np.save(output,pixels)
            row=dict(case=case,model=model,crop=crop,shape=list(noisy.shape),seconds=elapsed,
                     timing_scope='single cold filter request, source preloaded; quality evidence, not paired performance',
                     supplied_parameters=dict(sigma=case['sigma']),resolved=props,output=output.name,
                     output_sha256=sha(output),input_sha256=sha(out/(label+'-input.npy')),
                     quality=metrics(clean,pixels),noisy_quality=metrics(clean,noisy),
                     implementation='plugin' if core else 'independent_numpy_scipy',
                     plugin_sha256=sha(args.plugin) if core else None)
            rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(label,'DONE',elapsed,flush=True)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,passed=True,fixture_manifest_sha256=sha(root/'fixtures.json'),script_sha256=sha(__file__)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True)
    p.add_argument('--plugin');p.add_argument('--crop',type=int);p.add_argument('--sigma',type=float,default=25)
    p.add_argument('--case');p.add_argument('--models',nargs='+',default=['TWSC','NLH']);main(p.parse_args())
