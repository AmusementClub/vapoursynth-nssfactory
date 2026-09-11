#!/usr/bin/env python3
"""Calibrate global linear-NLH coefficients against default BM3D noise response.

Sigma always equals the injected noise standard deviation. Each case uses two
independent noise realizations. Search timing is diagnostic, not a benchmark.
No per-image sigma tuning, output blending, or zero-sigma bypass is permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from defaults_compare import plane_quality


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pair(args, case_id):
    roots=[Path(args.fixtures).resolve(),Path(args.probe).resolve()]
    cases=[next(c for c in json.loads((p/'fixtures.json').read_text())['cases'] if c['id']==case_id) for p in roots]
    a,b=cases
    for key in ('width','height','channels','sigma','clean_sha256'):
        if a[key]!=b[key]:raise ValueError('inconsistent paired fixture '+key)
    shape=(a['channels'],a['height'],a['width'])
    for root,case in zip(roots,cases):
        for key in ('clean','noisy'):
            if sha(root/case[key])!=case[key+'_sha256']:raise ValueError('fixture hash mismatch')
    clean=np.fromfile(roots[0]/a['clean'],dtype='<f4').reshape(shape)
    noisy=np.array([np.fromfile(root/case['noisy'],dtype='<f4').reshape(shape) for root,case in zip(roots,cases)])
    crop=None
    if args.crop:
        size=args.crop;y=(shape[1]-size)//2;x=(shape[2]-size)//2
        if size>min(shape[1:]):raise ValueError('crop exceeds fixture')
        crop=[x,y,size,size];clean=clean[:,y:y+size,x:x+size].copy();noisy=noisy[:,:,y:y+size,x:x+size].copy()
    if np.array_equal(noisy[0],noisy[1]):raise ValueError('noise pair must be independent')
    return cases,clean,noisy,crop


def worker(args):
    os.sched_setaffinity(0,{args.cpu})
    import vapoursynth as vs
    cases,clean,noisy,crop=read_pair(args,args.case)
    c,h,w=clean.shape
    if args.gray:
        weights=np.array([.299,.587,.114],float)
        original_channel=clean[0].astype(float)
        clean=np.einsum('c,chw->hw',weights,clean.astype(float))[None].astype(np.float32)
        # Gray fixtures retain the supplied sigma: independent channel 0 noise
        # is added to the clean luma, rather than silently reducing its sigma.
        noise=noisy[:,0].astype(float)-original_channel
        noisy=(clean[None].astype(float)+noise[:,None]).astype(np.float32);c=1
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'inputs.npz',clean=clean,noisy=noisy)
    sigma=cases[0]['sigma'] if args.dial_sigma is None else args.dial_sigma
    input_energy=float(np.sum((noisy[0].astype(float)-noisy[1])**2))
    core=vs.core;core.num_threads=1;core.max_cache_size=64
    plugin=Path(args.plugin).resolve();core.std.LoadPlugin(path=str(plugin))
    blank=core.std.BlankClip(width=w,height=h,length=2,format=vs.RGBS if c==3 else vs.GRAYS)
    fills=[0]
    def fill(n,f):
        fills[0]+=1;result=f.copy()
        for p in range(c):np.copyto(np.asarray(result[p]),noisy[n,p])
        return result
    source=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(source,mode=1,fixedsize=2,maxsize=2)
    held=[source.get_frame(i) for i in range(2)]
    grid=[('BM3D',None,None)]
    if args.frozen:grid.append(('NLH-v3',None,None))
    elif args.defaults:grid.append(('NLH',args.base_coefficient,args.cw[0]))
    else:grid.extend(('NLH',kh,cw) for kh in args.kh for cw in args.cw)
    rows=[]
    for model,kh,cw in grid:
        parameters=dict(sigma=sigma)
        if model=='NLH' and not args.defaults:parameters.update(hard_strength=kh/args.base_coefficient,wiener_sigma_scale=cw)
        node=getattr(core.nss,'NLH' if model=='NLH-v3' else model)(source,**parameters)
        before=fills[0];started=time.perf_counter();pixels=[];properties=[]
        for n in range(2):
            with node.get_frame(n) as frame:
                pixels.append(np.array([np.asarray(frame[p]) for p in range(c)]))
                properties.append({k:v for k,v in dict(frame.props).items() if k.startswith('_NSS') and isinstance(v,(int,float,list))})
        seconds=time.perf_counter()-started;pixels=np.array(pixels)
        if not np.isfinite(pixels).all() or fills[0]!=before:raise AssertionError('nonfinite output or timed source fill')
        if model=='NLH':
            if properties[0]['_NSSModelVersion']!=4 or not math.isclose(properties[0]['_NSSHardCoefficient'],kh,rel_tol=1e-14,abs_tol=1e-14) or properties[0]['_NSSWienerSigmaScale']!=cw:
                raise AssertionError('wrong model version or coefficient mapping')
        elif model=='NLH-v3' and properties[0]['_NSSModelVersion']!=3:raise AssertionError('wrong frozen model version')
        response=math.sqrt(float(np.sum((pixels[0].astype(float)-pixels[1])**2))/input_energy)
        label=model if kh is None else f'NLH-kh{kh:g}-cw{cw:g}'
        file=out/(label+'.npz');np.savez_compressed(file,pixels=pixels)
        quality=[plane_quality(clean,p) for p in pixels]
        row=dict(model=model,kh=kh,cw=cw,parameters=parameters,response=response,quality=quality,
                 mean_psnr=float(np.mean([q['psnr_db'] for q in quality])),mean_ssim=float(np.mean([q['ssim'] for q in quality])),
                 seconds=seconds,timing_scope='two source-preloaded requests; concurrent search timing, not performance admission',
                 timed_source_fills=fills[0]-before,output=file.name,output_sha256=sha(file),
                 pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),resolved=properties)
        rows.append(row)
        (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(args.case,label,response,row['mean_psnr'],flush=True)
        del node
    target=rows[0]['response']
    if target<=0:raise AssertionError('invalid BM3D response target')
    for row in rows[1:]:
        row['relative_response_error']=row['response']/target-1
        row['strength_matched_5pct']=abs(row['relative_response_error'])<=.05
    report=dict(passed=True,case=cases[0]['id'],sigma=sigma,injected_sigma=cases[0]['sigma'],dial_diagnostic=args.dial_sigma is not None,gray=args.gray,shape=list(clean.shape),crop=crop,rows=rows,
                source_cases=cases,plugin_sha256=sha(plugin),script_sha256=sha(__file__),input_pair_sha256=sha(out/'inputs.npz'),
                affinity=sorted(os.sched_getaffinity(0)),reference='BM3D, same requested sigma, other defaults, single Basic call')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


def campaign(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[c['id'] for c in json.loads((Path(args.fixtures)/'fixtures.json').read_text())['cases'] if c['sigma'] in args.sigmas and (not args.cases or c['id'] in args.cases)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def lane(cpu,subset):
        for case in subset:
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--plugin',str(Path(args.plugin).resolve()),
                     '--fixtures',str(Path(args.fixtures).resolve()),'--probe',str(Path(args.probe).resolve()),
                     '--case',case,'--out',str(out/case),'--cpu',str(cpu),'--base-coefficient',str(args.base_coefficient),
                     '--kh',*map(str,args.kh),'--cw',*map(str,args.cw)]
            if args.crop:command+=['--crop',str(args.crop)]
            if args.gray:command+=['--gray']
            if args.frozen:command+=['--frozen']
            if args.defaults:command+=['--defaults']
            if args.dial_sigma is not None:command+=['--dial-sigma',str(args.dial_sigma)]
            run=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (out/(case+'.log')).write_bytes(run.stdout)
            if run.returncode:raise RuntimeError(f'{case}: {run.stdout[-1500:].decode()}')
            print(case,'complete',flush=True)
    with ThreadPoolExecutor(len(args.cpus)) as pool:
        futures=[pool.submit(lane,cpu,cases[i::len(args.cpus)]) for i,cpu in enumerate(args.cpus)]
        for future in futures:future.result()
    reports=[json.loads((out/case/'summary.json').read_text()) for case in cases]
    candidates=[]
    if not args.frozen:
        for kh in args.kh:
            for cw in args.cw:
                selected=[next(r for r in d['rows'] if r['model']=='NLH' and r['kh']==kh and r['cw']==cw) for d in reports]
                log_ratio=[math.log(r['response']/d['rows'][0]['response']) for d,r in zip(reports,selected)]
                errors=[abs(r['relative_response_error']) for r in selected]
                candidates.append(dict(kh=kh,cw=cw,diagnostic_only=kh==0,
                    rms_log_response_error=math.sqrt(float(np.mean(np.square(log_ratio)))),
                    max_relative_response_error=max(errors),mean_relative_response_error=float(np.mean(errors)),
                    matched_cases=sum(r['strength_matched_5pct'] for r in selected),cases=len(selected),
                    mean_psnr=float(np.mean([r['mean_psnr'] for r in selected])),mean_ssim=float(np.mean([r['mean_ssim'] for r in selected]))))
        candidates.sort(key=lambda c:(c['rms_log_response_error'],c['max_relative_response_error'],-c['mean_psnr']))
    report=dict(passed=True,scope='Global coefficient screening against BM3D paired-noise response; development images, not general calibration proof',
                selection_rule='Minimize equally weighted RMS log response ratio; then worst relative error; quality only breaks ties. kh=0 is diagnostic only.',
                cases=reports,candidates=candidates,selected=next((c for c in candidates if not c['diagnostic_only']),None))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['selected']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('run','worker'):
        p=sub.add_parser(mode)
        for name in ('plugin','fixtures','probe','out'):p.add_argument('--'+name,required=True)
        p.add_argument('--crop',type=int);p.add_argument('--gray',action='store_true');p.add_argument('--frozen',action='store_true')
        p.add_argument('--defaults',action='store_true',help='Verify final public defaults without advanced parameters')
        p.add_argument('--dial-sigma',type=float,help='Fixed-input knob diagnostic only; never used to select calibration coefficients')
        p.add_argument('--base-coefficient',type=float,default=2.025)
        p.add_argument('--kh',type=float,nargs='+',default=[0,1,1.5,2.025,2.7,3.5,5])
        p.add_argument('--cw',type=float,nargs='+',default=[.5])
        if mode=='run':p.add_argument('--sigmas',type=float,nargs='+',default=[5,15,25,50]);p.add_argument('--cases',nargs='+');p.add_argument('--cpus',type=int,nargs='+',default=[0,2,4,6])
        else:p.add_argument('--case',required=True);p.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args()
    if args.defaults:
        if len(args.cw)!=1:parser.error('--defaults requires one expected --cw')
        args.kh=[args.base_coefficient]
    worker(args) if args.mode=='worker' else campaign(args)
