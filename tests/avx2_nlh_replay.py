#!/usr/bin/env python3
"""Causal reference crossover for two NLH calls (each call has two internal passes)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from bm_numerics import compare, psnr, ssim

def worker(a):
    import vapoursynth as vs
    from profile_cpu_all import make_source
    core=vs.core;core.num_threads=1
    core.std.LoadPlugin(path=str(a.plugin))
    config=json.loads(a.config.read_text());w,h=config['size'];kw=config['kwargs']
    if kw.get('radius',0)!=0:raise ValueError('This crossover is spatial only')
    algorithm=config.get('algorithm','nlh')
    source=make_source(core,algorithm,3,w,h,42,config.get('sample'))
    fn=getattr(core.nss,'BM3D' if algorithm=='bm3d' else 'NLH')
    if a.reference:
        values=np.load(a.reference)
        def fill(n,f):
            out=f.copy();np.asarray(out[0])[:]=values;return out
        ref=core.std.ModifyFrame(source,source,fill)
        output=fn(source,**{'ref' if algorithm=='bm3d' else 'rclip':ref},**kw)
    else:output=fn(source,**kw)
    np.save(a.out,np.asarray(output.get_frame(1)[0]).copy())

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.screen/'summary.json').read_text())
    failed=[(i,r) for i,r in enumerate(rows) if not r['numerical']['passed'] and (a.index is None or i==a.index)]
    if len(failed)!=1:raise RuntimeError(f'expected one retained failing row, got {len(failed)}')
    index,row=failed[0];config=row['config'];p=a.out/'config.json';p.write_text(json.dumps(config,indent=2))
    def invoke(side,name,ref=None):
        out=a.out/(name+'.npy')
        cmd=['taskset','-c',str(a.cpu),sys.executable,__file__,'worker','--plugin',str(getattr(a,side)),
             '--config',str(p),'--out',str(out)]
        if ref:cmd+=['--reference',str(ref)]
        subprocess.run(cmd,check=True)
        return out
    pilots={s:invoke(s,s+'-pilot') for s in ('baseline','candidate')}
    outputs={s+'_'+r:invoke(s,s+'-using-'+r,pilots[r]) for s in pilots for r in pilots}
    pilot_arrays={s:np.load(p) for s,p in pilots.items()}
    arrays={s:np.load(p) for s,p in outputs.items()}
    pilot=compare(pilot_arrays['baseline'],pilot_arrays['candidate'])
    fixed={r:compare(arrays['baseline_'+r],arrays['candidate_'+r]) for r in pilots}
    propagated={s:compare(arrays[s+'_baseline'],arrays[s+'_candidate']) for s in pilots}
    reproduce={s:bool(np.array_equal(arrays[s+'_'+s],np.load(a.screen/f'{index}-{s}.npy')[0,0])) for s in pilots}
    w,h=config['size']
    if config.get('sample'):clean=np.fromfile(config['sample'],dtype=np.uint8).reshape(h,w).astype(np.float32)*np.float32(1/255)
    else:
        yy,xx=np.mgrid[0:h,0:w];clean=(.35+.2*np.sin(xx/37)*np.cos(yy/29)+.05*(xx+yy)/(w+h)).astype(np.float32)
    quality={s:dict(psnr=psnr(arrays[s+'_'+s],clean),ssim=ssim(arrays[s+'_'+s],clean)) for s in pilots}
    loss={m:quality['baseline'][m]-quality['candidate'][m] for m in ('psnr','ssim')}
    pilot_approved=False
    if not pilot['passed'] and config.get('algorithm')=='bm3d' and a.baseline_probe and a.candidate_probe:
        review_dir=a.out/'pilot-stage-replay'
        cmd=[sys.executable,str(Path(__file__).with_name('avx2_dct_replay.py')),'--baseline',str(a.baseline_probe),
            '--candidate',str(a.candidate_probe),'--out',str(review_dir),'--case-config',str(p),
            '--baseline-output',str(pilots['baseline']),'--candidate-output',str(pilots['candidate']),'--cpu',str(a.cpu)]
        subprocess.run(cmd,stdout=subprocess.DEVNULL,check=False)
        rp=review_dir/'replay.json'
        if rp.exists():
            review=json.loads(rp.read_text())
            pilot_approved=review['passed'] and all(review['output_hashes'][side]==hashlib.sha256(pilot_arrays[side].tobytes()).hexdigest() for side in pilots)
    passed=all(reproduce.values()) and (pilot['passed'] or pilot_approved) and all(r['passed'] for r in fixed.values()) and loss['psnr']<=.05 and loss['ssim']<=.001
    result=dict(passed=passed,classification='first_call_reference_rounding_amplification' if passed else 'unresolved',
        original_row=index,original=row['numerical'],reproduced_original=reproduce,pilot=pilot,pilot_stage_review_passed=pilot_approved,
        fixed_reference_second_call=fixed,reference_perturbation=propagated,quality=quality,loss=loss,
        scope='Crossover isolates propagation from pilot perturbation. Independent scalar group oracle is a separate CTest requirement.',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.baseline,a.candidate,Path(__file__),*pilots.values(),*outputs.values()]})
    (a.out/'replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    w=sub.add_parser('worker');w.add_argument('--plugin',type=Path,required=True);w.add_argument('--config',type=Path,required=True);w.add_argument('--out',type=Path,required=True);w.add_argument('--reference',type=Path)
    r=sub.add_parser('run')
    for name in ('baseline','candidate','screen','out'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--cpu',type=int,default=0);r.add_argument('--index',type=int)
    r.add_argument('--baseline-probe',type=Path);r.add_argument('--candidate-probe',type=Path)
    a=p.parse_args();sys.exit(worker(a) if a.command=='worker' else run(a))
