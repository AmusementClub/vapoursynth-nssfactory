#!/usr/bin/env python3
"""Pinned BM3DCPU comparison; calibration, quality and performance are separate."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import vapoursynth as vs
from bm_numerics import compare,psnr,ssim

def arrclip(core,values):
    n,h,w=values.shape;b=core.std.BlankClip(width=w,height=h,format=vs.GRAYS,length=n)
    def fill(n,f):
        out=f.copy();np.asarray(out[0])[:]=values[n];return out
    return core.std.ModifyFrame(b,b,fill)
def output(core,label,src,sigma,r,stage,reference_range=7):
    api=getattr(core,label)
    common=dict(sigma=sigma,block_step=8,bm_range=7,radius=r,ps_num=2,ps_range=4)
    def bm(src,ref=None):
        kw=dict(common)
        if ref is not None:kw['ref']=ref
        if label=='bm3dcpu':
            kw['bm_range']=reference_range
            if sigma == 0:
                return src
            raw = api.BM3D(src, chroma=False, **kw)
            # The pinned BM3Dv2 wrapper appends three planes even for GRAYS.
            return api.VAggregate(raw, src, planes=[0]) if r else raw
        raw=api.BM3D(src,block_size=8,group_size=8,**kw)
        return api.VAggregate(raw,src,radius=r) if r else raw
    if stage=='basic':return bm(src)
    if stage=='wiener':return bm(src,src)
    return bm(src,bm(src))
def shifted(base,delta):
    return base[:,np.clip(np.arange(base.shape[1])-delta,0,base.shape[1]-1)]
def sequence(base,kind):
    out=[]
    for t in range(5):
        a=shifted(base,2*(t-2)).copy() if kind=='translation' else base.copy()
        if kind=='brightness':a+=.03*(t-2)
        if kind=='occlusion' and t>=2:a[30:85,50:120]=.7
        if kind=='cut' and t>=2:a=1-a
        out.append(a)
    return np.stack(out).astype(np.float32)
def trms(out,clean,kind):
    e=out.astype(np.float64)-clean
    if len(e)<2:return None
    if kind=='translation':return float(np.sqrt(np.mean([(e[t+1,:,8:-8]-e[t,:,6:-10])**2 for t in range(len(e)-1)])))
    return float(np.sqrt(np.mean(np.diff(e,axis=0)**2)))
def run(a):
    core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=a.nss);core.std.LoadPlugin(path=a.reference)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifests={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.nss,a.reference,__file__,str(Path(__file__).with_name('bm_numerics.py'))]}
    (out/'manifest.json').write_text(json.dumps(dict(files=manifests,reference_commit='e869cfae8d2322cf7a2b3c8056ed57e9308b2d33',chroma=False,crop=a.crop,crop_origin=[800,450] if a.crop else None,bm_range=7,reference_bm_range=a.reference_range,step=8,block=8,group=8,temporal_boundary='nss truncate versus reference clamp'),indent=2))
    reports=[]
    for image_id,sample in enumerate(sorted(Path(a.samples).glob('*.gray8'))):
        if a.sample and sample.name!=a.sample:continue
        base=np.fromfile(sample,np.uint8).reshape(1080,1920).astype(np.float32)/255
        if a.crop:
            crop_width, crop_height = map(int, a.crop.split(','))
            if not (1 <= crop_width <= 1120 and 1 <= crop_height <= 630):
                raise ValueError('crop must fit sample starting at x=800,y=450')
            base=base[450:450+crop_height,800:800+crop_width]
        for kind in (['static'] if a.spatial_only else a.kinds.split(',')):
            clean=sequence(base if kind=='static' else base[::6,::6],kind)
            radius=0 if kind=='static' else 1
            for sig in map(float,a.sigmas.split(',')):
                rng=np.random.default_rng(42+image_id)
                noisy=(clean+rng.standard_normal(clean.shape).astype(np.float32)*np.float32(sig/255)).astype(np.float32)
                src=arrclip(core,noisy)
                input_hashes=dict(clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                                  noisy_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),noise_seed=42+image_id)
                for stage in ['basic','wiener','two_stage']:
                    results={};row=dict(sample=sample.name,input_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),kind=kind,size=list(clean.shape[1:]),radius=radius,sigma=sig,stage=stage,**input_hashes)
                    for label in ['nss','bm3dcpu']:
                        node=output(core,label,src,sig,radius,stage,a.reference_range)
                        # Quality over the entire sequence, including explicitly different boundaries.
                        start=time.perf_counter();frames=[np.array(node.get_frame(n)[0]) for n in range(len(clean))];elapsed=time.perf_counter()-start
                        values=np.stack(frames);results[label]=values
                        if not np.isfinite(values).all():raise RuntimeError(f'nonfinite {label} {row}')
                        e=values.astype(np.float64)-clean
                        row[label]=dict(psnr_clean=psnr(values,clean),ssim_clean=ssim(values,clean),residual_std=float(e.std()),bias=float(e.mean()),motion_compensated_error_rms=trms(values,clean,kind),cold_sequence_ms=elapsed*1000)
                    row['difference']=compare(results['nss'],results['bm3dcpu'])
                    # Diagnostic only: external differences do not use the internal triage gate.
                    row['difference'].pop('passed',None)
                    reports.append(row)
                    (out/'quality.json').write_text(json.dumps(reports,indent=2))
            print(f'{sample.name} {kind} completed',flush=True)
    (out/'complete.json').write_text(json.dumps(dict(completed=True,cases=len(reports),natural_video_verified=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nss',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--samples',default='/opt/nss-c4/samples/gray8');p.add_argument('--spatial-only',action='store_true');p.add_argument('--crop');p.add_argument('--sample');p.add_argument('--reference-range',type=int,default=7);p.add_argument('--sigmas',default='0,.5,1,3,5,10,20,40');p.add_argument('--kinds',default='static,translation,brightness,occlusion,cut');run(p.parse_args())
