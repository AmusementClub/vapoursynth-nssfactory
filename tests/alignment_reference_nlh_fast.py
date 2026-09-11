#!/usr/bin/env python3
"""Independent vectorized full-image NLH reference (Gray/RGB spatial).

Explicit Haar matrices, FP64 row distances and complete q*m scatter aggregation.
No production primitive is used. Box-window matching is shared only with the
independent NumPy reference; --verify-small checks the readable reference too.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from alignment_reference_fast import patch_matrix,strip_matches
from alignment_reference import spatial_reference
from test_alignment_math import haar, NLH_HARD_COEFFICIENT, NLH_WIENER_SIGMA_SCALE
from alignment_campaign import metrics,sha


def stage_parameters(parameters, rgb, sigma):
    """Resolve public spatial controls without importing production resolvers."""
    p=dict(parameters or {})
    allowed={'noise_model','block_size','block_step','group_size','q','search_window',
             'basic_iters','lambda_basic','hard_strength','wiener_iters','wiener_sigma_scale'}
    if set(p)-allowed: raise ValueError('unsupported reference controls: '+str(sorted(set(p)-allowed)))
    mode=p.get('noise_model','auto')
    if mode not in ('auto','awgn','real'): raise ValueError('invalid noise_model')
    real=mode=='real' or (mode=='auto' and rgb)
    default_block=7 if real else 8 if sigma<=50 else 10
    defaults={'block_size':default_block,'block_step':1,'group_size':16,'q':4,'search_window':40}
    stages=[{},{}]
    for key,default in defaults.items():
        values=p.get(key,default)
        values=list(values) if isinstance(values,(list,tuple)) else [values]
        if len(values)==1: values*=2
        if len(values)!=2 or any(isinstance(v,bool) or int(v)!=v for v in values):
            raise ValueError('expected one or two integers for '+key)
        for s,value in zip(stages,values): s[key]=int(value)
    for s in stages:
        b,q,g,step,window=(s[k] for k in ('block_size','q','group_size','block_step','search_window'))
        if not (2<=b<=16 and q in (2,4,8,16) and q<=b*b and
                g in (2,4,8,16,32,64) and 1<=step<=b and 1<=window<=129):
            raise ValueError('invalid stage shape')
    rounds=p.get('basic_iters',2 if real else 4 if sigma<=50 else 5)
    gains=p.get('wiener_iters',2)
    if any(isinstance(v,bool) or int(v)!=v or not 1<=v<=64 for v in (rounds,gains)):
        raise ValueError('invalid iteration count')
    mix=float(p.get('lambda_basic',.6));strength=float(p.get('hard_strength',1))
    scale=float(p.get('wiener_sigma_scale',NLH_WIENER_SIGMA_SCALE))
    if not (np.isfinite([mix,strength,scale]).all() and 0<=mix<=1 and strength>=0 and scale>=0):
        raise ValueError('invalid filtering coefficients')
    return stages,int(rounds),int(gains),mix,strength,scale


def query_indices(guide,b,requested,window,y0,rows,xs,ys):
    """Include the last raster query and handle small, nonuniform edge groups."""
    _,h,w=guide.shape;ny,nx=h-b+1,w-b+1
    n=1 << (min(requested,nx*ny).bit_length()-1)
    side=min(window//2+1,window-window//2)
    if min(nx,side)*min(ny,side)>=n:
        matched=strip_matches(guide,b,n,window,y0,rows).reshape(rows,nx,n)
        return [matched[y-y0,x] for y in ys for x in xs]
    patches=patch_matrix(guide,b).astype(float)
    lo=window//2;hi=window-lo-1;result=[]
    for y in ys:
        for x in xs:
            candidates=np.array([yy*nx+xx for yy in range(max(0,y-lo),min(ny,y+hi+1))
                                 for xx in range(max(0,x-lo),min(nx,x+hi+1))])
            difference=patches[candidates]-patches[y*nx+x]
            scores=np.sum(difference*difference,axis=1).astype(np.float32)
            scores[candidates==y*nx+x]=-1
            count=1 << (min(requested,len(candidates)).bit_length()-1)
            result.append(candidates[np.argsort(scores,kind='stable')[:count]].astype(np.int32))
    return result


def denoise(image,sigma,strip_rows=8,progress=None,parameters=None):
    if not np.isfinite(sigma) or sigma<0: raise ValueError('finite nonnegative explicit sigma required')
    image=np.asarray(image,np.float32);c,h,w=image.shape;rgb=c==3
    if c not in (1,3) or not np.isfinite(image).all(): raise ValueError('finite Gray/RGB input required')
    stages,rounds,gains,mix,strength,scale=stage_parameters(parameters,rgb,sigma)
    if sigma==0: return image.copy(),[]
    transform=np.array([[.299,.587,.114],[-.168736607142857,-.331263392857143,.5],[.5,-.4186875,-.0813125]])
    source=(transform@image.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32) if rgb else image.copy()
    noise=np.full(c,np.float32(sigma)/np.float32(255),np.float32)
    if rgb: noise=np.sqrt((transform*transform)@(noise.astype(float)**2)).astype(np.float32)
    basic=source.copy();logs=[]
    for iteration in range(rounds+1):
        started=time.perf_counter();wiener=iteration==rounds
        s=stages[int(wiener)]
        b,q,requested,step,window=(s[k] for k in ('block_size','q','group_size','block_step','search_window'))
        ny,nx=h-b+1,w-b+1;m=b*b
        if min(nx,ny)<1: raise ValueError('image smaller than block')
        xs=sorted(set(range(0,nx,step))|{nx-1});all_ys=set(range(0,ny,step))|{ny-1}
        hq=haar(q)
        data=source if wiener else (mix*basic.astype(float)+(1-mix)*source.astype(float)).astype(np.float32)
        guide=basic[:1] if wiener else data[:1]
        p=patch_matrix(data,b).reshape(nx*ny,c,m)
        gp=patch_matrix(guide,b)
        bp=patch_matrix(basic,b).reshape(nx*ny,c,m) if wiener else None
        patch_num=np.zeros((nx*ny,c*m),float);patch_den=np.zeros((nx*ny,m),float)
        block_hash=hashlib.sha256();pixel_hash=hashlib.sha256()
        group_counts={}
        for y0 in range(0,ny,strip_rows):
            rows=min(strip_rows,ny-y0);ys=sorted(all_ys.intersection(range(y0,y0+rows)))
            if not ys: continue
            indices=query_indices(guide,b,requested,window,y0,rows,xs,ys)
            for selected in indices: block_hash.update(selected.tobytes())
            batches=[]
            for n in sorted({len(v) for v in indices}):
                same=np.array([v for v in indices if len(v)==n],np.int32)
                group_counts[n]=group_counts.get(n,0)+len(same)
                batch_limit=max(1,min(16,32_000_000//(m*m*n*8)))
                batches.extend(same[start:start+batch_limit] for start in range(0,len(same),batch_limit))
            for selected in batches:
                batch,n=selected.shape;hn=haar(n)
                packed=gp[selected].transpose(0,2,1).astype(float)
                distance=np.sum((packed[:,:,None,:]-packed[:,None,:,:])**2,axis=3)
                rr=np.arange(m);distance[:,rr,rr]=-1
                neighbors=np.argsort(distance,axis=2,kind='stable')[:,:,:q]
                pixel_hash.update(neighbors.astype(np.int32).tobytes())
                bi=np.arange(batch)[:,None,None,None];ci=np.arange(c)[None,:,None,None]
                data_group=p[selected].transpose(0,2,3,1).astype(float)
                matrices=data_group[bi,ci,neighbors[:,None,:,:],:]
                coefficients=hq@matrices@hn.T
                if wiener:
                    reference=bp[selected].transpose(0,2,3,1).astype(float)[bi,ci,neighbors[:,None,:,:],:]
                    r=hq@reference@hn.T;r2=r*r
                    variance=((scale*noise.astype(float))**2)[None,:,None,None,None]
                    gain=np.ones_like(r2) if scale==0 else r2/(r2+variance)
                    for _ in range(gains): coefficients*=gain
                else:
                    threshold=(NLH_HARD_COEFFICIENT*strength*noise.astype(float))[None,:,None,None,None]
                    coefficients[np.abs(coefficients)<threshold]=0
                    coefficients[:,:,:,q-2:,1:]=0
                restored=hq.T@coefficients@hn
                local_num=np.zeros((batch,c,m,n),float);local_den=np.zeros((batch,m),float)
                for k in range(q):
                    np.add.at(local_num,(np.arange(batch)[:,None,None],np.arange(c)[None,:,None],neighbors[:,None,:,k]),restored[:,:,:,k,:])
                    np.add.at(local_den,(np.arange(batch)[:,None],neighbors[:,:,k]),1)
                values=local_num.transpose(0,3,1,2).reshape(batch,n,c*m)
                np.add.at(patch_num,selected,values)
                np.add.at(patch_den,selected,np.broadcast_to(local_den[:,None,:],(batch,n,m)))
            if progress: progress(iteration,y0,ny,time.perf_counter()-started)
        num=np.zeros_like(source,dtype=float);den=np.zeros((h,w),float)
        pn=patch_num.reshape(ny,nx,c,b,b);pd=patch_den.reshape(ny,nx,b,b)
        for dy in range(b):
            for dx in range(b):
                for channel in range(c): num[channel,dy:dy+ny,dx:dx+nx]+=pn[:,:,channel,dy,dx]
                den[dy:dy+ny,dx:dx+nx]+=pd[:,:,dy,dx]
        basic=np.divide(num,den,out=source.astype(float).copy(),where=den>0).astype(np.float32)
        logs.append(dict(iteration=iteration+1,stage='Wiener' if wiener else 'Basic',seconds=time.perf_counter()-started,
                         parameters=s,actual_groups=group_counts,
                         block_matching_sha256=block_hash.hexdigest(),pixel_matching_sha256=pixel_hash.hexdigest()))
    num=num.astype(np.float32);den=den.astype(np.float32)
    if rgb:
        inverse=np.array([[1,0,1.402],[1,-.114*1.772/.587,-.299*1.402/.587],[1,1.772,0]])
        num=(inverse@num.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32)
    result=(num.astype(float)/den).astype(np.float32)
    if not np.isfinite(result).all(): raise ArithmeticError('nonfinite reference output')
    return result,logs


def main(args):
    root=Path(args.fixtures).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('alignment_reference_fast.py'),Path(__file__).with_name('test_alignment_math.py')]
    identities={str(p.resolve()):sha(p) for p in files}
    cases=[c for c in json.loads((root/'fixtures.json').read_text())['cases'] if c['sigma']==args.sigma and (not args.case or c['id']==args.case)]
    if not cases: raise ValueError('no selected fixtures')
    rows=[]
    for case in cases:
        shape=(case.get('channels',1),case['height'],case['width'])
        if sha(root/case['noisy'])!=case['noisy_sha256'] or sha(root/case['clean'])!=case['clean_sha256']: raise ValueError('fixture identity mismatch')
        noisy=np.fromfile(root/case['noisy'],dtype='<f4').reshape(shape);clean=np.fromfile(root/case['clean'],dtype='<f4').reshape(shape);crop=None
        if args.crop:
            size=args.crop;x=(shape[2]-size)//2;y=(shape[1]-size)//2;crop=[x,y,size,size]
            noisy=noisy[:,y:y+size,x:x+size].copy();clean=clean[:,y:y+size,x:x+size].copy()
        def progress(i,y,ny,seconds):
            (out/(case['id']+'-progress.json')).write_text(json.dumps(dict(iteration=i+1,completed_query_rows=y+min(8,ny-y),query_rows=ny,seconds=seconds))+'\n')
        parameters=json.loads(args.parameters_json)
        started=time.perf_counter();pixels,iterations=denoise(noisy,args.sigma,progress=progress,parameters=parameters);elapsed=time.perf_counter()-started
        output=out/(case['id']+'-NLH.npy');np.save(output,pixels)
        row=dict(case=case,model='NLH',crop=crop,shape=list(noisy.shape),seconds=elapsed,iterations=iterations,
                 output=output.name,output_sha256=sha(output),quality=metrics(clean,pixels),implementation='independent_numpy_haar')
        if args.verify_small:
            if parameters: raise ValueError('--verify-small readable oracle supports defaults only')
            if args.crop is None or args.crop>24: raise ValueError('--verify-small requires --crop <=24')
            exact=spatial_reference(noisy,args.sigma,'NLH');row['readable_reference_difference']=metrics(exact,pixels)
            np.testing.assert_allclose(pixels,exact,atol=2e-5,rtol=2e-4)
        rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],'DONE',elapsed,flush=True)
    if any(sha(p)!=value for p,value in identities.items()): raise RuntimeError('reference source changed during execution')
    (out/'summary.json').write_text(json.dumps(dict(passed=True,rows=rows,source_identities=identities),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);p.add_argument('--case')
    p.add_argument('--sigma',type=float,default=25);p.add_argument('--crop',type=int);p.add_argument('--parameters-json',default='{}');p.add_argument('--verify-small',action='store_true');main(p.parse_args())
