#!/usr/bin/env python3
"""Independent full-image TWSC reference for uniform explicit channel noise.

NumPy/LAPACK economy SVD replaces the production QR/Jacobi implementation.
FP64 box sums evaluate exactly the same SSD candidate windows. For orthonormal
U and uniform W1, the full Sylvester equation separates by coefficient; every
ADMM update and all three stopping tests are still evaluated. This does not
call a production matcher, decomposition, solver or finisher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from alignment_campaign import metrics
from alignment_reference import spatial_reference,matches


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_matrix(image,b):
    view=np.lib.stride_tricks.sliding_window_view(image,(b,b),axis=(-2,-1))
    return view.transpose(1,2,0,3,4).reshape(-1,len(image)*b*b).copy()


def strip_matches(image,b,g,window,y0,rows):
    """Return self-first patch indices, with float32 SSD and stable raster ties."""
    _,h,w=image.shape;ny,nx=h-b+1,w-b+1;lo=window//2;hi=window-lo-1
    offsets=[(dx,dy) for dy in range(-min(lo,ny-1),min(hi,ny-1)+1)
             for dx in range(-min(lo,nx-1),min(hi,nx-1)+1)]
    scores=np.full((len(offsets),rows,nx),np.inf,np.float32)
    for k,(dx,dy) in enumerate(offsets):
        ay=max(y0,0,-dy);by=min(y0+rows,ny,ny-dy)
        ax=max(0,-dx);bx=min(nx,nx-dx)
        if ay>=by or ax>=bx: continue
        left=image[:,ay:by+b-1,ax:bx+b-1].astype(float)
        right=image[:,ay+dy:by+dy+b-1,ax+dx:bx+dx+b-1].astype(float)
        error=np.sum((left-right)**2,axis=0)
        integral=np.pad(error.cumsum(axis=0).cumsum(axis=1),((1,0),(1,0)))
        distance=integral[b:,b:]-integral[:-b,b:]-integral[b:,:-b]+integral[:-b,:-b]
        scores[k,ay-y0:by-y0,ax:bx]=distance.astype(np.float32)
    scores[offsets.index((0,0))]=-1 # self always ranks first, including tied zeros
    scores=scores.reshape(len(offsets),-1).T
    if np.any(np.sum(np.isfinite(scores),axis=1)<g): raise ValueError('this reference needs a uniform actual group size')
    selected=np.argpartition(scores,g-1,axis=1)[:,:g]
    selected_scores=np.take_along_axis(scores,selected,axis=1)
    cutoff=selected_scores.max(axis=1)
    ties=np.sum(scores==cutoff[:,None],axis=1)>np.sum(selected_scores==cutoff[:,None],axis=1)
    for row in np.flatnonzero(ties):
        less=np.flatnonzero(scores[row]<cutoff[row]);equal=np.flatnonzero(scores[row]==cutoff[row])
        selected[row]=np.concatenate([less,equal[:g-len(less)]])
    order=np.lexsort((selected,np.take_along_axis(scores,selected,axis=1)),axis=1)
    selected=np.take_along_axis(selected,order,axis=1)
    xy=np.array(offsets,dtype=np.int32)[selected]
    yy,xx=np.mgrid[y0:y0+rows,:nx]
    result=(yy.ravel()[:,None]+xy[:,:,1])*nx+xx.ravel()[:,None]+xy[:,:,0]
    return result.astype(np.int32)


def uniform_group(groups,original,sigma,iteration,admm_iter=10,rho=.5,mu=1.1,tol=1e-6):
    """Batches of group matrices shaped B,M,N; inputs are FP32-valued FP64."""
    batch,m,n=groups.shape
    mean=groups.mean(axis=2)
    centered=(groups-mean[:,:,None]).astype(np.float32).astype(float)
    u,s,vt=np.linalg.svd(centered,full_matrices=False)
    residual=0 if iteration==0 else np.mean((original-groups)**2,axis=1)
    column_sigma=np.broadcast_to(np.sqrt(np.abs(float(sigma)**2-residual)),(batch,n)).astype(np.float32).astype(float)
    spectrum=np.sqrt(np.maximum(s*s-n*column_sigma[:,0,None]**2,0))
    atoms=u*spectrum[:,None,:]
    gram=np.matmul(atoms.swapaxes(1,2),atoms)/max(float(sigma),1e-6)
    diagonal=spectrum*spectrum/max(float(sigma),1e-6)
    rr=np.arange(len(s[0]));off=gram.copy();off[:,rr,rr]-=diagonal
    gram_norm=np.linalg.norm(gram,axis=(1,2));off_norm=np.linalg.norm(off,axis=(1,2))
    if np.any(off_norm>1e-12*gram_norm+1e-300): raise ArithmeticError('LAPACK dictionary failed orthogonality bound')
    data=np.matmul(atoms.swapaxes(1,2),centered)/max(float(sigma),1e-6)
    columns=np.maximum(column_sigma,1e-6)[:,None,:]
    coefficients=np.zeros_like(data);aux=np.zeros_like(data);dual=np.zeros_like(data)
    active=np.ones(batch,bool);used=np.zeros(batch,np.int32)
    for k in range(admm_iter):
        new=(data+.5*(rho*aux-dual)*columns)/(diagonal[:,:,None]+.5*rho*columns)
        temporary=new+dual/rho
        z=np.sign(temporary)*np.maximum(np.abs(temporary)-1/rho,0)
        primal=np.linalg.norm(new-z,axis=(1,2));dc=np.linalg.norm(new-coefficients,axis=(1,2));dz=np.linalg.norm(z-aux,axis=(1,2))
        done=(primal<=tol)&(dc<=tol)&(dz<=tol)
        coefficients[active]=new[active];aux[active]=z[active];used[active]=k+1
        active&=~done
        dual[active]+=rho*(coefficients[active]-aux[active])
        if not active.any(): break
        rho*=mu
    result=(np.matmul(atoms,coefficients)+mean[:,:,None]).astype(np.float32)
    weights=(1/np.maximum(column_sigma,1e-6)).astype(np.float32)
    return result,weights,int(active.sum()),int(used.sum())


def denoise(image,sigma,*,strip_rows=8,batch_size=16,progress=None):
    if not np.isfinite(sigma) or sigma<=0: raise ValueError('this reference requires finite positive uniform sigma')
    image=np.asarray(image,np.float32);c,h,w=image.shape
    b=7 if sigma<=20 else 8 if sigma<=60 else 9
    requested_g=70 if sigma<=20 else 90 if sigma<=40 else 120 if sigma<=60 else 140
    iterations=8 if sigma<=20 else 12 if sigma<=60 else 14
    nx,ny=w-b+1,h-b+1;g=min(requested_g,nx*ny)
    if nx<1 or ny<1: raise ValueError('image smaller than block')
    working_sigma=np.float32(sigma)/np.float32(255)
    original=patch_matrix(image,b);estimate=image.copy();logs=[]
    for iteration in range(iterations):
        started=time.perf_counter();patches=patch_matrix(estimate,b)
        patch_numerator=np.zeros((nx*ny,c*b*b),float);patch_denominator=np.zeros(nx*ny,float)
        digest=hashlib.sha256();capped=0;admm_steps=0
        for y0 in range(0,ny,strip_rows):
            rows=min(strip_rows,ny-y0);indices=strip_matches(estimate,b,g,60,y0,rows)
            digest.update(indices.tobytes())
            for start in range(0,len(indices),batch_size):
                selected=indices[start:start+batch_size]
                groups=patches[selected].transpose(0,2,1).astype(float)
                base=original[selected].transpose(0,2,1).astype(float)
                rebuilt,weights,uncapped,steps=uniform_group(groups,base,working_sigma,iteration)
                capped+=uncapped;admm_steps+=steps
                values=rebuilt.transpose(0,2,1).astype(float)*weights[:,:,None].astype(float)
                np.add.at(patch_numerator,selected,values)
                np.add.at(patch_denominator,selected.ravel(),weights.ravel())
            if progress: progress(iteration,y0,ny,time.perf_counter()-started)
        numerator=np.zeros_like(image,dtype=float);denominator=np.zeros_like(numerator)
        grouped=patch_numerator.reshape(ny,nx,c,b,b)
        counts=patch_denominator.reshape(ny,nx)
        for channel in range(c):
            for dy in range(b):
                for dx in range(b):
                    numerator[channel,dy:dy+ny,dx:dx+nx]+=grouped[:,:,channel,dy,dx]
                    denominator[channel,dy:dy+ny,dx:dx+nx]+=counts
        estimate=(numerator/denominator).astype(np.float32)
        if not np.isfinite(estimate).all(): raise ArithmeticError('nonfinite reference estimate')
        row=dict(iteration=iteration+1,seconds=time.perf_counter()-started,groups=nx*ny,
                 matching_sha256=digest.hexdigest(),capped_groups=capped,admm_steps=admm_steps,
                 estimate_sha256=hashlib.sha256(estimate.tobytes()).hexdigest())
        logs.append(row)
    # The plugin's final contribution storage is FP32 before its normalization.
    result=(numerator.astype(np.float32).astype(float)/denominator.astype(np.float32)).reshape(image.shape).astype(np.float32)
    return result,logs


def main(args):
    root=Path(args.fixtures).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((root/'fixtures.json').read_text())['cases']
    cases=[c for c in cases if c['sigma']==args.sigma and (not args.case or c['id']==args.case)]
    if not cases: raise ValueError('no selected fixtures')
    rows=[]
    for case in cases:
        shape=(case.get('channels',1),case['height'],case['width'])
        if sha(root/case['noisy'])!=case['noisy_sha256'] or sha(root/case['clean'])!=case['clean_sha256']: raise ValueError('fixture identity mismatch')
        noisy=np.fromfile(root/case['noisy'],dtype='<f4').reshape(shape);clean=np.fromfile(root/case['clean'],dtype='<f4').reshape(shape)
        crop=None
        if args.crop:
            size=args.crop;x=(shape[2]-size)//2;y=(shape[1]-size)//2;crop=[x,y,size,size]
            noisy=noisy[:,y:y+size,x:x+size].copy();clean=clean[:,y:y+size,x:x+size].copy()
        progress_path=out/(case['id']+'-progress.json')
        def progress(i,y,ny,seconds):
            progress_path.write_text(json.dumps(dict(iteration=i+1,completed_query_rows=y+min(args.strip_rows,ny-y),query_rows=ny,seconds=seconds))+'\n')
        started=time.perf_counter();pixels,iterations=denoise(noisy,args.sigma,strip_rows=args.strip_rows,progress=progress)
        elapsed=time.perf_counter()-started
        output=out/(case['id']+'-TWSC.npy');np.save(output,pixels)
        row=dict(case=case,model='TWSC',crop=crop,shape=list(noisy.shape),seconds=elapsed,iterations=iterations,
                 output=output.name,output_sha256=sha(output),quality=metrics(clean,pixels),implementation='independent_numpy_uniform_sylvester')
        if args.verify_small:
            if args.crop is None or args.crop>24: raise ValueError('--verify-small requires --crop <=24')
            exact=spatial_reference(noisy,args.sigma,'TWSC');row['schur_reference_difference']=metrics(exact,pixels)
            np.testing.assert_allclose(pixels,exact,atol=2e-5,rtol=2e-4)
            b=7 if args.sigma<=20 else 8 if args.sigma<=60 else 9
            ny,nx=noisy.shape[1]-b+1,noisy.shape[2]-b+1
            selected=strip_matches(noisy,b,min(90,ny*nx),60,0,ny)
            for q,(x,y) in enumerate(( (x,y) for y in range(ny) for x in range(nx))):
                expected=matches(noisy[None],0,x,y,b,min(90,ny*nx),60)
                assert selected[q].tolist()==[yy*nx+xx for _,xx,yy,_ in expected]
        rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(case['id'],'DONE',elapsed,flush=True)
    (out/'summary.json').write_text(json.dumps(dict(passed=True,rows=rows,script_sha256=sha(__file__),fixture_manifest_sha256=sha(root/'fixtures.json')),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True);p.add_argument('--case')
    p.add_argument('--sigma',type=float,default=25);p.add_argument('--crop',type=int);p.add_argument('--strip-rows',type=int,default=8);p.add_argument('--verify-small',action='store_true')
    main(p.parse_args())
