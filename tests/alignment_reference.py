"""Readable independent image reference for the documented TWSC v3 / NLH v4 models.

This implements the chosen mathematical contract, not the authors' closed MEX
program. Array transforms and Schur Sylvester solves are independent of C++.
"""
import numpy as np
from test_alignment_math import haar, row_neighbors, solve, NLH_HARD_COEFFICIENT, NLH_WIENER_SIGMA_SCALE


def raster(width, height, block, step):
    xs = list(range(0, width-block+1, step))
    ys = list(range(0, height-block+1, step))
    if xs[-1] != width-block: xs.append(width-block)
    if ys[-1] != height-block: ys.append(height-block)
    return [(x,y) for y in ys for x in xs]


def matches(guides, center, x, y, block, group, window, radius=0, ps_num=2, ps_range=1):
    """guides: T,C,H,W; every score compares to the original center query."""
    frames, _, height, width = guides.shape
    query = guides[center,:,y:y+block,x:x+block].astype(float)
    def score(t, xx, yy):
        d = query-guides[t,:,yy:yy+block,xx:xx+block].astype(float)
        return float(np.float32(np.sum(d*d)))
    def key(value):
        t,xx,yy,d = value
        return d,t,yy,xx
    lo,hi = window//2,window-window//2-1
    spatial = [(center,xx,yy,score(center,xx,yy))
               for yy in range(max(0,y-lo),min(height-block,y+hi)+1)
               for xx in range(max(0,x-lo),min(width-block,x+hi)+1) if (xx,yy)!=(x,y)]
    retained = sorted(spatial,key=key)[:group-1]
    first = (center,x,y,0.)
    if radius and group>1:
        seeds = ([first]+retained)[:ps_num]
        for direction in (-1,1):
            centers = seeds
            for dt in range(1,radius+1):
                t=center+direction*dt
                if t<0 or t>=frames: break
                coords=set()
                for _,xx,yy,_ in centers:
                    coords.update((a,b) for b in range(max(0,yy-ps_range),min(height-block,yy+ps_range)+1)
                                  for a in range(max(0,xx-ps_range),min(width-block,xx+ps_range)+1))
                centers=sorted([(t,xx,yy,score(t,xx,yy)) for xx,yy in coords],key=key)[:ps_num]
                retained=sorted(retained+centers,key=key)[:group-1]
    return [first]+retained


def pack(images, found, block):
    return np.array([images[t,:,y:y+block,x:x+block].reshape(-1) for t,x,y,_ in found], dtype=float).T


def noise_estimate(channel, guide):
    """Independent fixed b8/g16/q4/W40 bootstrap; guide only selects positions."""
    estimates=[]
    h,w=channel.shape
    for x,y in raster(w,h,8,1):
        found=matches(guide[None,None],0,x,y,8,16,40)
        gp=pack(guide[None,None],found,8)
        pixels=pack(channel[None,None],found,8)
        idx=row_neighbors(gp,4)
        distance=pixels[:,None,:]-pixels[idx[:,1:]]
        estimates.append(np.mean(np.sqrt(np.mean(distance*distance,axis=2))))
    return np.float32(np.mean(estimates))


def twsc_contributions(images, sigmas, center, *, guide=None, block=4, group=4, step=4,
                       window=6, iterations=2, radius=0, ps_num=2, ps_range=1,
                       lambda2=1., delta=0., admm_iter=10, rho=.5, mu=1.1, tol=1e-6):
    images=np.asarray(images,np.float32)
    sigmas=np.asarray(sigmas,np.float32)
    frames,channels,height,width=images.shape
    estimate=images.copy()
    for iteration in range(iterations):
        if iteration:
            estimate=(estimate.astype(float)+delta*(images.astype(float)-estimate)).astype(np.float32)
        num=np.zeros(images.shape,float);den=np.zeros_like(num)
        for t0 in ([center] if iteration==iterations-1 else range(frames)):
            precision=1/np.maximum(np.repeat(sigmas[t0],block*block).astype(float),1e-6)
            for x,y in raster(width,height,block,step):
                found=matches(guide if guide is not None else estimate,t0,x,y,block,group,window,radius,ps_num,ps_range)
                ygroup=pack(estimate,found,block)
                original=pack(images,found,block)
                cols=[]
                for j,(t,_,_,_) in enumerate(found):
                    variance=float(np.mean(sigmas[t].astype(float)**2))
                    residual=0 if iteration==0 else float(np.mean((original[:,j]-ygroup[:,j])**2))
                    cols.append(lambda2*np.sqrt(abs(variance-residual)))
                cols=np.asarray(cols,np.float32).astype(float)
                mean=np.mean(ygroup,axis=1)
                centered=(ygroup-mean[:,None]).astype(np.float32).astype(float)
                d,s,_=np.linalg.svd(centered,full_matrices=False)
                s=np.sqrt(np.maximum(s*s-len(found)*cols[0]**2,0))
                result,_,_,_=solve(centered,d,s,precision,cols,admm_iter,rho,mu,tol)
                result=(result+mean[:,None]).astype(np.float32).astype(float)
                weights=(1/np.maximum(cols,1e-6)).astype(np.float32).astype(float)
                for j,(t,xx,yy,_) in enumerate(found):
                    patch=result[:,j].reshape(channels,block,block)
                    num[t,:,yy:yy+block,xx:xx+block]+=patch*weights[j]
                    den[t,:,yy:yy+block,xx:xx+block]+=weights[j]
        if iteration+1<iterations:
            estimate=np.divide(num,den,out=images.astype(float).copy(),where=den>0).astype(np.float32)
    return num,den


def nlh_contributions(images, sigmas, center, *, guide=None, blocks=(4,4), groups=(4,8), qs=(2,4),
                      steps=(4,4), windows=(6,7), basic_iters=2, radius=0, ps_num=2, ps_range=1,
                      basic_mix=.6, hard_strength=1., wiener_iters=2, wiener_scale=NLH_WIENER_SIGMA_SCALE):
    images=np.asarray(images,np.float32)
    sigmas=np.asarray(sigmas,np.float32)
    frames,channels,height,width=images.shape
    def run(data,basic,stage):
        b,g,q,step,window=blocks[stage],groups[stage],qs[stage],steps[stage],windows[stage]
        num=np.zeros(images.shape,float);den=np.zeros_like(num)
        match_source=guide if guide is not None else basic if stage else data
        match_source=match_source[:,:1]
        for t0 in ([center] if stage else range(frames)):
            for x,y in raster(width,height,b,step):
                found=matches(match_source,t0,x,y,b,g,window,radius,ps_num,ps_range)
                n=2**int(np.floor(np.log2(len(found))))
                found=found[:n]
                guide_group=pack(match_source,found,b)
                indices=row_neighbors(guide_group,q)
                hq,hn=haar(q),haar(n)
                for c in range(channels):
                    source=pack(data[:,c:c+1],found,b)
                    ref=pack(basic[:,c:c+1],found,b) if stage else None
                    local_num=np.zeros_like(source);local_den=np.zeros_like(source)
                    for row in range(b*b):
                        matrix=source[indices[row]]
                        sigma=float(sigmas[t0,c])
                        if sigma:
                            coefficients=hq@matrix@hn.T
                            if stage:
                                r=hq@ref[indices[row]]@hn.T
                                noise=(wiener_scale*sigma)**2
                                gain=np.ones_like(r) if noise==0 else r*r/(r*r+noise)
                                for _ in range(wiener_iters): coefficients*=gain
                            else:
                                coefficients[np.abs(coefficients)<NLH_HARD_COEFFICIENT*hard_strength*sigma]=0
                                coefficients[max(0,q-2):,1:]=0
                            matrix=hq.T@coefficients@hn
                        for k,destination in enumerate(indices[row]):
                            local_num[destination]+=matrix[k]
                            local_den[destination]+=1
                    for j,(t,xx,yy,_) in enumerate(found):
                        num[t,c,yy:yy+b,xx:xx+b]+=local_num[:,j].reshape(b,b)
                        den[t,c,yy:yy+b,xx:xx+b]+=local_den[:,j].reshape(b,b)
        return num,den
    basic=images.copy()
    for iteration in range(basic_iters):
        basic=(basic_mix*basic.astype(float)+(1-basic_mix)*images.astype(float)).astype(np.float32)
        num,den=run(basic,basic,0)
        basic=np.divide(num,den,out=images.astype(float).copy(),where=den>0).astype(np.float32)
    return run(images,basic,1)


def spatial_reference(image, sigma, model, **options):
    """RGB/Gray adapter with explicit FP32 color/output boundaries."""
    image=np.asarray(image,np.float32)
    sigmas=np.broadcast_to(np.asarray(sigma,np.float32),image.shape[:1]).copy()/np.float32(255)
    working=image.copy()
    color=model=='NLH' and len(image)==3
    if color:
        a=np.array([[.299,.587,.114],[-.168736607142857,-.331263392857143,.5],[.5,-.4186875,-.0813125]])
        working=(a@image.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32)
        sigmas=np.sqrt((a*a)@(sigmas.astype(float)**2)).astype(np.float32)
    if model=='TWSC':
        rms=float(np.sqrt(np.mean(np.asarray(sigma,float)**2)))
        defaults=dict(block=7 if rms<=20 else 8 if rms<=60 else 9,
                      group=70 if rms<=20 else 90 if rms<=40 else 120 if rms<=60 else 140,
                      step=1,window=60,iterations=8 if rms<=20 else 12 if rms<=60 else 14)
        num,den=twsc_contributions(working[None],sigmas[None],0,**dict(defaults,**options))
    else:
        block=7 if color else 8 if max(sigmas)*255<=50 else 10
        defaults=dict(blocks=(block,block),groups=(16,16),qs=(4,4),steps=(1,1),windows=(40,40),
                      basic_iters=2 if color else 4 if block==8 else 5)
        num,den=nlh_contributions(working[None],sigmas[None],0,**dict(defaults,**options))
    num=num[0].astype(np.float32);den=den[0].astype(np.float32)
    if color:
        b=np.array([[1,0,1.402],[1,-.114*1.772/.587,-.299*1.402/.587],[1,1.772,0]])
        num=(b@num.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32)
    return np.divide(num.astype(float),den,out=image.astype(float).copy(),where=den>0).astype(np.float32)
