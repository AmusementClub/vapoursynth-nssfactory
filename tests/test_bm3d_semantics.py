#!/usr/bin/env python3
"""Independent time-index and public-sigma oracles (not legacy=rolling alone)."""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import vapoursynth as vs
core=vs.core
core.num_threads=2
core.std.LoadPlugin(path=os.environ['NSS_SO'])

def clip(values,fmt=vs.GRAYS):
    a=np.asarray(values,dtype=np.float32)
    if a.ndim==3:a=a[:,None]
    n,p,h,w=a.shape
    base=core.std.BlankClip(width=w,height=h,format=fmt,length=n)
    def fill(n,f):
        out=f.copy()
        for p in range(out.format.num_planes):np.asarray(out[p])[:]=a[n,p]
        return out
    return core.std.ModifyFrame(base,base,fill)

def arrays(c,n):return [np.array(c.get_frame(n)[p]) for p in range(c.format.num_planes)]
def close(a,b,msg,tol=1e-5):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f'{msg}: nonfinite value')
    err=float(np.max(np.abs(np.asarray(a,dtype=np.float64)-b)))
    if err>tol:raise AssertionError(f'{msg}: max_abs={err}')

def temporal():
    rng=np.random.default_rng(42)
    clean=rng.uniform(.05,.95,(7,3,12,18)).astype(np.float32)
    src=clip(clean,vs.YUV444PS)
    # Construct center/slice values independently. Each valid target is identity,
    # multiplied by a center-dependent weight; wrong-axis reducers fail.
    for radius in (0,1,2,4,16):
        fats=np.zeros((7,3,12*(2*radius+1)*2,18),np.float32)
        for c in range(7):
            for t in range(max(0,c-radius),min(7,c+radius+1)):
                sl=t-c+radius
                fats[c,:,2*sl*12:(2*sl+1)*12]=clean[t]*(c+1)
                fats[c,:,(2*sl+1)*12:(2*sl+2)*12]=c+1
        fat=clip(fats,vs.YUV444PS)
        dst=core.nss.VAggregate(fat,src,radius=radius)
        for n in (0,6,3,1,5,2,4):close(arrays(dst,n),clean[n],f'identity r{radius} n{n}',2e-6)
    # Constant image per frame exposes disabled-plane contamination at boundaries.
    v=np.stack([np.full((3,8,8),.05+.1*n,np.float32) for n in range(5)])
    src=clip(v,vs.RGBS)
    for name in ('BM3D','WNNM','NLH','NCSR','TWSC','MCWNNM'):
        kw=dict(sigma=0,radius=2,block_size=4,block_step=4,group_size=4,bm_range=2)
        if name in ('TWSC','MCWNNM','NCSR'):kw['iters']=1
        fat=getattr(core.nss,name)(src,**kw)
        dst=core.nss.VAggregate(fat,src,radius=2)
        for n in range(5):close(arrays(dst,n),v[n],f'disabled {name} n{n}',0)
    for name in ('TWSC','MCWNNM'):
        for radius in (0,1):
            kw=dict(sigma=[3,0],radius=radius,block_size=4,block_step=4,group_size=4,bm_range=2,iters=1)
            out=getattr(core.nss,name)(src,**kw)
            if radius:out=core.nss.VAggregate(out,src,radius=radius)
            for n in range(5):
                values = arrays(out, n)
                if not np.isfinite(values).all():
                    raise AssertionError(f'partial-zero {name}: nonfinite active output')
                close(values[1:],v[n,1:],f'joint zero-channel {name} r{radius}',0)
    noisy = np.random.default_rng(73).uniform(.1, .8, (3, 3, 12, 12)).astype(np.float32)
    shared_src = clip(noisy, vs.RGBS)
    for name in ('WNNM', 'NLH', 'NCSR', 'TWSC', 'MCWNNM'):
        kw = dict(sigma=3, radius=1, block_size=4, block_step=2,
                  group_size=4, bm_range=2, ps_range=1)
        if name in ('TWSC', 'MCWNNM', 'NCSR'):
            kw['iters'] = 1
        fat = getattr(core.nss, name)(shared_src, **kw)
        target = core.nss.VAggregate(fat, shared_src, radius=1)
        for n in range(3):
            num = np.zeros((3, 12, 12), np.float64)
            den = np.zeros_like(num)
            for center in range(max(0, n-1), min(3, n+2)):
                values = np.stack(arrays(fat, center))
                slot = n-center+1
                num += values[:, 2*slot*12:(2*slot+1)*12]
                den += values[:, (2*slot+1)*12:(2*slot+2)*12]
            expected = np.divide(num, den, out=noisy[n].astype(np.float64).copy(), where=den>1e-12)
            close(arrays(target, n), expected, f'shared nonzero target oracle {name}')
    for radius in (1,2,4):
        for wiener in (False,True):
            kw=dict(sigma=[3,0],radius=radius,block_size=4,block_step=4,group_size=8,bm_range=2,ps_range=1)
            if wiener:kw['ref']=src
            fat=core.nss.BM3D(src,**kw)
            target=core.nss.VAggregate(fat,src,radius=radius)
            roll=core.nss.BM3D(src,temporal_mode='rolling',rolling_chunk=2,rolling_cache_limit=1,**kw)
            expected=[]
            for n in range(5):
                num=np.zeros((3,8,8),np.float64);den=np.zeros_like(num)
                for c in range(max(0,n-radius),min(5,n+radius+1)):
                    vals=np.stack(arrays(fat,c));sl=n-c+radius
                    num+=vals[:,2*sl*8:(2*sl+1)*8];den+=vals[:,(2*sl+1)*8:(2*sl+2)*8]
                want=np.divide(num,den,out=v[n].astype(np.float64).copy(),where=den>1e-12)
                close(arrays(target,n),want,'actual producer target oracle')
                close(arrays(target,n)[1:],v[n,1:],'sigma short-array disabled',0)
                expected.append(want)
            order=[4,0,2,1,3,0,4,2]
            with ThreadPoolExecutor(2) as pool:
                results=list(pool.map(lambda n:arrays(roll,n),order))
            for n,out in zip(order,results):close(out,expected[n],f'rolling oracle r{radius} n{n}')
    # Actual two-stage rolling has a second dependency expansion through ref.
    kw=dict(sigma=[3,0],radius=2,block_size=4,block_step=4,group_size=8,bm_range=2,ps_range=1)
    bfat=core.nss.BM3D(src,**kw);basic=core.nss.VAggregate(bfat,src,radius=2)
    finalfat=core.nss.BM3D(src,ref=basic,**kw);final=core.nss.VAggregate(finalfat,src,radius=2)
    rb=core.nss.BM3D(src,temporal_mode='rolling',rolling_chunk=2,rolling_cache_limit=1,**kw)
    rf=core.nss.BM3D(src,ref=rb,temporal_mode='rolling',rolling_chunk=2,rolling_cache_limit=1,**kw)
    for n in [4,0,2,1,3,0]:close(arrays(rf,n),arrays(final,n),'actual two-stage rolling')
    # Single-frame maximum-radius must not replicate source candidates.
    one=clip(v[:1],vs.RGBS)
    for mode in ('legacy','rolling'):
        out=core.nss.BM3D(one,sigma=[0,0,0],radius=16,temporal_mode=mode)
        if mode=='legacy':out=core.nss.VAggregate(out,one,radius=16)
        close(arrays(out,0),v[0],'one frame disabled',0)
    one = clip(noisy[:1], vs.RGBS)
    for wiener in (False, True):
        kw = dict(sigma=3, block_size=4, group_size=8, block_step=2, bm_range=2)
        if wiener:
            kw['ref'] = one
        spatial = core.nss.BM3D(one, **kw)
        for mode in ('legacy', 'rolling'):
            out = core.nss.BM3D(one, radius=16, temporal_mode=mode, **kw)
            if mode == 'legacy':
                out = core.nss.VAggregate(out, one, radius=16)
            close(arrays(out, 0), arrays(spatial, 0), 'one real nonzero frame at maximum radius')

def matrix(n):
    k=np.arange(n)[:,None];x=np.arange(n)[None,:]
    return np.cos(np.pi*(x+.5)*k/n)*np.sqrt(np.where(k==0,1.,2.)/n)
def oracle(p,ref,group,sigma,wiener):
    b=p.shape[0];d=matrix(b);g=matrix(group)
    cube=np.zeros((group,b,b));cube[0]=p
    f=np.einsum('kg,yv,xu,gvu->kyx',g,d,d,cube,optimize=True)
    if wiener:
        cube[0]=ref;r=np.einsum('kg,yv,xu,gvu->kyx',g,d,d,cube,optimize=True)
        with np.errstate(invalid='ignore',divide='ignore'):weight=r*r/(r*r+sigma*sigma)
        weight[0,0,0]=1;f*=weight
    else:
        mask=np.abs(f)>=2.7*sigma;mask[0,0,0]=True;f*=mask
    return np.einsum('kg,yv,xu,kyx->gvu',g,d,d,f,optimize=True)[0]

def sigma():
    rng=np.random.default_rng(59)
    for b in (1,2,4,8,12,16,32):
        p=rng.uniform(-.02,.03,(b,b)).astype(np.float32);ref=p*.8
        src=clip(p[None]);rc=clip(ref[None])
        for group in (1,2,4,8,16,32,64):
            for user in (0,.5,1,3,5,10,20,40):
                for wiener in (False,True):
                    kw=dict(sigma=user,block_size=b,block_step=b,group_size=group,bm_range=1)
                    if wiener:kw['ref']=rc
                    out=core.nss.BM3D(src,**kw)
                    want=p if user==0 else oracle(p,ref,group,user*.75/255,wiener)
                    close(arrays(out,0)[0],want,f'public sigma b{b} g{group} s{user} w{wiener}',3e-5)
    for bad in (-1,float('nan'),float('inf')):
        try:core.nss.BM3D(src,sigma=bad)
        except vs.Error:pass
        else:raise AssertionError('nonfinite/negative sigma accepted')

if __name__=='__main__':
    temporal();sigma();print('independent temporal, public sigma, disabled-plane and concurrency oracles passed')
