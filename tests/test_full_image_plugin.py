#!/usr/bin/env python3
"""Public TWSC v3 / NLH v5 gates, including independent image references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import traceback

import numpy as np
import vapoursynth as vs

from alignment_reference import nlh_contributions, twsc_contributions, noise_estimate
from test_alignment_math import NLH_HARD_COEFFICIENT, NLH_WIENER_SIGMA_SCALE


def make_clip(core, fmt, frames=5, width=20, height=18, seed=7, constant=None):
    blank=core.std.BlankClip(width=width,height=height,format=fmt,length=frames)
    format=blank.format
    rng=np.random.default_rng(seed)
    arrays=[]
    for n in range(frames):
        planes=[]
        for p in range(format.num_planes):
            w=width >> (format.subsampling_w if p else 0)
            h=height >> (format.subsampling_h if p else 0)
            y,x=np.mgrid[:h,:w]
            values=(rng.integers(-8,9,(h,w))/128+(x+2*y+n)/512+.3)
            if format.color_family==vs.YUV and p: values-=.3
            if constant is not None: values=np.full((h,w),constant)
            planes.append(np.asarray(values,np.float32))
        arrays.append(planes)
    def fill(n,f):
        out=f.copy()
        for p,values in enumerate(arrays[n]): np.copyto(np.asarray(out[p]),values)
        return out
    return core.std.ModifyFrame(blank,blank,fill),arrays


def values(node,n):
    with node.get_frame(n) as frame:
        return [np.array(frame[p],copy=True) for p in range(frame.format.num_planes)],dict(frame.props)


def kwargs(name,radius=0):
    if name=='TWSC':
        return dict(block_size=4,block_step=4,group_size=8,search_window=6,iters=2,radius=radius,ps_num=2,ps_range=1)
    # The independent mathematical fixtures use explicit v4 coefficients, so
    # changing the public presets cannot silently change their reference math.
    return dict(block_size=[4,4],block_step=[4,4],group_size=[4,8],q=[2,4],search_window=[6,7],
                basic_iters=2,wiener_iters=2,lambda_basic=.6,hard_strength=1.,wiener_sigma_scale=.08,
                radius=radius,ps_num=2,ps_range=1)


class Suite:
    def __init__(self,out,plugin):
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.plugin=Path(plugin).resolve();self.core=vs.core;self.core.max_cache_size=32
        self.core.std.LoadPlugin(path=str(self.plugin));self.rows=[]

    def case(self,name,call):
        try:
            detail=call() or {}
            self.rows.append(dict(name=name,passed=True,**detail));print(name,'PASS',flush=True)
        except Exception:
            error=traceback.format_exc();self.rows.append(dict(name=name,passed=False,error=error));print(name,'FAIL',error,flush=True)

    def matrix(self,name,fmt,radius,reference):
        core=self.core
        source,original=make_clip(core,fmt)
        guide,_=make_clip(core,fmt,seed=99)
        args=kwargs(name,radius);args['sigma']=3 if source.format.num_planes==1 else [3,4,5]
        expected=None
        order=[4,0,2,1,3]
        for threads in (1,2,4):
            core.num_threads=threads
            raw=getattr(core.nss,name)(source,**args,**({'rclip':guide} if reference else {}))
            node=core.nss.VAggregate(raw,source,radius=radius) if radius else raw
            core.std.SetVideoCache(node,mode=0)
            with ThreadPoolExecutor(max_workers=threads) as pool:
                rows=list(pool.map(lambda n: (n,values(node,n)),order))
            captured={n:row[0] for n,row in rows}
            for n,(planes,props) in rows:
                assert props['_NSSModelVersion']==(3 if name=='TWSC' else 5)
                for p,a in enumerate(planes):
                    assert a.shape==original[n][p].shape and np.isfinite(a).all()
            if expected is None: expected=captured
            else:
                for n in order:
                    for a,b in zip(expected[n],captured[n]): np.testing.assert_array_equal(a,b)
            if radius:
                fat,props=values(raw,0)
                for p,array in enumerate(fat):
                    h=original[0][p].shape[0]
                    np.testing.assert_array_equal(array[:2*radius*h],0)
                assert props['_NSSFatCenter']==0 and props['_NSSFatRadius']==radius
        path=self.out/f'matrix_{name}_{fmt}_{radius}_{reference}.npz'
        np.savez_compressed(path,**{f'n{n}_p{p}':a for n in order for p,a in enumerate(expected[n])})
        return dict(pixels_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def zero(self,name,fmt,radius):
        core=self.core;core.num_threads=2
        source,original=make_clip(core,fmt)
        sigmas=0 if source.format.num_planes==1 else [0,3,0]
        raw=getattr(core.nss,name)(source,sigma=sigmas,**kwargs(name,radius))
        node=core.nss.VAggregate(raw,source,radius=radius) if radius else raw
        for n in (0,2,4):
            planes,_=values(node,n)
            for p in range(source.format.num_planes):
                if source.format.num_planes==1 or p in (0,2): np.testing.assert_array_equal(planes[p],original[n][p])

    def oracle(self,name,radius,reference):
        core=self.core;core.num_threads=1
        source,original=make_clip(core,vs.GRAYS,frames=3,width=12,height=11,seed=111)
        guide,guides=make_clip(core,vs.GRAYS,frames=3,width=12,height=11,seed=222)
        images=np.array(original,np.float32);guide_images=np.array(guides,np.float32)
        args=kwargs(name,radius)
        if name=='TWSC': args.update(group_size=4,lambda2=.8,delta=.1)
        args['sigma']=3
        raw=getattr(core.nss,name)(source,**args,**({'rclip':guide} if reference else {}))
        center=1;first=max(0,center-radius);last=min(2,center+radius)
        window=images[first:last+1];ref=guide_images[first:last+1] if reference else None
        sigma=np.full((len(window),1),np.float32(3/255),np.float32)
        if name=='TWSC':
            num,den=twsc_contributions(window,sigma,center-first,guide=ref,radius=radius,lambda2=.8,delta=.1)
        else:
            num,den=nlh_contributions(window,sigma,center-first,guide=ref,radius=radius)
        got,_=values(raw,center)
        differences=[]
        if radius:
            h,w=window.shape[2:]
            for t in range(len(window)):
                sl=first+t-center+radius
                gn=got[0][2*sl*h:(2*sl+1)*h]
                gd=got[0][(2*sl+1)*h:(2*sl+2)*h]
                expected=np.divide(num[t,0],den[t,0],out=np.zeros((h,w)),where=den[t,0]>0)
                actual=np.divide(gn,gd,out=np.zeros_like(gn),where=gd>0)
                # Destination/count identity is exact for NLH; TWSC's adaptive
                # column weights can contain propagated FP32 solver rounding.
                if name=='NLH': np.testing.assert_array_equal(gd,den[t,0])
                else: np.testing.assert_allclose(gd,den[t,0],rtol=2e-4,atol=2e-5)
                np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
                differences.append(float(np.max(np.abs(actual-expected))))
        else:
            expected=np.divide(num[0,0],den[0,0],out=images[center,0].astype(float).copy(),where=den[0,0]>0)
            np.testing.assert_allclose(got[0],expected,rtol=2e-4,atol=2e-5)
            differences.append(float(np.max(np.abs(got[0]-expected))))
        path=self.out/f'oracle_{name}_{radius}_{reference}.npz'
        np.savez_compressed(path,input=images,guide=guide_images,expected_num=num,expected_den=den,actual=got[0])
        return dict(max_abs=max(differences))

    def parameters(self):
        core=self.core;core.num_threads=1
        source,_=make_clip(core,vs.GRAYS,frames=1,width=16,height=16)
        for sigma,block,group in [(3,7,70),(20,7,70),(20+1e-9,8,90),(20.01,8,90),(40,8,90),(40+1e-9,8,120),(40.01,8,120),(60,8,120),(60+1e-9,9,140),(60.01,9,140),(100,9,140)]:
            _,props=values(core.nss.TWSC(source,sigma=sigma,iters=1,block_step=7,search_window=1),0)
            assert props['_NSSBlockSize']==block and props['_NSSGroupSize']==group,(sigma,props)
        calibration=dict(lambda_basic=.6,hard_strength=1.,wiener_iters=2,wiener_sigma_scale=.08)
        a,props=values(core.nss.NLH(source,sigma=3,block_size=4,block_step=4,group_size=8,q=2,search_window=5,basic_iters=1,**calibration),0)
        b,_=values(core.nss.NLH(source,sigma=3,block_size=[4,4],block_step=[4,4],group_size=[8,8],q=[2,2],bm_range=2,basic_iters=1,**calibration),0)
        np.testing.assert_array_equal(a[0],b[0])
        assert props['_NSSHardStrength']==1 and props['_NSSHardCoefficient']==NLH_HARD_COEFFICIENT
        assert props['_NSSWienerSigmaScale']==NLH_WIENER_SIGMA_SCALE
        _,props=values(core.nss.NLH(source,sigma=3,block_size=4,block_step=4,group_size=8,q=2,search_window=5,basic_iters=1,hard_strength=1.5),0)
        assert props['_NSSHardCoefficient']==NLH_HARD_COEFFICIENT*1.5
        presets=json.loads((Path(__file__).parent/'data/nlh_presets_v5.json').read_text())['profiles']
        for sigma,lane,noise_model in [(50,'gray-low','awgn'),(50+1e-9,'gray-high','awgn'),
                                       (50.01,'gray-high','awgn'),(3,'rgb','real')]:
            _,props=values(core.nss.NLH(source,sigma=sigma,noise_model=noise_model,search_window=1),0)
            expected=presets[lane]
            assert props['_NSSBlockSize']==expected['block_size']
            assert props['_NSSIterations']==[expected['basic_iters'],expected['wiener_iters']]
        # Omitted sigma is an estimate, not the old constant 3.
        _,props=values(core.nss.NLH(source,block_size=4,block_step=4,basic_iters=1,search_window=5),0)
        assert props['_NSSSigma']>0 and abs(props['_NSSSigma']-3)>1e-3
        _,props=values(core.nss.TWSC(source,estimate_sigma=1,block_size=4,block_step=4,iters=1,search_window=5),0)
        assert props['_NSSSigma']>0
        guide,guides=make_clip(core,vs.GRAYS,frames=1,width=16,height=16,seed=99)
        original,_=values(source,0)
        expected=float(noise_estimate(original[0],guides[0][0]))*255
        for name,extra in [('TWSC',dict(estimate_sigma=1,iters=1)),('NLH',dict(basic_iters=1))]:
            _,props=values(getattr(core.nss,name)(source,rclip=guide,block_size=4,block_step=4,search_window=5,**extra),0)
            np.testing.assert_allclose(props['_NSSSigma'],expected,rtol=1e-6,atol=1e-6)

    def invalid(self):
        core=self.core;source,_=make_clip(core,vs.GRAYS,frames=1,width=16,height=16)
        cases=[('TWSC',dict(lambda1=0)),('TWSC',dict(sigma=3,estimate_sigma=1)),('TWSC',dict(rho=0)),
               ('TWSC',dict(mu=.9)),('TWSC',dict(tol=0)),('TWSC',dict(group_size=257)),
               ('TWSC',dict(bm_range=2,search_window=6)),('NLH',dict(bm_range=2,search_window=[6,6])),
               ('NLH',dict(group_size=[16,17])),('NLH',dict(block_size=[4,4,4])),('NLH',dict(q=[2,3])),
               ('NLH',dict(block_size=2,q=8)),('NLH',dict(noise_model='unknown')),('NLH',dict(wiener_iters=0)),
               ('NLH',dict(lambda_basic=1.01)),('NLH',dict(search_window=[])),('NLH',dict(hard_strength=-1)),
               ('NLH',dict(hard_strength=float('inf'))),('NLH',dict(hard_strength=float('nan'))),
               ('NLH',dict(hard_tau=2)),('NLH',dict(hard_tau=30.5))]
        for name in ('TWSC','NLH'):
            cases.extend((name,dict(sigma=x)) for x in (-1,-1e-300,1e-300,float('nan'),float('inf')))
        for name,args in cases:
            try: values(getattr(core.nss,name)(source,**args),0)
            except (vs.Error,TypeError): pass
            else: raise AssertionError((name,args,'must fail'))
        for name in ('TWSC','NLH'):
            raw=getattr(core.nss,name)(source,sigma=0,**kwargs(name,1))
            incompatible=core.std.SetFrameProps(raw,_NSSModelVersion=4 if name=='NLH' else 2)
            try: values(core.nss.VAggregate(incompatible,source,radius=1),0)
            except vs.Error: pass
            else: raise AssertionError('old model contribution accepted')
        rgb,_=make_clip(core,vs.RGBS,frames=1,width=32,height=32)
        try: values(core.nss.TWSC(rgb,sigma=3,block_size=16,group_size=256,iters=1,memory_limit_mb=1),0)
        except vs.Error as e: assert 'memory' in str(e) or 'resource' in str(e),e
        else: raise AssertionError('memory budget did not reject oversized group')

    def extended(self):
        core=self.core
        source,_=make_clip(core,vs.GRAYS,frames=1,width=20,height=20)
        for name,args in [('TWSC',dict(block_size=4,block_step=4,group_size=256,iters=1,search_window=129)),
                          ('NLH',dict(block_size=[7,10],block_step=[7,10],group_size=[16,64],q=[4,16],basic_iters=1,search_window=[40,129]))]:
            result,props=values(getattr(core.nss,name)(source,sigma=3,**args),0)
            assert np.isfinite(result[0]).all()
        for name in ('TWSC','NLH'):
            zero,_=make_clip(core,vs.GRAYS,frames=1,constant=0)
            reference,_=make_clip(core,vs.GRAYS,frames=1,seed=88)
            result,_=values(getattr(core.nss,name)(zero,sigma=3,rclip=reference,**kwargs(name)),0)
            np.testing.assert_array_equal(result[0],0)

    def run(self):
        self.case('parameters',self.parameters);self.case('invalid_and_version',self.invalid);self.case('extended_shapes',self.extended)
        for name,radius,reference in itertools.product(('TWSC','NLH'),(0,1),(False,True)):
            label=f'oracle_{name}_r{radius}_ref{reference}'
            self.case(label,lambda name=name,radius=radius,reference=reference:self.oracle(name,radius,reference))
        for name,fmt,radius,reference in itertools.product(('TWSC','NLH'),(vs.GRAYS,vs.RGBS,vs.YUV444PS,vs.YUV422PS,vs.YUV420PS),(0,1,2),(False,True)):
            label=f'matrix_{name}_{fmt}_r{radius}_ref{reference}'
            self.case(label,lambda name=name,fmt=fmt,radius=radius,reference=reference:self.matrix(name,fmt,radius,reference))
        for name,fmt,radius in itertools.product(('TWSC','NLH'),(vs.GRAYS,vs.RGBS,vs.YUV420PS),(0,2)):
            label=f'zero_{name}_{fmt}_r{radius}'
            self.case(label,lambda name=name,fmt=fmt,radius=radius:self.zero(name,fmt,radius))
        report=dict(passed=all(r['passed'] for r in self.rows),cases=self.rows,platform=platform.platform(),numpy=np.__version__,vapoursynth=str(vs.__version__),
                    plugin_sha256=hashlib.sha256(self.plugin.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(sum(r['passed'] for r in self.rows),'/',len(self.rows),'passed',flush=True)
        return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);parser.add_argument('--plugin',default=os.environ.get('NSS_SO'))
    args=parser.parse_args()
    if not args.plugin: parser.error('--plugin or NSS_SO is required')
    raise SystemExit(0 if Suite(args.out,args.plugin).run() else 1)
