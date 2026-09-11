#!/usr/bin/env python3
"""Independent FP64 ADMM/SVD and TWSC reconstruction for actual plugin groups."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from group_oracle import read, ncsr_oracle


def reconstruction(actual, expected):
    error=np.abs(actual.astype(float)-expected)
    ratio=float(np.max(error/(2e-5+2e-4*np.abs(expected))))
    return dict(max_abs=float(error.max()),tolerance_ratio=ratio,passed=ratio<=1)


def mc_oracle(row):
    original=row['input'].astype(float).T;m,n=original.shape
    mean=original.mean(axis=1,keepdims=True) if row['residual'] else np.zeros((m,1))
    y=original-mean;z=np.zeros_like(y);dual=np.zeros_like(y)
    sigmas=row['sigmas'].astype(float);smin=min(s for s in sigmas if s>0)
    safe=np.where(sigmas>0,sigmas,smin*1e-6)
    weights=np.repeat((smin/safe)**2,m//row['nch'])[:,None]
    rho=row['rho'];c=8*np.sqrt(2*n)*smin*smin
    minimum_margin=float('inf');kept=0
    for _ in range(row['iters']):
        x=(weights*y+rho*.5*(z-dual/rho))/(weights+rho*.5)
        temp=x+dual/rho
        u,s,vt=np.linalg.svd(temp,full_matrices=False)
        threshold=c*2/rho
        start=0 if row['residual'] else 1
        minimum_margin=min(minimum_margin,float(np.min(np.abs(s[start:]**2-threshold))))
        shrunk=s.copy()
        shrunk[start:]=np.where(s[start:]**2>threshold,
            (s[start:]+np.sqrt(np.maximum(s[start:]**2-threshold,0)))*.5,0)
        kept=int(np.count_nonzero(shrunk))
        z=(u*shrunk)@vt
        dual+=rho*(x-z)
        rho=min(1e4,row['mu']*rho)
    result=reconstruction(row['output'].T,z+mean)
    result.update(minimum_squared_singular_threshold_margin=minimum_margin,oracle_kept=kept)
    if row['adaptive']:
        result['weight_error']=abs(row['weight']-(1/kept if kept else 1))
        result['passed'] &= result['weight_error']<=1e-7
    return result


def twsc_oracle(row):
    original=row['input'].astype(float).T
    weights=row['row_weight'].astype(float)[:,None]
    x=original*weights;mean=x.mean(axis=1,keepdims=True);centered=x-mean
    u=row['u'].astype(float);b=u.T@centered
    singular=np.linalg.svd(centered,compute_uv=False)
    projection=float(np.linalg.norm(u@b-centered)/max(np.linalg.norm(centered),1e-30))
    # S is the stored post-noise-subtraction spectrum. Validate against the
    # independent SVD before using that measured state to isolate finishing.
    noise=x.shape[1]*float(row['reference'][0])**2
    expected_s=np.sqrt(np.maximum(singular**2-noise,0))
    spectrum=float(np.linalg.norm(row['s']-expected_s)/max(np.linalg.norm(singular),1e-30))
    threshold=row['reference'].astype(float)[None,:]**2/(row['s'].astype(float)[:,None]+float(np.float32(1e-6)))
    codes=np.sign(b)*np.maximum(np.abs(b)-threshold,0)
    divisor=np.copysign(np.maximum(np.abs(weights),1e-12),weights)
    result=reconstruction(row['output'].T,(u@codes+mean)/divisor)
    result.update(projection_relative_error=projection,post_shrink_spectrum_relative_error=spectrum)
    result['passed'] &= projection<=2e-4 and spectrum<=2e-5
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();groups=read(a.capture);results=[]
    for row in groups:
        oracle={2:ncsr_oracle,3:mc_oracle,4:twsc_oracle,5:ncsr_oracle}[row['kind']]
        results.append(dict(kind=row['kind'],batch=row['batch'],index=row['index'],**oracle(row)))
    report=dict(passed=bool(results) and all(r['passed'] for r in results),groups=results,
                policy='Existing group reconstruction 2e-5 + 2e-4 abs(reference); spectrum 2e-5; projection 2e-4.',
                hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.capture,Path(__file__),Path(__file__).with_name('group_oracle.py')]})
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report['passed'],groups=len(results),failures=sum(not r['passed'] for r in results))))
