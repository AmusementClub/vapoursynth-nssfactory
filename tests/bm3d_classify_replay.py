#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

def dct(cube):
    g,b,_=cube.shape
    def matrix(n):
        k=np.arange(n)[:,None];x=np.arange(n)[None,:]
        return np.cos(np.pi*(x+.5)*k/n)*np.sqrt(np.where(k==0,1.,2.)/n)
    return np.einsum('kg,yv,xu,gvu->kyx',matrix(g),matrix(b),matrix(b),cube,optimize=True)
def run(a):
    read=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s]
    old,new=read(a.baseline),read(a.candidate)
    if len(old)!=len(new) or not old:raise RuntimeError('replay group count mismatch/empty')
    reports=[]
    for x,y in zip(old,new):
        if x['query']!=y['query'] or x['matches']!=y['matches'] or x['input']!=y['input']:raise RuntimeError('matching/input is the first divergence')
        shape=(x['group'],x['block'],x['block'])
        # The probe's nine significant digits round-trip float32. Restore
        # those exact payloads before evaluating the independent double DCT.
        restore=lambda values:np.asarray(values,dtype=np.float32).astype(np.float64).reshape(shape)
        coeff=dct(restore(x['input']))
        before=dct(restore(x['output']));after=dct(restore(y['output']))
        # Recover hard-threshold masks from inverse outputs using a double DCT.
        m0=np.abs(before)>2e-6;m1=np.abs(after)>2e-6;m0[0,0,0]=m1[0,0,0]=True
        changed=m0!=m1;threshold=float(np.float32(2.7)*np.float32(x['sigma_eff']))
        margins=np.abs(np.abs(coeff[changed])-threshold)
        passed=not changed.any() or float(margins.max())<=2e-6
        # Outside the mask transition, both transform reconstructions must remain near the same coefficient.
        stable=m0&m1;continuous=float(np.max(np.abs(before[stable]-after[stable]))) if stable.any() else 0.
        passed=passed and continuous<=2e-5
        reports.append(dict(query=x['query'],changed_masks=int(changed.sum()),max_threshold_margin=float(margins.max()) if len(margins) else 0.,continuous_coefficient_delta=continuous,baseline_weight=x['weight'],candidate_weight=y['weight'],passed=passed))
    result=dict(passed=all(x['passed'] for x in reports),classification='threshold_boundary_rounding' if any(x['changed_masks'] for x in reports) else 'continuous_rounding',groups=reports)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);run(p.parse_args())
