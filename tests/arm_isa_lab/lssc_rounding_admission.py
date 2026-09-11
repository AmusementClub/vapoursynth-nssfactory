#!/usr/bin/env python3
"""Independent DCT-seed and OMP near-tie checks for retained compiler differences."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import psnr,ssim
p=argparse.ArgumentParser(description=__doc__);p.add_argument('captures',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
if np.finfo(np.longdouble).nmant <= 53:raise RuntimeError('requires genuinely extended long-double NumPy arithmetic')
prefix='nss-release-r3-lssc-';report=dict(passed=False,cases=[],policy='Existing DCT 2e-5 limit, independent long-double dots and IEEE FP64 gamma bounds; existing clean-quality loss .05 dB/.001. Original divergent arrays retained; no tolerance or production comparator changes.')
def directory(side,b,suffix=''):return a.captures/f'{prefix}{side}-b{b}{suffix}'
def array(side,b,n,kind='output',suffix=''):return np.fromfile(directory(side,b,suffix)/f'frame{n}-{kind}.f32',np.float32)
y,x=np.mgrid[:34,:36];clean=np.stack([((x*3+y*5+n*7)%97).astype(np.float32)/128 for n in range(3)])
for b in [4,8]:
 sides=['neon','portable-test'];m=b*b;traces={s:[json.loads(l) for l in (directory(s,b)/'trace.jsonl').read_text().splitlines()] for s in sides};steps={s:{(r['frame'],r['call'],r['step']):r for r in rows if r['kind']=='step'} for s,rows in traces.items()}
 assert steps[sides[0]].keys()==steps[sides[1]].keys()
 key=next(k for k in steps[sides[0]] if steps[sides[0]][k]['selected']!=steps[sides[1]][k]['selected']);entries=[steps[s][key] for s in sides];atoms=entries[0]['atoms'];D=[np.fromfile(directory(s,b)/f'call{key[1]}-D.f32',np.float32).reshape(atoms,m).astype(np.longdouble) for s in sides];res=[np.array(t['residual'],np.longdouble) for t in entries]
 u=np.longdouble(2)**-53;gamma=m*u/(1-m*u);correlations=[np.sum(d*r,axis=1,dtype=np.longdouble) for d,r in zip(D,res)];dotbounds=[gamma*np.sum(abs(d*r),axis=1,dtype=np.longdouble) for d,r in zip(D,res)]
 selections=[t['selected'] for t in entries];valid=[]
 for t,c,bounds in zip(entries,correlations,dotbounds):
  k=t['selected'];best=t['oracle_selected'];gap=abs(c[best])-abs(c[k]);valid.append(dict(selected=k,long_double_best=best,selection_gap=float(gap),dot_roundoff_bound=float(bounds[best]+bounds[k]),passed=bool(gap<=bounds[best]+bounds[k])))
 perturb=np.sum(abs(D[0]-D[1])*abs(res[0]),axis=1)+np.sum(abs(D[1])*abs(res[0]-res[1]),axis=1)+dotbounds[0]+dotbounds[1]
 i,j=selections;margin=abs(abs(correlations[0][i])-abs(correlations[0][j]));bound=perturb[i]+perturb[j]
 order=sorted((u+v,u+v*b) for v in range(b) for u in range(b));basis=[]
 for _,index in order[:min(atoms,m)]:
  v,u=divmod(index,b);basis.append(np.outer(np.cos(np.pi*(np.arange(b)+.5)*v/b),np.cos(np.pi*(np.arange(b)+.5)*u/b)).reshape(-1))
 basis=np.stack(basis);basis/=np.sqrt(np.sum(basis*basis,axis=1))[:,None]
 initial=[array(s,b,key[0],'initial-D').reshape(atoms,m) for s in sides];errors=[float(np.max(abs(d[:len(basis)]-basis))) for d in initial]
 assert all(np.array_equal(d.astype(np.float32),orig) for d,orig in zip(D,initial))
 frozen={}
 for stage in ['initial','final']:
  for ref in sides:
   left,right=[np.stack([array(s,b,n,suffix=f'-{stage}-{ref}') for n in range(3)]) for s in sides]
   frozen[stage+'-'+ref]=float(np.max(abs(left-right)))
 outputs={s:np.stack([array(s,b,n).reshape(34,36) for n in range(3)]) for s in sides};quality={s:dict(psnr=float(psnr(v,clean)),ssim=ssim(v,clean)) for s,v in outputs.items()};loss={k:quality['portable-test'][k]-quality['neon'][k] for k in ['psnr','ssim']}
 passed=all(v['passed'] for v in valid) and margin<=bound and max(errors)<=2e-5 and max(frozen.values())<=1e-5 and loss['psnr']<=.05 and loss['ssim']<=.001
 report['cases'].append(dict(block=b,public_index=659 if b==4 else 660,passed=bool(passed),first_divergence=dict(frame=key[0],call=key[1],step=key[2]),initial_dictionary_only_dct_atoms_differ=bool(np.array_equal(initial[0][m:],initial[1][m:])),initial_dct_max_error=errors,selection_checks=valid,rank_reversal=dict(margin=float(margin),perturbation_bound=float(bound)),original_max_abs=float(np.max(abs(outputs['neon']-outputs['portable-test']))),fixed_dictionary_max_abs=frozen,quality_against_exact_dyadic_source=quality,neon_quality_loss=loss,output_sha256={s:hashlib.sha256(v.tobytes()).hexdigest() for s,v in outputs.items()}))
report['passed']=all(r['passed'] for r in report['cases']);report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=report['passed'],cases=report['cases'])))
