#!/usr/bin/env python3
"""Independent double SSD checks for same-pilot BM3D matching divergences."""
import argparse,json
from pathlib import Path
import numpy as np
from match_replay import records
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('replay',type=Path);p.add_argument('crossover',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
prime=1099511628211;mask=(1<<64)-1;u=float(np.finfo(np.float32).eps)/2

def fnv(array):
 h=14695981039346656037
 for word in array.astype(np.float32).reshape(-1).view(np.uint32):h=((h^int(word))*prime)&mask
 return h
report=dict(passed=False,cases=[])
for directory in sorted(a.replay.glob('c*')):
 case=json.loads((a.crossover/directory.name/'case.json').read_text());b=case['block'];radius=0 if case['mode']=='spatial' else 1
 for ref in ['baseline','candidate']:
  data=np.load(a.crossover/directory.name/(ref+'-pilot.npz'));frames=[data[f'n{n}'] for n in range(3)];hashed=[fnv(x) for x in frames];mapping={}
  for n in range(3):
   # Host reference slots are clamped at clip boundaries. Candidate validity
   # is enforced separately; the capture hashes every physical slot.
   ids=[max(0,min(2,n+t)) for t in range(-radius,radius+1)]
   if radius:
    h=0
    for i in ids:h=((h*prime)^hashed[i])&mask
   else:h=hashed[n]
   mapping[(2 if radius else 1,h)]=(n,ids)
   # Predictive matching also records its nested spatial seed searches.
   mapping[(1,hashed[n])]=(n,[n])
  sides=[records(directory/(side+'-ref-'+ref+'.bin')) for side in ['baseline','candidate']]
  if len(sides[0])!=len(sides[1]):raise RuntimeError('trace counts differ')
  checks=[]
  for call,(left,right) in enumerate(zip(*sides)):
   ha,ia,pa=left;hb,ib,pb=right
   if ia!=ib:raise RuntimeError('pilot inputs differ')
   if ha[:7]!=hb[:7]:raise RuntimeError('query geometry differs')
   xa=pa.reshape(-1,5);xb=pb.reshape(-1,5)
   ca=[tuple(int(t) for t in v[:3]) for v in xa];cb=[tuple(int(t) for t in v[:3]) for v in xb]
   if ca==cb:continue
   n,ids=mapping[(ha[0],ia)];x,y=ha[1:3];query=frames[n][y:y+b,x:x+b].astype(float)
   distances={};errors=[]
   gamma=(4*b*b+8)*u/(1-(4*b*b+8)*u)
   for coords,values in [(ca,xa),(cb,xb)]:
    for key,v in zip(coords,values):
     xx,yy,t=key;frame=ids[t] if radius else n
     exact=float(np.sum((query-frames[frame][yy:yy+b,xx:xx+b].astype(float))**2))
     score=float(np.asarray([v[4]],np.uint32).view(np.float32)[0]);bound=gamma*abs(exact)+1e-30
     distances[key]=(exact,bound);errors.append(abs(score-exact)/bound)
   swaps=[]
   for first in set(ca)|set(cb):
    for second in set(ca)|set(cb):
     if first>=second:continue
     ai=ca.index(first) if first in ca else len(ca);aj=ca.index(second) if second in ca else len(ca)
     bi=cb.index(first) if first in cb else len(cb);bj=cb.index(second) if second in cb else len(cb)
     if (ai-aj)*(bi-bj)>=0:continue
     d0,e0=distances[first];d1,e1=distances[second]
     swaps.append(dict(candidates=[first,second],double_margin=abs(d0-d1),roundoff_bound=e0+e1,ratio=abs(d0-d1)/(e0+e1)))
   checks.append(dict(call=call,max_score_error_ratio=max(errors),reversed_pairs=swaps,passed=max(errors)<=1 and all(s['ratio']<=1 for s in swaps)))
  report['cases'].append(dict(case=directory.name,reference=ref,changed_calls=len(checks),checks=checks,passed=all(c['passed'] for c in checks)))
report['passed']=all(c['passed'] for c in report['cases']);report['policy']='Independent double SSD; conservative gamma_(4*b*b+8) relative FP32 roundoff for subtraction, squares, and sum. Reversed pair separation must fit both score error bounds.'
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),changed_calls=sum(c['changed_calls'] for c in report['cases']))))
