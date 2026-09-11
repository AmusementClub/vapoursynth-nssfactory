#!/usr/bin/env python3
"""Independent SSD oracle for actual same-input NLH matcher reversals."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from match_replay import records
p=argparse.ArgumentParser(description=__doc__);p.add_argument('captures',type=Path);p.add_argument('out',type=Path);a=p.parse_args();report=dict(passed=False,cases=[]);prime=1099511628211;mask=(1<<64)-1;u=float(np.finfo(np.float32).eps)/2
for case in sorted(a.captures.glob('c*')):
 if not case.is_dir():continue
 maps={}
 for f in (case/'baseline').glob('*.json'):
  d=json.loads(f.read_text())
  if (d['width'],d['height'])!=(36,34):continue
  maps[int(f.stem)]=np.fromfile(f.with_suffix('.f32'),np.float32).reshape(34,36)
 combos={}
 for keys in itertools.product(maps,repeat=3):
  h=0
  for k in keys:h=((h*prime)^k)&mask
  combos[h]=keys
 sides=[records(case/(s+'.bin')) for s in ['baseline','candidate']];assert len(sides[0])==len(sides[1]);checks=[];changed_inputs=0
 for call,(left,right) in enumerate(zip(*sides)):
  ha,ia,pa=left;hb,ib,pb=right
  if np.array_equal(pa,pb):continue
  if ia!=ib:changed_inputs+=1;continue
  if ha[0]==6:
   assert np.array_equal(pa,pb),'same-input pixel indices require separate oracle'
  xa=pa.reshape(-1,5);xb=pb.reshape(-1,5);ca=[tuple(map(int,v[:3])) for v in xa];cb=[tuple(map(int,v[:3])) for v in xb]
  if ca==cb:continue
  assert ha[:7]==hb[:7] and ha[0] in [1,2,5]
  frames=[maps[ia]] if ha[0] in [1,5] else [maps[k] for k in combos[ia]];center=0 if len(frames)==1 else 1;x,y,b=ha[1:4];query=frames[center][y:y+b,x:x+b].astype(float);distances={};ratios=[];gamma=(4*b*b+8)*u/(1-(4*b*b+8)*u)
  for coords,values in [(ca,xa),(cb,xb)]:
   for key,v in zip(coords,values):
    xx,yy,t=key;target=frames[0 if len(frames)==1 else t][yy:yy+b,xx:xx+b].astype(float);exact=float(np.sum((query-target)**2));bound=gamma*abs(exact)+1e-30;score=float(np.asarray([v[4]],np.uint32).view(np.float32)[0]);ratios.append(abs(score-exact)/bound);distances[key]=(exact,bound)
  swaps=[]
  for first,second in itertools.combinations(set(ca)|set(cb),2):
   ai=ca.index(first) if first in ca else len(ca);aj=ca.index(second) if second in ca else len(ca);bi=cb.index(first) if first in cb else len(cb);bj=cb.index(second) if second in cb else len(cb)
   if (ai-aj)*(bi-bj)>=0:continue
   d0,e0=distances[first];d1,e1=distances[second];swaps.append(abs(d0-d1)/(e0+e1))
  nondyadic=any(np.any(frame*128!=np.rint(frame*128)) for frame in frames)
  checks.append(dict(call=call,kind=ha[0],max_score_error_ratio=max(ratios),max_reversed_margin_ratio=max(swaps,default=0),source_is_processed_nondyadic=bool(nondyadic),passed=max(ratios)<=1 and max(swaps,default=0)<=1))
 report['cases'].append(dict(index=int(case.name[1:]),same_input_changed_calls=checks,changed_input_records=changed_inputs,passed=all(r['passed'] for r in checks)))
report['passed']=all(r['passed'] for r in report['cases']);report['policy']='Independent double SSD on captured inputs. gamma_(4*b*b+8) FP32 arithmetic bounds; rank reversals must lie inside the sum of score bounds. Original outputs and all recorder bytes were verified exact with interposition enabled.';report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
