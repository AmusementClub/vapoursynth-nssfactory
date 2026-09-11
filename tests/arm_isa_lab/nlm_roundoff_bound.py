#!/usr/bin/env python3
"""Check FP32 rolling-sum roundoff and FastExp against captured NLM stages."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('captures',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
u=float(np.finfo(np.float32).eps)/2;radius=4;rows=[]
for case in sorted(a.captures.glob('c*-n*')):
 for side in ['baseline','candidate']:
  folder=case/side;scale=float(np.fromfile(folder/'scale.f32',np.float32)[0])
  for index in range(12):
   horizontal=np.fromfile(folder/f'h{index}.f32',np.float32).reshape(128,192)
   captured=np.fromfile(folder/f'w{index}.f32',np.float32).reshape(horizontal.shape).astype(float)
   h,w=horizontal.shape;values=horizontal.astype(float)
   actual=(horizontal[0]*np.float32(radius)).astype(np.float32)
   total=radius*values[0];absolute=abs(total);operations=1
   for y in range(radius):
    actual=(actual+horizontal[min(y,h-1)]).astype(np.float32)
    total=total+values[min(y,h-1)];absolute+=abs(values[min(y,h-1)]);operations+=1
   sums=[];bounds=[];expected=[]
   for y in range(h):
    add=min(y+radius,h-1);sub=max(y-radius,0)
    actual_sum=(actual+horizontal[add]).astype(np.float32)
    total=total+values[add];absolute+=abs(values[add]);operations+=1
    # gamma_n sum(abs(terms)) bounds arbitrary rounding order of this
    # add/sub history; this is not a fitted absolute image tolerance.
    bound=(operations*u)/(1-operations*u)*absolute+operations*float(np.nextafter(np.float32(0),np.float32(1)))
    sums.append(actual_sum.astype(float));bounds.append(bound);expected.append(total.copy())
    actual=(actual_sum-horizontal[sub]).astype(np.float32)
    total=total-values[sub];absolute+=abs(values[sub]);operations+=1
   sums=np.stack(sums);expected=np.stack(expected);bounds=np.stack(bounds)
   error=abs(sums-expected);ratio=float(np.max(error/np.maximum(bounds,1e-45)))
   exponent=(sums.astype(np.float32)*np.float32(-scale)).astype(np.float32).astype(float)
   exp_error=float(np.max(abs(captured-np.exp(exponent))))
   rows.append(dict(case=case.name,side=side,map=index,roundoff_bound_ratio=ratio,max_sum_error=float(error.max()),fast_exp_error=exp_error,passed=ratio<=1 and exp_error<=8e-6))
result=dict(passed=bool(rows) and all(r['passed'] for r in rows),unit_roundoff=u,
 policy='IEEE FP32 gamma_n sum(abs(terms)) for rolling sums; existing 8e-6 NLM weight-unit error bound for FastExp after the reproduced FP32 sum.',
 maps=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],maps=len(rows),max_roundoff_ratio=max(r['roundoff_bound_ratio'] for r in rows),max_fast_exp_error=max(r['fast_exp_error'] for r in rows))))
