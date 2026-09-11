#!/usr/bin/env python3
"""Compare identical public fixtures without promoting triage to admission."""
import argparse,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare
p=argparse.ArgumentParser(description=__doc__)
for k in ['left','right','out']:p.add_argument(k,type=Path)
a=p.parse_args();metadata=[json.loads((q/'summary.json').read_text()) for q in [a.left,a.right]]
if not all(q['passed'] for q in metadata) or metadata[0]['fixture_sha256']!=metadata[1]['fixture_sha256']:raise RuntimeError('fixture mismatch or gate failure')
pixels=[np.load(q/'pixels.npz') for q in [a.left,a.right]]
if set(pixels[0].files)!=set(pixels[1].files):raise RuntimeError('pixel keys differ')
if [q['parameters'] for q in metadata[0]['cases']] != [q['parameters'] for q in metadata[1]['cases']]:raise RuntimeError('parameters differ')
report=dict(numerical_admission=False,fixture_sha256=metadata[0]['fixture_sha256'],plugins=[q['plugin_sha256'] for q in metadata],results=[])
for row in metadata[0]['cases']:
 index=row['index'];keys=sorted(k for k in pixels[0].files if k.startswith(f'c{index}_'))
 if not keys:raise RuntimeError('missing case pixels')
 # Chroma planes can have different dimensions; preserve spatial axes for
 # each SSIM call and aggregate squared error by the actual sample count.
 parts=[(k,compare(pixels[0][k],pixels[1][k]),pixels[0][k].size) for k in keys]
 count=sum(n for _,_,n in parts)
 mse=sum(v['rmse']**2*n for _,v,n in parts)/count
 key,peak,_=max(parts,key=lambda q:q[1]['max_abs'])
 maximum=peak['max_abs'];rmse=math.sqrt(mse);passed=maximum<=1e-5 and rmse<=1e-6
 result=dict(passed=passed,reason='within_triage_band' if passed else 'requires_stage_replay',max_abs=maximum,rmse=rmse,
             max_key=key,max_index=peak['max_index'],psnr_between=-10*math.log10(mse) if mse else None,
             ssim_between=sum(v['ssim_between']*n for _,v,n in parts)/count)
 report['results'].append(dict(index=index,parameters=row['parameters'],byte_exact=all(np.array_equal(pixels[0][k],pixels[1][k]) for k in keys),mse=mse,**result))
report.update(cases=len(report['results']),byte_exact=sum(q['byte_exact'] for q in report['results']),requires_stage_replay=sum(not q['passed'] for q in report['results']),pixel_files={str(q/'pixels.npz'):hashlib.sha256((q/'pixels.npz').read_bytes()).hexdigest() for q in [a.left,a.right]})
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['cases','byte_exact','requires_stage_replay']}))
