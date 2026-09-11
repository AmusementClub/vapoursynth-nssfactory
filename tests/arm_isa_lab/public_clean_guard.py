#!/usr/bin/env python3
"""Against-source guard on the exact deterministic public matrix fixtures."""
import argparse,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import ssim
p=argparse.ArgumentParser(description=__doc__)
for name in ['baseline','candidate','out']:p.add_argument(name,type=Path)
a=p.parse_args();folders=[a.baseline,a.candidate];meta=[json.loads((p/'summary.json').read_text()) for p in folders]
assert all(d['passed'] for d in meta) and meta[0]['fixture_sha256']==meta[1]['fixture_sha256']
arrays=[]
for folder,d in zip(folders,meta):
 path=folder/'pixels.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==d['pixels_sha256'];arrays.append(np.load(path))
report=dict(passed=False,scope='Noise-free dyadic public fixtures, compared with their exact source. Separate noisy-fixture and external-reference quality reports are required.',cases=[])
for case in meta[0]['cases']:
 index=case['index'];keys=[k for k in arrays[0].files if k.startswith(f'c{index}_')];quality=[]
 for values in arrays:
  squared=0.;count=0;similarity=0.
  for key in keys:
   _,frame,plane=key.split('_');frame=int(frame[1:]);plane=int(plane[1:]);v=values[key];y,x=np.indices(v.shape);clean=((x*3+y*5+frame*7+plane*11)%97).astype(np.float32)/np.float32(128)
   squared+=float(np.sum((v.astype(float)-clean)**2));count+=v.size;similarity+=ssim(v,clean)*v.size
  quality.append(dict(psnr=-10*math.log10(squared/count) if squared else None,ssim=similarity/count,mse=squared/count))
 psnr_loss=quality[0]['psnr']-quality[1]['psnr'] if quality[0]['psnr'] is not None and quality[1]['psnr'] is not None else 0 if quality[0]['mse']==quality[1]['mse'] else float('inf')
 loss=dict(psnr=psnr_loss,ssim=quality[0]['ssim']-quality[1]['ssim']);report['cases'].append(dict(index=index,parameters=case['parameters'],quality=quality,loss=loss,passed=loss['psnr']<=.05 and loss['ssim']<=.001))
report.update(passed=all(v['passed'] for v in report['cases']),max_psnr_loss=max(v['loss']['psnr'] for v in report['cases']),max_ssim_loss=max(v['loss']['ssim'] for v in report['cases']),plugins=[d['plugin_sha256'] for d in meta],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
