#!/usr/bin/env python3
"""Frozen BM3DCPU b8/g8 spatial reference on the existing clean/noisy pack."""
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import vapoursynth as vs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import psnr,ssim
p=argparse.ArgumentParser(description=__doc__)
for key in ['plugin','pack','out']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
metadata=json.loads((a.pack/'manifest.json').read_text());packfile=a.pack/'fixtures.npz'
assert hashlib.sha256(packfile.read_bytes()).hexdigest()==metadata['data_sha256']
data=np.load(packfile);core=vs.core;core.num_threads=1
core.std.LoadPlugin(path=str(a.plugin.resolve()))
report=dict(passed=False,fixture_sha256=metadata['data_sha256'],plugin_sha256=hashlib.sha256(a.plugin.read_bytes()).hexdigest(),
            reference_commit='e869cfae8d2322cf7a2b3c8056ed57e9308b2d33',
            scope='Spatial b8/g8 grayscale BM3D two-stage; radius0 only; original sigma with frozen reference noise profile.',cases=[])
outputs={}
for index,case in enumerate(metadata['cases']):
 if case['algorithm']!='BM3D' or case['radius']:continue
 values=data[f'{case["fixture"]}_s{case["sigma"]:g}_gray'];clean=data[case['fixture']+'_clean_gray']
 length,_,h,w=values.shape
 blank=core.std.BlankClip(width=w,height=h,length=length,format=vs.GRAYS)
 def fill(n,f,values=values):
  out=f.copy();np.asarray(out[0])[:]=values[n,0];return out
 source=core.std.ModifyFrame(blank,blank,fill)
 kw=dict(sigma=case['sigma'],block_step=8,bm_range=7,radius=0,chroma=False,ps_num=2,ps_range=4)
 pilot=core.bm3dcpu.BM3D(source,**kw);node=core.bm3dcpu.BM3D(source,ref=pilot,**kw)
 actual=np.stack([np.array(node.get_frame(n)[0])[None] for n in [0,2,4]])
 if not np.isfinite(actual).all():raise RuntimeError('nonfinite reference output')
 outputs[f'c{index}']=actual
 report['cases'].append(dict(index=index,parameters=case,reference_kwargs=kw,metrics=dict(psnr_against_clean=psnr(actual,clean[[0,2,4]]),ssim_against_clean=ssim(actual,clean[[0,2,4]])),output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
np.savez_compressed(a.out/'pixels.npz',**outputs)
report.update(passed=bool(outputs),pixels_sha256=hashlib.sha256((a.out/'pixels.npz').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=report['passed'],cases=len(outputs))))
