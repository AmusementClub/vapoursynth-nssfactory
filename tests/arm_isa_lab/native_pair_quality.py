#!/usr/bin/env python3
"""Against-clean quality of the actual saved same-host performance outputs."""
import argparse,hashlib,json,random
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import psnr,ssim
p=argparse.ArgumentParser(description=__doc__);p.add_argument('pairs',type=Path);p.add_argument('out',type=Path);p.add_argument('--sample',type=Path);a=p.parse_args()
d=json.loads((a.pairs/'summary.json').read_text());assert d['complete']
report=dict(passed=False,numerical_admission=False,cases=[],policy=dict(max_psnr_loss_db=.05,max_ssim_loss=.001))
for row in d['results']:
 c=row['config'];w,h=c['size'];raw=np.fromfile(a.sample or c['sample'],np.uint8).reshape(1080,1920)
 base=raw[np.arange(h)*1080//h][:,np.arange(w)*1920//w].astype(np.float32)/np.float32(255)
 values={s:np.load(a.pairs/c['name']/(s+'.npy')) for s in ['baseline','candidate']}
 metadata=row['pairs'][0]['baseline'];first=metadata['timed_first'];order=list(range(first,first+metadata['timed_frames']))
 first_clean=base[:,np.clip(np.arange(w)-(first%7-3)*2,0,w-1)] if c.get('motion') else base
 rng=np.random.RandomState(42+first if c.get('motion') else 42)
 payload=np.stack([first_clean+rng.randn(h,w).astype(np.float32)*np.float32(3/255) for _ in range(3 if c['algorithm']=='mcwnnm' else 1)])
 if hashlib.sha256(payload.tobytes()).hexdigest()!=metadata['input_sha256']:raise RuntimeError('source payload does not reproduce measured input')
 if c.get('access')=='random':random.Random(42).shuffle(order)
 shape=values['baseline'].shape;assert shape==values['candidate'].shape
 clean=np.stack([np.stack([base[:,np.clip(np.arange(w)-(n%7-3)*2,0,w-1)] if c.get('motion') else base]*shape[1]) for n in order[:shape[0]]])
 quality={s:dict(psnr=psnr(x,clean),ssim=ssim(x,clean)) for s,x in values.items()}
 losses={m:quality['baseline'][m]-quality['candidate'][m] for m in ['psnr','ssim']}
 passed=losses['psnr']<=.05 and losses['ssim']<=.001
 report['cases'].append(dict(name=c['name'],passed=passed,quality=quality,loss=losses,frames=order[:shape[0]],
     output_hashes={s:hashlib.sha256(v.tobytes()).hexdigest() for s,v in values.items()}))
report['passed']=all(x['passed'] for x in report['cases']);report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),max_psnr_loss=max(r['loss']['psnr'] for r in report['cases']),max_ssim_loss=max(r['loss']['ssim'] for r in report['cases']))))
