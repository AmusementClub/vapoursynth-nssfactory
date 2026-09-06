#!/usr/bin/env python3
"""Fixed-match group replay of a Basic BM3D DCT threshold deviation."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from bm_numerics import psnr,ssim,compare

def run(a):
 a.out.mkdir(parents=True,exist_ok=False)
 if a.screen:
  row=json.loads((a.screen/'summary.json').read_text())[a.index];cfg=row['config']
  arrays={side:np.load(a.screen/f'{a.index}-{side}.npy')[0,0] for side in ('baseline','candidate')}
 else:
  cfg=json.loads(a.case_config.read_text());cfg['stage']='basic'
  arrays={side:np.load(getattr(a,side+'_output')) for side in ('baseline','candidate')}
  row=dict(config=cfg,numerical=compare(arrays['baseline'][None,None],arrays['candidate'][None,None]))
 kw=cfg['kwargs']
 if cfg['stage']!='basic' or kw['radius']!=0:raise ValueError('Basic spatial replay only')
 w,h=cfg['size'];b=kw['block_size'];g=kw['group_size'];step=kw['block_step']
 clean=np.fromfile(cfg['sample'],dtype=np.uint8).reshape(h,w).astype(np.float32)*np.float32(1/255)
 source=clean+np.random.RandomState(42).randn(h,w).astype(np.float32)*np.float32(3/255)
 input_file=a.out/'source.f32';source.tofile(input_file)
 py,px=row['numerical']['max_index'][-2:]
 sigma=float(np.float32(np.float32(kw['sigma'])*np.float32(1/255))*np.float32(.75))
 logs=[]
 for side in ('baseline','candidate'):
  log=a.out/(side+'.jsonl');logs.append(log)
  cmd=['taskset','-c',str(a.cpu),str(getattr(a,side)),str(input_file),str(w),str(h),str(b),str(g),str(step),str(sigma),str(px),str(py)]
  with log.open('w') as out:subprocess.run(cmd,stdout=out,check=True)
 classification=a.out/'classification.json'
 subprocess.run([sys.executable,str(Path(__file__).with_name('bm3d_classify_replay.py')),'--baseline',str(logs[0]),'--candidate',str(logs[1]),'--out',str(classification)],check=True,stdout=subprocess.DEVNULL)
 details=json.loads(classification.read_text())
 reproduced={}
 for side,log in zip(('baseline','candidate'),logs):
  num=np.float32(0);den=np.float32(0)
  for group in [json.loads(line) for line in log.read_text().splitlines()]:
   weight=np.float32(group['weight']);patches=np.asarray(group['output'],np.float32).reshape(g,b,b)
   for j,(x,y,_) in enumerate(group['matches']):
    if x<=px<x+b and y<=py<y+b:
     num=np.float32(float(weight)*float(patches[j,py-y,px-x])+float(num));den=np.float32(den+weight)
  predicted=float(num/den);observed=float(arrays[side][py,px])
  reproduced[side]=dict(predicted=predicted,observed=observed,error=abs(predicted-observed),passed=abs(predicted-observed)<=2e-6)
 quality={s:dict(psnr=psnr(x,clean),ssim=ssim(x,clean)) for s,x in arrays.items()}
 loss={m:quality['baseline'][m]-quality['candidate'][m] for m in ('psnr','ssim')}
 result=dict(passed=details['passed'] and all(x['passed'] for x in reproduced.values()) and loss['psnr']<=.05 and loss['ssim']<=.001,
   reproduced_max_pixel=reproduced,
   classification=details['classification'],groups=len(details['groups']),config=cfg,quality=quality,loss=loss,original=row['numerical'],
   output_hashes={side:hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for side,array in arrays.items()},
   hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.baseline,a.candidate,input_file,*logs]},
   scope='Exact matches and packed inputs; independent double 3D DCT around the retained maximum-error pixel.')
 (a.out/'replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('out','baseline','candidate'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--screen',type=Path)
 for n in ('case-config','baseline-output','candidate-output'):p.add_argument('--'+n,type=Path)
 p.add_argument('--index',type=int,default=0);p.add_argument('--cpu',type=int,default=0);sys.exit(run(p.parse_args()))
