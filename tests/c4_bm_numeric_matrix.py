#!/usr/bin/env python3
"""Cache immutable baseline arrays, compare each candidate, preserve failures."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from c4_bm_matrix import configs
from bm_numerics import compare

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    rows=configs();here=Path(__file__).resolve().parent
    # Add new temporal boundary and maximum-radius cases to the retained matrix.
    rows += [dict(name=f'new_r{r}_{stage}',size=[19,17],frames=2,stage=stage,kwargs=dict(block_size=8,group_size=8,block_step=8,bm_range=1,sigma=3,radius=r,ps_range=1)) for r in [1,2,4,16] for stage in ['basic','wiener','two_stage']]
    rows += [dict(name=f'pairwise_{b}_{g}_{step}_{rad}_{stage}',size=[75,49],frames=2,stage=stage,kwargs=dict(block_size=b,group_size=g,block_step=step,bm_range=2,sigma=3,radius=rad,ps_range=1)) for b,g,step,rad in [(4,8,2,1),(8,16,4,2),(12,32,8,0),(16,8,2,1),(32,64,4,4)] for stage in ['basic','wiener','two_stage']]
    rows += [dict(name=f'minimum_step_b{b}_{stage}',size=[19,17],frames=2,stage=stage,
                  kwargs=dict(block_size=b,group_size=16,block_step=1,bm_range=2,
                              sigma=3,radius=2,ps_range=1))
             for b in (4,8) for stage in ('basic','wiener','two_stage')]
    token=hashlib.sha256((sha(a.baseline)+sha(here/'c4_paired_bm.py')+sha(here/'profile_cpu_all.py')+sha(here/'bm_numerics.py')+json.dumps(rows,sort_keys=True)).encode()).hexdigest()
    cache=Path(a.cache)/token;cache.mkdir(parents=True,exist_ok=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);results=[]
    manifest=dict(baseline=sha(a.baseline),candidate=sha(a.candidate),driver=sha(here/'c4_paired_bm.py'),cache_key=token,cases=len(rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for i,cfg in enumerate(rows):
        name=cfg['name'];base=cache/f'{i}.npy';cand=out/f'{i}.npy'
        for plugin,path in [(a.baseline,base),(a.candidate,cand)]:
            if path==base and path.exists():continue
            config=dict(cfg,_dump=str(path),warmup=1)
            proc=subprocess.run(['taskset','-c','0',sys.executable,str(here/'c4_paired_bm.py'),'worker','--plugin',plugin,'--config',json.dumps(config)],text=True,capture_output=True)
            if proc.returncode:raise RuntimeError(f'{name}: {proc.stderr}\n{proc.stdout}')
        report=compare(np.load(base),np.load(cand));results.append(dict(name=name,config=cfg,**report))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        if report['passed']:cand.unlink()
        else:
            import shutil
            shutil.copy2(base,out/f'{i}-baseline.npy')
            print(json.dumps(results[-1]),flush=True)
            if not a.keep_going:break
    passed=len(results)==len(rows) and all(r['passed'] for r in results)
    (out/'decision.json').write_text(json.dumps(dict(passed=passed,cases=len(results),expected=len(rows)),indent=2))
    print(json.dumps(dict(passed=passed,cases=len(results),expected=len(rows))),flush=True)
    return 0 if passed else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',default='/tmp/nss-numeric-cache');p.add_argument('--keep-going',action='store_true');sys.exit(run(p.parse_args()))
