#!/usr/bin/env python3
"""Frozen-tree/current-tree output identity for selected unchanged models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from alignment_campaign import metrics,sha


def main(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(144)
    frames=np.array(rng.integers(32,96,(5,3,24,24))/128,np.float32)
    path=out/'input.npy';np.save(path,frames)
    rows=[]
    for model in args.models:
        captures=[]
        for name,plugin in [('baseline',args.baseline),('current',args.current)]:
            target=out/f'{model}-{name}.npy'
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('alignment_campaign.py')),'worker',
                '--plugin',str(Path(plugin).resolve()),'--model',model,'--input',str(path),
                '--output',str(target),'--params',json.dumps({} if model=='NLM' else dict(sigma=3))],
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True)
            (out/f'{model}-{name}.log').write_text(run.stdout+run.stderr)
            if run.returncode: raise RuntimeError(f'{model} {name} failed')
            captures.append(np.load(target))
        row=dict(model=model,**metrics(*captures));rows.append(row);print(model,row['exact'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(passed=all(r['exact'] for r in rows),rows=rows,
        baseline_sha256=sha(args.baseline),current_sha256=sha(args.current),script_sha256=sha(__file__)),indent=2)+'\n')
    return all(r['exact'] for r in rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True);p.add_argument('--current',required=True);p.add_argument('--out',required=True)
    p.add_argument('--models',nargs='+',default=['NLM','BM3D','WNNM','MCWNNM','NCSR','LSSC'])
    raise SystemExit(0 if main(p.parse_args()) else 1)
