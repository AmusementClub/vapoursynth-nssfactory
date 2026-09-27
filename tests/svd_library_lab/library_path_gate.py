#!/usr/bin/env python3
"""Exercise actual n=48 library calls under concurrent VS frame requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

def worker(a):
    import vapoursynth as vs
    core=vs.core;core.num_threads=a.threads;core.std.LoadPlugin(path=a.plugin)
    rng=np.random.default_rng(9320)
    y,x=np.mgrid[:32,:32]
    clean=np.stack([.3+.15*np.sin(x/3+p)+.1*np.cos(y/5-p) for p in range(3)]).astype(np.float32)
    sigma=[10,25,40] if a.unequal else [25,25,25]
    frames=[clean+rng.standard_normal(clean.shape).astype(np.float32)*np.asarray(sigma,np.float32)[:,None,None]/255 for _ in range(5)]
    blank=core.std.BlankClip(width=32,height=32,format=vs.RGBS,length=5)
    def fill(n,f):
        out=f.copy()
        for p in range(3):np.asarray(out[p])[:]=frames[n][p]
        return out
    source=core.std.ModifyFrame(blank,blank,fill)
    node=core.nss.TWSC(source,sigma=sigma,block_size=4,block_step=4,group_size=48,iters=2,admm_iter=5,radius=a.radius)
    if a.radius:node=core.nss.VAggregate(node,source,radius=a.radius)
    order=[4,1,3,0,2] if a.threads>1 else list(range(5))
    pending={n:node.get_frame_async(n) for n in order}
    pixels=np.stack([np.stack([np.array(pending[n].result()[p]) for p in range(3)]) for n in range(5)])
    assert np.isfinite(pixels).all()
    np.save(a.out,pixels)
    print(json.dumps(dict(sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),threads=a.threads,radius=a.radius,unequal=a.unequal)))

def main(a):
    root=Path(a.out);root.mkdir(exist_ok=False);rows=[]
    (root/'identity.json').write_text(json.dumps(dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plugin_sha256=hashlib.sha256(Path(a.plugin).read_bytes()).hexdigest()),indent=2)+'\n')
    for mode in ('baseline','tight','gesdd'):
        for radius in (0,1):
            for unequal in (False,True):
                previous=None
                for threads in (1,2,4):
                    name=f'{mode}-r{radius}-u{int(unequal)}-t{threads}'
                    cmd=[sys.executable,__file__,'worker','--plugin',a.plugin,'--out',str(root/(name+'.npy')),
                         '--threads',str(threads),'--radius',str(radius)]
                    if unequal:cmd.append('--unequal')
                    run=subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,NSS_SVD_LAB=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=120)
                    (root/(name+'.log')).write_text(run.stdout+run.stderr)
                    row=dict(name=name,returncode=run.returncode)
                    if run.returncode:raise RuntimeError(name+' failed: '+run.stderr)
                    result=json.loads(run.stdout.splitlines()[-1]);row.update(result)
                    statistics=[json.loads(line[len('SVD_LAB '):]) for line in run.stderr.splitlines() if line.startswith('SVD_LAB ')]
                    row['statistics']=statistics
                    if mode=='gesdd' and not any(s['calls']>0 for s in statistics):raise RuntimeError('library path was not exercised')
                    if previous is not None and previous!=result['sha256']:raise RuntimeError('thread/request-order mismatch '+name)
                    previous=result['sha256'];rows.append(row)
                    (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'complete.json').write_text(json.dumps(dict(passed=True,cases=len(rows)))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('--plugin',required=True);p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--radius',type=int,default=0);p.add_argument('--unequal',action='store_true')
    a=p.parse_args();worker(a) if a.action=='worker' else main(a)
