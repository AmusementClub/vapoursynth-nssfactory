#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Real captured groups, known-spectrum controls, independent factor checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from campaign import ticks, environment, qualified

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_factors(path, matrices, m, n):
    r=min(m,n);raw=np.fromfile(path,'<f8');offset=0;result=[]
    for a in matrices:
        u=raw[offset:offset+m*r].reshape((m,r),order='F');offset+=m*r
        s=raw[offset:offset+r];offset+=r
        v=raw[offset:offset+r*n].reshape((r,n),order='F');offset+=r*n
        mask=s>0;active=int(mask.sum())
        reconstruction=np.linalg.norm((u*s)@v-a)/max(np.linalg.norm(a),1e-300)
        orth=max(np.max(np.abs(u[:,mask].T@u[:,mask]-np.eye(active)),initial=0),
                 np.max(np.abs(v[mask]@v[mask].T-np.eye(active)),initial=0))
        exact_s=np.linalg.svd(a.astype(np.float64),compute_uv=False)
        spectral=float(np.max(np.abs(s-exact_s))/max(float(exact_s[0]),1e-300))
        result.append(dict(relative_reconstruction=float(reconstruction),orthogonality=float(orth),spectrum_normalized_max_abs=spectral,rank=active))
        if not np.isfinite(raw).all() or np.any(s<0) or np.any(np.diff(s)>0) or reconstruction>5e-5 or orth>5e-4 or spectral>5e-5:
            raise RuntimeError(f'independent factor check failed: {result[-1]}')
    if offset!=raw.size:raise RuntimeError('factor output length')
    return result

def main(a):
    root=Path(a.out);root.mkdir(exist_ok=False);cases=[]
    (root/'policy.json').write_text(json.dumps(dict(driver_sha256=sha(__file__),bench_sha256=sha(a.bench),
        environment_helper_sha256=sha(Path(__file__).with_name('campaign.py')),pairs=7,
        modes=['baseline','tight','gesdd','gesvd'],idle_min=.99,steal_max=0,
        interpretation='fixed-FP64 replay; coding uses uniform nominal noise; failed environment windows remain unqualified'),indent=2)+'\n')
    assert json.loads((Path(a.frames)/'complete.json').read_text())['completed']
    # A separate environment epoch, after all primary frame timing is complete.
    # Earlier failed windows remain failed; this does not repair their evidence.
    commands=[['systemctl','status','snapd.service','snapd.socket','--no-pager'],
              ['journalctl','-u','snapd.service','--since','2 hours ago','--no-pager'],
              ['sudo','systemctl','mask','--runtime','--now','snapd.service','snapd.socket'],
              ['systemctl','is-active','snapd.service','snapd.socket']]
    quiet=[]
    for command in commands:
        p=subprocess.run(command,capture_output=True,text=True,timeout=30)
        quiet.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    (root/'quiet-epoch.json').write_text(json.dumps(quiet,indent=2)+'\n')
    if quiet[2]['returncode'] or 'active' in quiet[3]['stdout'].splitlines():raise RuntimeError('background service did not quiesce')
    screens=Path(a.frames)/'screen'
    for directory in sorted(screens.glob('*/baseline/capture')):
        shapes={p.name.split('-')[0] for p in directory.glob('*.f32')}
        for shape in sorted(shapes):
            m,n=map(int,shape.split('x'))
            files=sorted(directory.glob(shape+'-*.f32'),key=lambda p:int(p.stem.split('-')[1]))
            if len(files)<4:continue
            ids=np.linspace(0,len(files)-1,min(16,len(files)),dtype=int)
            selected=[files[i] for i in ids]
            name=directory.parents[1].name+'-'+shape
            data=b''.join(p.read_bytes() for p in selected);p=root/(name+'.f32');p.write_bytes(data)
            case=json.loads((directory.parent/'config.json').read_text());sig=case['kwargs']['sigma']
            cases.append(dict(name=name,m=m,n=n,count=len(selected),sigma=float(np.mean(sig)),input=str(p),
                              capture_files=[str(x) for x in selected],source_hashes=[sha(x) for x in selected],
                              coding_policy='uniform sigma mean; unequal-noise production case has separate full-frame evidence'))
    rng=np.random.default_rng(9273)
    for m,n in [(48,48),(147,70),(192,90),(768,256)]:
        r=min(m,n);matrices=[]
        for kind in range(4):
            u=np.linalg.qr(rng.standard_normal((m,r)))[0];v=np.linalg.qr(rng.standard_normal((n,r)))[0]
            spectrum=np.ones(r) if kind==0 else np.geomspace(1,1e-9,r)
            if kind==2:spectrum[r//2:]=0
            if kind==3:spectrum[:]=0
            matrices.append((u*spectrum)@v.T)
        p=root/f'synthetic-{m}x{n}.f32';p.write_bytes(b''.join(x.astype('<f4').tobytes(order='F') for x in matrices))
        cases.append(dict(name=p.stem,m=m,n=n,count=4,sigma=25,input=str(p),synthetic=True))
    (root/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    rows=[];expected={};began=time.monotonic()
    for c in cases:
        m,n,count=c['m'],c['n'],c['count']
        matrices=np.fromfile(c['input'],'<f4').reshape(count,n,m).transpose(0,2,1)
        pilot_times=[]
        for mode in ('baseline','tight','gesdd','gesvd'):
            d=root/c['name']/'pilot'/mode;d.mkdir(parents=True)
            cmd=['taskset','-c','0',a.bench,str(m),str(n),str(count),'1',str(c['sigma']),c['input'],str(d/'output')]
            with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
                p=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,NSS_SVD_LAB=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=180)
            if p.returncode:raise RuntimeError(f'kernel calibration failed {c["name"]}/{mode}')
            pilot_times.append(json.loads((d/'stdout').read_text())['total_s'])
        # One fixed count for every method and round, long enough for /proc/stat.
        loops=max(2,min(10000,math.ceil(.6/min(pilot_times))))
        (root/c['name']/'calibration.json').write_text(json.dumps(dict(loops=loops,pilot_total_s=pilot_times,target_fastest_combined_s=1.2))+'\n')
        for repeat in range(7):
            modes=['baseline','tight','gesdd','gesvd']
            if repeat%2:modes.reverse()
            for mode in modes:
                if time.monotonic()-began>2400:raise RuntimeError('kernel campaign time cap')
                d=root/c['name']/str(repeat)/mode;d.mkdir(parents=True)
                cmd=['taskset','-c','0',a.bench,str(m),str(n),str(count),str(loops),str(c['sigma']),c['input'],str(d/'output')]
                start=ticks()
                with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
                    p=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,NSS_SVD_LAB=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=180)
                e=environment(start,ticks())
                record=dict(case=c['name'],repeat=repeat,mode=mode,returncode=p.returncode,environment=e,qualified=qualified(e))
                if p.returncode==0:
                    record['result']=json.loads((d/'stdout').read_text())
                    record['factors_sha256']=sha(d/'output.factors');record['pixels_sha256']=sha(d/'output.f32')
                    key=(c['name'],mode)
                    identity=(record['factors_sha256'],record['pixels_sha256'])
                    if key in expected and expected[key]!=identity:raise RuntimeError('kernel repeat mismatch')
                    expected[key]=identity
                    if repeat==0:record['independent']=check_factors(d/'output.factors',matrices,m,n)
                (d/'record.json').write_text(json.dumps(record,indent=2)+'\n')
                rows.append(record);(root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(case=c['name'],repeat=repeat,mode=mode,returncode=p.returncode,qualified=record['qualified'])),flush=True)
                if p.returncode:raise RuntimeError('kernel failed')
    (root/'complete.json').write_text(json.dumps(dict(completed=True,elapsed=time.monotonic()-began))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('frames','bench','out'):p.add_argument('--'+x,required=True)
    main(p.parse_args())
