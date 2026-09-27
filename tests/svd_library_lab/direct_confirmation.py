#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Confirm library timing directly against the untouched production plugin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from campaign import ticks, environment, qualified

def main(a):
    root=Path(a.out);root.mkdir(exist_ok=False);fixtures=Path(a.fixtures).resolve()
    names=['rgb-g48-0874','gray-g48','rgb-default25','rgb-default75']
    cases=[x for x in json.loads((fixtures/'cases.json').read_text()) if x['name'] in names]
    (root/'policy.json').write_text(json.dumps(dict(cases=names,pairs=7,candidate='gesdd',control='unmodified stock plugin',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        worker_driver_sha256=hashlib.sha256(Path(__file__).with_name('campaign.py').read_bytes()).hexdigest(),
        timed_worker_sha256=hashlib.sha256((Path(a.source)/'tests/c4_integration.py').read_bytes()).hexdigest(),
        stock_sha256=hashlib.sha256(Path(a.stock).read_bytes()).hexdigest(),
        plugin_sha256=hashlib.sha256(Path(a.plugin).read_bytes()).hexdigest(),
        order='alternating stock/library then library/stock',time_cap_s=3600,idle_min=.99,steal_max=0,
        frames={'rgb-g48-0874':2,'gray-g48':7,'rgb-default25':1,'rgb-default75':1},
        duration_reason='Pre-confirmation adjustment: retain original short-window failures; longer windows reduce sensitivity to brief sibling activity; Gray capped at seven frames so unchanged worker retains every timed output',
        cases_selected_before_primary_paired_results=True),indent=2)+'\n')
    rows=[];began=time.monotonic()
    for c in cases:
        c['sample']=str(fixtures/c['sample'])
        c['frames']={'rgb-g48-0874':2,'gray-g48':7}.get(c['name'],1)
        for repeat in range(7):
            for mode in (('stock','gesdd') if repeat%2==0 else ('gesdd','stock')):
                if time.monotonic()-began>3600:raise RuntimeError('direct confirmation cap')
                d=root/c['name']/str(repeat)/mode;d.mkdir(parents=True)
                config=dict(c,_dump=str(d/'pixels.npy'))
                (d/'config.json').write_text(json.dumps(config,indent=2)+'\n')
                plugin=a.stock if mode=='stock' else a.plugin
                cmd=['taskset','-c','0',sys.executable,str(Path(__file__).with_name('campaign.py')),'worker',
                     '--source',a.source,'--plugin',plugin,'--config',str(d/'config.json')]
                start=ticks();stamp=time.monotonic()
                with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
                    p=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,NSS_SVD_LAB='baseline' if mode=='stock' else mode,
                        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=600)
                e=environment(start,ticks());row=dict(case=c['name'],repeat=repeat,mode=mode,returncode=p.returncode,whole_environment=e,elapsed=time.monotonic()-stamp)
                if p.returncode==0:
                    row['result']=json.loads((d/'stdout').read_text());t=row['result']['timed_environment']
                    row['qualified']=qualified(e) and t['cpu1_idle'] is not None and t['cpu1_idle']>=.99 and t['cpu0_steal_ticks']==0 and t['cpu1_steal_ticks']==0
                else:row['qualified']=False
                (d/'record.json').write_text(json.dumps(row,indent=2)+'\n')
                rows.append(row);(root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(case=c['name'],repeat=repeat,mode=mode,returncode=p.returncode,qualified=row['qualified'])),flush=True)
                if p.returncode:raise RuntimeError('direct confirmation failed')
    (root/'complete.json').write_text(json.dumps(dict(completed=True,elapsed=time.monotonic()-began))+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('source','plugin','stock','fixtures','out'):p.add_argument('--'+x,required=True)
    main(p.parse_args())
