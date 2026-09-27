#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Bounded sequential experiment; preserves every run, including failed gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ticks():
    return {x.split()[0]:list(map(int,x.split()[1:])) for x in Path('/proc/stat').read_text().splitlines() if x.startswith(('cpu0 ','cpu1 '))}
def environment(a,b):
    d=[y-x for x,y in zip(a['cpu1'],b['cpu1'])]
    return dict(cpu1_idle=d[3]/sum(d[:8]) if sum(d[:8]) else None,
                cpu0_steal=b['cpu0'][7]-a['cpu0'][7],cpu1_steal=d[7],
                cpu0_ticks=[y-x for x,y in zip(a['cpu0'],b['cpu0'])],cpu1_ticks=d)
def qualified(e):return e['cpu1_idle'] is not None and e['cpu1_idle']>=.99 and e['cpu0_steal']==0 and e['cpu1_steal']==0

def worker(args):
    sys.path.insert(0,str(Path(args.source)/'tests'))
    import c4_integration
    config=json.loads(Path(args.config).read_text())
    result=c4_integration.worker(args.plugin,config)
    print(json.dumps(result),flush=True)

def main(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    fixtures=Path(args.fixtures).resolve();lab=Path(args.source).resolve()
    cases=json.loads((fixtures/'cases.json').read_text())
    for c in cases:c['sample']=str(fixtures/c['sample'])
    (root/'policy.json').write_text(json.dumps(dict(pairs=7,modes=['baseline','tight','gesdd'],
        order='alternate baseline,tight,gesdd / gesdd,tight,baseline; baseline shared within round',
        idle_min=.99,steal_max=0,whole_run_gate=True,timeout_s=600,total_cap_s=9000,
        admission='Exploratory numeric-equivalent candidate; paired median CI lower >1.02; no automatic production admission',
        plugin_sha256=sha(args.plugin),stock_sha256=sha(args.stock),cases=cases),indent=2)+'\n')
    began=time.monotonic();rows=[]
    def run(c,mode,label,plugin=None,capture=False):
        if time.monotonic()-began>9000:raise RuntimeError('campaign cap reached')
        d=root/label;d.mkdir(parents=True,exist_ok=False)
        config=dict(c,_dump=str(d/'pixels.npy'))
        (d/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        env=dict(os.environ,NSS_SVD_LAB=mode,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        if capture:
            (d/'capture').mkdir();env['NSS_SVD_CAPTURE']=str(d/'capture')
        cmd=['taskset','-c','0',sys.executable,__file__,'worker','--source',str(lab),'--plugin',plugin or args.plugin,'--config',str(d/'config.json')]
        a=ticks();start=time.monotonic()
        with (d/'stdout').open('w') as out,(d/'stderr').open('w') as err:
            try:p=subprocess.run(cmd,stdout=out,stderr=err,env=env,timeout=600);code=p.returncode
            except subprocess.TimeoutExpired:code=124
        b=ticks();record=dict(case=c['name'],mode=mode,label=label,returncode=code,elapsed=time.monotonic()-start,whole_environment=environment(a,b))
        if code==0:
            record['result']=json.loads((d/'stdout').read_text().splitlines()[-1])
            t=record['result']['timed_environment']
            record['qualified']=qualified(record['whole_environment']) and t['cpu1_idle'] is not None and t['cpu1_idle']>=.99 and t['cpu0_steal_ticks']==0 and t['cpu1_steal_ticks']==0
        else:record['qualified']=False
        (d/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        rows.append(record);(root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ('label','returncode','elapsed','qualified')}),flush=True)
        if code:raise RuntimeError(f'worker failed {label}')
        return record
    # Verify instrumentation baseline and stock on every case; retain captures
    # from untimed diagnostic runs separately from paired evidence.
    for c in cases:
        stock=run(c,'baseline','screen/'+c['name']+'/stock',args.stock)
        baseline=run(c,'baseline','screen/'+c['name']+'/baseline',capture=True)
        if stock['result']['sha256']!=baseline['result']['sha256']:raise RuntimeError('lab baseline differs from stock')
        for mode in ('tight','gesdd'):run(c,mode,'screen/'+c['name']+'/'+mode)
    for c in (cases[0], cases[2]):
        for repeat in range(7):
            order=('stock','lab') if repeat%2==0 else ('lab','stock')
            for label in order:
                run(c,'baseline',f'instrumentation/{c["name"]}/{repeat}/{label}',args.stock if label=='stock' else args.plugin)
    for c in cases:
        screen=next(x for x in rows if x['label']=='screen/'+c['name']+'/stock')
        # Fixed frame count for all modes/rounds; no per-candidate calibration.
        c['frames']=max(1,min(4,int(750/max(screen['result']['ms'],1))+1))
        for repeat in range(7):
            order=('baseline','tight','gesdd') if repeat%2==0 else ('gesdd','tight','baseline')
            for mode in order:run(c,mode,f'paired/{c["name"]}/{repeat}/{mode}')
    (root/'complete.json').write_text(json.dumps(dict(completed=True,elapsed=time.monotonic()-began))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','run'])
    for name in ('source','plugin','stock','fixtures','out','config'):p.add_argument('--'+name)
    a=p.parse_args()
    worker(a) if a.action=='worker' else main(a)
