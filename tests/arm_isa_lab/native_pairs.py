#!/usr/bin/env python3
"""Same-host native release pairs with frame latency and resource evidence.

This produces measurements, not automatic numerical or release admission.
Linux pins CPU0 (SMT topology is recorded); macOS has no guaranteed CPU pinning.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from c4_paired_bm import cpu_stat, interval
from bm_numerics import compare

parser=argparse.ArgumentParser(description=__doc__)
for n in ('baseline','candidate','configs','out'):parser.add_argument('--'+n,type=Path,required=True)
parser.add_argument('--pairs',type=int,choices=(7,15,31),default=15)
parser.add_argument('--seconds',type=float,default=.5)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
linux=platform.system()=='Linux'
driver=Path(__file__).resolve().parents[1]/'c4_integration.py'
inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.baseline,args.candidate,args.configs,Path(__file__),driver,driver.with_name('c4_paired_bm.py'),driver.with_name('bm_numerics.py'))}
report=dict(schema='nssfactory.native-release-pairs.v1',complete=False,performance_admission=False,
            platform=platform.platform(),architecture=platform.machine(),inputs=inputs,results=[],
            affinity='CPU0' if linux else 'macOS scheduler, no CPU pinning',
            latency_scope='Observed frame requests; small-sample p95/p99 are descriptive, not an SLA guarantee.')
if linux:
    report['cpu0_siblings']=Path('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list').read_text().strip()
else:
    for name,cmd in [('power',['pmset','-g','batt']),('thermal',['pmset','-g','therm']),('host',['sysctl','hw.model','hw.memsize','hw.ncpu'])]:
        p=subprocess.run(cmd,text=True,capture_output=True);(args.out/(name+'.txt')).write_text(p.stdout+p.stderr)

def worker(plugin,config,directory,name):
    command=(['taskset','-c','0'] if linux else [])+[sys.executable,str(driver),'worker','--plugin',str(plugin.resolve()),'--config',json.dumps(config)]
    before=cpu_stat() if linux else None
    proc=subprocess.run(command,text=True,capture_output=True,timeout=240)
    after=cpu_stat() if linux else None
    (directory/(name+'.stdout')).write_text(proc.stdout);(directory/(name+'.stderr')).write_text(proc.stderr)
    (directory/(name+'.command.json')).write_text(json.dumps(command,indent=2));proc.check_returncode()
    value=json.loads(proc.stdout)
    if value['timed_source_fills']!=0:raise RuntimeError('source generation entered timing')
    if len(value.get('frame_ms',[]))!=value['timed_frames']:raise RuntimeError('missing per-frame latency')
    if linux:
        delta=[b-a for a,b in zip(before['cpu1'],after['cpu1'])]
        value['environment']=dict(cpu1_idle=delta[3]/max(1,sum(delta[:8])),cpu0_steal_ticks=after['cpu0'][7]-before['cpu0'][7])
    return value

try:
    for original in json.loads(args.configs.read_text()):
        directory=args.out/original['name'];directory.mkdir()
        config=dict(original,_frame_times=True,_cpu_environment=linux)
        calibration=[worker(p,dict(config,frames=1),directory,'cal-'+s) for s,p in [('baseline',args.baseline),('candidate',args.candidate)]]
        w,h=config['size'];planes=3 if config['algorithm']=='mcwnnm' else 1
        # Bound retained source/output payloads while giving small tail cases
        # a useful timing interval and observable scheduler tick coverage.
        frame_cap=max(1,min(1024,int(256*1024*1024/(w*h*planes*8))))
        frames=max(1,min(frame_cap,math.ceil(args.seconds/(max(x['ms'] for x in calibration)/1000))))
        config['frames']=frames
        row=dict(config=config,calibration=calibration,pairs=[]);report['results'].append(row)
        for index in range(args.pairs):
            values={}
            for side,plugin in ([('baseline',args.baseline),('candidate',args.candidate)] if index%2==0 else [('candidate',args.candidate),('baseline',args.baseline)]):
                current=dict(config)
                if index==0:current['_dump']=str(directory/(side+'.npy'))
                values[side]=worker(plugin,current,directory,f'{index}-{side}')
            if values['baseline']['input_sha256']!=values['candidate']['input_sha256'] or values['baseline']['timed_frames']!=values['candidate']['timed_frames']:
                raise RuntimeError('paired inputs/frame counts differ')
            row['pairs'].append(values)
            (args.out/'summary.json').write_text(json.dumps(report,indent=2))
        hashes={side:{p[side]['sha256'] for p in row['pairs']} for side in ('baseline','candidate')}
        if any(len(h)!=1 for h in hashes.values()):raise RuntimeError('repeat pixels changed within one binary')
        ratios=[p['baseline']['ms']/p['candidate']['ms'] for p in row['pairs']]
        row.update(paired_speedup=statistics.median(ratios),ci95=interval(ratios),ratios=ratios,
                   baseline_ms=statistics.median(p['baseline']['ms'] for p in row['pairs']),
                   candidate_ms=statistics.median(p['candidate']['ms'] for p in row['pairs']),
                   exact=hashes['baseline']==hashes['candidate'],
                   numeric_triage=compare(np.load(directory/'baseline.npy'),np.load(directory/'candidate.npy')),
                   resources={},latency={})
        for side in ('baseline','candidate'):
            latencies=[v for pair in row['pairs'] for v in pair[side]['frame_ms']]
            row['latency'][side]=dict(samples=len(latencies),p95_ms=float(np.percentile(latencies,95)),p99_ms=float(np.percentile(latencies,99)))
            row['resources'][side]=dict(peak_rss_bytes=max(p[side]['peak_rss_kib']*1024 for p in row['pairs']))
        if linux:
            idle=[p[s]['timed_environment']['cpu1_idle'] for p in row['pairs'] for s in hashes]
            row['minimum_cpu1_idle']=min((x for x in idle if x is not None),default=None)
            row['scheduler_ticks_observed']=all(x is not None for x in idle)
            row['cpu0_no_steal']=all(p[s]['timed_environment']['cpu0_steal_ticks']==0 for p in row['pairs'] for s in hashes)
        print(json.dumps(dict(name=config['name'],speedup=row['paired_speedup'],ci95=row['ci95'],exact=row['exact'],numeric=row['numeric_triage']['max_abs'])),flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in inputs.items()):raise RuntimeError('measurement inputs changed')
    report['complete']=True
except Exception as error:
    report['error']=str(error);raise
finally:
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
