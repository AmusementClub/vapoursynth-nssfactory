#!/usr/bin/env python3
"""C4A compiler/ISA diagnostic: same-host alternating full-filter pairs.

No release or optimization admission is granted here. Timing and exact/triage
pixel comparisons stay separate; every deviation is preserved for stage replay.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c4_paired_bm import cpu_stat, interval
from bm_numerics import compare
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
for name in ('baseline','candidate','configs','out'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--pairs',type=int,choices=(7,15),default=7)
parser.add_argument('--seconds',type=float,default=.3)
args=parser.parse_args()
assert platform.machine()=='aarch64'
assert Path('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list').read_text().strip()=='0'
args.out.mkdir(parents=True,exist_ok=False)
driver=Path(__file__).resolve().parents[1]/'c4_integration.py'
inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.baseline,args.candidate,args.configs,Path(__file__),driver,driver.with_name('c4_paired_bm.py'))}
report=dict(schema='nssfactory.c4a-isa-pairs.v1',complete=False,performance_admission=False,
            cpu=0,independent_other_core=1,inputs=inputs,results=[])

def worker(plugin,config,directory,name):
    command=['taskset','-c','0',sys.executable,str(driver),'worker','--plugin',str(plugin.resolve()),'--config',json.dumps(config)]
    before=cpu_stat()
    proc=subprocess.run(command,text=True,capture_output=True,timeout=180)
    after=cpu_stat()
    (directory/(name+'.stdout')).write_text(proc.stdout)
    (directory/(name+'.stderr')).write_text(proc.stderr)
    (directory/(name+'.command.json')).write_text(json.dumps(command,indent=2))
    proc.check_returncode()
    result=json.loads(proc.stdout)
    delta=[b-a for a,b in zip(before['cpu1'],after['cpu1'])]
    result['environment']=dict(cpu1_idle=delta[3]/max(1,sum(delta[:8])),
        cpu0_steal_ticks=after['cpu0'][7]-before['cpu0'][7])
    assert result['timed_source_fills']==0
    return result

try:
    for original in json.loads(args.configs.read_text()):
        directory=args.out/original['name'];directory.mkdir()
        calibration=[worker(p,dict(original,frames=1),directory,'cal-'+name)
                     for name,p in [('baseline',args.baseline),('candidate',args.candidate)]]
        frames=max(1,min(16,math.ceil(args.seconds/(max(r['ms'] for r in calibration)/1000))))
        config=dict(original,frames=frames)
        row=dict(config=config,calibration=calibration,pairs=[])
        report['results'].append(row)
        for pair in range(args.pairs):
            results={}
            for label,plugin in ([('baseline',args.baseline),('candidate',args.candidate)] if pair%2==0
                                  else [('candidate',args.candidate),('baseline',args.baseline)]):
                kw=dict(config)
                if pair==0:kw['_dump']=str(directory/(label+'.npy'))
                results[label]=worker(plugin,kw,directory,f'{pair}-{label}')
            assert results['baseline']['input_sha256']==results['candidate']['input_sha256']
            assert results['baseline']['timed_frames']==results['candidate']['timed_frames']
            row['pairs'].append(results)
            (args.out/'summary.json').write_text(json.dumps(report,indent=2))
        ratios=[p['baseline']['ms']/p['candidate']['ms'] for p in row['pairs']]
        hashes={label:{p[label]['sha256'] for p in row['pairs']} for label in ('baseline','candidate')}
        if any(len(h)!=1 for h in hashes.values()):raise RuntimeError('within-binary repeat pixels changed')
        row.update(paired_speedup=statistics.median(ratios),ci95=interval(ratios),ratios=ratios,
                   baseline_ms=statistics.median(p['baseline']['ms'] for p in row['pairs']),
                   candidate_ms=statistics.median(p['candidate']['ms'] for p in row['pairs']),
                   exact=hashes['baseline']==hashes['candidate'],
                   numeric_triage=compare(np.load(directory/'baseline.npy'),np.load(directory/'candidate.npy')),
                   cpu0_no_steal=all(p[l]['environment']['cpu0_steal_ticks']==0 for p in row['pairs'] for l in hashes),
                   minimum_cpu1_idle=min(p[l]['environment']['cpu1_idle'] for p in row['pairs'] for l in hashes))
        print(json.dumps({k:row[k] for k in ('paired_speedup','ci95','exact','baseline_ms','candidate_ms')}|{'name':config['name']}),flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in inputs.items()):raise RuntimeError('input changed')
    report['complete']=True
except Exception as error:
    report['error']=str(error)
    raise
finally:
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
