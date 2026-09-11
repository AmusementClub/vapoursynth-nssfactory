#!/usr/bin/env python3
"""Supported C4 profiles and alternating same-host public NLH A/B runs."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
from nlh_defaults_inputs import save_json,file_sha


class InvalidTiming(RuntimeError):
    pass


def run_worker(plugin,job,base,frames,mode=None,require_quiet=True):
    command=[sys.executable,str(Path(__file__).with_name('nlh_candidate_profile.py')),
             '--plugin',str(plugin),'--input',job['input'],'--parameters-json',json.dumps(job['parameters']),
             '--frames',str(frames)]
    if job.get('metadata'):command+=['--fixture-meta',job['metadata']]
    fifos=[]
    if mode:
        control,ack=Path(str(base)+'.control'),Path(str(base)+'.ack')
        for path in (control,ack):os.mkfifo(path);fifos.append(path)
        command+=['--control',str(control),'--ack',str(ack)]
        if mode=='record':
            command=['perf','record','-e','cycles:u','-F','499','-D','-1',
                     '--control=fifo:'+str(control)+','+str(ack),'-o',str(base)+'.perf.data','--',*command]
        else:
            command=['perf','stat','-j','-M','Topdown'+mode,'-D','-1',
                     '--control=fifo:'+str(control)+','+str(ack),'-o',str(base)+'.stat.json','--',*command]
    try:
        with Path(str(base)+'.stdout').open('w') as stdout,Path(str(base)+'.stderr').open('w') as stderr:
            completed=subprocess.run(command,stdout=stdout,stderr=stderr)
        if completed.returncode:raise RuntimeError('failed worker: '+str(base))
        messages=[json.loads(line) for line in Path(str(base)+'.stdout').read_text().splitlines() if line.startswith('{')]
        result=next(r for r in messages if r.get('schema')=='nss.nlh-candidate-profile.v1')
        if result['timed_source_fills'] or result['affinity']!=[0]:raise AssertionError('invalid timing boundary')
        activity=result['cpu_activity']
        quiet=not any(r['steal_ticks'] for r in activity.values()) and activity['cpu1']['busy_fraction'] is not None and activity['cpu1']['busy_fraction']<=.01
        result.update(cpu_valid=quiet,measurement_role='measurement' if require_quiet else 'frame-count pilot',
                      accepted_for_timing=quiet and require_quiet)
        if require_quiet and not quiet:
            result['exclusion_reason']='CPU1 interference or steal'
            save_json(Path(str(base)+'.json'),result)
            raise InvalidTiming('CPU1 interference/steal: '+str(base))
        if mode in ('L1','L2'):
            events=[json.loads(line.rstrip(',')) for line in Path(str(base)+'.stat.json').read_text().splitlines() if line.startswith('{')]
            counted=[r for r in events if 'counter-value' in r]
            valid=bool(counted) and all(not str(r['counter-value']).startswith('<') and float(r.get('pcnt-running',0))>=99.9 for r in counted)
            result['pmu_valid']=valid
            if not valid:raise RuntimeError('unsupported or multiplexed PMU group: '+str(base))
        if mode=='record':
            data=Path(str(base)+'.perf.data')
            with Path(str(base)+'.report.txt').open('w') as stream:
                subprocess.run(['perf','report','--stdio','--header','--no-children','--percent-limit','0.05',
                                '--sort','symbol','-i',str(data)],stdout=stream,check=True)
            report=Path(str(base)+'.report.txt').read_text()
            if not data.stat().st_size or 'Total Lost Samples: 0' not in report:
                raise RuntimeError('missing samples or losses: '+str(base))
        save_json(Path(str(base)+'.json'),result)
        return result
    finally:
        for path in fifos:path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('profile','bench'))
    for key in ('baseline','jobs','out'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--candidate');parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--target-seconds',type=float,default=8.)
    parser.add_argument('--comparison', choices=('kernel', 'parameters'), default='kernel',
                        help='Parameter comparisons retain separate output identities and frame counts')
    args=parser.parse_args()
    if os.sched_getaffinity(0)!={0}:parser.error('pin the whole campaign to CPU0')
    if args.mode=='bench' and not args.candidate:parser.error('bench requires --candidate')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    jobs=json.loads(Path(args.jobs).read_text())['jobs'];rows=[]
    for job in jobs:
        label=job['label'];directory=root/label;directory.mkdir()
        pilot=run_worker(args.baseline,job,directory/'pilot',1,require_quiet=False)
        frames=min(128,max(1,math.ceil(args.target_seconds/pilot['seconds_per_frame'])))
        candidate_job=dict(job)
        expected={'baseline':pilot, 'candidate':pilot}
        frame_counts={'baseline':frames, 'candidate':frames}
        if args.comparison=='parameters':
            if args.mode!='bench':parser.error('parameter comparisons require bench mode')
            candidate_job['parameters']=job['candidate_parameters']
            expected['candidate']=run_worker(args.candidate,candidate_job,directory/'candidate-pilot',1,require_quiet=False)
            if expected['candidate']['input_sha256']!=pilot['input_sha256']:
                raise AssertionError('parameter comparison input mismatch')
            if not pilot.get('host') or expected['candidate']['host']!=pilot['host']:
                raise AssertionError('parameter comparison host or boot changed')
            frame_counts['candidate']=min(128,max(1,math.ceil(args.target_seconds/expected['candidate']['seconds_per_frame'])))
        elif 'candidate_parameters' in job:
            raise ValueError('different parameters require --comparison parameters')
        if args.mode=='profile':
            results=[]
            for mode in ('L1','L2','record'):
                result=run_worker(args.baseline,job,directory/mode,frames,mode)
                if result['output_sha256']!=pilot['output_sha256']:raise AssertionError('profile output changed')
                results.append(result)
            rows.append(dict(label=label,pilot=pilot,frames=frames,profiles=results))
        else:
            pairs=[];excluded=[]
            for repeat in range(args.pairs):
                order=[('baseline',args.baseline),('candidate',args.candidate)]
                if repeat%2:order.reverse()
                for attempt in range(3):
                    pair={}
                    try:
                        for variant,plugin in order:
                            variant_job=job if variant=='baseline' else candidate_job
                            result=run_worker(plugin,variant_job,directory/f'{repeat}-attempt{attempt}-{variant}',frame_counts[variant])
                            if not pilot.get('host') or result['host']!=pilot['host']:
                                raise AssertionError('paired comparison host or boot changed')
                            if result['input_sha256']!=pilot['input_sha256'] or result['output_sha256']!=expected[variant]['output_sha256']:
                                raise AssertionError('A/B input/output identity mismatch')
                            pair[variant]=result
                    except InvalidTiming as error:
                        excluded.append(dict(repeat=repeat,attempt=attempt,reason=str(error)))
                        save_json(directory/'excluded-attempts.json',excluded)
                        if attempt==2:raise
                        continue
                    pairs.append(pair)
                    break
            ratios=[p['baseline']['seconds_per_frame']/p['candidate']['seconds_per_frame'] for p in pairs]
            rng=np.random.default_rng(20260909)
            bootstrap=np.median(rng.choice(ratios,size=(10000,len(ratios)),replace=True),axis=1)
            rows.append(dict(label=label,frames=frame_counts,pairs=pairs,paired_speedups=ratios,
                             comparison=args.comparison, numerical_equivalence_required=args.comparison=='kernel',
                             expected_output_sha256={k:v['output_sha256'] for k,v in expected.items()},
                             excluded_attempts=excluded,
                             median_speedup=statistics.median(ratios),ci95=np.quantile(bootstrap,[.025,.975]).tolist()))
        save_json(root/'summary.json',dict(mode=args.mode,rows=rows,jobs_sha256=file_sha(args.jobs),script_sha256=file_sha(__file__)))
        print(label,'complete',rows[-1].get('median_speedup'),flush=True)


if __name__=='__main__':main()
