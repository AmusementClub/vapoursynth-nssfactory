#!/usr/bin/env python3
"""Bounded oracle PSNR tuning plus independent-noise response matching.

Search timings are not benchmarks. Selected outputs are re-run with warmup and
three measured repeats. Only selected pixel arrays are retained permanently.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

from defaults_compare import ALGORITHMS, eligible
from paper_compare import cpu_activity, cpu_ticks, save_json, sha


class Search:
    def __init__(self,args,case):
        self.args=args;self.case=case;self.out=Path(args.out).resolve()
        self.scratch=self.out/'scratch'/case['id'];self.scratch.mkdir(parents=True,exist_ok=True)
        self.cache={};self.probes={};self.responses={}
        self.worker=Path(__file__).with_name('defaults_compare.py').resolve()
        shape=(3,case['height'],case['width'])
        first=np.fromfile(Path(args.fixtures)/case['noisy'],dtype='<f4').reshape(shape).astype(np.float64)
        second=np.fromfile(Path(args.probe)/case['noisy'],dtype='<f4').reshape(shape).astype(np.float64)
        self.input_difference_energy=float(np.sum((first-second)**2))

    def call(self,algorithm,strength,probe,output,cold=True):
        output.parent.mkdir(parents=True,exist_ok=True)
        fixture_root=self.args.probe if probe else self.args.fixtures
        cmd=[sys.executable,str(self.worker),'worker','--fixtures',str(Path(fixture_root).resolve()),
             '--case',self.case['id'],'--algorithm',algorithm,'--plugin',str(Path(self.args.plugin).resolve()),
             '--strength',repr(strength),'--output',str(output)]
        if cold:cmd+=['--cold-evaluation']
        profile=getattr(self.args,'profiles',{}).get(algorithm,{})
        if profile:cmd+=['--parameters-json',json.dumps(profile)]
        before=cpu_ticks()
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=self.args.timeout)
        cpu=cpu_activity(before,cpu_ticks())
        if result.returncode:
            (self.out/'failure.log').write_bytes(result.stdout)
            raise RuntimeError(f'{algorithm} strength={strength}: {result.stdout[-2000:].decode(errors="replace")}')
        row=json.loads(Path(str(output)+'.json').read_text())
        row.update(cpu_activity_during_worker=cpu,timing_eligible=eligible({'cpu_activity_during_worker':cpu}),
                   tuned_parameter='h' if algorithm=='NLM' else 'sigma',strength=strength,probe=probe)
        return row

    def evaluate(self,algorithm,strength,probe=False):
        strength=float(strength);key=(algorithm,strength)
        table=self.probes if probe else self.cache
        if key in table:return table[key]
        token=hashlib.sha256(strength.hex().encode()).hexdigest()[:16]
        path=self.scratch/f'{algorithm}-{token}-{"probe" if probe else "main"}.f32'
        row=self.call(algorithm,strength,probe,path)
        row['scratch_output']=str(path)
        table[key]=row
        logged={k:v for k,v in row.items() if k!='scratch_output'}
        logged['pixel_retention']='temporary search pixels; only selected outputs retained'
        with (self.out/'search.jsonl').open('a') as stream:stream.write(json.dumps(logged)+'\n')
        return row

    def response(self,algorithm,strength):
        key=(algorithm,float(strength))
        if key in self.responses:return self.responses[key]
        a=self.evaluate(algorithm,strength);b=self.evaluate(algorithm,strength,probe=True)
        first=np.fromfile(a['scratch_output'],dtype='<f4').astype(np.float64)
        second=np.fromfile(b['scratch_output'],dtype='<f4').astype(np.float64)
        ratio=math.sqrt(float(np.sum((first-second)**2))/self.input_difference_energy)
        self.responses[key]=ratio
        with (self.out/'responses.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(case=self.case['id'],algorithm=algorithm,strength=strength,
                response=ratio,primary_output_sha256=a['output_sha256'],probe_output_sha256=b['output_sha256']))+'\n')
        return ratio

    def best(self,algorithm):
        sigma=self.case['sigma']
        initial=[sigma*f for f in getattr(self.args,'coarse_factors',(.25,.5,1,2,4,8,16))]
        initial.append(1.2 if algorithm=='NLM' else 0.)
        for value in sorted(set(initial)):self.evaluate(algorithm,value)
        def points():return sorted(v for a,v in self.cache if a==algorithm)
        def winner():return max(points(),key=lambda v:self.cache[(algorithm,v)]['quality']['psnr_db'])
        # PSNR is not assumed monotone. Keep all coarse peaks, and bisect the
        # wider adjacent interval around the best tested point each step.
        for _ in range(self.args.refinements):
            values=points();best=winner();index=values.index(best)
            neighbors=[]
            if index:neighbors.append((values[index-1],best))
            if index+1<len(values):neighbors.append((best,values[index+1]))
            lo,hi=max(neighbors,key=lambda pair:pair[1]-pair[0])
            if hi-lo < max(1e-6,sigma*1e-5):break
            self.evaluate(algorithm,(lo+hi)/2)
        value=winner();row=self.cache[(algorithm,value)]
        return dict(strength=value,quality=row['quality'],output_sha256=row['output_sha256'],
            structural_parameters=getattr(self.args,'profiles',{}).get(algorithm,{}),
            evaluated_points=len(points()),search_bounds=[min(initial),max(initial)],
            optimum_at_boundary=value in (min(initial),max(initial)),
            response=self.response(algorithm,value))

    def match(self,algorithm,best,target):
        low,high=best['search_bounds']
        if low==0:low=self.case['sigma']/256
        for value in (low,best['strength'],high):
            if value>0:self.response(algorithm,value)
        tolerance=max(.005,.05*target)
        for _ in range(self.args.match_steps):
            points=sorted((s,r) for (a,s),r in self.responses.items() if a==algorithm and s>0)
            nearest=min(points,key=lambda pair:abs(pair[1]-target))
            if abs(nearest[1]-target)<=tolerance:break
            brackets=[(a,b) for a,b in zip(points,points[1:]) if (a[1]-target)*(b[1]-target)<=0]
            if not brackets:break
            a,b=min(brackets,key=lambda pair:pair[1][0]-pair[0][0])
            self.response(algorithm,(a[0]+b[0])/2)
        points=[(s,r) for (a,s),r in self.responses.items() if a==algorithm and s>0]
        value,response=min(points,key=lambda pair:abs(pair[1]-target))
        row=self.cache[(algorithm,value)]
        return dict(strength=value,response=response,target_response=target,
            structural_parameters=getattr(self.args,'profiles',{}).get(algorithm,{}),
            absolute_response_error=abs(response-target),tolerance=tolerance,
            matched=abs(response-target)<=tolerance,quality=row['quality'],output_sha256=row['output_sha256'])

    def retain(self,algorithm,mode,selection,dedup):
        strength=selection['strength'];rows=[]
        # Preserve both independent inputs' selected outputs so the response
        # calibration can be recomputed from downloaded evidence.
        probe=self.evaluate(algorithm,strength,True)
        probe_path=self.out/f'{self.case["id"]}-{algorithm}-{mode}-probe.f32'
        shutil.copy2(probe['scratch_output'],probe_path)
        save_json(str(probe_path)+'.json',{**probe,'output':probe_path.name,'scratch_output':None})
        for repeat in range(3):
            output=self.out/f'{self.case["id"]}-{algorithm}-{mode}-r{repeat}.f32'
            row=self.call(algorithm,strength,False,output,cold=False)
            assert row['output_sha256']==selection['output_sha256'], 'Warm final differs from quality search'
            digest=row['output_sha256']
            if digest in dedup:
                output.unlink();os.link(dedup[digest],output)
            else:dedup[digest]=output
            row.update(mode=mode,repeat=repeat,ok=True,selection=selection,
                probe_output=probe_path.name,probe_output_sha256=probe['output_sha256'],
                input_difference_energy=self.input_difference_energy)
            rows.append(row)
        return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','probe','out','plugin'):p.add_argument('--'+name,required=True)
    p.add_argument('--cases',nargs='+');p.add_argument('--refinements',type=int,default=6)
    p.add_argument('--match-steps',type=int,default=10);p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--profiles-file')
    args=p.parse_args()
    args.profiles=json.loads(Path(args.profiles_file).read_text())['profiles'] if args.profiles_file else {}
    assert os.sched_getaffinity(0)=={0}
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.cases and (out/'pause-after-phase').exists():
        print('PAUSED AFTER REQUESTED CASE PHASE',flush=True)
        return
    cases=json.loads((Path(args.fixtures)/'fixtures.json').read_text())['cases']
    if args.cases:cases=[c for c in cases if c['id'] in args.cases]
    rows=[];selections=[];dedup={}
    if (out/'results.json').exists():
        previous=json.loads((out/'results.json').read_text());rows=previous['rows'];selections=previous['selections']
        for r in rows:dedup[r['output_sha256']]=out/r['output']
    done={s['case'] for s in selections}
    save_json(out/'environment.json',dict(uname=list(os.uname()),cpuinfo=Path('/proc/cpuinfo').read_text(),
        affinity=sorted(os.sched_getaffinity(0)),plugin_sha256=sha(args.plugin),search_harness_sha256=sha(__file__),
        worker_harness_sha256=sha(Path(__file__).with_name('defaults_compare.py')),
        fixtures_manifest_sha256=sha(Path(args.fixtures)/'fixtures.json'),probe_manifest_sha256=sha(Path(args.probe)/'fixtures.json')))
    for case in cases:
        if case['id'] in done:continue
        search=Search(args,case);best={};matched={}
        print('START '+case['id'],flush=True)
        for algorithm in ALGORITHMS:
            best[algorithm]=search.best(algorithm)
            b=best[algorithm]
            print(f"BEST {case['id']} {algorithm} parameter={b['strength']:.5g} PSNR={b['quality']['psnr_db']:.3f} response={b['response']:.4f}",flush=True)
        # Clamp the initial median target to the common measured reachable
        # response interval, so a sigma-only noise floor cannot masquerade as
        # equal-strength output. Preserve the constraint and any mismatch.
        ranges={}
        for algorithm in ALGORITHMS:
            lo,hi=best[algorithm]['search_bounds']
            if lo==0:lo=case['sigma']/256
            values=[search.response(algorithm,s) for s in (lo,best[algorithm]['strength'],hi) if s>0]
            ranges[algorithm]=[min(values),max(values)]
        common_low=max(v[0] for v in ranges.values())
        common_high=min(v[1] for v in ranges.values())
        median_target=statistics.median(b['response'] for b in best.values())
        assert common_low<=common_high, 'No common reachable noise response interval'
        target=min(max(median_target,common_low),common_high)
        print(f'TARGET {case["id"]} median={median_target:.4f} common=[{common_low:.4f},{common_high:.4f}] chosen={target:.4f}',flush=True)
        for algorithm in ALGORITHMS:
            matched[algorithm]=search.match(algorithm,best[algorithm],target)
            m=matched[algorithm]
            print(f"MATCH {case['id']} {algorithm} parameter={m['strength']:.5g} PSNR={m['quality']['psnr_db']:.3f} response={m['response']:.4f} target={target:.4f} ok={m['matched']}",flush=True)
        for mode,choices in (('best',best),('matched',matched)):
            for algorithm in ALGORITHMS:rows+=search.retain(algorithm,mode,choices[algorithm],dedup)
        selections.append(dict(case=case['id'],best=best,matched=matched,target_response=target,
            initial_median_target=median_target,common_response_interval=[common_low,common_high],
            reachable_response_ranges=ranges))
        save_json(out/'results.json',dict(schema='nss.oracle-tuning.v1',rows=rows,selections=selections,
            criterion='Best observed clean-reference PSNR within bounded coarse + midpoint refinement search',
            matching='Frobenius RMS ratio ||D(y1)-D(y2)||/||y1-y2|| for two independent AWGN inputs, same clean image and parameters',
            matching_target='Per-case median response clamped to common measured non-bypass interval; sigma0 discontinuity excluded; tolerance max0.005 or5percent of target',
            caveats=['oracle tuning uses clean image; not deployable noise estimation','two-noise response is an approximate residual-noise measure, includes noise-induced texture instability','only sigma or NLM h tuned; other public parameters remain default','only selected pixel arrays retained; full search metrics and hashes retained']))
        print(f"CASE COMPLETE {case['id']} rows={len(rows)}",flush=True)
        # This is a task-owned per-case scratch directory, never a source tree.
        assert search.scratch.parent==out/'scratch'
        shutil.rmtree(search.scratch)
    print('TUNING COMPLETE',flush=True)


if __name__=='__main__':main()
