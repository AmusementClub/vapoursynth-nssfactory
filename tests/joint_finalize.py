#!/usr/bin/env python3
"""Finalize sigma25 structural winners without repeating their completed searches."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import statistics

from defaults_compare import ALGORITHMS
from denoise_tune import Search
from paper_compare import save_json,sha


def local_match(search,algorithm,initial,target):
    value=initial['strength'];tol=max(.005,.05*target)
    search.response(algorithm,value)
    lo_limit=search.case['sigma']/256;hi_limit=search.case['sigma']*16
    for _ in range(12):
        points=sorted((s,r) for (a,s),r in search.responses.items() if a==algorithm and s>0)
        nearest=min(points,key=lambda point:abs(point[1]-target))
        if abs(nearest[1]-target)<=tol:break
        brackets=[(a,b) for a,b in zip(points,points[1:]) if (a[1]-target)*(b[1]-target)<=0]
        if brackets:
            a,b=min(brackets,key=lambda pair:pair[1][0]-pair[0][0]);candidate=(a[0]+b[0])/2
        else:
            # Local expansion around the already-tuned setting; avoids blindly
            # rerunning expensive far endpoints before a bracket is needed.
            s,r=nearest;candidate=min(hi_limit,s*1.25) if r>target else max(lo_limit,s/1.25)
            if candidate==s:break
        search.response(algorithm,candidate)
    points=[(s,r) for (a,s),r in search.responses.items() if a==algorithm and s>0]
    value,response=min(points,key=lambda point:abs(point[1]-target))
    r=search.cache[(algorithm,value)]
    return dict(strength=value,response=response,target_response=target,tolerance=tol,
        absolute_response_error=abs(response-target),matched=abs(response-target)<=tol,
        quality=r['quality'],output_sha256=r['output_sha256'],structural_parameters=search.args.profiles.get(algorithm,{}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','probe','plugin','baseline','structure','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--timeout',type=int,default=240);args=p.parse_args()
    assert os.sched_getaffinity(0)=={0}
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    structure=json.loads(Path(args.structure).read_text());args.profiles={}
    base={s['case']:s['best'] for s in json.loads(Path(args.baseline).read_text())['selections']}
    cases=[c for c in json.loads((Path(args.fixtures)/'fixtures.json').read_text())['cases'] if c['sigma']==25]
    rows=[];selections=[];profiles_by_case={}
    save_json(out/'environment.json',dict(uname=list(os.uname()),cpuinfo=Path('/proc/cpuinfo').read_text(),
        plugin_sha256=sha(args.plugin),finalizer_sha256=sha(__file__),profiles=args.profiles,
        timing_policy='Single filtered call, no warmup; quality and indicative cost only, not a performance gate'))
    for case in cases:
        args.profiles={}
        search=Search(args,case);initial={}
        for algorithm in ALGORITHMS:
            options=[({},base[case['id']][algorithm])]
            for trial in structure['trials']:
                if trial['algorithm']==algorithm:
                    seed=next(c for c in trial['cases'] if c['case']==case['id'])
                    options.append((trial['parameters'],seed))
            profile,seed=max(options,key=lambda option:option[1]['quality']['psnr_db'])
            args.profiles[algorithm]=profile
            r=search.evaluate(algorithm,seed['strength'])
            assert r['output_sha256']==seed['output_sha256']
            initial[algorithm]=dict(strength=seed['strength'],quality=r['quality'],output_sha256=r['output_sha256'],
                response=search.response(algorithm,seed['strength']),structural_parameters=profile)
            print(f"SEED {case['id']} {algorithm} {profile} sigma/h={seed['strength']:.5g} psnr={r['quality']['psnr_db']:.3f} rho={initial[algorithm]['response']:.4f}",flush=True)
        target=statistics.median(s['response'] for s in initial.values());matched={}
        for algorithm in ALGORITHMS:
            matched[algorithm]=local_match(search,algorithm,initial[algorithm],target)
            m=matched[algorithm]
            print(f"MATCH {case['id']} {algorithm} sigma/h={m['strength']:.5g} psnr={m['quality']['psnr_db']:.3f} rho={m['response']:.4f} target={target:.4f} ok={m['matched']}",flush=True)
        # If a measured floor blocks the median, select a common measured
        # interval and retry; never silently accept mismatched strength.
        if not all(m['matched'] for m in matched.values()):
            ranges={a:[min(r for (b,s),r in search.responses.items() if b==a and s>0),
                       max(r for (b,s),r in search.responses.items() if b==a and s>0)] for a in ALGORITHMS}
            low=max(r[0] for r in ranges.values());high=min(r[1] for r in ranges.values())
            if low<=high:
                target=min(max(target,low),high)
                matched={a:local_match(search,a,initial[a],target) for a in ALGORITHMS}
        best={}
        for algorithm in ALGORITHMS:
            r=max((r for (a,s),r in search.cache.items() if a==algorithm),key=lambda r:r['quality']['psnr_db'])
            best[algorithm]=dict(strength=r['strength'],quality=r['quality'],output_sha256=r['output_sha256'],
                response=search.response(algorithm,r['strength']),structural_parameters=args.profiles.get(algorithm,{}))
        measured={}
        for mode,choices in (('best',best),('matched',matched)):
            for algorithm in ALGORITHMS:
                choice=choices[algorithm];key=(algorithm,choice['strength'])
                if key not in measured:
                    filename=f"{case['id']}-{algorithm}-{len(measured):02d}.f32";path=out/filename
                    r=search.call(algorithm,choice['strength'],False,path,cold=True)
                    assert r['output_sha256']==choice['output_sha256'], 'Final independent run differs from tuning'
                    probe=search.evaluate(algorithm,choice['strength'],True)
                    probe_path=out/(filename+'.probe.f32');shutil.copy2(probe['scratch_output'],probe_path)
                    r.update(probe_output=probe_path.name,probe_output_sha256=probe['output_sha256'],
                        search_output_sha256=choice['output_sha256'],independent_repeat_hash_verified=True,
                        measurement_id=filename,input_difference_energy=search.input_difference_energy)
                    measured[key]=r
                row=copy.deepcopy(measured[key]);row.update(mode=mode,repeat=0,ok=True,selection=choice)
                rows.append(row)
        selections.append(dict(case=case['id'],best=best,matched=matched,target_response=target))
        profiles_by_case[case['id']]=copy.deepcopy(args.profiles)
        save_json(out/'results.json',dict(schema='nss.joint-tuning.v1',rows=rows,selections=selections,
            profiles_by_case=profiles_by_case,scope='three sigma25 RGB512 ROIs; per-image best observed bounded structural + sigma/h search',
            repeat_policy='Selected primary output independently rerun and hash-compared to search; identical best/matched configurations share the same measured run',
            timing_policy='single cold filter call, source preloaded; indicative cost only, not warm paired benchmarking'))
        print('JOINT CASE COMPLETE '+case['id'],flush=True)
        shutil.rmtree(search.scratch)
    print('JOINT FINALIZATION COMPLETE',flush=True)


if __name__=='__main__':main()
