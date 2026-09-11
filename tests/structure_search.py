#!/usr/bin/env python3
"""Bounded coordinate search of public structural controls on three sigma25 ROIs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

from denoise_tune import Search
from paper_compare import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','probe','plugin','baseline','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--algorithms',nargs='+',default=['WNNM','LSSC','NLH'])
    p.add_argument('--light-secondary',action='store_true')
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cases=[c for c in json.loads((Path(args.fixtures)/'fixtures.json').read_text())['cases'] if c['sigma']==25]
    baseline=json.loads(Path(args.baseline).read_text())['selections']
    base={s['case']:s['best'] for s in baseline}
    previous=json.loads((out/'structure-trials.json').read_text()) if (out/'structure-trials.json').exists() else {}
    trials=previous.get('trials',[]);profiles=previous.get('selected_profiles',{});trial_index=len(trials)
    def score(algorithm,profile):
        nonlocal trial_index
        for previous_trial in trials:
            if previous_trial['algorithm']==algorithm and previous_trial['parameters']==profile:
                return previous_trial
        trial_index+=1;values=[];measures=[]
        for case in cases:
            token=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:8]
            work=out/f'trial-{trial_index:02d}-{algorithm}-{token}'
            config=SimpleNamespace(fixtures=args.fixtures,probe=args.probe,plugin=args.plugin,out=str(work),
                timeout=240,profiles={algorithm:profile})
            search=Search(config,case)
            strengths=sorted({0.,12.5,25.,37.5,50.,75.,100.,base[case['id']][algorithm]['strength']})
            for strength in strengths:search.evaluate(algorithm,strength)
            for _ in range(2):
                strengths=sorted(s for a,s in search.cache if a==algorithm)
                best=max(strengths,key=lambda s:search.cache[(algorithm,s)]['quality']['psnr_db'])
                index=strengths.index(best);intervals=[]
                if index:intervals.append((strengths[index-1],best))
                if index+1<len(strengths):intervals.append((best,strengths[index+1]))
                lo,hi=max(intervals,key=lambda pair:pair[1]-pair[0]);search.evaluate(algorithm,(lo+hi)/2)
            best=max(search.cache.values(),key=lambda r:r['quality']['psnr_db'])
            values.append(best['quality']['psnr_db'])
            measures.append(dict(case=case['id'],strength=best['strength'],quality=best['quality'],
                cold_seconds=best['seconds'],output_sha256=best['output_sha256']))
            print(f"STRUCT {algorithm} {profile} {case['id']} sigma={best['strength']:.4g} PSNR={best['quality']['psnr_db']:.3f} cold={best['seconds']:.3f}s",flush=True)
            shutil.rmtree(search.scratch)
        trial=dict(algorithm=algorithm,parameters=profile,mean_psnr=statistics.mean(values),cases=measures)
        trials.append(trial);save_json(out/'structure-trials.json',dict(trials=trials,selected_profiles=profiles))
        return trial
    for algorithm in args.algorithms:
        champion=dict(parameters={},mean_psnr=statistics.mean(base[c['id']][algorithm]['quality']['psnr_db'] for c in cases))
        def consider(changes):
            nonlocal champion
            trial=score(algorithm,{**champion['parameters'],**changes})
            # Keep every measured point for the final per-image oracle, but
            # only route subsequent coordinate combinations through gains
            # exceeding0.05dB. This bounds expensive low-yield cross products.
            if trial['mean_psnr']>champion['mean_psnr']+.05:champion=trial
            print(f"CHAMPION {algorithm} {champion['parameters']} mean={champion['mean_psnr']:.4f}",flush=True)
        if algorithm=='WNNM':
            consider({'group_size':16});consider({'group_size':32})
            consider({'block_step':4});consider({'block_step':2})
            if args.light_secondary:
                # Screen independent switches at a less expensive group16
                # anchor; only promote combined profiles after measuring them.
                anchor={'group_size':16,'block_step':4}
                a=score(algorithm,anchor)
                b=score(algorithm,{**anchor,'residual':1})
                c=score(algorithm,{**anchor,'adaptive_aggregation':0})
                candidates=[champion,a,b,c]
                if b['mean_psnr']>a['mean_psnr'] and c['mean_psnr']>a['mean_psnr']:
                    candidates.append(score(algorithm,{**anchor,'residual':1,'adaptive_aggregation':0}))
                champion=max(candidates,key=lambda r:r['mean_psnr'])
            else:
                consider({'residual':1});consider({'adaptive_aggregation':0})
        elif algorithm=='LSSC':
            consider({'block_step':4});consider({'block_step':2})
            # b8/step1 exceeds the public grid-size guard on512x512; do not
            # silently bypass that guard or change implementation internals.
            consider({'block_size':4,'block_step':2})
        elif algorithm=='NLH':
            if args.light_secondary:
                # Screen q/group at default stride, then validate the best
                # combination with step4. Step2 is left outside this budget.
                consider({'q':2});consider({'q':8});consider({'group_size':8});consider({'block_step':4})
            else:
                consider({'block_step':4});consider({'block_step':2})
                consider({'q':2});consider({'q':8});consider({'group_size':8})
        profiles[algorithm]=champion['parameters']
        save_json(out/'selected-profiles.json',dict(profiles=profiles,
            criterion='shared public profile maximizing observed mean PSNR of3nativeRGB512 ROIs at injected sigma25; sigma separately refined per image',
            boundary='bounded sequential coordinate search, not exhaustive global optimization; structure held fixed across noise levels in final search'))
    save_json(out/'structure-trials.json',dict(trials=trials,selected_profiles=profiles))
    print('STRUCTURE SEARCH COMPLETE '+json.dumps(profiles),flush=True)


if __name__=='__main__':main()
