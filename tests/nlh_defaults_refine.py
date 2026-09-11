#!/usr/bin/env python3
"""Recheck screening candidates across new development photographs and refine.

One predetermined noise level per original is used in this development pass;
selection/test evaluation separately spans all six injected noise levels.
"""
import argparse
import copy
import json
from pathlib import Path

from nlh_defaults_search import Engine,LANES,baseline,change,identity,lane_cases,legal,shortlist,strength_variants,nondominated,fitted_structures
from nlh_defaults_inputs import save_json,validate_manifest,file_sha


def development_cases(manifest,lane):
    rows=lane_cases(manifest,lane,'development')
    images=list(dict.fromkeys(c['image'] for c in rows if c['dataset']=='DIV2K'))
    levels=(5,15,25,50) if lane=='gray-low' else (75,100) if lane=='gray-high' else (5,15,25,50,75,100)
    result=[next(c for c in rows if c['image']==image and c['sigma']==levels[i%len(levels)] and c['realization']==0)
            for i,image in enumerate(images)]
    return result+[c for c in rows if c['dataset']=='CC']


def neighbors(p):
    result=[]
    for stage in (0,1):
        for key in ('block_size','block_step','search_window'):
            for delta in (-1,1):result.append(change(p,key,p[key][stage]+delta,stage))
        for key in ('q','group_size'):
            for value in (p[key][stage]//2,p[key][stage]*2):result.append(change(p,key,value,stage))
    for key,bound in (('basic_iters',7),('wiener_iters',3)):
        for value in (p[key]-1,p[key]+1):
            if 1<=value<=bound:result.append(change(p,key,value))
    for value in (p['lambda_basic']-.2,p['lambda_basic']+.2):
        if 0<=value<=1:result.append(change(p,'lambda_basic',round(value,12)))
    if p['lambda_basic']==0:
        result.append(change(p,'basic_iters',1))
    return list({identity(v):v for v in result if legal(v)}.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plugin','inputs','out','profiles'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--lane',choices=LANES,required=True);parser.add_argument('--cpu',type=int,default=0)
    parser.add_argument('--size',type=int,default=256);parser.add_argument('--score-size',type=int,default=96)
    parser.add_argument('--save-images',action='store_true');parser.add_argument('--joint-passes',type=int,default=1)
    parser.add_argument('--refit-only',action='store_true')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Refit candidates only; dense v4 baseline remains in the primary comparison sets')
    parser.add_argument('--native-bridge')
    args=parser.parse_args()
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    driver=dict(script_sha256=file_sha(__file__),joint_passes=args.joint_passes,refit_only=args.refit_only,
                baseline_included=not args.skip_baseline)
    stamp=output/'refinement.json'
    if stamp.exists() and json.loads(stamp.read_text())!=driver:
        raise ValueError('refinement procedure changed; use a new output directory')
    save_json(stamp,driver)
    manifest=json.loads((Path(args.inputs)/'inputs.json').read_text());validate_manifest(manifest)
    seeds=json.loads(Path(args.profiles).read_text())['profiles']
    if any(not legal(p) for p in seeds.values()):raise ValueError('invalid refinement profile')
    engine=Engine(args,development_cases(manifest,args.lane));rows={}
    def evaluate(p,label):
        key=identity(p)
        if key not in rows:
            rows[key]=engine.evaluate(p,label)
            save_json(engine.out/'trials.json',list(rows.values()))
            r=rows[key];print(args.lane,len(rows),label,r['mean_psnr'],r['mean_ssim'],r['geomean_seconds'],flush=True)
        return rows[key]
    def fit(p,label):
        choices=[evaluate(v,label) for v in strength_variants(p)]
        return max(choices,key=lambda r:(r['mean_psnr'],r['mean_ssim']))
    if not args.skip_baseline:
        evaluate(baseline(args.lane),'current-defaults')
    for label,p in seeds.items():fit(p,label)
    for iteration in range(0 if args.refit_only else args.joint_passes):
        starts=list({r['id']:r for r in shortlist(list(rows.values())).values()}.values())
        for start in starts:
            for p in neighbors(start['parameters']):fit(p,f'joint-{iteration}')
    for start in {r['id']:r for r in shortlist(list(rows.values())).values()}.values():
        p=copy.deepcopy(start['parameters'])
        for key in ('hard_strength','wiener_sigma_scale'):
            choices=[evaluate(change(p,key,p[key]*factor),'fine-coefficients') for factor in (2**-.5,1.,2**.5)]
            p=copy.deepcopy(max(choices,key=lambda r:(r['mean_psnr'],r['mean_ssim']))['parameters'])
    chosen=shortlist(list(rows.values()))
    save_json(engine.out/'summary.json',dict(lane=args.lane,metadata=engine.metadata,refinement=driver,selected=chosen,
              frontier=nondominated(fitted_structures(list(rows.values()))),trials=len(rows),
              cohort_policy='all new development originals, deterministic noise-level rotation, first realization; all CC development scenes'))
    save_json(engine.out/'profiles.json',dict(profiles={k:v['parameters'] for k,v in chosen.items()},
              source=engine.metadata,selection='development-only candidates; not final defaults'))


if __name__=='__main__':main()
