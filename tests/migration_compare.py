#!/usr/bin/env python3
"""Thin old/current migration orchestrator; all filter/quality work uses balanced_campaign.

Alternates binaries in each repeat. Output equality is required only within a
model across repeated static frames, never between legacy and current models.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from paper_compare import save_json, sha


def report(root, identity, rows):
    groups = {}
    for row in rows:
        if row['ok']:
            groups.setdefault((row['case'],row['algorithm'],row['profile']),{}).setdefault(row['variant'],[]).append(row)
    results = []
    for (case, algorithm, profile), variants in groups.items():
        for items in variants.values():
            if len({r['output_sha256'] for r in items}) != 1: raise ValueError('within-model repeat hash changed')
            if len({json.dumps(r['supplied_parameters'],sort_keys=True) for r in items}) != 1:
                raise ValueError('within-model repeat parameters changed')
        if set(variants) != {'legacy','current'}: continue
        old, new = variants['legacy'], variants['current']
        pairs = [(a,b) for a in old for b in new if a['repeat']==b['repeat'] and
                 a['timing_eligible'] and b['timing_eligible'] and a['host']==b['host'] and
                 a['noisy_sha256']==b['noisy_sha256'] and a['clean_sha256']==b['clean_sha256']]
        ratios = [a['seconds']/b['seconds'] for a,b in pairs]
        results.append(dict(case=case, algorithm=algorithm, profile=profile,
            legacy_psnr_db=old[0]['quality']['psnr_db'], current_psnr_db=new[0]['quality']['psnr_db'],
            current_minus_legacy_psnr_db=new[0]['quality']['psnr_db']-old[0]['quality']['psnr_db'],
            current_minus_legacy_ssim=new[0]['quality']['ssim']-old[0]['quality']['ssim'],
            noisy_quality=old[0]['noisy_quality'], eligible_pairs=len(pairs),
            legacy_over_current_time=statistics.median(ratios) if ratios else None,
            ratios=ratios, paired_timing_complete=len(pairs)==identity['repeats'],
            legacy_seconds=statistics.median(r['seconds'] for r in old),
            current_seconds=statistics.median(r['seconds'] for r in new),
            attribution=old[0]['model_attribution'],
            interpretation='quality/runtime migration; no same-model efficiency or numerical-equivalence claim'))
    save_json(root/'summary.json', dict(schema='nss.nlh-twsc-migration.v1', identity=identity,
        comparisons=results, failed_or_incomplete=[r for r in rows if not r['ok']],
        cross_model_equality_required=False,
        scope='same saved static 128 crop pilot; no full-frame or fresh held-out quality claim'))
    lines = ['# Historical NLH/TWSC migration', '',
        'Different model/default comparison. Legacy/current time ratios do not establish implementation efficiency.', '',
        '| Case | Algorithm | Recipe | Old PSNR | Current PSNR | Current SSIM delta | Old/current time | Eligible pairs |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        ratio=f"{r['legacy_over_current_time']:.4f}" if r['legacy_over_current_time'] is not None else 'unqualified'
        lines.append(f"| {r['case']} | {r['algorithm']} | {r['profile']} | {r['legacy_psnr_db']:.3f} | {r['current_psnr_db']:.3f} | {r['current_minus_legacy_ssim']:+.5f} | {ratio} | {r['eligible_pairs']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy','current','fixtures','out','legacy-source-sha256','current-source-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--profiles',default=str(Path(__file__).with_name('migration_nlh_twsc.json')))
    p.add_argument('--split',choices=['search','validation'],default='search')
    p.add_argument('--sigma',type=float,default=25)
    p.add_argument('--noise-repeat',type=int,default=0)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--twsc-timeout',type=float,default=120)
    p.add_argument('--budget-seconds',type=float,default=1800)
    p.add_argument('--target-seconds',type=float,default=.5)
    p.add_argument('--max-frames',type=int,default=256)
    args=p.parse_args()
    if not hasattr(os,'sched_getaffinity') or os.sched_getaffinity(0)!={0}: p.error('run under taskset -c0 on verified C4')
    for name in ('repeats','timeout','twsc_timeout','budget_seconds','target_seconds','max_frames'):
        if getattr(args,name)<=0:p.error(name+' must be positive')
    for value in (args.legacy_source_sha256,args.current_source_sha256):
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('source SHA256 must be 64 lowercase hex digits')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    worker=Path(__file__).with_name('balanced_campaign.py').resolve()
    fixtures=Path(args.fixtures).resolve()
    cases=[c for c in json.loads((fixtures/'fixtures.json').read_text())['cases']
           if c['split']==args.split and c['sigma']==args.sigma and c['noise_repeat']==args.noise_repeat]
    if not cases:p.error('no cases match')
    profiles=json.loads(Path(args.profiles).read_text())['profiles']
    identity=dict(schema='nss.nlh-twsc-migration-identity.v1',
        sources_sha256={'legacy':args.legacy_source_sha256,'current':args.current_source_sha256},
        plugins_sha256={v:sha(getattr(args,v)) for v in ('legacy','current')},
        worker_sha256=sha(worker), orchestrator_sha256=sha(__file__), profiles_sha256=sha(args.profiles),
        helper_sha256={f:sha(worker.with_name(f)) for f in ('paper_compare.py','defaults_compare.py')},
        fixtures_sha256=sha(fixtures/'fixtures.json'), repeats=args.repeats,
        cases=[c['id'] for c in cases], timeout=args.timeout, twsc_timeout=args.twsc_timeout,
        target_seconds=args.target_seconds,max_frames=args.max_frames,
        old_model='archived pre-full-image NLH v2 / simplified TWSC',
        current_model='current B2 public full-image NLH/TWSC; frame properties recorded',
        cross_model_equality_required=False)
    ident=root/'identity.json'
    if ident.exists():
        if json.loads(ident.read_text())!=identity:raise ValueError('resume identity differs')
    else:save_json(ident,identity)
    raw=root/'results.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    done={(r['case'],r['algorithm'],r['profile'],r['repeat'],r['variant']) for r in rows}
    jobs=[]
    for c in cases:
        for profile in profiles:
            for repeat in range(args.repeats):
                for variant in (('legacy','current') if repeat%2==0 else ('current','legacy')):
                    jobs.append((c,profile,repeat,variant))
    started=time.monotonic(); deadline=started+args.budget_seconds
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case,profile,repeat,variant in jobs:
        key=(case['id'],profile['algorithm'],profile['id'],repeat,variant)
        if key in done:continue
        if time.monotonic()>=deadline:break
        output=root/('-'.join(map(str,key))+'.f32')
        candidate=dict(id=profile['id'],parameters=profile['parameters'])
        command=[sys.executable,str(worker),'worker','--plugin',str(Path(getattr(args,variant)).resolve()),
            '--fixtures',str(fixtures),'--case',case['id'],'--algorithm',profile['algorithm'],
            '--candidate-json',json.dumps(candidate),'--output',str(output.resolve()),
            '--target-seconds',str(args.target_seconds),'--max-frames',str(args.max_frames)]
        timeout=min(args.twsc_timeout if profile['algorithm']=='TWSC' else args.timeout,max(.01,deadline-time.monotonic()))
        row=dict(case=case['id'],algorithm=profile['algorithm'],profile=profile['id'],repeat=repeat,
            variant=variant,model_attribution=profile['attribution'],ok=False,status='pending')
        before=time.monotonic()
        try:
            result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
            output.with_suffix('.log').write_bytes(result.stdout)
            if result.returncode:raise RuntimeError(result.stdout[-3500:].decode(errors='replace'))
            row.update(json.loads(Path(str(output)+'.json').read_text()),ok=True,status='complete')
        except subprocess.TimeoutExpired as error:
            output.with_suffix('.log').write_bytes(error.stdout or b'')
            row.update(status='timeout',timeout_seconds=timeout,error='bounded worker killed; no comparison conclusion')
        except Exception as error:row.update(status='failed',error=str(error))
        row['worker_wall_seconds']=time.monotonic()-before
        with raw.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        rows.append(row);done.add(key)
        print(' '.join(map(str,key)),row['status'],flush=True)
        report(root,identity,rows)
    remaining=[dict(case=c['id'],algorithm=p['algorithm'],profile=p['id'],repeat=r,variant=v)
        for c,p,r,v in jobs if (c['id'],p['algorithm'],p['id'],r,v) not in done]
    save_json(root/'completion.json',dict(complete=not remaining,scheduled=len(jobs),recorded=len(rows),
        successful=sum(r['ok'] for r in rows),remaining=remaining,
        wall_seconds=time.monotonic()-started,budget_seconds=args.budget_seconds))
    report(root,identity,rows)


if __name__=='__main__':main()
