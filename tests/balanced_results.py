#!/usr/bin/env python3
"""Read-only consumer for frozen finalist phases; no selection or retuning.

`plan` reads fixture metadata/configs only. `report` reads future results solely
for explicitly requested --phases; unavailable phases remain pending.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from balanced_assess import assess, summarize, ALGORITHMS

SELECTION_SHA = 'ae4dd13d0cf410c229b2bdedba732975d9f6618f600142922a43fa277aaf8611'
METRICS_SHA = '2fc0be5dc39fd7973d88ccd4c618cafd30143575835dca403f9c9f6864a4f6ee'
DEFAULTS_SHA = '7362db7edb6302e6075e8d59696ab6967275b6e07dee94774f899f0e5328e17e'
PHASES = ('sealed', 'quality512', 'formal512', 'temporal1', 'temporal2', 'temporal-twsc1')
TIMING_POLICY = dict(minimum_sibling_ticks=10,maximum_sibling_busy_fraction=.01,maximum_steal_ticks=0,
    sibling_topology=['0-1','0,1'],minimum_valid_pairs_per_case=7,
    target_seconds='Calibration target only; no actual>=1-second minimum is present in the frozen worker',
    comparison='Joint NLM d/h recipe; repeated static noisy input gives default d1 no independent temporal noise information',
    confidence_interval='Deterministic20000-resample percentile95% CI of paired speed-ratio median; lower bound>1 supports a speedup claim per plan, separate from median>=1.2; no lower-bound>=1.2 requirement',
    implementation_policy='This is the parameter-worker <=1% sibling busy policy, not the separate implementation .999-idle policy')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def frozen_selection(root):
    if sha(root/'selection.json') != SELECTION_SHA: raise ValueError('frozen selection hash differs')
    selection = read(root/'selection.json')
    for name, digest in selection['files_sha256'].items():
        if name.startswith('._'):
            if not name.endswith('.json') or name[2:] not in selection['files_sha256']:
                raise ValueError('unexpected sidecar omission')
            continue
        if sha(root/name) != digest: raise ValueError('frozen config hash differs: '+name)
    return selection


def build_plan(args):
    root=Path(args.selection); selection=frozen_selection(root); fixtures=Path(args.fixtures)
    templates={size:read(fixtures/f'fixtures{size}'/'fixtures.json') for size in (256,512)}
    if sha(fixtures/'fixtures256/fixtures.json') != selection['evidence']['full_fixture_manifest_sha256']:
        raise ValueError('original256 fixture template differs')
    prior_fixtures={}
    for item in selection['evidence']['results']:
        identity_path=Path(item['path']).with_name('identity.json')
        if sha(identity_path)!=item['identity_sha256']:raise ValueError('prior validation identity changed')
        for fmt in ('gray','rgb'):
            if identity_path.parent.name==f'validation-{fmt}-256':prior_fixtures[fmt]=read(identity_path)['fixtures_sha256']
    stages=[]
    for phase in PHASES:
        profiles=['gray','rgb'] if phase=='formal512' else ['rgb-mid-high'] if phase=='temporal-twsc1' else list(selection['profiles'])
        for profile in profiles:
            fmt=profile.split('-')[0]; temporal=phase.startswith('temporal')
            size=512 if phase in ('quality512','formal512') else 256
            split='sealed' if phase=='sealed' else 'validation'
            if phase=='formal512':
                config=f'formal512-{fmt}-recipes.json';sigmas=[25,75] if fmt=='gray' else [5,25,75]
            elif phase=='temporal-twsc1':config='twsc-temporal-r1.json';sigmas=[25]
            else:
                scope=selection['profiles'][profile];sigmas=scope['sigmas']
                config=f'{profile}-temporal-r{phase[-1]}.json' if temporal else scope['recipes']
            cases=[r for r in templates[size]['cases'] if r['split']==split and r['format']==fmt and r['sigma'] in sigmas
                   and (split=='sealed' or r['noise_repeat']==0 and r['image'] in ('div2k-0874','div2k-0898'))]
            if temporal or phase=='quality512':cases=[r for r in cases if r['sigma'] in (5,25,75)]
            if phase=='temporal-twsc1':cases=[r for r in cases if r['image']=='div2k-0874']
            name=phase if phase=='temporal-twsc1' else f'{phase}-{profile}'
            cfg=read(root/config); repeats=7 if phase=='formal512' else 1
            jobs=[dict(case=c['id'],algorithm=a,candidate=r['id'],repeat=n) for c in cases
                  for a,recipes in cfg['algorithms'].items() for r in recipes for n in range(repeats)]
            stages.append(dict(name=name,phase=phase,format=fmt,size=size,split=split,temporal=temporal,
                config=config,config_sha256=sha(root/config),algorithms=list(cfg['algorithms']),cases=cases,
                jobs=jobs,repeats=repeats,radius=int(phase[-1]) if phase in ('temporal1','temporal2') else 1 if temporal else None,
                frames=11 if temporal else None,target_seconds=None if temporal else 1 if phase=='formal512' else .25,
                timeout=180,twsc_timeout=60 if phase=='sealed' else 180,max_frames=None if temporal else 4096,
                finalists=selection['profiles'][profile]['finalists'] if phase=='sealed' else None,
                fixture_manifest=f'inputs-{split}-{fmt}-{size}/fixtures.json' if not temporal else None,
                expected_temporal_fixture_sha256=prior_fixtures[fmt] if temporal else None,
                source_template_sha256=sha(fixtures/f'fixtures{size}'/'fixtures.json')))
    value=dict(schema='nss.frozen-result-consumption-plan.v1',selection_sha256=SELECTION_SHA,
        selection=str(root.resolve()),fixtures=str(fixtures.resolve()),stages=stages,
        reference_assessments={kind:dict(path=str(Path(path).resolve()),sha256=sha(path)) for kind,path in
            [('validation256',args.validation_assessment),('unequal256',args.unequal_assessment)]},
        qualification='Case metadata only; creation did not open prospective algorithm results. No retuning or inferred future outcomes.')
    output=Path(args.out)
    if output.exists(): raise ValueError('refusing to overwrite consumption plan')
    save(output,value);print(json.dumps(dict(plan=str(output),sha256=sha(output),stages=len(stages))))


def key(row, temporal=False):
    return row['case'],row['algorithm'],row['recipe' if temporal else 'candidate'],0 if temporal else row['repeat']


def coverage(stage, rows):
    """Derive coverage from the frozen job grid, never from a reported count."""
    expected={(r['case'],r['algorithm'],r['candidate'],r['repeat']) for r in stage['jobs']}
    actual=[key(r,stage['temporal']) for r in rows];counts=Counter(actual)
    missing=expected-set(actual);extra=set(actual)-expected;duplicates=[k for k,n in counts.items() if n>1]
    return dict(expected_jobs=len(expected),recorded_jobs=len(rows),
        missing=[dict(case=c,algorithm=a,candidate=p,repeat=n) for c,a,p,n in sorted(missing)],
        unexpected=[list(k) for k in sorted(extra)],duplicates=[list(k) for k in duplicates],
        status_counts=dict(Counter(r.get('status','missing-status') for r in rows)),
        all_successful=not missing and not extra and not duplicates and all(r.get('status')=='complete' for r in rows))


def expected_parameters(algorithm, recipe, case):
    value={} if algorithm=='NLM' else dict(sigma=case['sigma'])
    value.update(recipe['parameters'])
    if recipe.get('preset'): raise ValueError('consumer needs explicit resolution for preset '+recipe['preset'])
    if 'h_sigma' in value:value['h']=value.pop('h_sigma')*case['sigma']
    return value


def timing_eligible(row):
    activity=row.get('cpu_activity') or {}
    return bool(row.get('cpu0_thread_siblings') in ('0-1','0,1') and activity.get('cpu1') and
        activity['cpu1']['total_ticks']>=10 and activity['cpu1']['busy_fraction']<=.01 and
        all(v['steal_ticks']==0 for v in activity.values()))


def recipe_coverage(stage, checked, algorithm, candidate):
    """An unrelated recipe timeout must not hide a complete comparison."""
    candidates={candidate,'default'} if candidate else {'default'}
    expected=[r for r in stage['jobs'] if r['algorithm']==algorithm and r['candidate'] in candidates]
    missing=[r for r in checked['coverage']['missing'] if r['algorithm']==algorithm and r['candidate'] in candidates]
    failures=[r for r in checked.get('terminal_failures',[]) if r['algorithm']==algorithm and r['candidate'] in candidates]
    return dict(expected_jobs=len(expected),missing=missing,terminal_failures=failures,
        all_successful=bool(expected) and not missing and not failures and checked['integrity_pass'])


def formal_timing(rows, requested_repeats):
    complete=[r for r in rows if r.get('status')=='complete'];output=[]
    for case,algorithm,candidate in sorted({(r['case'],r['algorithm'],r['candidate']) for r in complete if r['candidate']!='default'}):
        candidates={r['repeat']:r for r in complete if (r['case'],r['algorithm'],r['candidate'])==(case,algorithm,candidate)}
        defaults={r['repeat']:r for r in complete if (r['case'],r['algorithm'],r['candidate'])==(case,algorithm,'default')}
        pairs=[];ratios=[]
        for repeat in range(requested_repeats):
            b=defaults.get(repeat);r=candidates.get(repeat);reasons=[]
            if b is None or r is None:reasons.append('missing complete pair')
            elif not timing_eligible(b) or not timing_eligible(r):reasons.append('frozen sibling/steal eligibility failed')
            elif b['host']!=r['host'] or b['noisy_sha256']!=r['noisy_sha256']:reasons.append('host/input identity differs')
            ratio=b['seconds']/r['seconds'] if not reasons else None
            if ratio is not None:ratios.append(ratio)
            pairs.append(dict(repeat=repeat,eligible=not reasons,exclusion_reasons=reasons,speed_ratio=ratio,
                default_window_seconds=b.get('elapsed_seconds') if b else None,candidate_window_seconds=r.get('elapsed_seconds') if r else None,
                default_timed_frames=b.get('frames') if b else None,candidate_timed_frames=r.get('frames') if r else None,
                default_cpu_activity=b.get('cpu_activity') if b else None,candidate_cpu_activity=r.get('cpu_activity') if r else None))
        interval=None
        if len(ratios)>=7:
            rng=random.Random(int.from_bytes(hashlib.sha256(f'{case}:{algorithm}:{candidate}'.encode()).digest()[:8],'little'))
            draws=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(20000))
            interval=[draws[int(.025*len(draws))],draws[int(.975*len(draws))-1]]
        windows={}
        for variant in ('default','candidate'):
            values=[p[variant+'_window_seconds'] for p in pairs if p[variant+'_window_seconds'] is not None]
            windows[variant]=dict(count=len(values),minimum=min(values,default=None),median=statistics.median(values) if values else None,
                maximum=max(values,default=None),below_one_second=sum(v<1 for v in values),values=values)
        output.append(dict(case=case,algorithm=algorithm,candidate=candidate,valid_pairs=len(ratios),
            seven_complete_valid_pairs=len(ratios)==requested_repeats and len(ratios)>=7,
            median_speed_ratio=statistics.median(ratios) if ratios else None,median_speed_ratio_ci95=interval,
            ci_lower_bound_above_one=bool(interval and interval[0]>1),actual_window_seconds=windows,pairs=pairs,policy=TIMING_POLICY))
    return output


def verify_stage(stage, directory, selection_root, selection, results_root, fixture_root):
    errors=[];warnings=[];rows=[]
    raw=directory/'results.jsonl';identity_path=directory/'identity.json'; identity=read(identity_path) if identity_path.exists() else None
    if raw.exists():
        for n,line in enumerate(raw.read_text().splitlines(),1):
            try:
                row=json.loads(line)
                key(row,stage['temporal'])
                rows.append(row)
            except (ValueError,TypeError) as e: errors.append(f'invalid raw row{n}: {e}')
            except KeyError as e: errors.append(f'missing job identity in raw row{n}: {e}')
    c=coverage(stage,rows)
    if c['unexpected']:errors.append('jobs outside frozen grid')
    if c['duplicates']:errors.append('duplicate job identities')
    if identity is None:errors.append('run identity missing')
    temporal=stage['temporal'];case_map={r['id']:r for r in stage['cases']}
    cfg=read(selection_root/stage['config']);recipes={(a,r['id']):r for a,v in cfg['algorithms'].items() for r in v}
    def match(got,expected,label):
        if got!=expected:errors.append(label+' differs')
    match(sha(selection_root/stage['config']),stage['config_sha256'],'frozen config')
    if identity is not None:
        expected=dict(plugin=selection['plugin_sha256'],recipes=stage['config_sha256'],harness=selection['temporal_worker_sha256'],
            fixtures=stage['expected_temporal_fixture_sha256'],
            cases=[r['id'] for r in stage['cases']],frames=11,radius=stage['radius'],timeout=180,twsc_timeout=180) if temporal else dict(
            plugin_sha256=selection['plugin_sha256'],harness_sha256=selection['worker_sha256'],candidates_sha256=stage['config_sha256'],
            metrics_sha256=METRICS_SHA,defaults_helper_sha256=DEFAULTS_SHA,split=stage['split'],
            cases=[r['id'] for r in stage['cases']],algorithms=stage['algorithms'],repeats=stage['repeats'],
            timeout=stage['timeout'],twsc_timeout=stage['twsc_timeout'],target_seconds=stage['target_seconds'],
            max_frames=4096,candidate_ids=None,finalists_sha256=sha(selection_root/stage['finalists']) if stage['finalists'] else None)
        for name,value in expected.items():match(identity.get(name),value,'identity.'+name)
        preset=sha(fixture_root/f'fixtures{stage["size"]}'/'nlh-v4.json')
        match(identity.get('nlh_v4' if temporal else 'nlh_v4_sha256'),preset,'NLH preset identity')
        if temporal:
            match(identity.get('helpers'),{'balanced_campaign.py':selection['worker_sha256'],
                'defaults_compare.py':DEFAULTS_SHA,'paper_compare.py':METRICS_SHA},'temporal helper identity')
        else:
            manifest=results_root/stage['fixture_manifest']
            if not manifest.exists():errors.append('downloaded materialized fixture manifest missing')
            else:
                match(identity.get('fixtures_sha256'),sha(manifest),'materialized fixture manifest identity')
                actual=read(manifest);template=read(fixture_root/f'fixtures{stage["size"]}'/'fixtures.json')
                original={r['id']:r for r in template['cases']}
                match(actual.get('parent_full_fixture_manifest_sha256'),stage['source_template_sha256'],'original full fixture identity')
                if any(r!=original.get(r['id']) for r in actual['cases']):errors.append('materialized fixture case differs from original')
                if not set(case_map).issubset({r['id'] for r in actual['cases']}):errors.append('materialized fixture grid missing selected cases')
    for r in rows:
        if r.get('status')!='complete':continue
        case=case_map.get(r['case']);recipe=recipes.get((r['algorithm'],r.get('recipe' if temporal else 'candidate')))
        if case is None or recipe is None:continue
        match(r.get('plugin_sha256'),selection['plugin_sha256'],'record plugin')
        match(r.get('harness_sha256'),selection['temporal_worker_sha256'] if temporal else selection['worker_sha256'],'record worker')
        match(r.get('supplied_parameters'),expected_parameters(r['algorithm'],recipe,case),'supplied parameters')
        match(r.get('sigma'),case['sigma'],'record sigma');match(r.get('split'),stage['split'],'record split')
        match(r.get('pipeline'),recipe.get('pipeline','single'),'record pipeline')
        if temporal:
            match(r.get('radius'),stage['radius'],'record radius');match(r.get('frames'),11,'temporal frames')
            match(r.get('shape'),[11,case['channels'],246,246],'temporal shape')
            match(r.get('parent_clean_sha256'),case['clean_sha256'],'temporal clean parent')
            effective=0 if r['algorithm']=='LSSC' else stage['radius']
            halo=effective if r['algorithm'] in ('NLM','LSSC') else 2*effective
            if recipe.get('pipeline')=='basic-final':halo*=2
            layout=r.get('temporal_layout',{})
            for name,value in [('requested_radius',stage['radius']),('effective_radius',effective),('source_halo',halo),
                ('interior_frames',list(range(halo,11-halo))),('boundary_frames',[n for n in range(11) if n<halo or n>=11-halo])]:
                match(layout.get(name),value,'temporal layout.'+name)
            if not r.get('repeat_exact') or not r.get('source_unchanged') or r.get('timed_source_fills')!=0:
                errors.append('temporal order/source invariant failed')
            match(r.get('final_sha256'),r.get('repeat_final_sha256'),'temporal repeat hash')
            if len(r.get('quality',[]))!=11:errors.append('missing per-frame quality')
            if len(r.get('source_frame_sha256',[]))!=11 or len(r.get('final_frame_sha256',[]))!=11:errors.append('missing temporal frame identities')
            if r['algorithm']=='LSSC' and (r.get('temporal_supported') or r['supplied_parameters'].get('radius')!=0):
                errors.append('spatial LSSC misattributed as temporal')
        else:
            match(r.get('shape'),[case['channels'],stage['size'],stage['size']],'record image shape')
            match(r.get('noisy_sha256'),case['noisy_sha256'],'record noisy input')
            match(r.get('clean_sha256'),case['clean_sha256'],'record clean input')
            match(r.get('timing_eligible'),timing_eligible(r),'record timing eligibility')
            if r.get('timed_source_fills')!=0 or not r.get('output_cache_disabled'):errors.append('uncached input invariant failed')
        pixels=directory/r['output']
        if pixels.exists():match(sha(pixels),r['output_sha256'],'downloaded output pixels')
    if rows and any(r.get('status')=='complete' and not (directory/r['output']).exists() for r in rows):
        warnings.append('output pixels not downloaded; recorded hashes retained, visual/hash-file verification pending')
    repeated={}
    for r in rows:
        if r.get('status')=='complete':repeated.setdefault(key(r,temporal)[:3],set()).add(r.get('final_sha256' if temporal else 'output_sha256'))
    if any(len(values)!=1 or None in values for values in repeated.values()):errors.append('repeat output identities differ or are missing')
    completion=directory/('summary.json' if temporal else 'completion.json')
    if not completion.exists():warnings.append('completion metadata missing; coverage reconstructed from frozen grid')
    else:
        value=read(completion)
        if temporal:
            match(value.get('all_completed'),c['all_successful'],'reported temporal completion')
        else:
            match(value.get('scheduled'),c['expected_jobs'],'reported job count');match(value.get('recorded'),len(rows),'reported record count')
            match(value.get('successful'),sum(r.get('status')=='complete' for r in rows),'reported success count')
            match(value.get('complete'),not c['missing'],'reported schedule completion')
    failures=[dict(case=r['case'],algorithm=r['algorithm'],candidate=r.get('recipe' if temporal else 'candidate'),
        repeat=0 if temporal else r['repeat'],status=r.get('status'),timeout_seconds=r.get('timeout_seconds'),
        reason=r.get('reason',r.get('error'))) for r in rows if r.get('status')!='complete']
    return dict(stage=stage['name'],phase=stage['phase'],coverage=c,terminal_failures=failures,integrity_pass=not errors,
        errors=sorted(set(errors)),warnings=warnings,rows=rows,identity=identity,
        source_results_sha256=sha(raw) if raw.exists() else None,source_identity_sha256=sha(identity_path) if identity_path.exists() else None)


def temporal_summary(rows):
    successful=[r for r in rows if r.get('status')=='complete'];results=[]
    for r in successful:
        defaults=[b for b in successful if b['case']==r['case'] and b['algorithm']==r['algorithm'] and b['recipe'].startswith('default-')]
        b=defaults[0] if defaults else None
        match=(not b or all(r[k]==b[k] for k in ('source_sha256','clean_sha256','source_frame_sha256')))
        domains={name:r['temporal_layout'][name+'_frames'] for name in ('interior','boundary')}
        metrics={}
        for domain,indices in domains.items():
            output=[r['quality'][n]['output']['psnr_db'] for n in indices]
            noisy=[r['quality'][n]['noisy']['psnr_db'] for n in indices]
            finite=all(v is not None and math.isfinite(v) for v in output+noisy)
            delta=[r['quality'][n]['output']['psnr_db']-b['quality'][n]['output']['psnr_db'] for n in indices] if b and finite and all(b['quality'][n]['output']['psnr_db'] is not None for n in indices) else []
            metrics[domain]=dict(frames=indices,median_psnr_over_noisy_db=statistics.median(a-c for a,c in zip(output,noisy)) if finite and indices else None,
                worst_psnr_over_noisy_db=min((a-c for a,c in zip(output,noisy)),default=None) if finite else None,
                median_output_psnr_db=statistics.median(output) if finite and indices else None,
                worst_output_psnr_db=min(output,default=None) if finite else None,
                worst_output_ssim=min((r['quality'][n]['output']['ssim'] for n in indices),default=None),
                worst_ssim_over_noisy=min((r['quality'][n]['output']['ssim']-r['quality'][n]['noisy']['ssim'] for n in indices),default=None),
                median_psnr_delta_db=statistics.median(delta) if delta else None,worst_psnr_delta_db=min(delta,default=None),
                worst_ssim_delta=min((r['quality'][n]['output']['ssim']-b['quality'][n]['output']['ssim'] for n in indices),default=None) if b else None)
        peaks=[p['_NSSResourcePeak'] for p in r.get('resource_props',{}).values() if isinstance(p.get('_NSSResourcePeak'),(int,float))]
        results.append(dict(case=r['case'],algorithm=r['algorithm'],recipe=r['recipe'],radius=r['radius'],
            matched_input_to_default=match,metrics=metrics,residual_variation=r['residual_variation'],
            max_peak_bytes=max(peaks,default=None),memory_scope=r.get('resource_scope'),
            worker_wall_seconds=r.get('worker_wall_seconds'),worker_render_seconds=r.get('worker_render_seconds'),
            timing_role='Diagnostic elapsed time only; no temporal performance admission',
            residual_std_ratio_to_default=(r['residual_variation']['output_temporal_std_rms']/b['residual_variation']['output_temporal_std_rms']) if b and b['residual_variation']['output_temporal_std_rms'] else None,
            verdict='Synthetic derivative evidence only; inspect clean quality and residual variation together; no natural-video or spatial-recipe admission'))
    return results


def attempt_statuses(root, phases):
    """Keep shell/preflight/outer-timeout evidence separate from worker rows."""
    results=[]
    for phase in phases:
        for directory in sorted((root/'attempts').glob(phase+'.*')):
            if not directory.is_dir() or directory.name.startswith('._'):continue
            files={}
            for path in sorted(directory.iterdir()):
                if path.name.startswith('._') or not path.is_file():continue
                if path.suffix=='.exit' or path.name in ('phase.started','phase.finished') or path.name.endswith('.status.json'):
                    files[path.name]=dict(sha256=sha(path),value=read(path) if path.suffix=='.json' else path.read_text().strip())
            results.append(dict(phase=phase,path=str(directory.resolve()),files=files,
                phase_exit=files.get('phase.exit',{}).get('value'),
                completion_role='Shell status evidence; recipe coverage is reconstructed independently from frozen jobs'))
    return results


def make_report(args):
    if sha(args.plan)!=args.expected_plan_sha256:raise ValueError('frozen consumption plan hash differs')
    plan=read(args.plan);root=Path(args.selection);s=frozen_selection(root);fixture_root=Path(args.fixtures)
    if plan['selection_sha256']!=SELECTION_SHA:raise ValueError('consumption plan selection differs')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);result_root=Path(args.results_root)
    references={}
    for name,item in plan['reference_assessments'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('reference assessment changed')
        a=read(item['path'])
        for source in a['source_results']:
            if sha(source['path'])!=source['sha256']:raise ValueError('reference raw results changed')
        if a['integrity_errors']:raise ValueError('reference assessment contains integrity errors')
        references[name]=dict(**item,summaries=a['summaries'],integrity_errors=a['integrity_errors'])
    statuses=[];spatial=[];temporal=[]
    for stage in plan['stages']:
        if stage['phase'] not in args.phases:
            statuses.append(dict(stage=stage['name'],phase=stage['phase'],status='not_requested',expected_jobs=len(stage['jobs'])))
            continue
        for size in (stage['size'],):
            if sha(fixture_root/f'fixtures{size}/fixtures.json')!=stage['source_template_sha256']:raise ValueError('fixture template changed')
        checked=verify_stage(stage,result_root/stage['name'],root,s,result_root,fixture_root)
        records=checked.pop('rows');identity=checked.pop('identity')
        checked['status']='complete' if checked['integrity_pass'] and checked['coverage']['all_successful'] else 'incomplete_or_invalid'
        statuses.append(checked)
        if not checked['integrity_pass'] or not records:continue
        if stage['temporal']:
            temporal_records=temporal_summary(records)
            temporal.append(dict(stage=stage['name'],records=temporal_records,coverage=checked['coverage'],
                matched_input_pass=all(r['matched_input_to_default'] for r in temporal_records)))
            continue
        # Normalize missing coverage in a NEW report-only copy. Raw bytes remain
        # unchanged; original partial/completion evidence is never rewritten.
        normalized=out/'normalized'/stage['name'];normalized.mkdir(parents=True)
        (normalized/'results.jsonl').write_bytes((result_root/stage['name']/'results.jsonl').read_bytes())
        save(normalized/'identity.json',identity);save(normalized/'completion.json',dict(remaining=checked['coverage']['missing']))
        destination=out/'assessments'/stage['name'];destination.parent.mkdir(exist_ok=True)
        assess(argparse.Namespace(results=[str(normalized)],fixtures=[str(fixture_root/f'fixtures{stage["size"]}')],
            cold_baselines=[],out=str(destination),min_pairs=7,max_frame_ms=None,max_memory_mb=None))
        analysis=read(destination/'assessment.json')
        if analysis['integrity_errors']:
            checked['integrity_pass']=False
            checked['status']='incomplete_or_invalid'
            checked['errors'].extend(analysis['integrity_errors'])
        spatial.append(dict(stage=stage['name'],phase=stage['phase'],profile=stage['name'].removeprefix(stage['phase']+'-'),
            coverage=checked['coverage'],summaries=analysis['summaries'],integrity_errors=analysis['integrity_errors'],
            recipe_coverage={a:{r['id']:recipe_coverage(stage,checked,a,r['id']) for r in read(root/stage['config'])['algorithms'][a]}
                for a in stage['algorithms']},
            cells=analysis['cells'] if stage['phase']=='formal512' else [],
            formal_timing=formal_timing(records,stage['repeats']) if stage['phase']=='formal512' else None))
    decisions=[]
    for profile,p in s['profiles'].items():
        finalists=read(root/p['finalists'])['algorithms']
        sealed=next((v for v in spatial if v['stage']=='sealed-'+profile),None)
        quality512=next((v for v in spatial if v['stage']=='quality512-'+profile),None)
        formal=next((v for v in spatial if v['stage']=='formal512-'+p['format']),None)
        for algorithm,chosen in finalists.items():
            candidate=chosen[0] if chosen else None
            evidence={}
            for name,stage_data in [('sealed',sealed),('quality512',quality512),('formal512',formal if algorithm=='NLM' else None)]:
                if stage_data is None:evidence[name]=dict(status='pending');continue
                if name=='formal512':
                    cells=[c for c in stage_data['cells'] if c['algorithm']==algorithm and c['candidate']==(candidate or 'default') and c['sigma'] in p['sigmas']]
                    summary=summarize(cells,7,None,None) if cells else None
                else:summary=next((r for r in stage_data['summaries'] if r['algorithm']==algorithm and r['candidate']==(candidate or 'default') and r['sigma']=='all'),None)
                evidence[name]=dict(status='available' if summary else 'missing_recipe',coverage=stage_data['coverage'],summary=summary,
                    recipe_coverage=stage_data['recipe_coverage'].get(algorithm,{}).get(candidate or 'default'))
                if name=='formal512':
                    selected_cases={c['case'] for c in cells}
                    details=[r for r in stage_data['formal_timing'] if r['algorithm']==algorithm and r['candidate']==candidate and r['case'] in selected_cases]
                    evidence[name]['paired_timing']=details
                    evidence[name]['ci_supports_speedup']=bool(details) and len(details)==len(selected_cases) and all(r['ci_lower_bound_above_one'] for r in details)
            if candidate is None:verdict='no alternative selected; default reference is not an automatic quality recommendation'
            elif not (evidence['sealed'].get('recipe_coverage') or {}).get('all_successful'):verdict='pending or incomplete sealed evidence; no admission'
            else:
                summary=evidence['sealed'].get('summary')
                if not summary:verdict='missing sealed candidate evidence'
                elif not summary['no_worse_than_noisy']:verdict='reject frozen scope on worse-than-noisy case/channel; no replacement is selected'
                elif algorithm=='TWSC':verdict='effective restoration only; public-default loss bound unavailable; no balanced-default admission'
                elif not summary['quality_bounds_pass']:verdict='reject frozen scope on sealed quality loss; no replacement is selected'
                elif not (evidence['quality512'].get('recipe_coverage') or {}).get('all_successful'):verdict='sealed quality passed;512 evidence pending/incomplete'
                elif not evidence['quality512'].get('summary',{}).get('quality_bounds_pass') or not evidence['quality512']['summary']['no_worse_than_noisy']:
                    verdict='sealed quality passed;512 quality does not qualify'
                elif algorithm=='NLM':
                    timing=evidence['formal512'].get('summary') or {}
                    verdict='measured spatial speed/quality gates passed; broader/default admission remains unestablished' if timing.get('timing_complete') and timing.get('quality_bounds_pass') and timing.get('no_worse_than_noisy') and (timing.get('median_speedup') or 0)>=1.2 and evidence['formal512'].get('ci_supports_speedup') else 'spatial quality gates passed; formal timing pending or unqualified'
                else:verdict='spatial quality gates passed for explicit quality/cost recipe; no public-default admission'
            decisions.append(dict(profile=profile,algorithm=algorithm,candidate=candidate,scope={k:p[k] for k in ('format','sigmas','measured_sigmas','prospective_unmeasured_sigmas')},verdict=verdict,evidence=evidence,public_default_admitted=False))
    value=dict(schema='nss.finalist-verdict.v1',selection_sha256=SELECTION_SHA,consumption_plan_sha256=sha(args.plan),
        consumer_sha256=sha(__file__),requested_phases=args.phases,phase_status=statuses,reference_evidence=references,
        spatial=spatial,temporal=temporal,attempts=attempt_statuses(result_root,args.phases),
        formal_timing_policy=TIMING_POLICY,decisions=decisions,all_algorithms=list(ALGORITHMS),
        unsupported=[dict(algorithm='MCWNNM',format='gray',verdict='unsupported genuine-Gray input')],
        policy='No retuning. Missing/timeout/invalid evidence never passes. One-repeat quality is not speed evidence. Temporal derivatives and unequal noise remain separate. Public defaults are never automatically changed.')
    save(out/'final-verdict.json',value)
    lines=['# Frozen finalist evidence','','This report performs no recipe selection. Missing phases stay pending; no public-default change is admitted.','',
        '| Stage | Status | Recorded / expected | Missing |','|---|---|---:|---:|']
    for r in statuses:
        c=r.get('coverage',{});lines.append(f"| {r['stage']} | {r['status']} | {c.get('recorded_jobs',0)} / {c.get('expected_jobs',r.get('expected_jobs'))} | {len(c['missing']) if 'missing' in c else 'not read'} |")
    lines+=['','| Profile | Algorithm / frozen alternative | Verdict |','|---|---|---|']
    lines += [f"| {r['profile']} | {r['algorithm']} / {r['candidate'] or 'none'} | {r['verdict']} |" for r in decisions]
    lines+=['','Observed default-reference damage at sigma 5 remains a quality failure, even when no replacement qualifies.','',
        '| Format / algorithm | Worst output minus noisy PSNR (dB) | Failing cases |', '|---|---:|---:|']
    for r in references['validation256']['summaries']:
        if r['candidate']=='default' and r['sigma']=='5' and not r['no_worse_than_noisy']:
            lines.append(f"| {r['format']} / {r['algorithm']} | {r['worst_psnr_over_noisy_db']:.4f} | {len(r['low_noise_worse_than_noisy_cases'])} |")
    lines+=['','Unequal-channel noise is separate evidence and does not modify the homogeneous selection. Positive aggregate PSNR does not excuse damage to the quiet channel.','',
        '| Noise / algorithm / recipe | Worst quiet-channel PSNR over noisy (dB) | Quality bounds / noisy guard |', '|---|---:|---|']
    for r in references['unequal256']['summaries']:
        if r['sigma']!='all':continue
        minimum=min((v['psnr_over_noisy_db'] for v in r['low_noise_channels']),default=None)
        lines.append(f"| {r['noise_family']} / {r['algorithm']} / {r['candidate']} | {minimum:.4f} | {r['quality_bounds_pass']} / {r['no_worse_than_noisy']} |")
    lines+=['','Formal parameter timing uses>=10 CPU1 ticks, CPU1 busy<=1%, zero steal and seven valid pairs per case. Target1s is calibration, not an actual>=1s gate. A supported speedup also needs CI lower bound>1; the selected median threshold remains1.2. This is not the separate implementation .999-idle policy.',
        '', '| Formal case | Valid pairs | Median ratio /95% CI | Actual default / candidate windows(s), min–max |',
        '|---|---:|---|---|']
    for stage in spatial:
        for r in stage.get('formal_timing') or []:
            d=r['actual_window_seconds']['default'];c=r['actual_window_seconds']['candidate']
            lines.append(f"| {r['case']} | {r['valid_pairs']} | {r['median_speed_ratio']} / {r['median_speed_ratio_ci95']} | {d['minimum']}–{d['maximum']} / {c['minimum']}–{c['maximum']} |")
    lines+=['','Per-noise quality, known default low-noise failures, quiet-channel checks, memory, pair counts, exact missing jobs and temporal interior/boundary metrics are in final-verdict.json.',
        'Output files must also be downloaded and hash-checked before visual conclusions. No unreceived sigma10/50,512 or temporal result is inferred.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(out=str(out),phases_read=args.phases,decisions=len(decisions))))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    for mode in ('plan','report'):
        parser=sub.add_parser(mode)
        parser.add_argument('--selection',required=True);parser.add_argument('--fixtures',required=True);parser.add_argument('--out',required=True)
        if mode=='plan':
            parser.add_argument('--validation-assessment',required=True);parser.add_argument('--unequal-assessment',required=True)
        else:
            parser.add_argument('--plan',required=True);parser.add_argument('--results-root',required=True)
            parser.add_argument('--expected-plan-sha256',required=True)
            parser.add_argument('--phases',nargs='*',choices=PHASES,default=[])
    args=p.parse_args();(build_plan if args.mode=='plan' else make_report)(args)


if __name__=='__main__':main()
