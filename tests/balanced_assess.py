#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Assess balanced campaigns by algorithm, format, noise and split without changing recipes.

Reads raw paired records, preserves missing/censored baselines, and accepts
explicitly cold quality-only references. Suggestions are evidence labels, never
public-default admission. Different plugins remain separate assessment groups.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ALGORITHMS=('NLM','BM3D','WNNM','MCWNNM','TWSC','NLH','NCSR','LSSC')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def median(values):return statistics.median(values) if values else None

def minimum(values):return min(values) if values else None

def maximum(values):return max(values) if values else None


def metadata(row, known):
    case=known.get(row['case'],{})
    channels=row.get('shape',[row.get('channels',case.get('channels'))])[0]
    vector=case.get('channel_sigma')
    supplied=row.get('supplied_parameters',{}).get('sigma')
    if vector:
        if row.get('ok') and supplied!=vector:
            raise ValueError('unequal input channel sigma differs from supplied recipe sigma')
        family='unequal-rgb-'+('-'.join(f'{v:g}' for v in vector))
    elif isinstance(supplied,list) and len(set(supplied))>1:
        family='unverified-vector-input-'+('-'.join(f'{v:g}' for v in supplied))
    else:
        family='homogeneous-awgn'
    return dict(format='gray' if channels==1 else 'rgb' if channels==3 else 'unknown',
        sigma=row.get('sigma',case.get('sigma')),split=row.get('split',case.get('split','unknown')),
        image=row.get('image',case.get('image',row['case'])),channel_sigma=vector,
        noise_family=family,input_noise_vector_verified=bool(vector),
        sentinel=case.get('known_low_noise_sentinel',False))


def read_runs(paths):
    loaded=[];seen=set()
    for item in paths:
        root=Path(item)
        raw=root/'results.jsonl' if root.is_dir() else root
        if raw.name=='summary.json':
            data=json.loads(raw.read_text());rows=data.get('rows',data.get('records'))
            if rows is None:
                sibling=raw.with_name('results.jsonl')
                if not sibling.exists():raise ValueError('summary lacks raw evidence; supply results.jsonl: '+str(raw))
                raw=sibling;rows=[json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
        else:rows=[json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
        digest=sha(raw)
        if digest in seen:raise ValueError('duplicate raw result input would double-count pairs')
        seen.add(digest)
        parent=raw.parent
        identity=json.loads((parent/'identity.json').read_text()) if (parent/'identity.json').exists() else {}
        completion=json.loads((parent/'completion.json').read_text()) if (parent/'completion.json').exists() else {}
        loaded.append(dict(path=str(raw.resolve()),sha256=sha(raw),rows=rows,identity=identity,completion=completion))
    return loaded


def summarize(items, min_pairs, max_latency_ms, max_memory_mb):
    # Never count the same scene/noise case twice when multiple runs are read.
    case_groups={}
    for r in items:case_groups.setdefault(r['case'],[]).append(r)
    unique=[]
    for case,records in case_groups.items():
        success=[r for r in records if r['ok']]
        chosen=dict(success[0] if success else records[0])
        if success:
            if len({r['output_sha256'] for r in success})!=1 or len({json.dumps(r['supplied_parameters'],sort_keys=True) for r in success})!=1:
                raise ValueError('same recipe/case differs across runs: '+case)
            paired=[r for r in success if r.get('baseline_quality')]
            if paired:
                chosen.update(baseline_quality=paired[0]['baseline_quality'],baseline_kind=paired[0]['baseline_kind'])
            chosen['paired_speed_ratios']=[v for r in success for v in r['paired_speed_ratios']]
            chosen['valid_pairs']=sum(r['valid_pairs'] for r in success)
            chosen['seconds']=median([r['seconds'] for r in success])
            chosen['peak_bytes']=maximum([r['peak_bytes'] for r in success if r['peak_bytes'] is not None])
        unique.append(chosen)
    items=unique
    good=[r for r in items if r['ok']]
    compared=[r for r in good if r.get('baseline_quality')]
    deltas=[r['quality']['psnr_db']-r['baseline_quality']['psnr_db'] for r in compared]
    ssim=[r['quality']['ssim']-r['baseline_quality']['ssim'] for r in compared]
    noisy=[r['quality']['psnr_db']-r['noisy_quality']['psnr_db'] for r in good]
    low=[r for r in good if r['sigma'] is not None and r['sigma']<=10]
    low_bad=[r['case'] for r in low if r['quality']['psnr_db'] < r['noisy_quality']['psnr_db']-.01]
    lows=[r['quality']['psnr_db']-r['noisy_quality']['psnr_db'] for r in low]
    low_planes=[]
    for r in good:
        vector=r.get('channel_sigma')
        if vector:
            for plane,sigma in enumerate(vector):
                if sigma<=10:
                    output=r['quality'].get('planes',[]);noisy_planes=r['noisy_quality'].get('planes',[])
                    if len(output)!=len(vector) or len(noisy_planes)!=len(vector):
                        raise ValueError('unequal noise requires per-plane quality metrics')
                    delta=output[plane]['psnr_db']-noisy_planes[plane]['psnr_db']
                    low_planes.append(dict(case=r['case'],plane=plane,sigma=sigma,psnr_over_noisy_db=delta))
    low_plane_bad=[r for r in low_planes if r['psnr_over_noisy_db']<-.01]
    worst_plane=[]
    for r in compared:
        bp=r['baseline_quality'].get('planes',[]);cp=r['quality'].get('planes',[])
        worst_plane.extend(c['psnr_db']-b['psnr_db'] for b,c in zip(bp,cp)
                           if c.get('psnr_db') is not None and b.get('psnr_db') is not None)
    ratios=[v for r in compared for v in r['paired_speed_ratios']]
    case_ratios=[median(r['paired_speed_ratios']) for r in compared if r['paired_speed_ratios']]
    seconds=[r['seconds'] for r in good]
    peaks=[r['peak_bytes'] for r in good if r['peak_bytes'] is not None]
    memory_missing=sum(r['peak_bytes'] is None for r in good)
    timing_complete=bool(compared) and all(r['valid_pairs']>=min_pairs for r in compared) and len(compared)==len(items)
    quality_complete=len(compared)==len(items) and bool(items)
    quality_safe=quality_complete and median(deltas)>=-.2 and min(deltas)>=-.5 and min(ssim)>=-.005
    noisy_safe=bool(good) and min(noisy)>=-.01 and not low_bad and not low_plane_bad
    limits_ok=(max_latency_ms is None or bool(seconds) and max(seconds)*1000<=max_latency_ms) and (
        max_memory_mb is None or not memory_missing and bool(peaks) and max(peaks)<=max_memory_mb*1024*1024)
    speed=median(case_ratios)
    if good and not noisy_safe:
        label='reject for measured worse-than-noisy case or low-noise channel'
    elif not quality_complete:
        label='unqualified: missing/failed baseline or candidate quality'
    elif quality_safe and noisy_safe and any(r.get('baseline_kind')=='cold-quality-only' for r in compared) and limits_ok:
        label='provisional quality-compatible recipe on measured cold-reference cases; no warm speed admission'
    elif quality_safe and noisy_safe and timing_complete and speed>=1.2 and limits_ok:
        label='meets measured speed/quality thresholds on this slice; broader admission pending'
    elif quality_safe and noisy_safe and median(deltas)>=.2 and limits_ok:
        label=('quality-improving cost tradeoff; no speed-balanced admission' if timing_complete else
               'quality improvement observed; timing or pair count unqualified')
    elif not quality_safe or not noisy_safe:
        label='reject for measured quality loss or worse-than-noisy case'
    else:
        label='retain current recipe or collect missing timing; no qualifying speed choice'
    if not compared and good and noisy_safe and min(noisy)>0:
        label='provisional effective-restoration evidence; default loss bound unavailable'
    if items and items[0]['candidate']=='default':
        label=('current default reference; low-noise/response checks still apply' if noisy_safe else
               'current default has missing evidence or worse-than-noisy cases; no qualified replacement implied')
    provenance_ok=not any(r.get('noise_family','').startswith('unverified-vector') for r in good)
    if not provenance_ok:label='unqualified: vector noise input provenance missing; supply fixture manifests'
    return dict(scheduled_cases=len(items),successful_cases=len(good),compared_cases=len(compared),
        input_noise_provenance_pass=provenance_ok,
        scenes=len({r['image'] for r in items}),failed_cases=[r['case'] for r in items if not r['ok']],
        missing_baseline_cases=[r['case'] for r in good if not r.get('baseline_quality')],
        median_psnr_delta_db=median(deltas),worst_psnr_delta_db=minimum(deltas),
        median_ssim_delta=median(ssim),worst_ssim_delta=minimum(ssim),worst_plane_psnr_delta_db=minimum(worst_plane),
        median_psnr_over_noisy_db=median(noisy),worst_psnr_over_noisy_db=minimum(noisy),
        low_noise_case_count=len(low),low_noise_worse_than_noisy_cases=low_bad,
        low_noise_worst_psnr_over_noisy_db=minimum(lows),
        low_noise_channels=low_planes,low_noise_channel_degradations=low_plane_bad,
        known_sentinel_cases=[r['case'] for r in good if r['sentinel']],
        median_speedup=speed,worst_speedup=minimum(case_ratios),valid_pairs=len(ratios),
        minimum_pairs_per_compared_case=minimum([r['valid_pairs'] for r in compared]),
        timing_complete=timing_complete,quality_complete=quality_complete,quality_bounds_pass=quality_safe,
        no_worse_than_noisy=noisy_safe,median_frame_ms=median(seconds)*1000 if seconds else None,
        worst_frame_ms=maximum(seconds)*1000 if seconds else None,
        max_peak_bytes=maximum(peaks),missing_memory_cases=memory_missing,
        memory_scope='plugin-reported _NSSResourcePeak, not process RSS',
        max_peak_ratio_over_default=maximum([r['peak_bytes']/r['baseline_peak_bytes'] for r in good
            if r['peak_bytes'] is not None and r.get('baseline_peak_bytes')]),
        optional_resource_limits_pass=limits_ok,
        quality_only_baseline_cases=sum(r.get('baseline_kind')=='cold-quality-only' for r in compared),
        recommendation=label,public_default_admitted=False)


def assess(args):
    known={}
    for fixture in args.fixtures:
        path=Path(fixture);path=path/'fixtures.json' if path.is_dir() else path
        for case in json.loads(path.read_text())['cases']:
            if case['id'] in known and known[case['id']]['noisy_sha256']!=case['noisy_sha256']:
                raise ValueError('fixture case ID collision')
            known[case['id']]=case
    runs=read_runs(args.results)
    for run in runs:
        for row in run['rows']:
            if row.get('ok'):
                known.setdefault(row['case'],dict(id=row['case'],image=row.get('image'),split=row.get('split','unknown'),
                    channels=row['shape'][0],sigma=row['sigma'],noisy_sha256=row['noisy_sha256']))
    cold={};cold_sources=[]
    for item in args.cold_baselines:
        path=Path(item)
        for file in ([p for p in sorted(path.glob('*.f32.json')) if not p.name.startswith('._')] if path.is_dir() else [path]):
            row=json.loads(file.read_text())
            if row.get('warmup_performed') is not False:raise ValueError('cold baseline not explicitly quality-only')
            key=(row['case'],row['algorithm'],row['plugin_sha256'])
            cold[key]=row;cold_sources.append(dict(path=str(file.resolve()),sha256=sha(file)))
    cells=[];errors=[]
    for run in runs:
        grouped={}
        for row in run['rows']:
            if row.get('status')=='unsupported':continue
            grouped.setdefault((row['case'],row['algorithm'],row['candidate']),[]).append(row)
        for (case,algorithm,candidate), records in grouped.items():
            successes=[r for r in records if r.get('ok')]
            representative=successes[0] if successes else records[0]
            meta=metadata(representative,known)
            plugin=representative.get('plugin_sha256',run['identity'].get('plugin_sha256','unknown'))
            output=dict(case=case,algorithm=algorithm,candidate=candidate,plugin_sha256=plugin,**meta,
                run=run['path'],ok=bool(successes),attempted_repeats=len(records),valid_pairs=0,
                paired_speed_ratios=[],baseline_kind=None,baseline_quality=None)
            if not successes:cells.append(output);continue
            hashes={r['output_sha256'] for r in successes}
            parameters={json.dumps(r['supplied_parameters'],sort_keys=True) for r in successes}
            if len(hashes)!=1 or len(parameters)!=1:
                output['ok']=False;errors.append(dict(case=case,algorithm=algorithm,candidate=candidate,
                    error='repeat outputs/parameters differ'));cells.append(output);continue
            default=[r for r in grouped.get((case,algorithm,'default'),[]) if r.get('ok')]
            if default and (len({r['output_sha256'] for r in default})!=1 or
                len({json.dumps(r['supplied_parameters'],sort_keys=True) for r in default})!=1):
                raise ValueError('default repeated outputs/parameters differ')
            baseline=default[0] if default else cold.get((case,algorithm,plugin))
            if baseline:
                for key in ('plugin_sha256','noisy_sha256','clean_sha256'):
                    if baseline[key]!=representative[key]:raise ValueError('baseline/candidate identity mismatch: '+key)
                output.update(baseline_quality=baseline['quality'],baseline_kind='paired-warm' if default else 'cold-quality-only',
                    baseline_peak_bytes=baseline.get('resource_props',{}).get('_NSSResourcePeak'))
            eligible=[]
            for r in successes:
                for b in default:
                    if candidate=='default' or r['repeat']!=b['repeat']:continue
                    if r['timing_eligible'] and b['timing_eligible'] and r['host']==b['host']:
                        eligible.append(b['seconds']/r['seconds'])
            props=[r.get('resource_props',{}) for r in successes]
            peaks=[p['_NSSResourcePeak'] for p in props if isinstance(p.get('_NSSResourcePeak'),(int,float))]
            output.update(quality=representative['quality'],noisy_quality=representative['noisy_quality'],
                supplied_parameters=representative['supplied_parameters'],valid_pairs=len(eligible),
                paired_speed_ratios=eligible,seconds=median([r['seconds'] for r in successes]),
                peak_bytes=maximum(peaks),memory_limits=[p.get('_NSSResourceLimit') for p in props],
                all_requested_repeats_complete=len(successes)==run['identity'].get('repeats',len(records)),
                output_sha256=representative['output_sha256'])
            cells.append(output)
        for row in run['completion'].get('remaining',[]):
            if (row['case'],row['algorithm'],row['candidate']) in grouped:continue
            meta=metadata(row,known)
            cells.append(dict(**row,**meta,run=run['path'],ok=False,plugin_sha256=run['identity'].get('plugin_sha256','unknown'),
                attempted_repeats=0,valid_pairs=0,paired_speed_ratios=[],baseline_kind=None,baseline_quality=None))
    groups={}
    for row in cells:
        noise_label='['+','.join(f'{v:g}' for v in row['channel_sigma'])+']' if row.get('channel_sigma') else str(row['sigma'])
        for noise in (noise_label,'all'):
            key=(row['plugin_sha256'],row['split'],row['algorithm'],row['candidate'],row['format'],row['noise_family'],noise)
            groups.setdefault(key,[]).append(row)
    summaries=[]
    for key,values in sorted(groups.items()):
        names=('plugin_sha256','split','algorithm','candidate','format','noise_family','sigma')
        summaries.append(dict(zip(names,key),**summarize(values,args.min_pairs,args.max_frame_ms,args.max_memory_mb)))
    decisions=[]
    for algorithm in ALGORITHMS:
        selected=[r for r in summaries if r['algorithm']==algorithm and r['sigma']=='all']
        decisions.append(dict(algorithm=algorithm,
            decision='No qualified public-default change; retain current public defaults. Assess explicit recipes by format/noise below.',
            recipe_assessments=selected,measured_candidates=sorted({r['candidate'] for r in selected})))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'assessment.json',dict(schema='nss.balanced-assessment.v1',
        source_results=[{k:r[k] for k in ('path','sha256','identity')} for r in runs],cold_baselines=cold_sources,
        policy=dict(min_pairs=args.min_pairs,speedup=1.2,median_psnr_loss_db=.2,worst_psnr_loss_db=.5,
                    worst_ssim_loss=.005,low_noise_sigma_max=10,low_noise_degradation_db=.01,
                    max_frame_ms=args.max_frame_ms,max_memory_mb=args.max_memory_mb),
        summaries=summaries,decisions=decisions,cells=cells,integrity_errors=errors,
        scope='Per-slice evidence labels only; visual texture, natural-video flicker and representative generalization are not automatic gates. Cold baselines never contribute timing ratios.'))
    lines=['# Balanced recipe assessment','','No public-default change is admitted by this report. Cold references provide quality only.','',
        '| Split | Format | Algorithm / recipe | Noise | Cases | Median/worst PSNR delta | Worst SSIM delta | Median speed | Pairs | Peak MiB | Decision |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x,digits=3):return 'unknown' if x is None else f'{x:.{digits}f}'
    for r in summaries:
        if r['sigma']!='all':continue
        lines.append(f"| {r['split']} | {r['format']} | {r['algorithm']} / {r['candidate']} | {r['noise_family']} | {r['compared_cases']}/{r['scheduled_cases']} | {fmt(r['median_psnr_delta_db'])}/{fmt(r['worst_psnr_delta_db'])} | {fmt(r['worst_ssim_delta'],5)} | {fmt(r['median_speedup'])} | {r['valid_pairs']} | {fmt(r['max_peak_bytes']/1048576 if r['max_peak_bytes'] is not None else None)} | {r['recommendation']} |")
    lines+=['','Noise-specific statistics, low-noise sentinels, all eight algorithm decisions, missing baselines and raw per-case evidence are in assessment.json.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),summaries=len(summaries),integrity_errors=len(errors))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',nargs='+',required=True);p.add_argument('--fixtures',nargs='*',default=[])
    p.add_argument('--cold-baselines',nargs='*',default=[]);p.add_argument('--out',required=True)
    p.add_argument('--min-pairs',type=int,default=3);p.add_argument('--max-frame-ms',type=float)
    p.add_argument('--max-memory-mb',type=float)
    args=p.parse_args()
    if args.min_pairs<1:p.error('minimum pairs must be positive')
    for value in (args.max_frame_ms,args.max_memory_mb):
        if value is not None and value<=0:p.error('optional resource limits must be positive')
    assess(args)


if __name__=='__main__':main()
