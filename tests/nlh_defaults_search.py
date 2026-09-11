#!/usr/bin/env python3
"""Resumable, shared-parameter NLH searches and explicit evaluation sets.

Search observations are single preloaded requests, not performance acceptance.
Every structural proposal is evaluated with shared strength alternatives before
ranking. Held-out cases can only be evaluated against a frozen profiles file.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from defaults_compare import plane_quality
from nlh_defaults_inputs import load_case, save_json, validate_manifest, file_sha

LANES = ('gray-low', 'gray-high', 'rgb')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:20]


def baseline(lane):
    b=7 if lane=='rgb' else 8 if lane=='gray-low' else 10
    return dict(block_size=[b,b],block_step=[1,1],group_size=[16,16],q=[4,4],
                search_window=[40,40],basic_iters=2 if lane=='rgb' else 4 if lane=='gray-low' else 5,
                wiener_iters=2,lambda_basic=.6,hard_strength=1.,wiener_sigma_scale=.08,
                noise_model='real' if lane=='rgb' else 'awgn')


def legal(profile):
    from alignment_reference_nlh_fast import stage_parameters
    try:
        stage_parameters(profile,profile.get('noise_model')=='real',25)
    except (ValueError,TypeError,OverflowError):
        return False
    return True


def structural_identity(profile):
    return identity({k:v for k,v in profile.items() if k not in ('hard_strength','wiener_sigma_scale')})


def strength_variants(profile):
    """Five shared settings: center and each coefficient doubled/halved."""
    variants=[copy.deepcopy(profile)]
    for key in ('hard_strength','wiener_sigma_scale'):
        for factor in (.5,2.):
            p=copy.deepcopy(profile);p[key]*=factor;variants.append(p)
    return variants


def summarize_measurements(measurements):
    """Equal injected noise levels within datasets, then equal datasets."""
    levels = {}
    for row in measurements:
        levels.setdefault((row['dataset'], row['sigma']), []).append(row)
    grouped = []
    for (dataset, sigma), rows in levels.items():
        grouped.append(dict(dataset=dataset, sigma=sigma, cases=len(rows),
                            images=len({r['image'] for r in rows}),
                            mean_psnr=statistics.mean(r['quality']['psnr_db'] for r in rows),
                            mean_ssim=statistics.mean(r['quality']['ssim'] for r in rows),
                            geomean_seconds=math.exp(statistics.mean(math.log(r['seconds']) for r in rows))))
    datasets = []
    for dataset in sorted({r['dataset'] for r in grouped}):
        rows = [r for r in grouped if r['dataset'] == dataset]
        datasets.append(dict(dataset=dataset, cases=sum(r['cases'] for r in rows),
                             mean_psnr=statistics.mean(r['mean_psnr'] for r in rows),
                             mean_ssim=statistics.mean(r['mean_ssim'] for r in rows),
                             geomean_seconds=math.exp(statistics.mean(math.log(r['geomean_seconds']) for r in rows))))
    return dict(mean_psnr=statistics.mean(r['mean_psnr'] for r in datasets),
                mean_ssim=statistics.mean(r['mean_ssim'] for r in datasets),
                geomean_seconds=math.exp(statistics.mean(math.log(r['geomean_seconds']) for r in datasets)),
                datasets=datasets, noise_levels=grouped)


def nondominated(rows):
    """Keep quality/latency tradeoffs without fixed gain or degradation cutoffs."""
    return [a for a in rows if not any(
        b['mean_psnr']>=a['mean_psnr'] and b['mean_ssim']>=a['mean_ssim'] and b['geomean_seconds']<=a['geomean_seconds'] and
        (b['mean_psnr']>a['mean_psnr'] or b['mean_ssim']>a['mean_ssim'] or b['geomean_seconds']<a['geomean_seconds'])
        for b in rows if a is not b)]


def fitted_structures(rows):
    """Finish shared strength fitting before ranking a structural candidate.

    Losing strength trials stay in the raw records. They must not become speed
    endpoints merely because an over-suppressed result took less time.
    """
    best = {}
    for row in rows:
        key = (row.get('algorithm', 'NLH'), structural_identity(row['parameters'])) if 'parameters' in row else row['id']
        if key not in best or (row['mean_psnr'], row['mean_ssim']) > (best[key]['mean_psnr'], best[key]['mean_ssim']):
            best[key] = row
    return list(best.values())


def shortlist(rows):
    """Fitted PSNR/log-latency knee and endpoints; equal distance prefers latency."""
    frontier=nondominated(fitted_structures(rows))
    if not frontier: raise ValueError('empty candidate frontier')
    qlo,qhi=min(r['mean_psnr'] for r in frontier),max(r['mean_psnr'] for r in frontier)
    tlo,thi=min(math.log(r['geomean_seconds']) for r in frontier),max(math.log(r['geomean_seconds']) for r in frontier)
    def distance(r):
        loss=(qhi-r['mean_psnr'])/(qhi-qlo) if qhi>qlo else 0
        latency=(math.log(r['geomean_seconds'])-tlo)/(thi-tlo) if thi>tlo else 0
        return (loss*loss+latency*latency,r['geomean_seconds'],-r['mean_ssim'],r['id'])
    return dict(balanced=min(frontier,key=distance),
                quality=max(frontier,key=lambda r:(r['mean_psnr'],r['mean_ssim'],-r['geomean_seconds'])),
                speed=min(frontier,key=lambda r:(r['geomean_seconds'],-r['mean_psnr'],-r['mean_ssim'])))


def lane_cases(manifest,lane,split,screen=False):
    rows=[c for c in manifest['cases'] if c['split']==split and not c['control']]
    if lane=='rgb':
        rows=[c for c in rows if c['format']=='rgb']
    else:
        rows=[c for c in rows if c['format']=='gray' and (c['sigma']<=50 if lane=='gray-low' else c['sigma']>50)]
    if screen:
        images=list(dict.fromkeys(c['image'] for c in rows if c['dataset']=='DIV2K'))[:2]
        sigmas=(15,50) if lane=='gray-low' else (75,100) if lane=='gray-high' else (25,75)
        selected=[next(c for c in rows if c['image']==im and c['sigma']==sigma and c['realization']==0)
                  for im,sigma in zip(images,sigmas)]
        if lane=='rgb': selected += [c for c in rows if c['dataset']=='CC'][:2]
        rows=selected
    if not rows: raise ValueError('no cases for '+lane+'/'+split)
    return rows


def evaluation_cases(cases, sigma_mode):
    if sigma_mode == 'explicit':
        return cases
    if sigma_mode != 'blind':
        raise ValueError('invalid sigma mode')
    return [dict(c, id=c['id']+'-blind', base_case=c['id'], estimate_sigma=True)
            if c['sigma'] is not None else c for c in cases]


def cpu_ticks():
    path=Path('/proc/stat')
    if not path.exists(): return {}
    return {p[0]:list(map(int,p[1:9])) for line in path.read_text().splitlines()
            if (p:=line.split()) and p[0] in ('cpu0','cpu1')}


def cpu_activity(before,after):
    result={}
    for key in before:
        delta=[b-a for a,b in zip(before[key],after[key])];total=sum(delta)
        result[key]=dict(ticks=total,busy_fraction=(total-delta[3]-delta[4])/total if total else None,steal_ticks=delta[7])
    return result


class Engine:
    def __init__(self,args,cases):
        import vapoursynth as vs
        self.vs=vs;self.core=vs.core;self.core.num_threads=1;self.core.max_cache_size=128
        self.core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
        if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,{args.cpu})
        self.args=args;self.root=Path(args.inputs).resolve();self.out=Path(args.out).resolve()
        self.out.mkdir(parents=True,exist_ok=True)
        self.manifest=json.loads((self.root/'inputs.json').read_text());validate_manifest(self.manifest)
        self.cases=cases;self.sources={};self.observations={}
        self.native_sources={};self.native=None;self.public_proof={}
        if getattr(args,'native_bridge',None):
            from nlh_search_native import Native
            if not 1<=args.size<=512:raise ValueError('cached screening is bounded to crops up to512')
            self.native=Native(Path(args.native_bridge).resolve())
        metadata=dict(schema='nss.nlh-defaults-search.v1',plugin_sha256=file_sha(args.plugin),
                      manifest_sha256=file_sha(self.root/'inputs.json'),script_sha256=file_sha(__file__),
                      inputs_script_sha256=file_sha(Path(__file__).with_name('nlh_defaults_inputs.py')),
                      cases=[c['id'] for c in cases],size=args.size,score_size=args.score_size,cpu=args.cpu,
                      scope='single source-preloaded request; search timing only',
                      selection_rule='equal noise levels and datasets; best shared strength per structure, then three-objective Pareto; normalized PSNR/log-latency knee, latency tie-break; no fixed gain threshold',
                      profiles_sha256=file_sha(args.profiles) if getattr(args,'profiles',None) else None)
        metadata['sigma_mode'] = getattr(args, 'sigma_mode', 'explicit')
        boot_id = Path('/proc/sys/kernel/random/boot_id')
        cpu_info = Path('/proc/cpuinfo')
        metadata['host'] = dict(hostname=platform.node(), kernel=platform.release(),
                                machine=platform.machine(),
                                boot_id=boot_id.read_text().strip() if boot_id.exists() else None,
                                cpu_model=next((line.split(':', 1)[1].strip()
                                                for line in cpu_info.read_text().splitlines()
                                                if line.startswith('model name')), None)
                                if cpu_info.exists() else platform.processor())
        metadata['backend'] = {k: v.decode() if isinstance(v, bytes) else v
                               for k, v in dict(self.core.nss.Backend()).items()}
        metadata['warmup_frames'] = getattr(args, 'warmups', 0)
        metadata['measurement_frames'] = getattr(args, 'measurements', 1)
        if metadata['warmup_frames'] or metadata['measurement_frames'] != 1:
            metadata['scope'] = 'warmed repeated public get_frame; quality-evaluation timing, not paired acceptance'
        if metadata['warmup_frames'] < 0 or metadata['measurement_frames'] < 1 or metadata['warmup_frames'] + metadata['measurement_frames'] > 4:
            raise ValueError('public evaluation needs one to four preloaded frame requests')
        if self.native and (metadata['warmup_frames'] or metadata['measurement_frames'] != 1):
            raise ValueError('warm repeated evaluation uses the public filter')
        if self.native:
            metadata.update(native_bridge_sha256=file_sha(args.native_bridge),
                native_adapter_sha256=file_sha(Path(__file__).with_name('nlh_search_native.py')),
                scope='cached input conversion/noise estimation plus measured native complete filter; screening time estimate, not public get_frame acceptance')
            reference_observations = getattr(args,'reference_observations',None) or getattr(args,'public_observations',None)
            if reference_observations:
                metadata['reference_observations_sha256']=file_sha(reference_observations)
                self.public_proof={r['id']:r for line in Path(reference_observations).read_text().splitlines()
                                   if (r:=json.loads(line))['algorithm']=='NLH'}
        prior_rows = {}
        if getattr(args, 'resume_from', None):
            metadata['reused_observations'] = []
            metadata['scope'] += '; individually labelled observations may be reused from matching frozen experiments'
            for directory in args.resume_from:
                directory = Path(directory)
                previous = json.loads((directory/'metadata.json').read_text())
                for field in ('plugin_sha256', 'manifest_sha256', 'inputs_script_sha256',
                              'cases', 'size', 'score_size', 'cpu'):
                    if previous[field] != metadata[field]:
                        raise ValueError('incompatible observation cache: '+field)
                if previous.get('sigma_mode', 'explicit') != metadata['sigma_mode']:
                    raise ValueError('incompatible observation cache: sigma mode')
                for field, default in (('warmup_frames', 0), ('measurement_frames', 1)):
                    if previous.get(field, default) != metadata[field]:
                        raise ValueError('incompatible observation cache: '+field)
                records = directory/'observations.jsonl'
                metadata['reused_observations'].append(dict(metadata=previous,
                    observations_sha256=file_sha(records)))
                for line in records.read_text().splitlines():
                    row = json.loads(line)
                    expected = identity(dict(profile=row['parameters'], case=row['case'],
                                             algorithm=row['algorithm']))
                    if row['id'] != expected or row['timed_source_fills']:
                        raise ValueError('invalid reusable observation')
                    if row['id'] in prior_rows and row['output_sha256'] != prior_rows[row['id']]['output_sha256']:
                        raise ValueError('reusable observations disagree')
                    row = dict(row, reused_from_metadata_sha256=file_sha(directory/'metadata.json'))
                    prior_rows.setdefault(row['id'], row)
        path=self.out/'metadata.json'
        if path.exists() and json.loads(path.read_text())!=metadata:
            raise ValueError('experiment identity changed; use a new output directory')
        save_json(path,metadata);self.metadata=metadata
        records=self.out/'observations.jsonl'
        if records.exists():
            for line in records.read_text().splitlines():
                r=json.loads(line);self.observations[r['id']]=r
        with records.open('a') as stream:
            for key, row in prior_rows.items():
                if key not in self.observations:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    self.observations[key] = row

    def source(self,case):
        if case['id'] not in self.sources:
            clean,noisy,crop=load_case(self.root,self.manifest,case,self.args.size)
            c,h,w=noisy.shape
            if self.args.score_size and self.args.score_size>min(h,w):
                raise ValueError('score crop exceeds input')
            blank=self.core.std.BlankClip(width=w,height=h,length=4,format=self.vs.RGBS if c==3 else self.vs.GRAYS)
            counter=[0]
            def fill(n,f):
                counter[0]+=1;result=f.copy()
                for p in range(c): np.copyto(np.asarray(result[p]),noisy[p])
                return result
            node=self.core.std.ModifyFrame(blank,blank,fill)
            self.core.std.SetVideoCache(node,mode=1,fixedsize=4,maxsize=4)
            held=[node.get_frame(n) for n in range(4)]
            self.sources[case['id']]=(clean,noisy,crop,node,held,counter)
        return self.sources[case['id']]

    def evaluate_case(self,profile,case,algorithm='NLH'):
        if 'sigma' in profile:raise ValueError('sigma is fixed by the input case, not a searched profile')
        key=identity(dict(profile=profile,case=case['id'],algorithm=algorithm))
        if key in self.observations: return self.observations[key]
        if self.native and algorithm=='NLH':return self.evaluate_native(profile,case,key)
        clean,noisy,crop,source,held,counter=self.source(case)
        kwargs=copy.deepcopy(profile)
        if case['sigma'] is not None and not case.get('estimate_sigma'): kwargs['sigma']=case['sigma']
        if algorithm=='BM3D' and case['sigma'] is None:
            raise ValueError('BM3D reference requires an explicit synthetic-noise case')
        node=getattr(self.core.nss,algorithm)(source,**kwargs)
        self.core.std.SetVideoCache(node,mode=0)
        warmups=self.metadata['warmup_frames'];measurements=self.metadata['measurement_frames']
        expected_hash=None
        before_fills=counter[0]
        for n in range(warmups):
            warm=node.get_frame(n)
            warm_pixels=np.stack([np.asarray(warm[p]).copy() for p in range(len(noisy))])
            expected_hash=hashlib.sha256(warm_pixels.tobytes()).hexdigest()
            del warm,warm_pixels
        seconds_list=[];before=cpu_ticks()
        requests=(1,) if not warmups and measurements==1 else range(warmups,warmups+measurements)
        for n in requests:
            start=time.perf_counter();frame=node.get_frame(n);seconds_list.append(time.perf_counter()-start)
            pixels=np.stack([np.asarray(frame[p]).copy() for p in range(len(noisy))])
            output_hash=hashlib.sha256(pixels.tobytes()).hexdigest()
            if expected_hash is not None and expected_hash!=output_hash:
                raise AssertionError('public repeated evaluation output changed')
            expected_hash=output_hash
        after=cpu_ticks();seconds=statistics.median(seconds_list)
        if counter[0]!=before_fills or not np.isfinite(pixels).all():
            raise AssertionError('timed source evaluation or nonfinite output')
        props={k:v for k,v in frame.props.items() if k.startswith('_NSS') and isinstance(v,(int,float,list))}
        inner=self.args.score_size
        if inner:
            _,h,w=clean.shape;y,x=(h-inner)//2,(w-inner)//2
            reference=clean[:,y:y+inner,x:x+inner];score=pixels[:,y:y+inner,x:x+inner]
        else: reference,score=clean,pixels
        quality=plane_quality(reference,score)
        result=dict(id=key,case=case['id'],image=case['image'],dataset=case['dataset'],sigma=case['sigma'],
                    algorithm=algorithm,parameters=profile,supplied_parameters=kwargs,quality=quality,seconds=seconds,
                    resolved=props,crop=crop,shape=list(clean.shape),score_size=inner,
                    input_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
                    output_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    timed_source_fills=counter[0]-before_fills,cpu_activity=cpu_activity(before,after),
                    measurement_seconds=seconds_list,warmup_frames=warmups,
                    timing_scope='median public get_frame; preloaded source; quality-evaluation timing, not paired acceptance')
        result['sigma_mode'] = 'estimated' if case['sigma'] is None or case.get('estimate_sigma') else 'explicit'
        result['base_case'] = case.get('base_case', case['id'])
        if self.args.save_images:
            directory=self.out/'pixels';directory.mkdir(exist_ok=True)
            arrays=dict(pixels=pixels) if getattr(self.args,'pixels_only',False) else dict(pixels=pixels,clean=clean,noisy=noisy)
            np.savez_compressed(directory/(key+'.npz'),**arrays)
            result['output_file']='pixels/'+key+'.npz'
        with (self.out/'observations.jsonl').open('a') as stream:
            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
        self.observations[key]=result
        del frame,node
        return result

    def evaluate_native(self,profile,case,key):
        if case['id'] not in self.native_sources:
            clean,noisy,crop=load_case(self.root,self.manifest,case,self.args.size)
            prepared=self.native.prepare(noisy,None if case.get('estimate_sigma') else case['sigma'])
            self.native_sources[case['id']]=(clean,noisy,crop,prepared)
        clean,noisy,crop,prepared=self.native_sources[case['id']]
        before=cpu_ticks();pixels,filter_seconds,groups=self.native.run(prepared,profile);after=cpu_ticks()
        if not np.isfinite(pixels).all():raise AssertionError('nonfinite native output')
        inner=self.args.score_size
        if inner:
            _,h,w=clean.shape;y,x=(h-inner)//2,(w-inner)//2
            reference=clean[:,y:y+inner,x:x+inner];score=pixels[:,y:y+inner,x:x+inner]
        else:reference,score=clean,pixels
        result=dict(id=key,case=case['id'],image=case['image'],dataset=case['dataset'],sigma=case['sigma'],
            algorithm='NLH',parameters=profile,quality=plane_quality(reference,score),
            seconds=prepared['prepare_seconds']+filter_seconds,filter_seconds=filter_seconds,
            cached_prepare_seconds=prepared['prepare_seconds'],
            timing_scope='preparation once per input plus fresh native complete filter; estimate for screening',
            resolved=dict(_NSSSigma=prepared['sigma'],_NSSGroups=groups),crop=crop,shape=list(clean.shape),score_size=inner,
            input_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
            output_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),timed_source_fills=0,
            cpu_activity=cpu_activity(before,after),evaluation_backend='native-cached-prepare')
        result['sigma_mode'] = 'estimated' if case['sigma'] is None or case.get('estimate_sigma') else 'explicit'
        result['base_case'] = case.get('base_case', case['id'])
        if key in self.public_proof:
            original=self.public_proof[key]
            equal=(original['input_sha256']==result['input_sha256'] and original['output_sha256']==result['output_sha256'])
            units=original['resolved']['_NSSSigma'];units=[units] if isinstance(units,(int,float)) else units
            equal=equal and units==prepared['sigma'] and original['resolved']['_NSSGroups']==groups
            result['reference_byte_identity']=equal
            result['reference_backend']=original.get('evaluation_backend','public-filter')
            if result['reference_backend']=='public-filter':result['public_byte_identity']=equal
            if not equal:
                save_json(self.out/'native-mismatch.json',dict(actual=result,expected=original))
                np.save(self.out/'native-mismatch.npy',pixels)
                raise AssertionError('cached preparation differs from public filter')
        if self.args.save_images:
            directory=self.out/'pixels';directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/(key+'.npz'),pixels=pixels,clean=clean,noisy=noisy)
            result['output_file']='pixels/'+key+'.npz'
        with (self.out/'observations.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
        self.observations[key]=result
        return result

    def close(self):
        if self.native:
            for _,_,_,prepared in self.native_sources.values():self.native.free(prepared)
        self.native_sources.clear();self.sources.clear()

    def evaluate(self,profile,label='',algorithm='NLH'):
        measurements=[self.evaluate_case(profile,c,algorithm) for c in self.cases
                      if algorithm!='BM3D' or c['sigma'] is not None]
        row=dict(id=identity(dict(profile=profile,algorithm=algorithm)),parameters=profile,label=label,algorithm=algorithm,
                 cases=[r['id'] for r in measurements])
        row.update(summarize_measurements(measurements))
        return row


def change(profile,key,value,stage=None):
    p=copy.deepcopy(profile)
    if stage is None: p[key]=value
    else: p[key][stage]=value
    if key=='block_size':
        b=p['block_size'][stage]
        p['block_step'][stage]=min(p['block_step'][stage],b)
        p['q'][stage]=min(p['q'][stage],1 << ((b*b).bit_length()-1))
    return p


def search(args,engine):
    rows={};events=[];coverage=[]
    def evaluate(p,label):
        key=identity(dict(profile=p,algorithm='NLH'))
        if key not in rows:
            rows[key]=engine.evaluate(p,label)
            save_json(engine.out/'trials.json',list(rows.values()))
            r=rows[key]
            print(f"{args.lane} {len(rows):04d} {label} PSNR={r['mean_psnr']:.5f} SSIM={r['mean_ssim']:.6f} seconds={r['geomean_seconds']:.5f}",flush=True)
        return rows[key]
    def fit(p,label):
        variants=[evaluate(v,label+':strength') for v in strength_variants(p)]
        # Strength alternatives have the same structure. Select shared quality,
        # without treating noisy sub-millisecond time differences as a benefit.
        return max(variants,key=lambda r:(r['mean_psnr'],r['mean_ssim']))
    anchor=baseline(args.lane)
    evaluate(anchor,'current-defaults')
    # Coefficients are shared across every case in this lane's development set.
    anchor=copy.deepcopy(fit(anchor,'baseline-fit')['parameters'])
    # First find useful sampling density; this makes later geometric screening
    # affordable, while the original dense baseline stays in every result set.
    for stage in (0,1):
        candidates=[]
        for value in range(1,anchor['block_size'][stage]+1):
            candidates.append(fit(change(anchor,'block_step',value,stage),f'step-stage{stage}-{value}'))
            coverage.append(dict(key='block_step',stage=stage,block=anchor['block_size'][stage],value=value))
        anchor=copy.deepcopy(shortlist(candidates)['balanced']['parameters'])
        events.append(dict(event='sampling-anchor',stage=stage,parameters=anchor))
        save_json(engine.out/'search-events.json',events)

    dimensions=[('block_size',list(range(2,17))),('q',[2,4,8,16]),('group_size',[2,4,8,16,32,64]),
                ('search_window',[20,30,40,43,50,64,129])]
    for key,values in dimensions:
        for stage in (0,1):
            candidates=[]
            for value in values:
                p=change(anchor,key,value,stage)
                if not legal(p): continue
                candidates.append(fit(p,f'{key}-stage{stage}-{value}'))
                coverage.append(dict(key=key,stage=stage,value=value))
            anchor=copy.deepcopy(shortlist(candidates)['balanced']['parameters'])
            events.append(dict(event='coordinate-anchor',key=key,stage=stage,parameters=anchor))
            save_json(engine.out/'search-events.json',events)
    for key,values in (('basic_iters',range(1,8)),('wiener_iters',range(1,4)),('lambda_basic',(0.,.2,.4,.6,.8,1.))):
        candidates=[fit(change(anchor,key,value),f'{key}-{value}') for value in values]
        coverage.extend(dict(key=key,value=v) for v in values)
        anchor=copy.deepcopy(shortlist(candidates)['balanced']['parameters'])
    # Cover every legal (block,step) pair around the selected generic settings,
    # separately in Basic and Wiener, with shared strength fitting throughout.
    if not args.skip_block_step_grid:
        for stage in (0,1):
            for block in range(2,17):
                for step in range(1,block+1):
                    p=change(change(anchor,'block_size',block,stage),'block_step',step,stage)
                    fit(p,f'block-step-{stage}-{block}-{step}')
                    coverage.append(dict(key='block_step',stage=stage,block=block,value=step))
    # Author Normal/Fast parameters transferable to the current fixed-stage
    # algorithm; differing equations and per-iteration schedules are not copied.
    author=baseline(args.lane)
    author.update(block_size=[6 if args.lane=='rgb' else 9 if args.lane=='gray-low' else 10,16],
                  block_step=[3,16],group_size=[16,64],q=[4,8],search_window=[43,129])
    fit(author,'author-stage-shape-transfer')
    starts=list({r['id']:r for r in shortlist(list(rows.values())).values()}.values())
    for start in starts:
        p=copy.deepcopy(start['parameters'])
        for stage in (0,1):
            for key in ('block_size','block_step','search_window'):
                center=p[key][stage]
                candidates=[fit(p,'joint-center')]
                for delta in (-1,1):
                    trial=change(p,key,center+delta,stage)
                    if legal(trial): candidates.append(fit(trial,'joint-neighbor'))
                p=copy.deepcopy(shortlist(candidates)['balanced']['parameters'])
        # Fine coefficient points between the logarithmic coarse samples.
        for key in ('hard_strength','wiener_sigma_scale'):
            candidates=[evaluate(change(p,key,p[key]*factor),'coefficient-refine') for factor in (2**-.5,1.,2**.5)]
            p=copy.deepcopy(max(candidates,key=lambda r:(r['mean_psnr'],r['mean_ssim']))['parameters'])
    front=nondominated(fitted_structures(list(rows.values())));chosen=shortlist(list(rows.values()))
    save_json(engine.out/'coverage.json',coverage)
    save_json(engine.out/'summary.json',dict(lane=args.lane,trials=len(rows),frontier=front,selected=chosen,
              search_scope='bounded multi-start coordinate search; not a Cartesian/global optimum claim',
              stage_shape_scope='all legal blocks/q/groups and requested windows; two stages independently configurable',
              block_step_complete=not args.skip_block_step_grid,metadata=engine.metadata))


def evaluate_profiles(args,engine):
    profiles=json.loads(Path(args.profiles).read_text())
    rows=[]
    for label,profile in profiles['profiles'].items():
        rows.append(engine.evaluate(profile,label))
        print(label,rows[-1]['mean_psnr'],rows[-1]['geomean_seconds'],flush=True)
        save_json(engine.out/'evaluation.json',dict(rows=rows,metadata=engine.metadata))
    if args.bm3d:
        rows.append(engine.evaluate({},'BM3D-same-sigma',algorithm='BM3D'))
    save_json(engine.out/'evaluation.json',dict(rows=rows,metadata=engine.metadata))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('search','evaluate'))
    for name in ('plugin','inputs','out'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--lane',choices=LANES,required=True)
    parser.add_argument('--cpu',type=int,default=0)
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--score-size',type=int,default=96)
    parser.add_argument('--split',choices=('development','selection','test'),default='development')
    parser.add_argument('--sigma-mode',choices=('explicit','blind'),default='explicit',
                        help='Blind evaluation keeps the injected noise unchanged and omits public sigma')
    parser.add_argument('--profiles');parser.add_argument('--cases',nargs='+')
    parser.add_argument('--native-bridge',help='Verified preparation-cache adapter; screening estimates only')
    parser.add_argument('--public-observations',help='Frozen matching public runs for additional byte-identity checks')
    parser.add_argument('--reference-observations',help='Frozen matching baseline runs, retaining public/native provenance')
    parser.add_argument('--resume-from',nargs='+',help='Frozen compatible experiment directories; preserve each row timing provenance')
    parser.add_argument('--screen',action='store_true');parser.add_argument('--save-images',action='store_true')
    parser.add_argument('--bm3d',action='store_true');parser.add_argument('--skip-block-step-grid',action='store_true')
    args=parser.parse_args()
    if args.mode=='search' and args.split!='development': parser.error('search is restricted to development images')
    if args.sigma_mode=='blind' and (args.mode!='evaluate' or args.bm3d):
        parser.error('blind mode is NLH evaluation only; BM3D uses the explicit-noise reference run')
    if args.mode=='evaluate' and not args.profiles: parser.error('evaluation requires a frozen profiles file')
    manifest=json.loads((Path(args.inputs)/'inputs.json').read_text());validate_manifest(manifest)
    cases=lane_cases(manifest,args.lane,args.split,screen=args.screen)
    if args.cases:
        requested=set(args.cases);cases=[c for c in cases if c['id'] in requested]
        if {c['id'] for c in cases}!=requested: parser.error('requested cases are not in the selected lane/split')
    cases = evaluation_cases(cases, args.sigma_mode)
    engine=Engine(args,cases)
    try:
        if args.mode=='search': search(args,engine)
        else: evaluate_profiles(args,engine)
    finally:engine.close()


if __name__=='__main__':
    main()
