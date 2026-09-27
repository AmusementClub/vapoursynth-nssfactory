#!/usr/bin/env python3
"""Offline analysis of retained pixels and paired timing; no plugin execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def ci(values):
    x=np.asarray(values,float);rng=np.random.default_rng(190920)
    boot=np.median(rng.choice(x,size=(20000,len(x)),replace=True),axis=1)
    return dict(pairs=len(x),median=float(np.median(x)),ci95=list(map(float,np.quantile(boot,[.025,.975]))),ratios=list(map(float,x)))
def psnr(x,y):
    mse=np.mean((x.astype(float)-y.astype(float))**2)
    return float(-10*np.log10(max(mse,1e-300)))
def ssim(x,y):
    scores=[]
    def mean7(z):return np.lib.stride_tricks.sliding_window_view(z,(7,7)).mean(axis=(-2,-1))
    for a,b in zip(x.astype(float),y.astype(float)):
        ux=mean7(a);uy=mean7(b)
        vx=(mean7(a*a)-ux*ux)*49/48
        vy=(mean7(b*b)-uy*uy)*49/48
        cov=(mean7(a*b)-ux*uy)*49/48
        z=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
        scores.append(z.mean())
    return float(np.mean(scores))
def main(a):
    root=Path(a.frames);rows=json.loads((root/'rows.json').read_text());policy=json.loads((root/'policy.json').read_text())
    fixtures=Path(a.fixtures);summary=dict(cases={},instrumentation={},errors=[],policy=policy)
    for c in policy['cases']:
        name=c['name'];clean=np.load(fixtures/c['clean']);data={}
        for mode in ('baseline','tight','gesdd'):
            selected=[r for r in rows if r['case']==name and r['mode']==mode and r['label'].startswith('paired/')]
            selected.sort(key=lambda x:int(x['label'].split('/')[2]))
            if len(selected)!=7:summary['errors'].append(f'{name}/{mode}: {len(selected)} of 7 rounds');continue
            arrays=[np.load(root/r['label']/'pixels.npy') for r in selected]
            first=arrays[0][0]
            exact=all(np.array_equal(x,first) for y in arrays for x in y)
            if not exact:summary['errors'].append(f'{name}/{mode}: repeat mismatch')
            for row,arr in zip(selected,arrays):
                if hashlib.sha256(arr.tobytes()).hexdigest()!=row['result']['sha256']:summary['errors'].append(f'{row["label"]}: payload hash')
                if row['result']['input_sha256']!=c['input_sha256']:summary['errors'].append(f'{row["label"]}: input hash')
                if not np.isfinite(arr).all():summary['errors'].append(f'{row["label"]}: nonfinite')
            data[mode]=dict(rows=selected,pixels=first,repeat_exact=exact,psnr=psnr(first,clean),ssim=ssim(first,clean))
            statistics=[]
            for row in selected:
                for line in (root/row['label']/'stderr').read_text().splitlines():
                    if line.startswith('SVD_LAB '):statistics.append(json.loads(line[len('SVD_LAB '):]))
            data[mode]['statistics']=statistics
        entry={}
        if 'baseline' in data:
            b=data['baseline'];entry['baseline_ms']=float(np.median([r['result']['ms'] for r in b['rows']]))
            entry['baseline_psnr']=b['psnr'];entry['baseline_ssim']=b['ssim']
            entry['baseline_peak_rss_kib']=float(np.median([r['result']['peak_rss_kib'] for r in b['rows']]))
            entry['baseline_resource_props']=b['rows'][0]['result'].get('resource_props',{})
            entry['baseline_statistics']=b['statistics']
            for mode in ('tight','gesdd'):
                if mode not in data:continue
                x=data[mode];diff=x['pixels'].astype(float)-b['pixels'].astype(float)
                ratios=[br['result']['ms']/cr['result']['ms'] for br,cr in zip(b['rows'],x['rows'])]
                entry[mode]=dict(**ci(ratios),all_environment_qualified=all(r['qualified'] for r in b['rows']+x['rows']),
                    repeat_exact=x['repeat_exact'],psnr_delta=x['psnr']-b['psnr'],ssim_delta=x['ssim']-b['ssim'],
                    max_abs=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff**2))),p99=float(np.quantile(np.abs(diff),.99)),
                    median_ms=float(np.median([r['result']['ms'] for r in x['rows']])),
                    statistics=x['statistics'],peak_rss_kib=float(np.median([r['result']['peak_rss_kib'] for r in x['rows']])),
                    resource_props=x['rows'][0]['result'].get('resource_props',{}),
                    invalid_windows=[r['label'] for r in b['rows']+x['rows'] if not r['qualified']])
        summary['cases'][name]=entry
    for name in {r['case'] for r in rows if r['label'].startswith('instrumentation/')}:
        pairs=[];q=[]
        for i in range(7):
            pair=[r for r in rows if r['label'].startswith(f'instrumentation/{name}/{i}/')]
            if len(pair)!=2:continue
            pair.sort(key=lambda r:r['label']) # lab, stock
            pairs.append(pair[1]['result']['ms']/pair[0]['result']['ms']);q.extend(r['qualified'] for r in pair)
        if pairs:summary['instrumentation'][name]=dict(**ci(pairs),all_environment_qualified=all(q))
    if a.kernels and (Path(a.kernels)/'rows.json').exists():
        kernel_rows=json.loads((Path(a.kernels)/'rows.json').read_text());summary['kernels']={}
        for name in {r['case'] for r in kernel_rows}:
            modes={mode:sorted([r for r in kernel_rows if r['case']==name and r['mode']==mode],key=lambda x:x['repeat']) for mode in ('baseline','tight','gesdd','gesvd')}
            cell={}
            for mode in ('tight','gesdd','gesvd'):
                b=modes['baseline'];x=modes[mode]
                if len(b)!=7 or len(x)!=7 or any(r['returncode'] for r in b+x):continue
                cell[mode]={phase:ci([br['result'][phase]/cr['result'][phase] for br,cr in zip(b,x)]) for phase in ('decomposition_s','validation_s','coding_s','total_s')}
                cell[mode]['all_environment_qualified']=all(r['qualified'] for r in b+x)
                bp=np.fromfile(Path(a.kernels)/name/'0/baseline/output.f32','<f4')
                cp=np.fromfile(Path(a.kernels)/name/'0'/mode/'output.f32','<f4')
                cell[mode]['max_group_output_difference']=float(np.max(np.abs(bp.astype(float)-cp.astype(float))))
                cell[mode]['independent_factor_checks']=x[0].get('independent',[])
            summary['kernels'][name]=cell
    if a.direct and (Path(a.direct)/'rows.json').exists():
        directory=Path(a.direct);direct_rows=json.loads((directory/'rows.json').read_text());summary['direct']={}
        for name in sorted({r['case'] for r in direct_rows}):
            b=sorted([r for r in direct_rows if r['case']==name and r['mode']=='stock'],key=lambda r:r['repeat'])
            x=sorted([r for r in direct_rows if r['case']==name and r['mode']=='gesdd'],key=lambda r:r['repeat'])
            if len(b)!=7 or len(x)!=7 or any(r['returncode'] for r in b+x):
                summary['errors'].append(f'incomplete direct cell {name}');continue
            c=next(c for c in policy['cases'] if c['name']==name);clean=np.load(fixtures/c['clean'])
            arrays={}
            for mode,records in (('stock',b),('gesdd',x)):
                pixels=[np.load(directory/name/str(r['repeat'])/mode/'pixels.npy') for r in records]
                arrays[mode]=pixels[0][0]
                if not all(np.array_equal(frame,pixels[0][0]) for run in pixels for frame in run):summary['errors'].append(f'direct repeat mismatch {name}/{mode}')
                for r,pixels_run in zip(records,pixels):
                    if pixels_run.shape[0] == r['result'].get('timed_frames', pixels_run.shape[0]):
                        if hashlib.sha256(pixels_run.tobytes()).hexdigest()!=r['result']['sha256']:summary['errors'].append(f'direct payload hash {name}/{mode}/{r["repeat"]}')
                    elif pixels_run.shape[0] != 7:
                        summary['errors'].append(f'direct retained prefix length {name}/{mode}/{r["repeat"]}')
                    if r['result']['input_sha256']!=c['input_sha256']:summary['errors'].append(f'direct input hash {name}/{mode}/{r["repeat"]}')
                original_mode='baseline' if mode=='stock' else mode
                original=np.load(root/f'paired/{name}/0/{original_mode}/pixels.npy')[0]
                if not np.array_equal(original,arrays[mode]):summary['errors'].append(f'direct versus primary pixels {name}/{mode}')
            delta=arrays['gesdd'].astype(float)-arrays['stock'].astype(float)
            summary['direct'][name]=dict(**ci([br['result']['ms']/cr['result']['ms'] for br,cr in zip(b,x)]),
                stock_ms=float(np.median([r['result']['ms'] for r in b])),gesdd_ms=float(np.median([r['result']['ms'] for r in x])),
                stock_peak_rss_kib=float(np.median([r['result']['peak_rss_kib'] for r in b])),
                gesdd_peak_rss_kib=float(np.median([r['result']['peak_rss_kib'] for r in x])),
                all_environment_qualified=all(r['qualified'] for r in b+x),
                invalid_windows=[dict(mode=r['mode'],repeat=r['repeat']) for r in b+x if not r['qualified']],
                psnr_delta=psnr(arrays['gesdd'],clean)-psnr(arrays['stock'],clean),
                ssim_delta=ssim(arrays['gesdd'],clean)-ssim(arrays['stock'],clean),
                max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))))
    Path(a.out).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(cases=summary['cases'],instrumentation=summary['instrumentation'],errors=summary['errors']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('frames','fixtures','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--kernels');p.add_argument('--direct');main(p.parse_args())
