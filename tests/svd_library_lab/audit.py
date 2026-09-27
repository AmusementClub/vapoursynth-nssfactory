#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Independent offline payload, repeated-output and captured-factor audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kernel_campaign import check_factors


def main(a):
    root=Path(a.root);errors=[];counts={};modes={};stats={}
    for stage in ('frames','kernels','direct'):
        complete=root/stage/'complete.json'
        if not complete.exists() or not json.loads(complete.read_text()).get('completed'):
            errors.append(f'{stage}: missing successful completion record')
    for stage in ('frames','direct'):
        directory=root/stage
        if not (directory/'rows.json').exists():continue
        rows=json.loads((directory/'rows.json').read_text());count=0
        for row in rows:
            if row['returncode']:
                errors.append(f'{stage} worker failure: {row}');continue
            label=row['label'] if stage=='frames' else f"{row['case']}/{row['repeat']}/{row['mode']}"
            d=directory/label;pixels=np.load(d/'pixels.npy');result=row['result']
            config=json.loads((d/'config.json').read_text())
            if pixels.shape[0] == result.get('timed_frames', pixels.shape[0]):
                if hashlib.sha256(pixels.tobytes()).hexdigest()!=result['sha256']:errors.append(f'{stage}/{label}: output hash')
            elif pixels.shape[0] < result.get('timed_frames', pixels.shape[0]):
                # The unchanged worker keeps only seven small-frame arrays; retain this as a prefix audit.
                if pixels.shape[0] != 7: errors.append(f'{stage}/{label}: unexpected retained prefix length')
            else: errors.append(f'{stage}/{label}: retained more frames than timed')
            if result['input_sha256']!=config['input_sha256']:errors.append(f'{stage}/{label}: input hash')
            if not np.isfinite(pixels).all():errors.append(f'{stage}/{label}: nonfinite')
            if result.get('timed_source_fills') != 0:errors.append(f'{stage}/{label}: timed source fill')
            # Keep instrumentation and screens separate from formal repeat checks.
            key=(stage,row['case'],row['mode'])
            if stage=='direct' or label.startswith('paired/'):
                first=pixels[0]
                digest=hashlib.sha256(first.tobytes()).hexdigest()
                if key in modes and modes[key]!=digest:errors.append(f'{stage}/{label}: cross-run nondeterminism')
                modes[key]=digest
                if not all(np.array_equal(frame,first) for frame in pixels):errors.append(f'{stage}/{label}: frame nondeterminism')
            for line in (d/'stderr').read_text().splitlines():
                if line.startswith('SVD_LAB '):
                    item=json.loads(line[8:]);k=f"{stage}/{row['mode']}"
                    dest=stats.setdefault(k,dict(calls=0,fallbacks=0,diagonal=0,dense=0,records=0))
                    for field in ('calls','fallbacks','diagonal','dense'):dest[field]+=item[field]
                    dest['records']+=1
                    if item['blas_threads']!=1:errors.append(f'{stage}/{label}: BLAS threads')
            count+=len(pixels)
        counts[stage]=dict(records=len(rows),frames=count,failed_environment=sum(not r['qualified'] for r in rows))
    kernel=root/'kernels';factor_checks=[]
    if (kernel/'rows.json').exists():
        cases={c['name']:c for c in json.loads((kernel/'cases.json').read_text())}
        seen={};rows=json.loads((kernel/'rows.json').read_text())
        for row in rows:
            c=cases[row['case']];d=kernel/row['case']/str(row['repeat'])/row['mode'];key=(row['case'],row['mode'])
            hashes=[]
            for file,field in [('output.factors','factors_sha256'),('output.f32','pixels_sha256')]:
                digest=hashlib.sha256((d/file).read_bytes()).hexdigest();hashes.append(digest)
                if digest!=row[field]:errors.append(f'{d/file}: hash')
            if key in seen and seen[key]!=hashes:errors.append(f'{key}: kernel nondeterminism')
            seen[key]=hashes
            if row['repeat']==0:
                matrices=np.fromfile(kernel/Path(c['input']).name,'<f4').reshape(c['count'],c['n'],c['m']).transpose(0,2,1)
                checks=check_factors(d/'output.factors',matrices,c['m'],c['n'])
                factor_checks.append(dict(case=c['name'],mode=row['mode'],matrices=len(checks),
                    reconstruction_max=max(x['relative_reconstruction'] for x in checks),
                    orthogonality_max=max(x['orthogonality'] for x in checks),
                    spectrum_error_max=max(x['spectrum_normalized_max_abs'] for x in checks)))
        counts['kernels']=dict(records=len(rows),factor_matrices=sum(x['matrices'] for x in factor_checks),
                              failed_environment=sum(not r['qualified'] for r in rows))
    out=dict(counts=counts,statistics=stats,factor_checks=factor_checks,errors=errors,passed=not errors)
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,errors=errors,passed=not errors)))
    if errors:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
