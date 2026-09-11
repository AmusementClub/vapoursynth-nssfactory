#!/usr/bin/env python3
"""Bounded, non-promoting P2 policy diagnostics on saved identical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat

from p2_reference import denoise_diagnostic
from run import metrics, preview, sha


POLICIES = {
    'p1_control': ('tropp_l1','disjoint'),
    'overlap_only': ('tropp_l1','overlap'),
    'energy_only': ('energy_gain','disjoint'),
    'energy_overlap': ('energy_gain','overlap'),
}


def verify_fit(result, image, dictionary, sigma):
    """Separate support LS/error and group-occurrence aggregation checks."""
    block=int(np.sqrt(dictionary.shape[0]))
    original=np.stack([image[y:y+block,x:x+block].ravel(order='F') for y,x in result.positions])
    expected=[]
    source_indices=[]
    for members,fit in zip(result.groups,result.pursuits):
        signals=original[members].T
        mean=signals.mean(axis=0,keepdims=True)
        centered=signals-mean
        if len(fit.support):
            active=dictionary[:,fit.support]
            reconstructed=active@(np.linalg.pinv(active)@centered)
        else:
            reconstructed=np.zeros_like(centered)
        np.testing.assert_allclose(reconstructed,fit.reconstruction,rtol=1e-10,atol=1e-10)
        energy=float(np.sum((centered-reconstructed)**2))
        assert energy<=fit.epsilon+1e-10*max(1.,np.sum(centered**2))
        expected.append((reconstructed+mean).T)
        source_indices.append(members)
    expected=np.concatenate(expected)
    source_indices=np.concatenate(source_indices)
    if result.grouping=='disjoint':
        order=np.argsort(source_indices)
        expected,source_indices=expected[order],source_indices[order]
    np.testing.assert_array_equal(source_indices,result.occurrence_sources)
    np.testing.assert_allclose(expected,result.reconstructed_occurrences,rtol=1e-10,atol=1e-10)
    numerator=np.zeros_like(image)
    denominator=np.zeros_like(image,dtype=np.int64)
    for patch,index in zip(expected,source_indices):
        y,x=result.positions[index]
        numerator[y:y+block,x:x+block]+=patch.reshape(block,block,order='F')
        denominator[y:y+block,x:x+block]+=1
    np.testing.assert_array_equal(denominator,result.coverage)
    np.testing.assert_allclose(numerator/denominator,result.output,rtol=1e-10,atol=1e-10)
    return {'groups':len(result.groups),'occurrences':len(source_indices),'independent_fit_and_aggregation_passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--dictionaries',required=True)
    parser.add_argument('--p1-results',required=True)
    parser.add_argument('--author-results',required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.repeats<1:
        raise ValueError('positive repeats required')
    fixtures=Path(args.fixtures)
    cases=json.loads((fixtures/'fixtures.json').read_text())['cases']
    if set(args.cases)-{c['id'] for c in cases}:
        raise ValueError('unknown case')
    cases=[c for c in cases if c['id'] in args.cases]
    p1=json.loads((Path(args.p1_results)/'results.json').read_text())['rows']
    authors=json.loads((Path(args.author_results)/'results.json').read_text())['rows']
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in cases:
        for name in ('clean','noisy'):
            assert sha(fixtures/case[name])==case[name+'_sha256']
        shape=(case['height'],case['width'])
        image=np.fromfile(fixtures/case['noisy'],dtype='<f4').reshape(shape).astype(float)
        clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(shape).astype(float)
        block=9 if case['sigma']<=25 else 12 if case['sigma']<=50 else 16
        dictionary_path=Path(args.dictionaries)/f'dict_n{block}.mat'
        dictionary=loadmat(dictionary_path,variable_names=['D'])['D']
        panels=[('Clean',clean)]
        for policy,(solver,grouping) in POLICIES.items():
            expected_hash=None
            for repeat in range(args.repeats):
                started=time.perf_counter()
                result=denoise_diagnostic(image,dictionary,case['sigma']/255,solver=solver,grouping=grouping)
                seconds=time.perf_counter()-started
                audit=verify_fit(result,image,dictionary,case['sigma']/255)
                pixels=result.output.astype('<f4')
                output=out/f"{case['id']}-{policy}-r{repeat}.f32"
                pixels.tofile(output)
                output_hash=sha(output)
                if expected_hash is not None and output_hash!=expected_hash:
                    raise AssertionError('repeat output changed')
                expected_hash=output_hash
                if policy=='p1_control':
                    original=next(r for r in p1 if r['case']==case['id'] and r['repeat']==0)
                    assert original['result']['input_sha256']==case['noisy_sha256']
                    assert output_hash==original['result']['output_sha256']
                members=np.concatenate(result.groups)
                group_offsets=np.r_[0,np.cumsum([len(g) for g in result.groups])]
                support_offsets=np.r_[0,np.cumsum([len(f.support) for f in result.pursuits])]
                support_indices=np.concatenate([f.support for f in result.pursuits])
                trace=out/f"{case['id']}-{policy}-r{repeat}.npz"
                np.savez_compressed(trace,pilot=result.pilot,coverage=result.coverage,positions=result.positions,
                    group_members=members,group_offsets=group_offsets,support_indices=support_indices,
                    support_offsets=support_offsets,epsilon=np.array([f.epsilon for f in result.pursuits]),
                    residual_squared=np.array([f.residual_squared for f in result.pursuits]),
                    occurrence_sources=result.occurrence_sources,occurrence_patches=result.reconstructed_occurrences)
                row={'case':case['id'],'policy':policy,'solver':solver,'grouping':grouping,'repeat':repeat,
                     'seconds_diagnostic_only':seconds,'quality':metrics(clean,pixels),'output':output.name,
                     'output_sha256':output_hash,'input_sha256':case['noisy_sha256'],
                     'dictionary_sha256':sha(dictionary_path),'trace_sha256':sha(trace),'audit':audit,
                     'matching_raw_ssd_threshold':result.threshold,'author_equivalence_verified':False,
                     'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in ('reference.py','p2_reference.py','p2_sweep.py')}}
                rows.append(row)
                with (out/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({'case':case['id'],'policy':policy,'repeat':repeat,'psnr':row['quality']['psnr_db'],
                                  'groups':len(result.groups),'occurrences':len(members),'audit':True}),flush=True)
                if repeat==0:
                    panels.append((policy,pixels))
        author=next(r for r in authors if r['case']==case['id'] and r['algorithm']=='LSSC' and r['variant']=='full')
        assert author['noisy_sha256']==case['noisy_sha256']
        author_path=Path(args.author_results)/author['output']
        assert sha(author_path)==author['output_sha256']
        panels.append(('Author full',np.fromfile(author_path,dtype='<f4').reshape(shape)))
        preview(out/(case['id']+'-comparison.png'),panels)
    summary={'schema':'nss.paper-p2-diagnostics.v1','rows':rows,'p1_controls_byte_identical':True,
             'all_repeat_outputs_byte_identical':True,'auto_promoted_policy':None,
             'author_equivalence_verified':False,'dictionary_learning':False}
    (out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
