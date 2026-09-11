#!/usr/bin/env python3
"""Localize full-NLH differences with fixed production inputs and selectors.

This is a diagnostic, not the standalone independent full-model oracle. It
checks a production replay against plugin pixels, then independently evaluates
SSD/row distances and Haar/aggregation at every recorded native stage. The
unconstrained independent full-image results remain separate and unchanged.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from alignment_campaign import metrics, sha
from alignment_reference_fast import patch_matrix, strip_matches
from test_alignment_math import haar, NLH_HARD_COEFFICIENT, NLH_WIENER_SIGMA_SCALE


def gamma(count):
    unit = np.finfo(np.float32).eps / 2
    return count * unit / (1 - count * unit)


def score_changes(native, expected, native_scores, expected_scores, operations):
    changed = native != expected
    gaps = np.abs(native_scores - expected_scores)
    # Each FP32 result may lie on either side of its exact SSD. The budget is
    # conservative for subtraction, squaring and summation of finite normal
    # samples; this fixture does not exercise underflow/overflow arithmetic.
    bound = 2 * gamma(operations) * np.maximum(np.maximum(native_scores, expected_scores), 1e-30)
    invalid = changed & (gaps > bound)
    first = None
    if changed.any():
        index = tuple(np.argwhere(changed)[0])
        first = dict(index=list(map(int, index)), native=int(native[index]), expected=int(expected[index]),
                     native_fp64_score=float(native_scores[index]), expected_fp64_score=float(expected_scores[index]),
                     gap=float(gaps[index]), bound=float(bound[index]))
    return dict(changed_entries=int(changed.sum()), outside_rounding_bound=int(invalid.sum()),
                max_changed_gap=float(gaps[changed].max()) if changed.any() else 0, first=first)


def main(args):
    trace, out = Path(args.trace).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    c, h, w = 3, args.height, args.width
    b, m, n, q = 7, 49, 16, 4
    ny, nx = h-b+1, w-b+1
    def load(name): return np.fromfile(trace/name, dtype='<f4').reshape(c,h,w)
    host = np.load(args.plugin_output)
    replay = load('output.f32')
    np.testing.assert_array_equal(replay, host)
    report = dict(scope=__doc__, passed=False, replay_exact=True, plugin_output_sha256=sha(args.plugin_output),
                  stages=[], source_sha256=sha(__file__))
    noise = np.fromfile(trace/'noise.f32', dtype='<f4')
    hq, hn = haar(q), haar(n)
    inverse = np.array([[1,0,1.402],[1,-.114*1.772/.587,-.299*1.402/.587],[1,1.772,0]])
    for stage in range(3):
        started=time.perf_counter();wiener=stage==2
        data=load(f'data{stage}.f32');basic=load(f'basic{stage}.f32')
        guide=basic[:1] if wiener else data[:1]
        p=patch_matrix(data,b).reshape(nx*ny,c,m)
        gp=patch_matrix(guide,b)
        bp=patch_matrix(basic,b).reshape(nx*ny,c,m) if wiener else None
        selected_native=np.memmap(trace/f'blocks{stage}.u32',dtype='<u4',mode='r',shape=(nx*ny,n))
        neighbors_native=np.memmap(trace/f'pixels{stage}.u8',dtype='u1',mode='r',shape=(nx*ny,m,q))
        patch_num=np.zeros((nx*ny,c*m),float);patch_den=np.zeros((nx*ny,m),float)
        row=dict(stage=stage, block_changed_queries=0, block_changed_entries=0, block_outside_bound=0,
                 pixel_changed_rows=0,pixel_changed_entries=0,pixel_outside_bound=0,
                 block_max_gap=0,pixel_max_gap=0,first_block=None,first_pixel=None)
        for y0 in range(0,ny,8):
            height=min(8,ny-y0);begin=y0*nx;end=(y0+height)*nx
            expected=strip_matches(guide,b,n,40,y0,height)
            selected=selected_native[begin:end]
            changed=np.any(selected!=expected,axis=1)
            row['block_changed_queries']+=int(changed.sum())
            if changed.any():
                query=np.arange(begin,end)[changed]
                actual=selected[changed];ideal=expected[changed]
                center=gp[query].astype(float)
                scores_a=np.sum((gp[actual].astype(float)-center[:,None,:])**2,axis=2)
                scores_b=np.sum((gp[ideal].astype(float)-center[:,None,:])**2,axis=2)
                detail=score_changes(actual,ideal,scores_a,scores_b,3*m)
                row['block_changed_entries']+=detail['changed_entries'];row['block_outside_bound']+=detail['outside_rounding_bound']
                row['block_max_gap']=max(row['block_max_gap'],detail['max_changed_gap'])
                if row['first_block'] is None:
                    detail['query_index']=int(query[detail['first']['index'][0]])
                    row['first_block']=detail
            for offset in range(begin,end,16):
                selected=selected_native[offset:min(offset+16,end)];batch=len(selected)
                neighbors=neighbors_native[offset:offset+batch].astype(int)
                packed=gp[selected].transpose(0,2,1).astype(float)
                distance=np.sum((packed[:,:,None,:]-packed[:,None,:,:])**2,axis=3)
                rr=np.arange(m);distance[:,rr,rr]=-1
                ideal=np.argsort(distance,axis=2,kind='stable')[:,:,:q]
                detail=score_changes(neighbors,ideal,np.take_along_axis(distance,neighbors,axis=2),
                                     np.take_along_axis(distance,ideal,axis=2),3*n)
                row['pixel_changed_rows']+=int(np.any(neighbors!=ideal,axis=2).sum())
                row['pixel_changed_entries']+=detail['changed_entries'];row['pixel_outside_bound']+=detail['outside_rounding_bound']
                row['pixel_max_gap']=max(row['pixel_max_gap'],detail['max_changed_gap'])
                if row['first_pixel'] is None and detail['first'] is not None:
                    detail['query_index']=offset+detail['first']['index'][0];row['first_pixel']=detail
                bi=np.arange(batch)[:,None,None,None];ci=np.arange(c)[None,:,None,None]
                data_group=p[selected].transpose(0,2,3,1).astype(float)
                coefficients=hq@data_group[bi,ci,neighbors[:,None,:,:],:]@hn.T
                if wiener:
                    reference=bp[selected].transpose(0,2,3,1).astype(float)[bi,ci,neighbors[:,None,:,:],:]
                    coeff_ref=hq@reference@hn.T;r2=coeff_ref*coeff_ref
                    gain=r2/(r2+((NLH_WIENER_SIGMA_SCALE*noise.astype(float))**2)[None,:,None,None,None])
                    coefficients*=gain;coefficients*=gain
                else:
                    coefficients[np.abs(coefficients)<(NLH_HARD_COEFFICIENT*noise.astype(float))[None,:,None,None,None]]=0
                    coefficients[:,:,:,q-2:,1:]=0
                restored=hq.T@coefficients@hn
                local_num=np.zeros((batch,c,m,n),float);local_den=np.zeros((batch,m),float)
                for k in range(q):
                    np.add.at(local_num,(np.arange(batch)[:,None,None],np.arange(c)[None,:,None],neighbors[:,None,:,k]),restored[:,:,:,k,:])
                    np.add.at(local_den,(np.arange(batch)[:,None],neighbors[:,:,k]),1)
                np.add.at(patch_num,selected,local_num.transpose(0,3,1,2).reshape(batch,n,c*m))
                np.add.at(patch_den,selected,np.broadcast_to(local_den[:,None,:],(batch,n,m)))
            (out/'progress.json').write_text(json.dumps(dict(stage=stage,query_rows=y0+height,seconds=time.perf_counter()-started))+'\n')
        num=np.zeros_like(data,dtype=float);den=np.zeros((h,w),float)
        pn=patch_num.reshape(ny,nx,c,b,b);pd=patch_den.reshape(ny,nx,b,b)
        for dy in range(b):
            for dx in range(b):
                for channel in range(c):num[channel,dy:dy+ny,dx:dx+nx]+=pn[:,:,channel,dy,dx]
                den[dy:dy+ny,dx:dx+nx]+=pd[:,:,dy,dx]
        if wiener:
            num=(inverse@num.astype(np.float32).astype(float).reshape(3,-1)).reshape(data.shape).astype(np.float32)
            result=(num.astype(float)/den.astype(np.float32)).astype(np.float32);expected=replay
        else:
            result=(num/den).astype(np.float32);expected=load(f'result{stage}.f32')
        row['fixed_input_selector_math_difference']=metrics(expected,result)
        row['seconds']=time.perf_counter()-started
        np.save(out/f'fixed-stage{stage}.npy',result)
        report['stages'].append(row)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        np.testing.assert_allclose(result,expected,atol=2e-5,rtol=2e-4)
        if row['block_outside_bound'] or row['pixel_outside_bound']:
            raise AssertionError('selector difference outside its FP32 SSD forward-error bound')
        print(stage,row['block_changed_queries'],row['pixel_changed_rows'],row['fixed_input_selector_math_difference']['max_abs'],flush=True)
    report['passed']=True
    report['trace_sha256']={p.name:sha(p) for p in sorted(trace.iterdir()) if p.is_file()}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('trace','out','plugin-output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--width',type=int,default=512);parser.add_argument('--height',type=int,default=512)
    main(parser.parse_args())
