#!/usr/bin/env python3
"""Independent double DCT and NCSR algebra on captured actual batch groups."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read(path):
    data = path.read_bytes(); cursor = 0; rows = []
    def floats(n):
        nonlocal cursor
        out = np.frombuffer(data, '<f4', n, cursor).copy(); cursor += n * 4
        return out
    batch = 0
    while cursor < len(data):
        magic, kind, count, status = struct.unpack_from('<4i', data, cursor); cursor += 16
        if magic != 0x4e534747 or kind not in (1,2,3,4,5) or count <= 0 or status:
            raise ValueError('invalid/failed group batch')
        for index in range(count):
            if kind == 3:
                m,n,ld,nch,iters,residual,adaptive,gemm = struct.unpack_from('<8i', data, cursor); cursor += 32
                rho,mu,weight = floats(3)
                row = dict(kind=kind,batch=batch,index=index,m=m,n=n,k=n,ld=ld,nch=nch,iters=iters,
                           residual=residual,adaptive=adaptive,gemm=gemm,rho=float(rho),mu=float(mu),weight=float(weight))
                row['sigmas'] = floats(nch)
                row['input'] = floats(m*n).reshape(n,m)
                row['output'] = floats(m*n).reshape(n,m)
                rows.append(row)
                continue
            m, n, k, ld, flag = struct.unpack_from('<5i', data, cursor); cursor += 20
            sigma, weight = floats(2)
            row = dict(kind=kind, batch=batch, index=index, m=m, n=n, k=k, ld=ld, flag=flag, sigma=float(sigma), weight=float(weight))
            row['input'] = floats(m*n).reshape(n,m)
            row['reference'] = floats(m*n).reshape(n,m) if kind == 1 else floats(n)
            if kind == 4: row['row_weight'] = floats(m)
            row['output'] = floats(m*n).reshape(n,m)
            if kind in (2,4,5): row['u'] = floats(m*n).reshape(n,m).T; row['s'] = floats(n)
            if kind == 5: row['actual_weights'] = floats(n)
            rows.append(row)
        batch += 1
    if cursor != len(data): raise ValueError('truncated group stream')
    return rows


def dct_matrix(n):
    k = np.arange(n)[:,None]; i = np.arange(n)[None,:]
    return np.cos(np.pi * (i + .5) * k / n) * np.sqrt(np.where(k == 0, 1., 2.) / n)


def dct(cube, inverse=False):
    result = cube.astype(np.float64)
    for axis in (2,1,0):
        matrix = dct_matrix(cube.shape[axis])
        result = np.moveaxis(np.moveaxis(result, axis, -1) @ (matrix if inverse else matrix.T), -1, axis)
    return result


def bm_compare(a,b):
    m,n=a['m'],a['n']; block=int(np.sqrt(m)); shape=(n,block,block)
    if not np.array_equal(a['input'], b['input']) or not np.array_equal(a['reference'], b['reference']):
        return dict(passed=False, reason='group_inputs_differ')
    coeff = dct(a['input'].reshape(shape))
    outputs = [dct(r['output'].reshape(shape)) for r in (a,b)]
    if a['flag']: raise ValueError('hard-threshold capture expected')
    threshold = float(np.float32(2.7) * np.float32(a['sigma']))
    masks = [np.abs(o) > 2e-6 for o in outputs]
    for mask in masks: mask[0,0,0] = True
    changed = masks[0] != masks[1]
    margin = float(np.max(np.abs(np.abs(coeff[changed]) - threshold))) if changed.any() else 0.
    stable = masks[0] & masks[1]
    continuous = float(np.max(np.abs(outputs[0][stable] - outputs[1][stable]))) if stable.any() else 0.
    # The retained Plan01 threshold-replay bounds, without relaxing them.
    passed = margin <= 2e-6 and continuous <= 2e-5
    per_backend = []
    for row, output, mask in zip((a,b), outputs, masks):
        expected = np.where(mask, coeff, 0.)
        expected[0,0,0] = coeff[0,0,0]
        reconstructed = dct(expected, inverse=True)
        residual = row['output'].reshape(shape).astype(float) - reconstructed
        ratio = float(np.max(np.abs(residual)/(2e-5+2e-4*np.abs(reconstructed))))
        # Weight counts can be recovered from the complete padded cube.
        expected_weight = 1/max(1,int(mask.sum()))
        weight_error = abs(row['weight'] - expected_weight)
        per_backend.append(dict(oracle_reconstruction_max_abs=float(np.max(np.abs(residual))),
                                reconstruction_tolerance_ratio=ratio, weight_error=weight_error))
        passed = passed and ratio <= 1 and weight_error <= 1e-7
    return dict(passed=passed, changed_masks=int(changed.sum()), max_threshold_margin=margin,
                stable_coefficient_delta=continuous, backends=per_backend)


def ncsr_oracle(row):
    # Validate SVD independently, then use the measured basis to isolate the
    # remaining weighting/centralization/reconstruction arithmetic. Repeated
    # singular subspaces need not choose the same vectors in NumPy and C++.
    x = row['input'].astype(float).T; u = row['u'].astype(float); m,n=x.shape
    centered = x - x.mean(axis=1,keepdims=True)
    singular = np.linalg.svd(centered,compute_uv=False)
    spectrum_error = float(np.linalg.norm(row['s'][:len(singular)]-singular)/max(np.linalg.norm(singular),1e-30))
    projection_error = float(np.linalg.norm(u@(u.T@centered)-centered)/max(np.linalg.norm(centered),1e-30))
    b = u.T@centered
    sigma = row['sigma']; h=max(2*m*sigma*sigma,1e-12)
    distances = row['reference'].astype(float) if row['flag'] else np.sum((centered-centered[:,0:1])**2,axis=0)
    ideal_weights = np.exp(-distances/h)
    weights = row.get('actual_weights', ideal_weights).astype(float)
    weight_error = float(np.max(np.abs(weights-ideal_weights)))
    weight_sum_error = float(abs(weights.sum()-ideal_weights.sum()))
    weights=weights/weights.sum()
    beta = b@weights
    variance = np.sum(weights[None,:]*(b-beta[:,None])**2,axis=1)
    threshold = float(np.float32(2.8284271247461903))*sigma*sigma/(np.sqrt(variance)+1e-12)
    codes = beta[:,None]+np.sign(b-beta[:,None])*np.maximum(np.abs(b-beta[:,None])-threshold[:,None],0.)
    expected = u@codes+x.mean(axis=1,keepdims=True)
    error = np.abs(row['output'].T.astype(float)-expected)
    ratio = float(np.max(error/(2e-5+2e-4*np.abs(expected))))
    return dict(spectrum_relative_error=spectrum_error, projection_relative_error=projection_error,
                finisher_max_abs=float(error.max()), finisher_tolerance_ratio=ratio,
                measured_weight_policy='FastExp' if 'actual_weights' in row else 'ideal exp',
                weight_max_abs=weight_error,weight_sum_error=weight_sum_error,
                passed=bool(spectrum_error<=2e-5 and projection_error<=2e-4 and ratio<=1
                       and weight_error<=3e-5 and weight_sum_error<=2e-4*ideal_weights.sum()+1e-6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left',type=Path);parser.add_argument('right',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args()
    left,right=read(args.left),read(args.right)
    if len(left)!=len(right) or not left: raise RuntimeError('unaligned/empty group capture')
    groups=[]
    for a,b in zip(left,right):
        if any(a[k]!=b[k] for k in ('kind','batch','index','m','n','k','ld','flag','sigma')): raise RuntimeError('group metadata differ')
        result=bm_compare(a,b) if a['kind']==1 else dict(backends=[ncsr_oracle(a),ncsr_oracle(b)])
        if a['kind']==2: result['passed']=all(x['passed'] for x in result['backends'])
        groups.append(dict(batch=a['batch'],index=a['index'],**result))
    report=dict(passed=all(r['passed'] for r in groups), groups=groups, numerical_admission=False,
                files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.left,args.right,Path(__file__))})
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report['passed'],groups=len(groups),failures=sum(not r['passed'] for r in groups))))
