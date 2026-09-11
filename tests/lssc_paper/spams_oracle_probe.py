#!/usr/bin/env python3
"""Compare math probes with a separately built SPAMS 2.1 executable.

The external binary is a research oracle, not a linked NSSFactory dependency and
not the ICCV denoising MEX. No SPAMS source or binary is redistributed here.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from p2_reference import simultaneous_ols
from reference import somp
from run import sha


def execute(oracle, directory, name, dictionary, signals, epsilon):
    path=directory/(name+'.input.bin')
    output=directory/(name+'.output.bin')
    m,k=dictionary.shape
    n=signals.shape[1]
    with path.open('wb') as stream:
        stream.write(struct.pack('<3qd',m,k,n,epsilon))
        stream.write(dictionary.astype('<f8').tobytes(order='F'))
        stream.write(signals.astype('<f8').tobytes(order='F'))
    run=subprocess.run([str(oracle),str(path),str(output)],capture_output=True,text=True,timeout=20)
    (directory/(name+'.log')).write_text(run.stdout+run.stderr)
    if run.returncode:
        raise RuntimeError(f'oracle exit {run.returncode}: {run.stderr}')
    data=output.read_bytes()
    count=struct.unpack_from('<q',data)[0]
    if count<0 or count>min(m,k) or len(data)!=8+count*8+count*n*8:
        raise ValueError('invalid oracle protocol response')
    support=np.frombuffer(data,dtype='<i8',count=count,offset=8).copy()
    coefficients=np.frombuffer(data,dtype='<f8',offset=8+count*8).reshape(count,n,order='F').copy()
    if np.any(support<0) or np.any(support>=k):
        raise ValueError('invalid oracle support')
    return support,dictionary[:,support]@coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    oracle=Path(args.oracle).resolve()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    cases=[('l1-versus-energy',np.eye(2),np.array([[1.,1.],[1.5,0.]]),2.3,False),
           ('below-budget',np.eye(3),np.ones((3,2))*.01,1.,False),
           ('exact-budget-boundary',np.eye(2),np.array([[1.,0.],[0.,1.]]),2.,True)]
    rng=np.random.default_rng(20260908)
    for group_size in [1,2,7,40]:
        for index in range(6):
            dictionary=rng.normal(size=(8,16))
            dictionary/=np.linalg.norm(dictionary,axis=0)
            signals=rng.normal(size=(8,group_size))
            epsilon=float(np.sum(signals**2)*.3)
            cases.append((f'random-g{group_size}-{index}',dictionary,signals,epsilon,False))
    rows=[]
    for name,dictionary,signals,epsilon,boundary in cases:
        support,reconstructed=execute(oracle,out,name,dictionary,signals,epsilon)
        candidates={'tropp_l1':somp(dictionary,signals,epsilon),
                    'energy_gain':simultaneous_ols(dictionary,signals,epsilon)}
        comparisons={}
        for policy,fit in candidates.items():
            comparisons[policy]={'support_equal':np.array_equal(support,fit.support),
                'max_abs_reconstruction_difference':float(np.max(np.abs(reconstructed-fit.reconstruction))),
                'support':fit.support.tolist()}
        row={'case':name,'boundary_case':boundary,'oracle_support':support.tolist(),
             'oracle_residual_squared':float(np.sum((signals-reconstructed)**2)),
             'epsilon':epsilon,'comparisons':comparisons}
        rows.append(row)
        print(json.dumps(row),flush=True)
    summary={'schema':'nss.spams21-policy-probe.v1','rows':rows,'oracle_sha256':sha(oracle),
             'source_sha256':sha(__file__),'iccv_denoise_mex_equivalence':False,
             'source_scope':'separately compiled SPAMS v2.1 coreSOMP with one syntax-only Clang compatibility edit',
             'nonboundary_cases':sum(not r['boundary_case'] for r in rows)}
    (out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
