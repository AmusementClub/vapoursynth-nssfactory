#!/usr/bin/env python3
"""Save concrete NLH hard-threshold rounding witnesses on native-stage inputs.

Diagnostic only: production Haar coefficients are compared with explicit FP64
Haar matrices. A witness must straddle the literal threshold inside the existing
transform error band; this does not replace the independent full-image oracle.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from alignment_campaign import sha
from alignment_reference_fast import patch_matrix
from test_alignment_math import haar, NLH_HARD_COEFFICIENT


def main(args):
    trace=Path(args.trace).resolve();out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    h=w=512;c=3;b=7;m=49;q=4;n=16;ny=nx=506
    noise=np.fromfile(trace/'noise.f32',dtype='<f4').astype(float)
    hq,hn=haar(q),haar(n);rows=[]
    for stage in (0,1):
        data=np.fromfile(trace/f'data{stage}.f32',dtype='<f4').reshape(c,h,w)
        p=patch_matrix(data,b).reshape(nx*ny,c,m)
        selected=np.memmap(trace/f'blocks{stage}.u32',dtype='<u4',mode='r',shape=(nx*ny,n))
        neighbors=np.memmap(trace/f'pixels{stage}.u8',dtype='u1',mode='r',shape=(nx*ny,m,q))
        found=None
        for offset in range(0,nx*ny,8):
            sel=selected[offset:offset+8];idx=neighbors[offset:offset+8].astype(int);batch=len(sel)
            group=p[sel].transpose(0,2,3,1).astype(float)
            matrices=group[np.arange(batch)[:,None,None,None],np.arange(c)[None,:,None,None],idx[:,None,:,:],:]
            reference=hq@matrices@hn.T
            request=out/'request.f64';response=out/'response.f64'
            matrices.swapaxes(-1,-2).copy().tofile(request)
            subprocess.run([str(Path(args.probe).resolve()),'haar-batch','4','16','0',str(batch*c*m),str(request),str(response)],check=True)
            native=np.fromfile(response,dtype='<f8').reshape(batch,c,m,n,q).swapaxes(-1,-2)
            np.testing.assert_allclose(native,reference,atol=2e-5,rtol=2e-4)
            threshold=(NLH_HARD_COEFFICIENT*noise)[None,:,None,None,None]
            flips=(np.abs(native)<threshold)!=(np.abs(reference)<threshold)
            flips[:,:,:,q-2:,1:]=False
            if flips.any():
                index=tuple(np.argwhere(flips)[0]);bb,cc,pixel,kr,j=index
                a=float(native[index]);z=float(reference[index]);t=float(threshold[0,cc,0,0,0])
                found=dict(stage=stage,query_index=offset+int(bb),channel=int(cc),pixel_query=int(pixel),
                           coefficient=[int(kr),int(j)],native=a,reference=z,threshold=t,
                           coefficient_error=abs(a-z),reference_threshold_distance=abs(abs(z)-t),
                           checked_queries=offset+batch,forward_error_band=2e-5)
                assert abs(a-z)<=2e-5 and abs(abs(z)-t)<=abs(a-z)
                np.savez(out/f'witness-stage{stage}.npz',matrix=matrices[bb,cc,pixel],
                         native=native[bb,cc,pixel],reference=reference[bb,cc,pixel])
                break
            if offset%1024==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,queries=offset+batch))+'\n')
        if found is None:raise AssertionError('no literal-threshold witness found in this native stage')
        rows.append(found);print(json.dumps(found),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(passed=True,scope=__doc__,rows=rows,
             source_sha256=sha(__file__),probe_sha256=sha(args.probe)),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('trace','out','probe'):parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
