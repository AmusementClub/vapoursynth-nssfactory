#!/usr/bin/env python3
"""Active subsampled-plane, independent reference-stride and boundary checks."""
import json
import os
from pathlib import Path
import numpy as np
import vapoursynth as vs
from test_plan01_plugin import clip, evaluate, options, reject


def main():
    core = vs.core
    core.num_threads = 8
    core.std.LoadPlugin(path=os.environ['NSS_SO'])
    rows = []
    source = clip(core, width=36, height=34, frames=3, fmt=vs.YUV420PS, parent_extra=64)
    reference = clip(core, width=36, height=34, frames=3, fmt=vs.YUV420PS, parent_extra=128)
    packed = clip(core, width=36, height=34, frames=3, fmt=vs.YUV420PS)
    for name in ('BM3D','WNNM','TWSC','NCSR','NLH','LSSC'):
        radius = 0 if name == 'LSSC' else 2
        kwargs = options(name, radius=radius, sigma=[0,3,0])
        key = 'ref' if name == 'BM3D' else 'rclip'
        if name != 'LSSC': kwargs[key] = reference
        candidate = getattr(core.nss,name)(source,**kwargs)
        if name != 'LSSC': kwargs[key] = packed
        expected = getattr(core.nss,name)(packed,**kwargs)
        if radius:
            candidate = core.nss.VAggregate(candidate,source,radius=radius)
            expected = core.nss.VAggregate(expected,packed,radius=radius)
        error = 0.
        for n in (2,0,1,2):
            actual,_ = evaluate(candidate,n)
            ordinary,_ = evaluate(expected,n)
            src = source.get_frame(n)
            for plane in (0,2):
                if not np.array_equal(actual[plane],np.asarray(src[plane])):
                    raise AssertionError(f'{name} changed an unselected subsampled-format plane')
            error=max(error,float(np.max(np.abs(actual[1].astype(np.float64)-ordinary[1]))))
        if error>2e-5:raise AssertionError((name,error))
        rows.append(dict(algorithm=name,format='YUV420PS',active_plane=1,max_abs=error))
    for d in (0,1,3):
        unusual=core.nss.NLM(source,rclip=reference,d=d,a=1,s=1,h=3,channels='UV')
        ordinary=core.nss.NLM(packed,rclip=packed,d=d,a=1,s=1,h=3,channels='UV')
        error=0.
        for n in (2,0,1):
            a,_=evaluate(unusual,n);b,_=evaluate(ordinary,n)
            if not np.array_equal(a[0],np.asarray(source.get_frame(n)[0])):
                raise AssertionError('NLM UV changed Y')
            error=max(error,max(float(np.max(np.abs(x.astype(np.float64)-y))) for x,y in zip(a,b)))
        if error>2e-5:raise AssertionError(('NLM UV stride',d,error))
        rows.append(dict(algorithm='NLM',format='YUV420PS',channels='UV',d=d,max_abs=error))
    reject(lambda:core.nss.MCWNNM(source), 'RGBS or YUV444PS')
    # Maximum group-filter radius on a short clip must not replicate boundary
    # candidates. Compare the same actual support at radius 2 and 16.
    for name in ('BM3D','WNNM','TWSC','NCSR','NLH','MCWNNM'):
        src=clip(core,width=16,height=16,frames=3,fmt=vs.RGBS if name=='MCWNNM' else vs.GRAYS)
        outputs=[]
        for radius in (2,16):
            fat=getattr(core.nss,name)(src,**options(name,radius=radius))
            outputs.append(core.nss.VAggregate(fat,src,radius=radius))
        error=0.
        for n in (0,2,1):
            a,_=evaluate(outputs[0],n);b,_=evaluate(outputs[1],n)
            error=max(error,max(float(np.max(np.abs(x.astype(np.float64)-y))) for x,y in zip(a,b)))
        if error>2e-5:raise AssertionError(('duplicated boundary support',name,error))
        rows.append(dict(algorithm=name,radius_pair=[2,16],max_abs=error))
    result=dict(passed=True,cases=rows,
                source_strides=[source.get_frame(0).get_stride(p) for p in range(3)],
                reference_strides=[reference.get_frame(0).get_stride(p) for p in range(3)])
    print(json.dumps(result))
    if len(os.sys.argv)>1:Path(os.sys.argv[1]).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
