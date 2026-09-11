#!/usr/bin/env python3
"""Require byte identity between cached preparation and the public filter."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import vapoursynth as vs

from nlh_defaults_search import baseline
from nlh_defaults_verify import cases
from nlh_defaults_inputs import load_case,file_sha,save_json
from nlh_search_native import Native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plugin','bridge','inputs','out'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--real-size',type=int,default=64)
    args=parser.parse_args();os.sched_setaffinity(0,{0})
    root=Path(args.inputs);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'inputs.json').read_text());native=Native(args.bridge)
    core=vs.core;core.num_threads=1;core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    inputs=list(cases())
    for case in [c for c in manifest['cases'] if c['dataset']=='CC' and c['split']=='development'][:2]:
        _,noisy,_=load_case(root,manifest,case,args.real_size)
        inputs.append((case['id'],noisy,None,dict(block_size=[6,10],block_step=[3,5],basic_iters=2,
                                               wiener_sigma_scale=.32,hard_strength=.7)))
    rows=[]
    for name,image,sigma,changes in inputs:
        c,h,w=image.shape;p=baseline('rgb' if c==3 else 'gray-high' if sigma and sigma>50 else 'gray-low');p.update(changes)
        prepared=native.prepare(image,sigma)
        actual,seconds,groups=native.run(prepared,p)
        blank=core.std.BlankClip(width=w,height=h,length=1,format=vs.RGBS if c==3 else vs.GRAYS)
        def fill(n,f):
            result=f.copy()
            for plane in range(c):np.asarray(result[plane])[:]=image[plane]
            return result
        source=core.std.ModifyFrame(blank,blank,fill);kwargs=dict(p)
        if sigma is not None:kwargs['sigma']=sigma
        node=core.nss.NLH(source,**kwargs);frame=node.get_frame(0)
        expected=np.stack([np.asarray(frame[plane]).copy() for plane in range(c)])
        sigmas=frame.props['_NSSSigma'];sigmas=[sigmas] if isinstance(sigmas,(int,float)) else list(sigmas)
        row=dict(case=name,exact=bool(np.array_equal(actual,expected)),
                 sigma_exact=prepared['sigma']==sigmas,groups_exact=groups==frame.props['_NSSGroups'],
                 max_abs=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
                 native_filter_seconds=seconds,prepare_seconds=prepared['prepare_seconds'])
        rows.append(row);native.free(prepared);del frame,node,source
        print(name,row,flush=True)
    passed=all(r['exact'] and r['sigma_exact'] and r['groups_exact'] for r in rows)
    save_json(out/'verification.json',dict(passed=passed,rows=rows,plugin_sha256=file_sha(args.plugin),
              bridge_sha256=file_sha(args.bridge),script_sha256=file_sha(__file__),
              boundary='screening cache only; performance acceptance uses independent public get_frame runs'))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
