#!/usr/bin/env python3
"""Locate NLH branch changes using a production-primitive diagnostic replay.

The independent oracle remains alignment_reference.py. This mixed replay is
only a localization tool: it is never counted as independent correctness proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from alignment_reference import matches,pack,raster
from test_alignment_math import Suite,haar,row_neighbors,NLH_HARD_COEFFICIENT,NLH_WIENER_SIGMA_SCALE
from alignment_campaign import metrics


def main(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    probe=Suite(args.probe,out/'primitives')
    reference_probe=Suite(args.reference_probe,out/'reference-primitives') if args.reference_probe else None
    image=np.load(args.input).astype(np.float32)
    c,h,w=image.shape
    a=np.array([[.299,.587,.114],[-.168736607142857,-.331263392857143,.5],[.5,-.4186875,-.0813125]])
    rgb=c==3
    working=(a@image.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32) if rgb else image
    noise=np.full(c,np.float32(args.sigma/255),np.float32)
    if rgb: noise=np.sqrt((a*a)@(noise.astype(float)**2)).astype(np.float32)
    b=7 if rgb else 8;g=16;q=4;rounds=2 if rgb else 4
    traces={};outputs={};counter=0
    for native in (False,True):
        selected_probe=probe if native else reference_probe
        use_primitives=selected_probe is not None
        if args.estimate:
            noise=np.array([selected_probe.call(f'noise-{channel}','noise',w,h,0,0,
                            [working[channel].ravel(),working[0].ravel()])[0] for channel in range(c)],np.float32)
        basic=working.copy();trace=[]
        for iteration in range(rounds+1):
            stage=int(iteration==rounds)
            data=working if stage else (.6*basic.astype(float)+.4*working.astype(float)).astype(np.float32)
            guide=basic[:1] if stage else data[:1]
            num=np.zeros_like(working,dtype=float);den=np.zeros_like(num)
            for x,y in raster(w,h,b,1):
                if use_primitives:
                    raw=selected_probe.call(f'match-{counter}','match',w,h,b,g,[[1,1,0,40,2,4,0,x,y],guide.ravel()]);counter+=1
                    found=[(int(t),int(xx),int(yy),float(distance)) for xx,yy,t,distance in raw[1:].reshape(-1,4)]
                else: found=matches(guide[None],0,x,y,b,g,40)
                n=2**int(np.log2(len(found)));found=found[:n]
                gp=pack(guide[None],found,b);idx=row_neighbors(gp,q)
                if use_primitives:
                    for channel in range(c):
                        source=pack(data[None,channel:channel+1],found,b)
                        ref=pack(basic[None,channel:channel+1],found,b)
                        raw=selected_probe.call(f'group-{counter}','nlh-wiener' if stage else 'nlh-hard',b*b,n,q,2,
                                       [source,ref,gp,[noise[channel],1,NLH_WIENER_SIGMA_SCALE]]);counter+=1
                        ln=raw[:b*b*n].reshape(b*b,n,order='F');ld=raw[b*b*n:2*b*b*n].reshape(b*b,n,order='F')
                        idx=raw[2*b*b*n:].astype(int).reshape(b*b,q)
                        for j,(_,xx,yy,_) in enumerate(found):
                            num[channel,yy:yy+b,xx:xx+b]+=ln[:,j].reshape(b,b)
                            den[channel,yy:yy+b,xx:xx+b]+=ld[:,j].reshape(b,b)
                else:
                    hq,hn=haar(q),haar(n)
                    for channel in range(c):
                        source=pack(data[None,channel:channel+1],found,b);ref=pack(basic[None,channel:channel+1],found,b)
                        ln=np.zeros_like(source);ld=np.zeros_like(source)
                        for row in range(b*b):
                            coeff=hq@source[idx[row]]@hn.T
                            if stage:
                                rc=hq@ref[idx[row]]@hn.T;gain=rc*rc/(rc*rc+(NLH_WIENER_SIGMA_SCALE*float(noise[channel]))**2)
                                coeff*=gain;coeff*=gain
                            else:
                                coeff[np.abs(coeff)<NLH_HARD_COEFFICIENT*float(noise[channel])]=0;coeff[q-2:,1:]=0
                            rebuilt=hq.T@coeff@hn
                            for k,destination in enumerate(idx[row]): ln[destination]+=rebuilt[k];ld[destination]+=1
                        for j,(_,xx,yy,_) in enumerate(found):
                            num[channel,yy:yy+b,xx:xx+b]+=ln[:,j].reshape(b,b)
                            den[channel,yy:yy+b,xx:xx+b]+=ld[:,j].reshape(b,b)
                trace.append(dict(iteration=iteration,x=x,y=y,coords=[list(v[:3]) for v in found],idx=idx.tolist(),
                                  guide=gp.tolist(),scores=[v[3] for v in found]))
            basic=np.divide(num,den,out=working.astype(float).copy(),where=den>0).astype(np.float32)
            np.save(out/f'{native}-round{iteration}.npy',basic)
        traces[native]=trace
        num=num.astype(np.float32);den=den.astype(np.float32)
        if rgb:
            inv=np.array([[1,0,1.402],[1,-.114*1.772/.587,-.299*1.402/.587],[1,1.772,0]])
            num=(inv@num.astype(float).reshape(3,-1)).reshape(image.shape).astype(np.float32)
        outputs[native]=np.divide(num.astype(float),den,out=image.astype(float).copy(),where=den>0).astype(np.float32)
        np.save(out/f'{native}-output.npy',outputs[native])
    rows=[]
    for iteration in range(rounds+1):
        pairs=[(x,y) for x,y in zip(traces[False],traces[True]) if x['iteration']==iteration]
        block=[(x,y) for x,y in pairs if x['coords']!=y['coords']]
        pixel=[(x,y) for x,y in pairs if x['coords']==y['coords'] and x['idx']!=y['idx']]
        detail=dict(iteration=iteration,queries=len(pairs),changed_block_groups=len(block),changed_pixel_groups=len(pixel),
                    intermediate_difference=metrics(np.load(out/f'False-round{iteration}.npy'),np.load(out/f'True-round{iteration}.npy')))
        if block:
            x,y=block[0];gp=np.array(x['guide']);gn=np.array(y['guide'])
            first=next(k for k in range(len(x['coords'])) if x['coords'][k]!=y['coords'][k])
            detail['first_block_change']=dict(x=x['x'],y=x['y'],rank=first,
                reference_candidate=x['coords'][first],native_candidate=y['coords'][first],
                reference_float_score=x['scores'][first],native_float_score=y['scores'][first],
                reference_fp64_score=float(np.sum((gp[:,0]-gp[:,first])**2)),
                native_fp64_score=float(np.sum((gn[:,0]-gn[:,first])**2)))
        if pixel:
            x,y=pixel[0];ix=np.array(x['idx']);iy=np.array(y['idx']);rr=int(np.flatnonzero(np.any(ix!=iy,axis=1))[0])
            gp=np.array(x['guide']);gn=np.array(y['guide'])
            choices=sorted(set(ix[rr]).symmetric_difference(iy[rr]))
            if not choices: choices=list(dict.fromkeys([*ix[rr],*iy[rr]]))
            detail['first_pixel_change']=dict(x=x['x'],y=x['y'],row=rr,reference_indices=ix[rr].tolist(),native_indices=iy[rr].tolist(),
                guide_max_abs=float(np.max(np.abs(gp-gn))),reference_squared_distances={str(k):float(np.sum((gp[rr]-gp[k])**2)) for k in choices},
                native_input_squared_distances={str(k):float(np.sum((gn[rr]-gn[k])**2)) for k in choices})
        rows.append(detail)
    report=dict(scope='diagnostic production-primitive replay, not an independent oracle',iterations=rows,
                output_difference=metrics(outputs[False],outputs[True]))
    if args.native_output: report['replay_to_plugin']=metrics(np.load(args.native_output),outputs[True])
    if args.reference_output: report['reference_replay_to_plugin']=metrics(np.load(args.reference_output),outputs[False])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sigma',type=float,required=True);p.add_argument('--native-output');p.add_argument('--reference-probe');p.add_argument('--reference-output')
    p.add_argument('--estimate',action='store_true');args=p.parse_args()
    if args.estimate and not args.reference_probe: p.error('--estimate needs --reference-probe for both diagnostic estimator replays')
    main(args)
