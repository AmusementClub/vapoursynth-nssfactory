#!/usr/bin/env python3
"""Independent FP64 NLM stages for frozen grayscale, spatial Welsch cases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def shift(x, dx, dy):
    h,w=x.shape
    return x[np.clip(np.arange(h)+dy,0,h-1)][:,np.clip(np.arange(w)+dx,0,w-1)]

def box(x, radius, axis):
    indices=np.arange(x.shape[axis])
    return sum(np.take(x,np.clip(indices+i,0,x.shape[axis]-1),axis=axis) for i in range(-radius,radius+1))

def synthesize(source, maps, a):
    denominator=np.zeros_like(source);numerator=np.zeros_like(source)
    maximum=np.full_like(source,np.finfo(np.float32).eps)
    index=0
    for dy in range(-a,a+1):
        for dx in range(-a,a+1):
            if dy*(2*a+1)+dx>=0:continue
            weights=maps[index];index+=1
            opposite=shift(weights,-dx,-dy)
            denominator+=weights+opposite
            numerator+=weights*shift(source,dx,dy)+opposite*shift(source,-dx,-dy)
            maximum=np.maximum(maximum,np.maximum(weights,opposite))
    return (numerator+maximum*source)/(denominator+maximum)

parser=argparse.ArgumentParser(description=__doc__)
for n in ('pack','comparison','left-capture','right-capture','baseline','candidate','out'):
    parser.add_argument('--'+n,type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
pack=np.load(args.pack/'fixtures.npz')
comparison=json.loads(args.comparison.read_text())
captures=[np.load(p/'pixels.npz') for p in (args.left_capture,args.right_capture)]
report=dict(numerical_admission=False,cases=[])
for row in comparison['cases']:
    case=row['parameters']
    if case['algorithm']!='NLM' or row['pixel_comparison']['passed']:continue
    if case['radius']!=0:raise ValueError('spatial NLM oracle required')
    index=row['index'];values=pack[f'{case["fixture"]}_s{case["sigma"]:g}_gray'];a,s=2,4
    details=[]
    for position,frame in enumerate((0,2,4)):
        source=values[frame,0];h,w=source.shape
        directory=args.out/f'c{index}-n{frame}';directory.mkdir()
        sourcepath=directory/'input.f32';source.tofile(sourcepath)
        stages=[]
        for side,binary,capture in zip(('baseline','candidate'),(args.baseline,args.candidate),captures):
            target=directory/side
            subprocess.run([str(binary.resolve()),str(sourcepath),str(w),str(h),str(a),str(s),str(max(.1,case['sigma']/2.5)),str(target)],check=True)
            load=lambda name:np.fromfile(target/name,np.float32).astype(float).reshape(h,w)
            output=load('output.f32');expected=capture[f'c{index}'][position,0]
            scale=float(np.fromfile(target/'scale.f32',np.float32)[0])
            exact=bool(np.array_equal(output.astype(np.float32),expected))
            ownmaps=[];truemaps=[];horizontal_error=[];weight_error=[];i=0
            for dy in range(-a,a+1):
                for dx in range(-a,a+1):
                    if dy*(2*a+1)+dx>=0:continue
                    distance=3*(source.astype(float)-shift(source.astype(float),dx,dy))**2
                    horizontal=box(distance,s,1)
                    ownmaps.append(load(f'w{i}.f32'))
                    truemaps.append(np.exp(-scale*box(horizontal,s,0)))
                    horizontal_error.append(float(np.max(abs(load(f'h{i}.f32')-horizontal))))
                    weight_error.append(float(np.max(abs(ownmaps[-1]-truemaps[-1]))));i+=1
            from_weights=synthesize(source.astype(float),ownmaps,a)
            oracle=synthesize(source.astype(float),truemaps,a)
            stages.append(dict(reproduced_public_output=exact,max_horizontal_error=max(horizontal_error),
                                max_weight_error=max(weight_error),max_output_vs_double=float(np.max(abs(output-oracle))),
                                final_arithmetic_error=float(np.max(abs(output-from_weights)))))
        details.append(dict(frame=frame,backends=stages))
    report['cases'].append(dict(index=index,parameters=case,frames=details))
report['reproduced']=all(v['reproduced_public_output'] for r in report['cases'] for f in r['frames'] for v in f['backends'])
report['max_final_arithmetic_error']=max(v['final_arithmetic_error'] for r in report['cases'] for f in r['frames'] for v in f['backends'])
report['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),args.baseline,args.candidate)}
(args.out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(reproduced=report['reproduced'],cases=len(report['cases']),max_final_arithmetic_error=report['max_final_arithmetic_error'])))
