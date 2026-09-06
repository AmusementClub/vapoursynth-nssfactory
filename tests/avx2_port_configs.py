#!/usr/bin/env python3
"""Frozen, <=8 group per candidate AVX2 port screens."""
import argparse
import copy
import json
from pathlib import Path

def case(algorithm='bm3d',block=8,group=16,stage='two_stage',radius=0,size=None,**kw):
    size=size or ([1920,1080] if algorithm=='bm3d' and radius==0 else [320,180])
    args=dict(sigma=3,block_size=block,group_size=group,block_step=min(block,8),bm_range=7,radius=radius)
    if algorithm=='nlh':args.update(q=4)
    if algorithm in ('twsc','mcwnnm','ncsr'):args['iters']=1
    args.update(kw)
    name=f'{algorithm}_b{block}_g{group}_{stage}_r{radius}'
    if 'q' in args:name+=f"_q{args['q']}"
    row=dict(name=name,algorithm=algorithm,size=size,frames=2,warmup=1,stage=stage,kwargs=args)
    if size==[1920,1080]:row['sample']='/opt/nss-c4/samples/gray8/MAPPA.gray8'
    return row

def matrices():
    top=[case(group=g) for g in (16,32,64)]+[case(a,group=32,stage='basic') for a in ('wnnm','twsc','ncsr')]+[case('nlh',radius=1,stage='wiener'),case(group=8)]
    match={}
    for b in (4,12,16):
        match[b]=[case(block=b,group=g) for g in (8,16,32)]+[case(block=b,stage=s) for s in ('basic','wiener')]+[case(block=b,radius=1),case(block=b,group=1),case('wnnm',block=b,stage='basic')]
    nlh=[case('nlh',stage=s,radius=r) for r in (0,1) for s in ('basic','wiener')]+[case('nlh',stage='basic',q=2),case('nlh',group=8,stage='basic'),case('nlh',block=4,stage='basic'),case('nlh',stage='two_stage')]
    dct=[case(block=16,group=g,stage=s) for g in (1,8) for s in ('basic','wiener','two_stage')]+[case(block=16,group=16),case(block=8)]
    nlhmatch=[case('nlh',stage=s,bm_range=search) for search in (7,20) for s in ('basic','wiener')]+[case('nlh',radius=1,stage='basic'),case('nlh',block=4,stage='basic')]
    for i,row in enumerate(nlhmatch):row['name']+=f'_search{row["kwargs"]["bm_range"]}'
    gemm=[case('wnnm',group=g,stage='basic',size=[640,360]) for g in (8,16,32)]+[case('twsc',group=g,stage='basic') for g in (8,32)]+[case('mcwnnm',group=g,stage='basic',size=[160,96]) for g in (8,16)]+[dict(name='lssc_default',algorithm='lssc',size=[640,360],frames=2,warmup=1)]
    return {1:top,2:match[4],4:match[12],8:match[16],16:nlh,32:nlh,64:nlh,96:nlh,128:dct,256:nlhmatch,512:match[12],1024:gemm}

def write(out,samples):
    out.mkdir(parents=True,exist_ok=True)
    for mask,rows in matrices().items():
        assert len(rows)<=8 and len({r['name'] for r in rows})==len(rows)
        rows=copy.deepcopy(rows)
        for row in rows:
            if 'sample' in row:row['sample']=str(samples/'MAPPA.gray8')
        (out/f'{mask}.json').write_text(json.dumps(rows,indent=2))
        small=copy.deepcopy(rows)
        for row in small:
            row.pop('sample',None);row['size']=[75,49];row['frames']=2
            if 'kwargs' in row:row['kwargs']['bm_range']=3
        (out/f'{mask}-numeric.json').write_text(json.dumps(small,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--samples',type=Path,default=Path('/opt/nss-c4/samples/gray8'))
    a=p.parse_args();write(a.out,a.samples)
