#!/usr/bin/env python3
"""Bounded cross-algorithm selection matrix, with an explicit per-case policy."""
import json
from pathlib import Path
import sys


def bm(name, group=8, block=8, size=(1920,1080), sample='MAPPA.gray8', **kwargs):
    kw=dict(sigma=3, block_size=block, group_size=group, block_step=min(8,block), bm_range=7, radius=0)
    kw.update(kwargs)
    row=dict(name=name, frames=2, warmup=1, size=list(size), stage='two_stage', kwargs=kw)
    if size==(1920,1080):
        row['sample']='/opt/nss-c4/samples/gray8/'+sample
    return row


def configurations():
    shared=[dict(name='shared_'+a, algorithm=a, size=[640,360], frames=1, warmup=1)
            for a in ('wnnm','twsc','ncsr','lssc','nlh','mcwnnm','nlm')]
    for row in shared:
        if row['algorithm'] in ('lssc','nlm'):
            continue
        row['kwargs']=dict(sigma=3,block_size=8,block_step=8,group_size=32,bm_range=7,radius=0)
        if row['algorithm']=='nlh':
            row['kwargs'].update(group_size=16,radius=1)
            row['size']=[320,180]
        if row['algorithm']=='mcwnnm':
            row['size']=[320,180]
        if row['algorithm'] in ('twsc','ncsr','mcwnnm'):
            row['kwargs']['iters']=2
        row['name'] += '_temporal_g16' if row['algorithm']=='nlh' else '_g32'
    topk=[bm('bm_b8_g16',16),bm('bm_b8_g32',32),bm('bm_b8_g8'),
          bm('bm_b4_g32',32,4,size=(320,180)),bm('bm_b16_g8',8,16,size=(320,180)),
          bm('bm_temporal_g16',16,size=(640,360),radius=1)]+shared
    buffers=[bm('bm_g16',16),bm('bm_g32',32),bm('bm_g8'),
             bm('bm_temporal',16,size=(640,360),radius=1)]+shared
    ring=[]
    for radius,chunk,size in [(1,8,(1920,1080)),(1,2,(640,360)),(4,8,(320,180))]:
        for access in ('sequential','random'):
            row=bm(f'rolling_r{radius}_chunk{chunk}_{access}',size=size,radius=radius,
                   temporal_mode='rolling',rolling_chunk=chunk,rolling_cache_limit=1)
            row['access']=access
            ring.append(row)
    return dict(topk=topk,buffers=buffers,ring=ring)


if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    for name, rows in configurations().items():
        (out/(name+'.json')).write_text(json.dumps(rows,indent=2))
