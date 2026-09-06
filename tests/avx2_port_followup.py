#!/usr/bin/env python3
"""Build bounded corroboration/replay queues from completed screens."""
import argparse
import copy
import json
import hashlib
from pathlib import Path

def followup(screen, samples, images):
    rows=json.loads((screen/'summary.json').read_text())
    # Admission after causal replay lives beside the immutable raw summary.
    replay_path=screen.parent/'crossover/replay.json'
    if replay_path.exists():
        replay=json.loads(replay_path.read_text())
        index=replay.get('original_row',-1)
        if replay.get('passed') and 0<=index<len(rows):
            row=rows[index]
            if row['numerical']==replay['original'] and row['environment']['valid'] and row['paired_speedup']>1.02:
                row['selected']=True
                row['numerical_review']='causal crossover and clean-image quality passed'
    reviewed_path=screen/'selection-reviewed.json'
    if reviewed_path.exists():
        reviewed=json.loads(reviewed_path.read_text())
        actual=hashlib.sha256((screen/'summary.json').read_bytes()).hexdigest()
        if reviewed['summary_sha256']!=actual:raise ValueError('reviewed selection does not match raw summary')
        for row in rows:row['selected']=row['config']['name'] in reviewed['selected']
    selected=[r for r in rows if r.get('selected')]
    result=[]
    # Two different real images for one representative accepted target per mask.
    targets=[r for r in selected if r['config'].get('sample')]
    target=next((r for r in targets if r['config'].get('stage')=='two_stage'), targets[0] if targets else None)
    if target is None and selected:
        target=next((r for r in selected if r['config'].get('stage')=='two_stage'), selected[0])
    if target:
        images.mkdir(parents=True,exist_ok=True)
        provenance={}
        for sample in ('8bit.gray8','Ufotable.gray8'):
            row=copy.deepcopy(target['config'])
            w,h=row.get('size',[1920,1080])
            source=samples/sample
            if [w,h]!=[1920,1080]:
                import numpy as np
                raw=source.read_bytes();image=np.frombuffer(raw,dtype=np.uint8).reshape(1080,1920)
                x,y=(1920-w)//2,(1080-h)//2
                destination=images/sample
                destination.write_bytes(image[y:y+h,x:x+w].copy().tobytes())
                provenance[sample]=dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),
                    crop=[x,y,w,h],sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
                source=destination
            row.update(name=row['name']+'_'+sample.removesuffix('.gray8'),sample=str(source))
            result.append(row)
        (images/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('screen',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--samples',type=Path,default=Path('/opt/nss-c4/samples/gray8'))
    a=p.parse_args();a.out.write_text(json.dumps(followup(a.screen,a.samples,a.out.parent/'followup-images'),indent=2))
