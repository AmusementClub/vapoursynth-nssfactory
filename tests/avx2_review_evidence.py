#!/usr/bin/env python3
"""Create derived per-configuration selections without changing raw A/B records."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def verified(replay):
    if not replay.get('passed'):return False
    reproduced=replay.get('reproduced_original')
    pixel=replay.get('reproduced_max_pixel')
    if not ((reproduced and all(reproduced.values())) or (pixel and all(p.get('passed') for p in pixel.values()))):return False
    loss=replay.get('loss',{})
    return loss.get('psnr',1) <= .05 and loss.get('ssim',1)<=.001

def review_phase(phase):
    reports=[]
    for p in sorted(phase.glob('**/replay.json')):
        if (p.parent/'INVALID.txt').exists():continue
        r=json.loads(p.read_text())
        if verified(r):reports.append((p,r))
    result={}
    for run in ('bench','bench-remaining','followup'):
        p=phase/run/'summary.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());reviewed=[]
        for index,row in enumerate(rows):
            raw_hashes={}
            for side in ('baseline','candidate'):
                raw=p.parent/f'{index}-{side}.npy'
                if raw.exists():raw_hashes[side]=hashlib.sha256(np.ascontiguousarray(np.load(raw)).tobytes()).hexdigest()
            matching=[]
            for path,r in reports:
                if r.get('original')!=row['numerical'] or len(raw_hashes)!=2:continue
                hashes=r.get('output_hashes',{})
                if not hashes:
                    saved={side:path.parent/f'{side}-using-{side}.npy' for side in ('baseline','candidate')}
                    if all(x.exists() for x in saved.values()):
                        hashes={side:hashlib.sha256(np.ascontiguousarray(np.load(x)).tobytes()).hexdigest() for side,x in saved.items()}
                if hashes==raw_hashes:matching.append(str(path.relative_to(phase)))
            numerical=bool(row['numerical']['passed'] or matching)
            accepted=numerical and row['environment']['valid'] and row['paired_speedup']>1.02
            reviewed.append(dict(name=row['config']['name'],accepted=accepted,numerical_passed=numerical,
                numerical_basis='original_triage' if row['numerical']['passed'] else 'causal_replay' if matching else 'unresolved',
                replay_paths=matching,speedup=row['paired_speedup'],pairs=row['pairs'],ci95=row['ci95'],
                environment_valid=row['environment']['valid']))
        selection=dict(summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),threshold=1.02,
            selected=[r['name'] for r in reviewed if r['accepted']],rows=reviewed,
            policy='Per-configuration median >1.02, valid environment and original numeric band or independently verified causal replay plus clean quality.')
        (p.parent/'selection-reviewed.json').write_text(json.dumps(selection,indent=2));result[run]=selection
    (phase/'reviewed.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',type=Path);a=p.parse_args()
    review_phase(a.phase)
