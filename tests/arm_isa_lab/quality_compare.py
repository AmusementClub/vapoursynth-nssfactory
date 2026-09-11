#!/usr/bin/env python3
"""Apply the existing clean-quality guard; stage admission remains separate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('left', type=Path); parser.add_argument('right', type=Path); parser.add_argument('out', type=Path)
parser.add_argument('--algorithms', nargs='+')
args = parser.parse_args()
metadata = [json.loads((p / 'summary.json').read_text()) for p in (args.left, args.right)]
if not all(d['passed'] for d in metadata) or metadata[0]['fixture_sha256'] != metadata[1]['fixture_sha256']:
    raise RuntimeError('incomplete captures or fixture mismatch')
arrays = []
for p, d in zip((args.left, args.right), metadata):
    path = p / 'pixels.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != d['pixels_sha256']: raise RuntimeError('pixel hash mismatch')
    arrays.append(np.load(path))
rows = [{r['index']:r for r in d['cases'] if not args.algorithms or r['parameters']['algorithm'] in args.algorithms} for d in metadata]
if rows[0].keys() != rows[1].keys() or not rows[0]: raise RuntimeError('case sets differ or empty')
result = dict(schema='nssfactory.arm-quality-guard.v1', passed=False, numerical_admission=False,
              quality_policy=dict(max_psnr_loss_db=.05, max_ssim_loss=.001),
              fixture_sha256=metadata[0]['fixture_sha256'], plugins=[d['plugin_sha256'] for d in metadata], cases=[])
for index in rows[0]:
    left, right = [r[index] for r in rows]
    if left['parameters'] != right['parameters'] or left['kwargs'] != right['kwargs']: raise RuntimeError('parameters differ')
    a, b = [d[f'c{index}'] for d in arrays]
    if any(hashlib.sha256(v.tobytes()).hexdigest() != r['output_sha256'] for v, r in ((a,left),(b,right))):
        raise RuntimeError('case pixel hash mismatch')
    losses = {metric:left['metrics'][metric] - right['metrics'][metric] for metric in left['metrics']}
    passed = losses['psnr_against_clean'] <= .05 and losses['ssim_against_clean'] <= .001
    result['cases'].append(dict(index=index, parameters=left['parameters'], quality_passed=passed,
                                clean_quality_loss=losses, baseline_quality=left['metrics'], candidate_quality=right['metrics'],
                                pixel_comparison=compare(a,b)))
result['passed'] = all(r['quality_passed'] for r in result['cases'])
result['maximum_psnr_loss_db'] = max(r['clean_quality_loss']['psnr_against_clean'] for r in result['cases'])
result['maximum_ssim_loss'] = max(r['clean_quality_loss']['ssim_against_clean'] for r in result['cases'])
args.out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))
