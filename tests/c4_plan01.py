#!/usr/bin/env python3
"""Plan01 full-filter paired timing with separate first-pair clean diagnostics."""
from pathlib import Path
import hashlib
import json
import random
import sys

import c4_integration as integration
import c4_paired_bm as paired


def worker(plugin, config):
    result = integration.worker(plugin, config)
    if config.get('_dump'):
        import numpy as np
        from bm_numerics import psnr, ssim
        actual = np.load(config['_dump'])
        width, height = config['size']
        raw = np.fromfile(config['sample'], np.uint8).reshape(1080, 1920)
        base = raw[np.arange(height) * 1080 // height][:, np.arange(width) * 1920 // width].astype(np.float32) / np.float32(255)
        order = list(range(result['timed_first'], result['timed_first'] + result['timed_frames']))
        if config.get('access') == 'random':
            random.Random(42).shuffle(order)
        clean = []
        for frame in order[:len(actual)]:
            plane = base[:, np.clip(np.arange(width) - (frame % 7 - 3) * 2, 0, width-1)] if config.get('motion') else base
            clean.append(np.stack([plane] * actual.shape[1]))
        clean = np.stack(clean)
        error = np.abs(actual.astype(np.float64) - clean)
        result['clean_quality'] = dict(psnr=float(psnr(actual, clean)), ssim=float(ssim(actual, clean)),
                                       max_abs=float(error.max()), rms=float(np.sqrt(np.mean(error*error))),
                                       p99=float(np.quantile(error, .99)), frames=len(actual))
    return result


def configs():
    rows = integration.configs()
    sample = '/opt/nss-c4/samples/gray8/MAPPA.gray8'
    # Make Basic/Final and the inherited NLH fallback controls explicit.
    for stage in ('wiener', 'two_stage'):
        rows.append(dict(name='bm3d_default_'+stage, algorithm='bm3d', stage=stage,
                         kwargs=dict(sigma=3,block_size=8,block_step=8,group_size=8,bm_range=7),
                         size=[1920,1080],sample=sample,frames=2))
    for group, q in ((8,4),(16,2)):
        rows.append(dict(name=f'nlh_g{group}_q{q}_control',algorithm='nlh',
                         kwargs=dict(sigma=3,block_size=8,block_step=8,group_size=group,bm_range=20,q=q),
                         size=[640,360],sample=sample,frames=2))
    for row in rows:
        if row['size'] == [1920, 1080]:
            row['frames'] = 8
    return rows


original_run = paired.run

def audited_run(args):
    paths = [Path(__file__).with_name(name) for name in ('c4_integration.py','c4_paired_bm.py')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    try:
        return original_run(args)
    finally:
        unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
        destination = Path(args.out)
        if destination.exists():
            (destination/'helper-inputs.json').write_text(json.dumps(dict(files=hashes,unchanged=unchanged),indent=2))
        if not unchanged:
            raise RuntimeError('paired worker/helper changed during measurement')

if __name__ == '__main__':
    if sys.argv[1:] == ['configs']:
        print(json.dumps(configs(),indent=2))
    else:
        paired.worker = worker
        paired.run = audited_run
        paired.__file__ = str(Path(__file__).resolve())
        paired.main()
