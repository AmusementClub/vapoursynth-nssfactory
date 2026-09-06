#!/usr/bin/env python3
"""Numerical migration probe: old generic sigma=3 versus new sigma=4.

The input noise remains 3/255. This is not a same-parameter speed comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from bm_numerics import compare, psnr


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).with_name('c4_integration.py')
    sample = Path('/opt/nss-c4/samples/gray8/MAPPA.gray8')
    clean = np.fromfile(sample, np.uint8).reshape(1, 1, 1080, 1920).astype(np.float32) / np.float32(255)
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.baseline, args.candidate, driver, sample, Path(__file__))}
    (args.out / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for group in (16, 32):
        outputs = {}
        records = {}
        for label, plugin, sigma in (('baseline', args.baseline, 3), ('candidate', args.candidate, 4)):
            target = args.out / f'g{group}-{label}.npy'
            config = dict(name=f'sigma_compensation_g{group}', algorithm='bm3d', size=[1920, 1080],
                          sample=str(sample), frames=1, stage='two_stage', _dump=str(target),
                          kwargs=dict(sigma=sigma, block_size=8, block_step=8, group_size=group, bm_range=7))
            command = ['taskset', '-c', '0', sys.executable, str(driver), 'worker',
                       '--plugin', str(plugin), '--config', json.dumps(config)]
            result = json.loads(subprocess.check_output(command, text=True, timeout=60))
            records[label] = dict(config=config, worker=result)
            outputs[label] = np.load(target)
        row = dict(group=group, baseline_sigma=3, candidate_sigma=4, noise_sigma=3,
                   baseline_psnr_clean=psnr(outputs['baseline'], clean),
                   candidate_psnr_clean=psnr(outputs['candidate'], clean),
                   comparison=compare(outputs['baseline'], outputs['candidate']), records=records)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.out / 'results.json').write_text(json.dumps(rows, indent=2))
    for path, digest in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
