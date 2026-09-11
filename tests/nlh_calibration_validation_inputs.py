#!/usr/bin/env python3
"""New independent noise seeds on the existing clean calibration images.

This validates noise-realization stability, not unseen-image generalization.
The original training pair and its manifest are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    source=Path(args.fixtures).resolve();out=Path(args.out).resolve();out.mkdir(exist_ok=False)
    original=json.loads((source/'fixtures.json').read_text())
    for role in ('first','second'):
        root=out/role;root.mkdir();cases=[]
        for case in original['cases']:
            target=dict(case);shape=(case['channels'],case['height'],case['width'])
            clean_path=source/case['clean']
            if sha(clean_path)!=case['clean_sha256']:raise ValueError('clean fixture mismatch')
            clean=np.fromfile(clean_path,dtype='<f4').reshape(shape)
            seed=int.from_bytes(hashlib.sha256(('nlh-v4-noise-validation-20260908/'+role+'/'+case['id']).encode()).digest()[:8],'little')
            if seed==case['seed']:raise AssertionError('seed reused')
            noisy=(clean.astype(float)+np.random.default_rng(seed).normal(0,case['sigma']/255,shape)).astype('<f4')
            noisy.tofile(root/case['noisy'])
            if not (root/case['clean']).exists():shutil.copyfile(clean_path,root/case['clean'])
            target.update(seed=seed,noisy_sha256=sha(root/case['noisy']),validation='new noise realization, same clean image')
            cases.append(target)
        (root/'fixtures.json').write_text(json.dumps(dict(cases=cases,source_manifest_sha256=sha(source/'fixtures.json'),scope=__doc__),indent=2)+'\n')
    print('created two independent validation noise realizations for',len(original['cases']),'cases')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fixtures',required=True);parser.add_argument('--out',required=True)
    main(parser.parse_args())
