#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Freeze real DIV2K crops and deterministic AWGN for the experiment."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
g48 = dict(block_size=4, block_step=4, group_size=48, iters=4, admm_iter=5)
g24 = dict(block_size=4, block_step=4, group_size=24, iters=6, admm_iter=5)
recipes = [
    ('rgb-g48-0874', '0874', 256, 3, 25, g48),
    ('rgb-g48-0859', '0859', 256, 3, 25, g48),
    ('gray-g48', '0874', 256, 1, 3, g48),
    ('rgb-unequal-g48', '0874', 128, 3, [10,25,40], g48),
    ('rgb-g24', '0859', 128, 3, 25, g24),
    ('rgb-default25', '0874', 40, 3, 25, {}),
    ('rgb-default3', '0859', 40, 3, 3, {}),
    ('rgb-default75', '0874', 24, 3, 75, {}),
    ('gray-default3', '0859', 64, 1, 3, {}),
]
cases=[]
for index,(name,image,size,channels,sigma,kw) in enumerate(recipes):
    original=source/f'div2k-{image}-rgb8.npz'
    rgb=np.load(original)['rgb8'].astype(np.float32)/np.float32(255)
    y=(rgb.shape[1]-size)//2;x=(rgb.shape[2]-size)//2
    clean=rgb[:,y:y+size,x:x+size].copy()
    if channels==1:clean=(clean*np.array([.2126,.7152,.0722],np.float32)[:,None,None]).sum(axis=0,keepdims=True)
    noise=np.random.default_rng(190919+index).standard_normal(clean.shape).astype(np.float32)
    sigmas=np.broadcast_to(np.asarray(sigma,np.float32), (channels,))
    noisy=clean+noise*(sigmas/255)[:,None,None]
    np.save(out/f'{name}-clean.npy',clean)
    noisy.astype('<f4').tofile(out/f'{name}.f32')
    cases.append(dict(name=name,algorithm='twsc',size=[size,size],channels=channels,
        sample_format='planar-f32',sample=f'{name}.f32',clean=f'{name}-clean.npy',
        kwargs=dict(sigma=sigma,**kw),warmup=1,frames=1,_cpu_environment=True,
        _frame_times=True,_resource_props=True,input_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
        original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),seed=190919+index))
(out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
