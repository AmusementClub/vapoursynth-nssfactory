#!/usr/bin/env python3
"""Prepare native256/512 held-out campaign crops from locally archived DIV2K originals.

Preserves the historical NLH selection/test scene split, adds new noise, and
never opens denoiser outputs. Historical reuse is explicitly recorded; the known
0830 low-noise regression is a labeled sentinel, not a fresh unseen sample.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from paper_compare import save_json, sha


def prepare(args):
    source = Path(args.source)
    parent_path = source/'inputs.json'
    parent = json.loads(parent_path.read_text())
    selected = []
    seen = {}
    for old_split, split in [('selection','validation'), ('test','sealed')]:
        scenes = [r for r in parent['images'] if r['dataset']=='DIV2K' and r['split']==old_split and not r.get('control')]
        scenes = scenes[:args.per_split]
        for row in scenes:
            if row['clean_sha256'] in seen: raise ValueError('duplicate photograph identity across splits')
            seen[row['clean_sha256']] = split
            selected.append((row, split))
    if not selected: raise ValueError('no images selected')
    maximum = max(args.sizes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    all_cases = {size: [] for size in args.sizes}
    for size in args.sizes:
        directory = out/f'fixtures{size}'; directory.mkdir()
        for file in ('candidates.json','nlh-v4.json'):
            shutil.copyfile(Path(args.recipes)/file, directory/file)
    for row, split in selected:
        path = source/row['clean']
        if sha(path) != row['clean_sha256']: raise ValueError('source image hash mismatch')
        with Image.open(path) as pic:
            full = np.asarray(pic.convert('RGB'), dtype=np.float64).transpose(2,0,1)/255
        _, h, w = full.shape
        if min(h,w)<maximum: raise ValueError('native crop larger than source: '+row['id'])
        top,left=(h-maximum)//2,(w-maximum)//2
        rgb = full[:,top:top+maximum,left:left+maximum].astype('<f4')
        del full
        gray = np.einsum('c,chw->hw',np.array([.299,.587,.114]),rgb.astype(np.float64))[None].astype('<f4')
        for fmt, clean_large in [('rgb',rgb),('gray',gray)]:
            paths = {}
            for size in args.sizes:
                offset=(maximum-size)//2
                clean=np.ascontiguousarray(clean_large[:,offset:offset+size,offset:offset+size])
                directory=out/f'fixtures{size}'
                clean_path=directory/f"{row['id']}-{fmt}-clean.f32"; clean.tofile(clean_path)
                paths[size]=(clean_path,sha(clean_path))
            for sigma in args.sigmas:
                for repeat in range(args.seeds):
                    label=f"balanced-validation-20260919:{row['id']}:{fmt}:{sigma:g}:{repeat}"
                    seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')
                    noise=np.random.Generator(np.random.PCG64(seed)).standard_normal(clean_large.shape)
                    noisy_large=(clean_large.astype(np.float64)+noise*(sigma/255)).astype('<f4')
                    for size in args.sizes:
                        offset=(maximum-size)//2
                        noisy=np.ascontiguousarray(noisy_large[:,offset:offset+size,offset:offset+size])
                        directory=out/f'fixtures{size}'
                        name=f"{row['id']}-{fmt}-{size}-s{sigma:g}-n{repeat}"
                        noisy_path=directory/(name+'-noisy.f32');noisy.tofile(noisy_path)
                        clean_path,clean_hash=paths[size]
                        all_cases[size].append(dict(id=name,image=row['id'],split=split,
                            channels=len(clean_large),width=size,height=size,format=fmt,
                            sigma=sigma,noise_repeat=repeat,seed=seed,clean=clean_path.name,
                            noisy=noisy_path.name,clean_sha256=clean_hash,noisy_sha256=sha(noisy_path),
                            original_sha256=row['clean_sha256'],original_size=[w,h],
                            crop=[left+offset,top+offset,size,size],historical_split=row['split'],
                            historically_used_by_nlh=True,known_low_noise_sentinel=row['id']=='div2k-0830'))
    for size,cases in all_cases.items():
        save_json(out/f'fixtures{size}'/'fixtures.json',dict(schema='nss.balanced-fixtures.v1',
            cases=cases,parent_manifest_sha256=sha(parent_path),preparation_sha256=sha(__file__),
            scene_split={row['id']:split for row,split in selected},
            historically_used_scenes=True,
            scope='new scenes to current balanced campaign; historically used NLH selection/test images',
            input_policy=f'Native center crops; RGB8/255 and BT.601 Gray; independent new channel AWGN generated on {maximum}px crop before nested crop; unclipped float32',
            split_policy='historical selection→validation, historical test→sealed; no old3 controls; all formats/noise seeds/crop sizes of each scene stay in its split',
            unsupported=['native YUV','moving clips','real noise','unequal-channel noise','blind estimation']))
    save_json(out/'preparation.json',dict(source_manifest_sha256=sha(parent_path),script_sha256=sha(__file__),
        sizes=args.sizes,sigmas=args.sigmas,seeds=args.seeds,scenes=len(selected),
        cases_by_size={str(k):len(v) for k,v in all_cases.items()},
        local_sources_only=True,denoiser_outputs_consulted=False,
        originals=[dict(id=row['id'],sha256=row['clean_sha256'],split=split) for row,split in selected]))
    print(json.dumps(dict(out=str(out),scenes=len(selected),cases_by_size={k:len(v) for k,v in all_cases.items()})))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='artifacts/nlh-defaults-opt-20260909/inputs')
    p.add_argument('--recipes',default='artifacts/balanced-20260919/fixtures128')
    p.add_argument('--out',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[256,512])
    p.add_argument('--sigmas',nargs='+',type=float,default=[5,10,25,50,75])
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--per-split',type=int,default=6)
    args=p.parse_args()
    if min(args.sizes)<16 or len(set(args.sizes))!=len(args.sizes):p.error('sizes must be unique and at least16')
    if min(args.sigmas)<=0 or args.seeds<1 or not 1<=args.per_split<=6:p.error('invalid noise/split configuration')
    prepare(args)


if __name__=='__main__':main()
