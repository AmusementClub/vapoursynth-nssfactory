#!/usr/bin/env python3
"""Twelve validation-only unequal-channel RGB AWGN cases; unchanged campaign worker.

Reuse the compact clean-pack already on the guest. Optional expected manifests
make every regenerated clean/noisy byte identity match the locally frozen cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

SCENES=('div2k-0874','div2k-0898','div2k-0875')
PROFILES={'rgb-5-25-50':[5.,25.,50.], 'rgb-50-25-5':[50.,25.,5.]}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def noisy_input(clean,channel_sigma,seed):
    vector=np.asarray(channel_sigma,dtype=np.float64)
    if vector.shape!=(3,) or not np.isfinite(vector).all() or (vector<=0).any():
        raise ValueError('three positive finite RGB sigma values required')
    noise=np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape)
    return (clean.astype(np.float64)+noise*(vector[:,None,None]/255)).astype('<f4')


def prepare(args):
    root=Path(args.pack);transfer=json.loads((root/'clean-pack.json').read_text())
    template_info=transfer['templates']['256']
    if sha(root/template_info['file'])!=template_info['sha256']:raise ValueError('source template hash mismatch')
    template=json.loads((root/template_info['file']).read_text())
    original={r['image']:r for r in template['cases'] if r['split']=='validation' and r['format']=='rgb'}
    recipes=json.loads(Path(args.recipes).read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for profile,vector in PROFILES.items():
        destination=out/profile;destination.mkdir()
        shutil.copyfile(root/'nlh-v4.json',destination/'nlh-v4.json')
        selected={}
        for algorithm in ('MCWNNM','TWSC','NLH'):
            selected[algorithm]=[]
            for recipe in recipes['algorithms'][algorithm]:
                entry=json.loads(json.dumps(recipe));entry['parameters']['sigma']=vector
                selected[algorithm].append(entry)
        save(destination/'candidates.json',dict(schema='nss.balanced-candidates.v1',algorithms=selected,
            channel_sigma=vector,noise_profile=profile,source_recipes_sha256=sha(args.recipes),
            qualification='Unequal-channel validation diagnostic; no homogeneous sigma25 pooling or default admission'))
        cases=[]
        expected={}
        if args.expected:
            expected={r['id']:r for r in json.loads((Path(args.expected)/profile/'fixtures.json').read_text())['cases']}
        for scene in SCENES:
            asset=transfer['assets'][scene]
            if asset['split']!='validation':raise ValueError('sealed or search source forbidden')
            if sha(root/asset['file'])!=asset['sha256']:raise ValueError('clean asset hash mismatch')
            with np.load(root/asset['file']) as data:
                large=(data['rgb8'].astype(np.float64)/255).astype('<f4')
            maximum=transfer['maximum_size'];offset=(maximum-256)//2
            clean=np.ascontiguousarray(large[:,offset:offset+256,offset:offset+256])
            clean_path=destination/(scene+'-rgb-clean.f32');clean.tofile(clean_path)
            if sha(clean_path)!=original[scene]['clean_sha256']:raise ValueError('original clean crop hash mismatch')
            for repeat in (0,1):
                label=f'balanced-unequal-20260919:{scene}:{profile}:{repeat}'
                seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')
                noisy=noisy_input(clean,vector,seed)
                name=f'{scene}-unequal-{profile}-256-n{repeat}'
                noisy_path=destination/(name+'-noisy.f32');noisy.tofile(noisy_path)
                row=dict(id=name,image=scene,split='validation',format='rgb',channels=3,width=256,height=256,
                    sigma=25.,sigma_role='runner selection tag only; actual per-channel noise is channel_sigma',
                    channel_sigma=vector,noise_profile=profile,noise_repeat=repeat,seed=seed,
                    clean=clean_path.name,noisy=noisy_path.name,clean_sha256=sha(clean_path),noisy_sha256=sha(noisy_path),
                    original_sha256=original[scene]['original_sha256'],crop=original[scene]['crop'],
                    historically_used_by_nlh=True,input_color_order='RGB')
                if args.expected:
                    previous=expected.get(name)
                    if previous is None or any(previous[k]!=row[k] for k in ('seed','channel_sigma','clean_sha256','noisy_sha256')):
                        raise ValueError('regenerated unequal fixture differs from frozen manifest: '+name)
                cases.append(row)
        save(destination/'fixtures.json',dict(schema='nss.balanced-fixtures.v1',cases=cases,
            scene_split={s:'validation' for s in SCENES},noise_profile=profile,channel_sigma=vector,
            do_not_pool_with_homogeneous_noise=True,source_template_sha256=template_info['sha256'],
            input_policy='Native256 genuine RGB clean crop; independent channel PCG64 Gaussian noise; specified RGB sigma/255; unclipped float32',
            runner_policy='Every recipe explicitly supplies sigma=channel_sigma, overriding case sigma25 selection tag',
            scope='Three validation scenes x two opposite noise vectors x two seeds; no sealed access'))
    save(out/'preparation.json',dict(schema='nss.unequal-inputs.v1',profiles=PROFILES,scenes=list(SCENES),cases=12,
        source_clean_pack_sha256=sha(root/'clean-pack.json'),source_recipes_sha256=sha(args.recipes),
        preparer_sha256=sha(__file__),local_sources_only=True,verified_against_expected=bool(args.expected)))
    print(json.dumps(dict(out=str(out),cases=12,verified_against_expected=bool(args.expected))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',required=True);p.add_argument('--recipes',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected')
    prepare(p.parse_args())


if __name__=='__main__':main()
