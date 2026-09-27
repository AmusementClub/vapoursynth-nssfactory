#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Compact clean-only transfer and exact-hash reconstruction of frozen fixtures.

Needs NumPy only. Encoded RGB8 reconstructs the frozen float32 clean image;
Gray and PCG64 AWGN are reproduced and checked against pre-existing hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pack(args):
    root=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    maximum=max(int(p.name.removeprefix('fixtures')) for p in root.glob('fixtures[0-9]*') if p.is_dir())
    source=root/f'fixtures{maximum}'
    manifest=json.loads((source/'fixtures.json').read_text())
    cases=[r for r in manifest['cases'] if r['split'] in args.splits]
    scenes={r['image']:r for r in cases if r['channels']==3}
    assets={}
    for scene,row in scenes.items():
        clean=np.fromfile(source/row['clean'],dtype='<f4').reshape(3,maximum,maximum)
        rgb8=np.rint(clean*255).astype(np.uint8)
        if not np.array_equal((rgb8.astype(np.float64)/255).astype('<f4'),clean):
            raise ValueError('clean cannot be represented by original RGB8')
        target=out/(scene+'-rgb8.npz');np.savez_compressed(target,rgb8=rgb8)
        assets[scene]=dict(file=target.name,sha256=sha(target),split=row['split'])
    templates={}
    for directory in root.glob('fixtures[0-9]*'):
        if not directory.is_dir():continue
        size=int(directory.name.removeprefix('fixtures'))
        target=out/f'fixtures{size}.json';shutil.copyfile(directory/'fixtures.json',target)
        templates[str(size)]=dict(file=target.name,sha256=sha(target))
    recipes={}
    for file in ('candidates.json','nlh-v4.json'):
        shutil.copyfile(source/file,out/file);recipes[file]=sha(out/file)
    save(out/'clean-pack.json',dict(schema='nss.balanced-clean-transfer.v1',maximum_size=maximum,
        assets=assets,templates=templates,recipes_sha256=recipes,splits=args.splits,
        scope='Clean crops and previously frozen hashes only; no noisy arrays or denoiser output transferred'))
    print(json.dumps(dict(pack=str(out),scenes=len(assets),bytes=sum(p.stat().st_size for p in out.iterdir()))))


def materialize(args):
    root=Path(args.pack);transfer=json.loads((root/'clean-pack.json').read_text())
    template=transfer['templates'][str(args.size)]
    if sha(root/template['file'])!=template['sha256']:raise ValueError('template hash mismatch')
    manifest=json.loads((root/template['file']).read_text())
    cases=[r for r in manifest['cases'] if r['split']==args.split and r['format'] in args.formats and
        (not args.sigmas or r['sigma'] in args.sigmas) and
        (args.noise_repeat is None or r['noise_repeat']==args.noise_repeat) and
        (not args.scenes or r['image'] in args.scenes)]
    if not cases:raise ValueError('no cases selected')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for file,expected in transfer['recipes_sha256'].items():
        if sha(root/file)!=expected:raise ValueError('recipe hash mismatch')
        shutil.copyfile(root/file,out/file)
    maximum=transfer['maximum_size'];offset=(maximum-args.size)//2
    overrides_path=Path(args.gray_overrides) if args.gray_overrides else None
    overrides=json.loads(overrides_path.read_text()) if overrides_path else None
    if overrides and (overrides['size']!=args.size or overrides['split']!=args.split or
                      overrides['fixture_template_sha256']!=template['sha256']):
        raise ValueError('exact Gray override does not match size/split/frozen template')
    cache={};written=set();gray_cache={}
    for row in cases:
        scene=row['image'];fmt=row['format']
        if fmt=='gray' and overrides:
            if scene not in gray_cache:
                asset=overrides['assets'][scene];path=overrides_path.parent/asset['file']
                if sha(path)!=asset['sha256']:raise ValueError('exact Gray transfer hash mismatch')
                gray_cache[scene]=np.fromfile(path,dtype='<f4').reshape(1,args.size,args.size)
            clean=np.ascontiguousarray(gray_cache[scene])
        else:
            if scene not in cache:
                asset=transfer['assets'][scene];path=root/asset['file']
                if sha(path)!=asset['sha256']:raise ValueError('clean transfer hash mismatch')
                with np.load(path) as data:rgb=(data['rgb8'].astype(np.float64)/255).astype('<f4')
                if rgb.shape!=(3,maximum,maximum):raise ValueError('unexpected clean transfer shape')
                # Preserve the original layout for platforms where this path
                # reproduces it. Cross-platform ties require exact Gray assets.
                rgb=np.ascontiguousarray(rgb.transpose(1,2,0)).transpose(2,0,1)
                gray=np.einsum('c,chw->hw',np.array([.299,.587,.114]),rgb.astype(np.float64))[None].astype('<f4')
                cache[scene]={'rgb':rgb,'gray':gray}
            clean_large=cache[scene][fmt]
            clean=np.ascontiguousarray(clean_large[:,offset:offset+args.size,offset:offset+args.size])
        if hashlib.sha256(clean.tobytes()).hexdigest()!=row['clean_sha256']:
            raise ValueError('reconstructed clean hash mismatch: '+row['id'])
        if row['clean'] not in written:clean.tofile(out/row['clean']);written.add(row['clean'])
        # Draw the original full512 grid, then crop. Crop-before-add uses the
        # same independent arithmetic at each retained pixel and avoids needing
        # an approximate reconstruction of unneeded Gray pixels outside256.
        noise_large=np.random.Generator(np.random.PCG64(row['seed'])).standard_normal(
            (row['channels'],maximum,maximum))
        noise=noise_large[:,offset:offset+args.size,offset:offset+args.size]
        noisy=np.ascontiguousarray((clean.astype(np.float64)+noise*(row['sigma']/255)).astype('<f4'))
        if hashlib.sha256(noisy.tobytes()).hexdigest()!=row['noisy_sha256']:
            raise ValueError('reconstructed noise hash mismatch; RNG/platform differs: '+row['id'])
        noisy.tofile(out/row['noisy'])
    manifest['cases']=cases
    manifest['parent_full_fixture_manifest_sha256']=template['sha256']
    manifest['materialization']=dict(clean_transfer_sha256=sha(root/'clean-pack.json'),
        materializer_sha256=sha(__file__),numpy_version=np.__version__,
        exact_gray_override_sha256=sha(overrides_path) if overrides_path else None,
        exact_all_clean_and_noisy_hashes_verified=True,selected_split=args.split)
    save(out/'fixtures.json',manifest)
    print(json.dumps(dict(out=str(out),cases=len(cases),exact_hashes_verified=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('pack');s.add_argument('--source',required=True);s.add_argument('--out',required=True)
    s.add_argument('--splits',nargs='+',choices=['validation','sealed'],default=['validation','sealed'])
    s=sub.add_parser('materialize');s.add_argument('--pack',required=True);s.add_argument('--out',required=True)
    s.add_argument('--size',type=int,choices=[256,512],required=True)
    s.add_argument('--split',choices=['validation','sealed'],required=True)
    s.add_argument('--sigmas',nargs='+',type=float);s.add_argument('--noise-repeat',type=int)
    s.add_argument('--gray-overrides',help='Manifest of exact frozen Gray float32 crops; never approximate')
    s.add_argument('--scenes',nargs='+')
    s.add_argument('--formats',nargs='+',choices=['gray','rgb'],default=['gray','rgb'])
    args=p.parse_args();globals()[args.mode](args)


if __name__=='__main__':main()
