#!/usr/bin/env python3
"""Freeze clean/noisy image and synthetic-motion fixtures for native gates.

PNG samples are interpreted as encoded RGB test values, without an inferred
linear-light or camera-noise model. Motion and scene cuts are synthetic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--samples', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
arrays, fixtures, source_hashes = {}, [], {}
w, h, frames = 192, 128, 5
images = []
for path in sorted(args.samples.glob('*.png')):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    rgb = rgb[np.arange(h) * rgb.shape[0] // h][:, np.arange(w) * rgb.shape[1] // w]
    images.append((path.stem, np.moveaxis(rgb.astype(np.float32) / np.float32(255), -1, 0)))
    source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
yy, xx = np.mgrid[:h, :w]
synthetics = {
    'constant': np.full((h, w), .35, dtype=np.float32),
    'edges': (.15 + .45 * (xx > 73) + .15 * (yy > 47)).astype(np.float32),
    'texture': (.45 + .18 * np.sin(xx / 3.) * np.cos(yy / 4.) + .08 * ((xx // 3 + yy // 5) % 2)).astype(np.float32),
}
images += [(name, np.stack([a, a, a])) for name, a in synthetics.items()]
if not images: raise RuntimeError('no fixtures')
for index, (name, rgb) in enumerate(images):
    for mode in ('static', 'motion', 'cut') if index == 0 else ('static',):
        clean_rgb = np.stack([rgb[:, :, np.clip(np.arange(w) - (n - 2) * 2, 0, w - 1)]
                              if mode == 'motion' else (1 - rgb if mode == 'cut' and n >= 2 else rgb)
                              for n in range(frames)]).astype(np.float32)
        clean_gray = np.sum(clean_rgb * np.asarray([.2126, .7152, .0722], np.float32)[None, :, None, None], axis=1, keepdims=True)
        fixture = f'f{len(fixtures)}'
        arrays[fixture + '_clean_rgb'] = clean_rgb
        arrays[fixture + '_clean_gray'] = clean_gray
        sigmas = (0.1, 1., 3., 10., 25., 50.) if name == 'edges' else (3., 25.)
        for sigma in sigmas:
            for color, clean in (('rgb', clean_rgb), ('gray', clean_gray)):
                noise = np.random.RandomState(42000 + index * 101).standard_normal(clean.shape).astype(np.float32)
                arrays[f'{fixture}_s{sigma:g}_{color}'] = clean + noise * np.float32(sigma / 255.)
        fixtures.append(dict(id=fixture, name=name, mode=mode, sigmas=sigmas, size=[w, h], frames=frames))
pack = args.out / 'fixtures.npz'; np.savez_compressed(pack, **arrays)
cases = []
for fixture in fixtures:
    for sigma in fixture['sigmas']:
        algorithms = ('BM3D',) if sigma not in (3., 25.) else ('NLM', 'BM3D', 'WNNM', 'MCWNNM', 'TWSC', 'NCSR', 'NLH', 'LSSC')
        for algorithm in algorithms:
            color = 'rgb' if algorithm in ('MCWNNM', 'TWSC') else 'gray'
            radius = 1 if fixture['mode'] != 'static' and algorithm != 'LSSC' else 0
            cases.append(dict(fixture=fixture['id'], name=fixture['name'], mode=fixture['mode'], algorithm=algorithm,
                               sigma=sigma, radius=radius, color=color, stage='two-stage' if algorithm == 'BM3D' else 'default'))
manifest = dict(schema='nssfactory.arm-quality-pack.v1', data_sha256=hashlib.sha256(pack.read_bytes()).hexdigest(),
                sources=source_hashes, numpy=np.__version__, pillow=pillow_version,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), fixtures=fixtures, cases=cases,
                interpretation='Encoded RGB/luma test intensities; additive frozen FP32 Gaussian noise; synthetic translations and cuts.')
(args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(fixtures=len(fixtures), cases=len(cases), bytes=pack.stat().st_size, sha256=manifest['data_sha256'])))
