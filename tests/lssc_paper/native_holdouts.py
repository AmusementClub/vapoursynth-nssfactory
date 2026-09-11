#!/usr/bin/env python3
"""Prepare unseen-image/noise-seed fixtures; external research assets stay external."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from run import sha
from scaling import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--side', type=int, default=128)
    parser.add_argument('--sigmas', nargs='+', type=float, default=[25])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20260918, 20260919])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for raw in args.images:
        path = Path(raw)
        name = path.stem.lower()
        if not re.fullmatch(r'[a-z0-9_-]+', name):
            raise ValueError('image basename must be a simple identifier')
        pixels = np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255
        if not 16 <= args.side <= min(pixels.shape):
            raise ValueError('holdout requires a native center crop, not upsampling or tiling')
        y, x = (pixels.shape[0] - args.side) // 2, (pixels.shape[1] - args.side) // 2
        crop = pixels[y:y + args.side, x:x + args.side]
        clean_path = out / (name + '-clean.f32')
        if clean_path.exists():
            raise FileExistsError('duplicate held-out image basename')
        crop.astype('<f4').tofile(clean_path)
        for sigma in args.sigmas:
            for seed in args.seeds:
                case = f'{name}-{args.side}-s{sigma:g}-seed{seed}'
                clean, noisy = fixture(crop, args.side, args.side, sigma, seed)
                np.testing.assert_array_equal(clean, crop.astype('<f4'))
                noisy_path = out / (case + '-noisy.f32')
                if noisy_path.exists():
                    raise FileExistsError('duplicate holdout configuration')
                noisy.tofile(noisy_path)
                cases.append({'id': case, 'image': name, 'width': args.side, 'height': args.side,
                              'sigma': sigma, 'seed': seed, 'clean': clean_path.name, 'noisy': noisy_path.name,
                              'clean_sha256': sha(clean_path), 'noisy_sha256': sha(noisy_path),
                              'original_sha256': sha(path), 'crop': [x, y, args.side, args.side],
                              'fixture_kind': 'heldout_image_native_center_crop_new_noise_seed'})
    record = {'schema': 'nss.paper-fixtures.v1', 'scope': 'unseen-image/new-seed validation, not tuning data', 'cases': cases}
    (out / 'fixtures.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': len(cases), 'manifest_sha256': sha(out / 'fixtures.json')}))


if __name__ == '__main__':
    main()
