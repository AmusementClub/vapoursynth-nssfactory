#!/usr/bin/env python3
"""Scientific comparisons of saved numerical outputs, with provenance checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from run import metrics, preview, read_image, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--scaling-root', required=True)
    parser.add_argument('--author-report', required=True)
    parser.add_argument('--author-output', required=True)
    args = parser.parse_args()
    root, scaling = Path(args.root), Path(args.scaling_root)
    out = root / 'visuals'
    out.mkdir(exist_ok=False)
    record = json.loads((root / 'pressure512/result.json').read_text())
    author = json.loads(Path(args.author_report).read_text())
    data = scaling / 'native-size/512x512-s25'
    if record['input_sha256'] != author['input_sha256'] or sha(data / 'noisy.f32') != record['input_sha256']:
        raise ValueError('author/native noisy inputs differ')
    full_row = next(row for row in author['rows'] if row['variant'] == 'full' and row['pair'] >= 0)
    if sha(args.author_output) != full_row['output_sha256']:
        raise ValueError('author output hash mismatch')
    clean = read_image(data / 'clean.f32', 512, 512)
    noisy = read_image(data / 'noisy.f32', 512, 512)
    full = np.fromfile(args.author_output, dtype='<f8').reshape(512, 512)
    for key, value in metrics(clean, full).items():
        np.testing.assert_allclose(value, full_row['metrics'][key], atol=1e-11, rtol=1e-12)
    panels = [('Clean', clean), ('Noisy', noisy)]
    for name, label in [('mixed_sme', 'Fixed-D mixed'), ('match2', 'Wider groups'),
                         ('learn1_match2', 'Learn + wider groups')]:
        item = record['summary'][name]
        pixels = np.load(root / 'pressure512' / (name + '-output.npy'), allow_pickle=False)
        panels.append((f"{label}\n{item['metrics']['psnr_db']:.2f} dB / {item['median_seconds']:.2f} s", pixels))
    panels.append((f"Author full (prior)\n{full_row['metrics']['psnr_db']:.2f} dB / timing N/A", full))
    preview(out / 'barbara512-quality.png', panels)
    preview(out / 'barbara512-texture.png', [(name, pixels[256:384, 256:384]) for name, pixels in panels])

    record_hd = json.loads((root / 'pressure1080/result.json').read_text())
    data_hd = scaling / 'native-1080p/1920x1080-s25'
    if sha(data_hd / 'noisy.f32') != record_hd['input_sha256']:
        raise ValueError('HD input mismatch')
    clean_hd = read_image(data_hd / 'clean.f32', 1920, 1080)
    noisy_hd = read_image(data_hd / 'noisy.f32', 1920, 1080)
    hd_panels = [('Clean', clean_hd), ('Noisy', noisy_hd)]
    for name, label in [('mixed_sme', 'Fixed-D mixed'), ('match2', 'Wider groups'),
                         ('learn1_match2', 'Learn + wider groups')]:
        item = record_hd['summary'][name]
        pixels = np.load(root / 'pressure1080' / (name + '-output.npy'), allow_pickle=False)
        hd_panels.append((f"{label}\n{item['metrics']['psnr_db']:.2f} dB / {item['median_seconds']:.2f} s", pixels))
    for name, x, y in [('mappa-text', 288, 440), ('mappa-flat', 960, 260)]:
        preview(out / (name + '.png'), [(label, pixels[y:y + 128, x:x + 128]) for label, pixels in hd_panels])
    report = {'schema': 'nss.native-visuals.v1', 'display_clipping_only': True,
              'author_input_matches_native': True, 'author_output_sha256': sha(args.author_output),
              'author_timing_not_compared': True, 'blind_visual_acceptance': False,
              'crops_xywh': {'barbara512-texture': [256, 256, 128, 128],
                            'mappa-text': [288, 440, 128, 128], 'mappa-flat': [960, 260, 128, 128]},
              'source_sha256': sha(__file__),
              'images': {path.name: sha(path) for path in sorted(out.glob('*.png'))}}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
