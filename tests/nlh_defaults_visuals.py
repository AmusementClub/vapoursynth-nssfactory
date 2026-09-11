#!/usr/bin/env python3
"""Render hash-verified clean/noisy/denoised details and shared-scale residuals.

The configuration names frozen experiments and trial labels. Display clipping
does not affect the stored float outputs or the unclipped quality measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nlh_defaults_inputs import file_sha, load_case, save_json, validate_manifest


def display(pixels):
    values = np.clip(pixels, 0, 1)
    return np.repeat(values[0, :, :, None], 3, axis=2) if len(values) == 1 else values.transpose(1, 2, 0)


def detail(pixels, size):
    _, height, width = pixels.shape
    y, x = (height-size)//2, (width-size)//2
    return pixels[:, y:y+size, x:x+size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    inputs, out = Path(config['inputs']), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs/'inputs.json').read_text())
    validate_manifest(manifest)
    cases = {row['id']: row for row in manifest['cases']}
    variants = []
    for item in config['variants']:
        root = Path(item['experiment'])
        evaluation = json.loads((root/'evaluation.json').read_text())
        trial = next(row for row in evaluation['rows'] if row['label'] == item['profile'])
        records = {row['id']: row for line in (root/'observations.jsonl').read_text().splitlines()
                   if (row := json.loads(line))}
        indexed = {records[key].get('base_case', records[key]['case']): records[key] for key in trial['cases']}
        variants.append((item, root, evaluation['metadata'], indexed))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pages = []
    for case_id in config['cases']:
        case = cases[case_id]
        size = variants[0][2]['size']
        clean, noisy, crop = load_case(inputs, manifest, case, size)
        titles, pixels, records = ['Clean', 'Noisy'], [clean, noisy], []
        input_hash = hashlib.sha256(noisy.tobytes()).hexdigest()
        score_size = None
        for item, root, metadata, indexed in variants:
            row = indexed[case_id]
            if metadata['size'] != size or row['input_sha256'] != input_hash or row['crop'] != crop:
                raise ValueError('comparison inputs or crop differ')
            if score_size is None:
                score_size = row['score_size']
            if score_size != row['score_size']:
                raise ValueError('comparison score regions differ')
            with np.load(root/row['output_file']) as stored:
                output = stored['pixels']
            if hashlib.sha256(output.tobytes()).hexdigest() != row['output_sha256']:
                raise ValueError('comparison output hash mismatch')
            titles.append(item['label'] + '\n' + f"{row['quality']['psnr_db']:.3f} dB / SSIM {row['quality']['ssim']:.4f}")
            pixels.append(output); records.append(row)
        zoom = min(config.get('zoom', 128), score_size or min(clean.shape[1:]), min(clean.shape[1:]))
        # The same scale is used for every removed-noise and error panel.
        noise_rms = float(np.sqrt(np.mean(np.square(noisy.astype(float)-clean))))
        residual_scale = 3 * (case['sigma']/255 if case['sigma'] is not None else noise_rms)
        if residual_scale <= 0:
            raise ValueError('residual comparison needs nonzero noise')
        figure, axes = plt.subplots(4, len(pixels), figsize=(3.2*len(pixels), 11.4), layout='constrained')
        for column, (title, output) in enumerate(zip(titles, pixels)):
            regions = [display(output), display(detail(output, zoom)),
                       display(.5 + detail(noisy-output, zoom)/(2*residual_scale)),
                       display(.5 + detail(output-clean, zoom)/(2*residual_scale))]
            for row, region in enumerate(regions):
                axes[row, column].imshow(region, interpolation='nearest')
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_color('#C4CDD5')
            axes[0, column].set_title(title, fontsize=10)
            _, height, width = output.shape
            axes[0, column].add_patch(Rectangle(((width-zoom)//2-.5, (height-zoom)//2-.5),
                                              zoom, zoom, fill=False, edgecolor='#E58A27', linewidth=1.))
        for row, label in enumerate(('Overview', f'Central {zoom} px', 'Removed: noisy - output', 'Error: output - clean')):
            axes[row, 0].set_ylabel(label, fontsize=10)
        noise_label = f'injected sigma {case["sigma"]}' if case['sigma'] is not None else 'CC real pair / blind estimate'
        score_label = f'central {score_size} px' if score_size else 'whole image'
        figure.suptitle(f'{case_id} | {noise_label}\n'
                       f'Quality: {score_label}; residual/error shared range +/- {residual_scale*255:.2f}/255; display clipping only', fontsize=12)
        path = out/(case_id+'.png')
        figure.savefig(path, dpi=150)
        plt.close(figure)
        pages.append(dict(case=case_id, file=path.name, sha256=file_sha(path), crop=crop,
                          score_size=score_size, zoom=zoom, residual_display_scale=residual_scale,
                          input_sha256=input_hash, observations=[r['id'] for r in records],
                          output_sha256=[r['output_sha256'] for r in records]))
    save_json(out/'visuals.json', dict(pages=pages, config_sha256=file_sha(args.config),
              script_sha256=file_sha(__file__), input_manifest_sha256=file_sha(inputs/'inputs.json')))
    print('rendered comparisons:', len(pages))


if __name__ == '__main__':
    main()
