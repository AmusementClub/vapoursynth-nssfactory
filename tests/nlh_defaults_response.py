#!/usr/bin/env python3
"""Measure paired-seed output response without changing injected sigma.

The ratio is ||output(seed0)-output(seed1)|| / ||input(seed0)-input(seed1)||.
BM3D uses the same noisy inputs and explicit sigma. This is a descriptive noise
response, never a gain threshold or a substitute for image quality metrics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from nlh_defaults_inputs import file_sha, load_case, save_json, validate_manifest


def response(noisy0, noisy1, output0, output1, score_size=0):
    if not (noisy0.shape == noisy1.shape == output0.shape == output1.shape):
        raise ValueError('paired response shape mismatch')
    if score_size:
        _, h, w = noisy0.shape
        if score_size > min(h, w):
            raise ValueError('paired score crop exceeds input')
        y, x = (h-score_size)//2, (w-score_size)//2
        crop = (slice(None), slice(y, y+score_size), slice(x, x+score_size))
        noisy0, noisy1, output0, output1 = [a[crop] for a in (noisy0, noisy1, output0, output1)]
    incoming = np.subtract(noisy0, noisy1, dtype=np.float64)
    outgoing = np.subtract(output0, output1, dtype=np.float64)
    input_energy = float(np.sum(incoming*incoming))
    if not input_energy:
        raise ValueError('paired response needs distinct noise realizations')
    output_energy = float(np.sum(outgoing*outgoing))
    return dict(ratio=(output_energy/input_energy)**.5,
                input_change_rmse=(input_energy/incoming.size)**.5,
                output_change_rmse=(output_energy/outgoing.size)**.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, inputs, out = Path(args.experiment), Path(args.inputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs/'inputs.json').read_text())
    validate_manifest(manifest)
    cases = {c['id']: c for c in manifest['cases']}
    evaluation = json.loads((root/'evaluation.json').read_text())
    if not evaluation.get('complete', True):
        raise ValueError('paired response requires a complete frozen evaluation')
    observations = {r['id']: r for line in (root/'observations.jsonl').read_text().splitlines()
                    if (r := json.loads(line))}
    rows = []
    for trial in evaluation['rows']:
        groups = {}
        for key in trial['cases']:
            row = observations[key]
            case = cases[row.get('base_case', row['case'])]
            if case['dataset'] != 'DIV2K':
                continue
            group = (case['image'], case['format'], case['sigma'])
            groups.setdefault(group, {})[case['realization']] = (case, row)
        for (image, fmt, sigma), pair in groups.items():
            if set(pair) != {0, 1}:
                continue
            noisy, pixels, ids = [], [], []
            for seed in (0, 1):
                case, row = pair[seed]
                _, incoming, _ = load_case(inputs, manifest, case, evaluation['metadata']['size'])
                with np.load(root/row['output_file']) as stored:
                    outgoing = stored['pixels']
                for array, field in ((incoming, 'input_sha256'), (outgoing, 'output_sha256')):
                    if hashlib.sha256(array.tobytes()).hexdigest() != row[field]:
                        raise ValueError('paired input/output hash mismatch')
                noisy.append(incoming); pixels.append(outgoing); ids.append(row['id'])
            full = response(*noisy, *pixels)
            score = response(*noisy, *pixels, evaluation['metadata']['score_size'])
            rows.append(dict(label=trial['label'], algorithm=trial['algorithm'],
                             image=image, format=fmt, sigma=sigma,
                             sigma_mode=pair[0][1]['sigma_mode'], observations=ids,
                             response_full=full['ratio'], response_score=score['ratio'],
                             score_input_change_rmse=score['input_change_rmse'],
                             score_output_change_rmse=score['output_change_rmse']))
    save_json(out/'response.json', dict(rows=rows, procedure=__doc__,
              driver_sha256=file_sha(__file__), evaluation_sha256=file_sha(root/'evaluation.json'),
              observations_sha256=file_sha(root/'observations.jsonl')))
    if rows:
        with (out/'response.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print('paired responses:', len(rows))


if __name__ == '__main__':
    main()
