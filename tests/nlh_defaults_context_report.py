#!/usr/bin/env python3
"""Compare nested input contexts on identical central pixels and parameters."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from defaults_compare import plane_quality
from nlh_defaults_compare import read_trial
from nlh_defaults_inputs import file_sha, load_case, save_json, validate_manifest


def center(array, size):
    if size > min(array.shape[1:]):
        raise ValueError('common score region exceeds context')
    _, height, width = array.shape
    y, x = (height-size)//2, (width-size)//2
    return array[:, y:y+size, x:x+size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    inputs = Path(config['inputs'])
    manifest = json.loads((inputs/'inputs.json').read_text())
    validate_manifest(manifest)
    cases = {c['id']: c for c in manifest['cases']}
    datasets, parameters = [], {}
    for experiment in config['experiments']:
        trials = {}
        for label in config['profiles']:
            source_label = experiment.get('profiles', {}).get(label, label)
            evaluation, trial, records = read_trial(experiment['path'], source_label)
            if label in parameters and parameters[label] != trial['parameters']:
                raise ValueError('a context comparison changed parameters')
            parameters[label] = trial['parameters']
            trials[label] = records
        datasets.append((experiment, evaluation['metadata'], trials))
    rows = []
    for case_id in config['cases']:
        common_input = None
        for experiment, metadata, trials in datasets:
            clean, noisy, crop = load_case(inputs, manifest, cases[case_id], metadata['size'])
            incoming = center(noisy, config['common_size'])
            if common_input is None:
                common_input = incoming.copy()
            elif not np.array_equal(common_input, incoming):
                raise ValueError('nested contexts use different noisy pixels')
            for label, records in trials.items():
                record = records[case_id]
                if record['input_sha256'] != hashlib.sha256(noisy.tobytes()).hexdigest() or record['crop'] != crop:
                    raise ValueError('context input hash or crop mismatch')
                with np.load(Path(experiment['path'])/record['output_file']) as stored:
                    output = stored['pixels']
                if record['output_sha256'] != hashlib.sha256(output.tobytes()).hexdigest():
                    raise ValueError('context output hash mismatch')
                common_quality = plane_quality(center(clean, config['common_size']),
                                               center(output, config['common_size']))
                full_quality = plane_quality(clean, output)
                steps = parameters[label].get('block_step')
                if isinstance(steps, int):
                    steps = [steps, steps]
                phases = [[crop[0] % step, crop[1] % step] for step in steps] if steps else None
                rows.append(dict(case=case_id, profile=label, context=experiment['label'],
                                 width=noisy.shape[2], height=noisy.shape[1], common_size=config['common_size'],
                                 global_sampling_phase_xy=json.dumps(phases),
                                 common_psnr=common_quality['psnr_db'], common_ssim=common_quality['ssim'],
                                 whole_psnr=full_quality['psnr_db'], whole_ssim=full_quality['ssim'],
                                 input_common_sha256=hashlib.sha256(incoming.tobytes()).hexdigest(),
                                 output_common_sha256=hashlib.sha256(center(output, config['common_size']).tobytes()).hexdigest(),
                                 output_sha256=record['output_sha256'], observation=record['id']))
                del output
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'context.json', dict(rows=rows, parameters=parameters,
              config_sha256=file_sha(args.config), driver_sha256=file_sha(__file__),
              scope='Identical noisy central pixels across nested contexts; crop boundaries and reference-grid phase can both change; descriptive stability, no fixed threshold',
              experiments=[dict(config=e, metadata=m) for e, m, _ in datasets]))
    with (out/'context.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('hash-verified context comparisons:', len(rows))


if __name__ == '__main__':
    main()
