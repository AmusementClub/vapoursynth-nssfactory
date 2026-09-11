#!/usr/bin/env python3
"""Inspect a saved mixed-precision difference without retuning the algorithm."""
import argparse
import json
from pathlib import Path

import numpy as np

import native
from native_campaign import digest64, matrix
from run import read_image, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--variant', default='mixed_sme')
    args = parser.parse_args()
    root = Path(args.run)
    destination = root / 'drift-probe.json'
    if destination.exists():
        raise FileExistsError('drift probe already exists')
    record = json.loads((root / 'result.json').read_text())
    if sha(args.input) != record['input_sha256'] or sha(args.dictionary) != record['dictionary_file_sha256']:
        raise ValueError('probe provenance mismatch')
    library = root / record['library_file']
    if sha(library) != record['library_sha256']:
        raise ValueError('probe library mismatch')
    policy = record['policies'][args.variant]
    if any(policy.get(name, 0) for name in ['image_passes', 'group_passes', 'max_group', 'max_support', 'dictionary_atoms']):
        raise ValueError('probe requires a fixed-dictionary uncapped precision candidate')
    engine = native.Native(library)
    image = read_image(args.input, record['width'], record['height'])
    dictionary = matrix(args.dictionary)
    block = int(np.sqrt(dictionary.shape[0]))
    sigma = record['sigma'] / 255
    base = {kind: np.load(root / ('fp64_direct-' + kind + '.npy'), allow_pickle=False) for kind in ('output', 'pilot')}
    mixed = {kind: np.load(root / (args.variant + '-' + kind + '.npy'), allow_pickle=False) for kind in ('output', 'pilot')}
    groups_base = engine.groups(base['pilot'], block, sigma, match_precision=0)
    groups_mixed = engine.groups(mixed['pilot'], block, sigma, match_precision=policy.get('match_precision', 0))
    same = len(groups_base) == len(groups_mixed) and all(np.array_equal(a, b) for a, b in zip(groups_base, groups_mixed))
    refined = engine.denoise(image, dictionary, sigma, **(policy | {'precision': 2}))
    report = {'schema': 'nss.native-drift-probe.v1', 'complete': True, 'input_sha256': sha(args.input),
              'dictionary_sha256': sha(args.dictionary), 'library_sha256': sha(library),
              'source_sha256': sha(__file__), 'adapter_sha256': sha(native.__file__),
              'diagnostic_only_not_paired_timing': True, 'candidate': args.variant,
              'candidate_policy': policy, 'refined_policy': policy | {'precision': 2},
              'control_groups': len(groups_base), 'candidate_groups': len(groups_mixed),
              'all_group_memberships_exact': same,
              'candidate_pilot_max_abs': float(np.max(np.abs(mixed['pilot'] - base['pilot']))),
              'candidate_output_max_abs': float(np.max(np.abs(mixed['output'] - base['output']))),
              'refined_pilot_max_abs': float(np.max(np.abs(refined.pilot - base['pilot']))),
              'refined_output_max_abs': float(np.max(np.abs(refined.output - base['output']))),
              'refined_output_sha256': digest64(refined.output),
              'refined_candidates': refined.stats['refined_candidates']}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
