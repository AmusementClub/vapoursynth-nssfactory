#!/usr/bin/env python3
"""Independent frozen-SVD/whole-image checks for a saved fixed-D native control.

This checks the P2 diagnostic contract, not author-MEX or full-paper equivalence.
Approximate/learned variants are evaluated separately, not assigned control parity.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from native import Native
from native_campaign import digest64, matrix
import p2_reference
from reference import extract_patches, noise_budget
from run import read_image, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--control', default='fp64_direct')
    parser.add_argument('--probes', type=int, default=16)
    args = parser.parse_args()
    root = Path(args.run)
    destination = root / 'native-audit.json'
    if destination.exists():
        raise FileExistsError('audit output already exists; use a new run directory')
    if args.probes < 1:
        raise ValueError('positive probe count required')
    record = json.loads((root / 'result.json').read_text())
    if sha(args.input) != record['input_sha256'] or sha(args.dictionary) != record['dictionary_file_sha256']:
        raise ValueError('input/dictionary provenance mismatch')
    policy = record['policies'][args.control]
    if any(policy.get(key, 0) for key in ('precision', 'match_precision', 'image_passes', 'group_passes',
                                         'max_group', 'max_support', 'dictionary_atoms')):
        raise ValueError('audit control must be an uncapped fixed-D FP64 configuration')
    if any(policy.get(key, value) != value for key, value in
           [('solver', 1), ('grouping', 1), ('window', 32), ('threshold_multiplier', 1), ('epsilon_scale', 1)]):
        raise ValueError('audit control must retain the P2 diagnostic definitions')
    library = root / record['library_file']
    if sha(library) != record['library_sha256']:
        raise ValueError('frozen native binary hash mismatch')
    engine = Native(library)
    image = read_image(args.input, record['width'], record['height'])
    dictionary = matrix(args.dictionary)
    sigma = record['sigma'] / 255
    report = {'schema': 'nss.native-p2-audit.v1', 'complete': False, 'control': args.control,
              'scope': 'frozen P2 diagnostic/SVD contract only, not author-MEX equivalence',
              'input_sha256': sha(args.input), 'dictionary_file_sha256': sha(args.dictionary),
              'library_sha256': sha(library), 'audit_source_sha256': sha(__file__),
              'reference_source_sha256': sha(Path(__file__).with_name('p2_reference.py'))}
    if report['reference_source_sha256'] != record['source_sha256']['tests/lssc_paper/p2_reference.py']:
        raise ValueError('frozen reference source differs from recorded campaign')
    started = time.perf_counter()
    try:
        expected = p2_reference.denoise_diagnostic(image, dictionary, sigma, solver='energy_gain', grouping='overlap')
        output = np.load(root / (args.control + '-output.npy'), allow_pickle=False)
        pilot = np.load(root / (args.control + '-pilot.npy'), allow_pickle=False)
        warm = next(row for row in record['rows'] if row['variant'] == args.control and row['pair'] == -1)
        if digest64(output) != warm['output_sha256'] or digest64(pilot) != warm['pilot_sha256']:
            raise ValueError('saved control output/pilot hash mismatch')
        report['output_max_abs'] = float(np.max(np.abs(output - expected.output)))
        report['pilot_max_abs'] = float(np.max(np.abs(pilot - expected.pilot)))
        report['output_rmse'] = float(np.sqrt(np.mean((output - expected.output)**2)))
        report['reference_output_sha256'] = digest64(expected.output)
        report['reference_pilot_sha256'] = digest64(expected.pilot)
        np.testing.assert_allclose(output, expected.output, atol=2e-10, rtol=2e-11)
        np.testing.assert_allclose(pilot, expected.pilot, atol=2e-10, rtol=2e-11)
        block = int(np.sqrt(dictionary.shape[0]))
        groups = engine.groups(pilot, block, sigma)
        assert len(groups) == len(expected.groups)
        for actual, golden in zip(groups, expected.groups):
            np.testing.assert_array_equal(actual, golden)
        report['all_group_memberships_exact'] = True
        report['groups'] = len(groups)
        patches, _ = extract_patches(image, block)
        # Include boundaries, largest groups, then a deterministic spread.
        choices = set(np.linspace(0, len(groups) - 1, min(args.probes, len(groups)), dtype=int))
        choices.add(int(np.argmax([len(g) for g in groups])))
        probes = []
        for index in sorted(choices):
            signals = patches[groups[index]].T
            signals = signals - signals.mean(axis=0, keepdims=True)
            epsilon = noise_budget(block * block, signals.shape[1], sigma)
            fit = engine.solve(dictionary, signals, epsilon, **policy)
            golden = expected.pursuits[index]
            np.testing.assert_array_equal(fit.support, golden.support)
            if len(fit.support):
                coefficients, _, rank, _ = np.linalg.lstsq(dictionary[:, fit.support], signals, rcond=None)
                independent = dictionary[:, fit.support] @ coefficients
                assert rank == len(fit.support)
            else:
                independent = np.zeros_like(signals)
            np.testing.assert_allclose(fit.reconstruction, independent, atol=2e-10, rtol=2e-11)
            residual = float(np.sum((signals - independent)**2))
            assert residual <= epsilon + 64 * np.finfo(float).eps * max(1., float(np.sum(signals**2)))
            probes.append({'index': int(index), 'size': len(groups[index]), 'support': len(fit.support),
                           'svd_max_abs': float(np.max(np.abs(fit.reconstruction - independent))),
                           'residual': residual, 'epsilon': epsilon})
        report['probes'] = probes
        report['complete'] = True
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['diagnostic_seconds'] = time.perf_counter() - started
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
