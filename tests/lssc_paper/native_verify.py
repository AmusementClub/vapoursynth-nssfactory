#!/usr/bin/env python3
"""Recompute saved-campaign provenance, hashes, quality and paired summaries.

Reads original fixtures and serialized outputs independently of worker timing.
Does not rerun denoising, change completed records or promote any policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np

from native_campaign import digest64, matrix, selected_dictionary, summarize
from run import metrics, read_image, sha


def fixture_index(manifests, raw_cases):
    result = {}
    for source in manifests:
        path = Path(source).resolve()
        for case in json.loads(path.read_text())['cases']:
            clean = (path.parent / case['clean']).resolve()
            noisy = (path.parent / case['noisy']).resolve()
            if not clean.is_relative_to(path.parent) or not noisy.is_relative_to(path.parent):
                raise ValueError('fixture path escaped manifest directory')
            if sha(clean) != case['clean_sha256'] or sha(noisy) != case['noisy_sha256']:
                raise ValueError('fixture hash mismatch')
            result[case['noisy_sha256']] = case | dict(clean_path=clean, noisy_path=noisy)
    for source in raw_cases:
        root = Path(source).resolve()
        record = json.loads((root / 'result.json').read_text())
        if sha(root / 'noisy.f32') != record['input_sha256']:
            raise ValueError('raw-case input hash mismatch')
        result[record['input_sha256']] = dict(width=record['width'], height=record['height'], sigma=record['sigma'],
                                             clean_path=root / 'clean.f32', noisy_path=root / 'noisy.f32')
    return result


def archives(root, record):
    archive = root / record.get('source_archive', 'source-v1.tar.gz')
    if 'source_archive_sha256' in record and sha(archive) != record['source_archive_sha256']:
        raise ValueError('source archive hash mismatch')
    with tarfile.open(archive, 'r:gz') as package:
        for name, expected in record['source_sha256'].items():
            source = package.extractfile(name)
            if source is None or hashlib.sha256(source.read()).hexdigest() != expected:
                raise ValueError(f'archived source mismatch: {name}')
        if 'library_file' in record:
            if Path(record['library_file']).name != record['library_file']:
                raise ValueError('library is not a local artifact basename')
            library_hash = sha(root / record['library_file'])
        else:
            # First pilot campaign archived its exact binary within source-v1.
            name = next(name for name in package.getnames() if name.endswith('/libnss_lssc_explore.dylib'))
            library_hash = hashlib.sha256(package.extractfile(name).read()).hexdigest()
    if library_hash != record['library_sha256']:
        raise ValueError('archived native binary hash mismatch')
    return len(record['source_sha256'])


def verify_case(path, fixtures, dictionaries):
    root = path.parent
    record = json.loads(path.read_text())
    if not record.get('complete') or not record.get('source_unchanged') or not record.get('library_unchanged'):
        raise ValueError('campaign incomplete or changed during timing')
    fixture = fixtures[record['input_sha256']]
    for name in ('width', 'height', 'sigma'):
        if fixture[name] != record[name]:
            raise ValueError('fixture geometry/noise mismatch')
    clean = read_image(fixture['clean_path'], record['width'], record['height'])
    if 'clean_sha256' in record and sha(fixture['clean_path']) != record['clean_sha256']:
        raise ValueError('campaign clean hash mismatch')
    dictionary = dictionaries[record['dictionary_file_sha256']]
    source_count = archives(root, record)
    variants = list(record['policies'])
    if len(record['rows']) != len(variants) * (record['pairs'] + 1):
        raise ValueError('missing/extra scheduled rows')
    recomputed = summarize(record['rows'], variants, 'C4 CPU0' in record['host_gate'])
    outputs = {}
    details = []
    for name in variants:
        rows = [row for row in record['rows'] if row['variant'] == name]
        if sorted(row['pair'] for row in rows) != list(range(-1, record['pairs'])):
            raise ValueError('duplicate/missing pair id')
        if any(row.get('failed') for row in rows):
            raise ValueError('failed row in completed campaign')
        arrays = {kind: np.load(root / f'{name}-{kind}.npy', allow_pickle=False)
                  for kind in ('output', 'pilot', 'dictionary')}
        initial = selected_dictionary(dictionary, record['policies'][name].get('dictionary_atoms'))
        if arrays['dictionary'].shape != initial.shape:
            raise ValueError('dictionary shape mismatch')
        for kind in ('output', 'pilot'):
            if arrays[kind].shape != clean.shape:
                raise ValueError('output geometry mismatch')
        for kind, array in arrays.items():
            if not np.isfinite(array).all() or array.dtype != np.float64:
                raise ValueError('nonfinite/non-FP64 saved array')
            expected = digest64(array)
            if any(row[kind + '_sha256'] != expected for row in rows):
                raise ValueError('repeat or serialization hash mismatch')
        if any(row['input_dictionary_sha256'] != digest64(initial) for row in rows):
            raise ValueError('initial dictionary/atom-subset mismatch')
        pixels32 = np.fromfile(root / (name + '.f32'), dtype='<f4').reshape(clean.shape)
        np.testing.assert_array_equal(pixels32, arrays['output'].astype('<f4'))
        calculated = metrics(clean, arrays['output'])
        expected = record['summary'][name]
        for key, value in calculated.items():
            np.testing.assert_allclose(value, expected['metrics'][key], atol=1e-11, rtol=1e-12)
        for key in ('median_seconds', 'median_native_seconds', 'paired_speedup_median'):
            if recomputed[name][key] != expected[key]:
                raise ValueError('paired summary mismatch')
        warm = next(row for row in rows if row['pair'] == -1)
        stats = warm['stats']
        if not stats['capped_groups'] and stats['worst_budget_ratio'] > 1 + 1e-9:
            raise ValueError('unreported residual-budget failure')
        if any(not row.get('repeat_fp64_exact') for row in rows):
            raise ValueError('repeat flag mismatch')
        if np.max(np.linalg.norm(arrays['dictionary'], axis=0)) > 1 + 1.1e-8:
            raise ValueError('dictionary unit-ball constraint violated')
        outputs[name] = arrays['output']
        details.append(dict(variant=name, seconds=expected['median_seconds'], psnr_db=calculated['psnr_db'],
                            ssim=calculated.get('ssim'), peak_rss_bytes=expected['peak_rss_bytes'],
                            native_tracked_peak_bytes=stats['tracked_peak_bytes'],
                            coding_batches=stats['learning_codes'], coding_converged=stats['learning_converged'],
                            coding_pg_max=stats['learning_stationarity_max'],
                            max_abs_vs_control=expected['max_abs_vs_baseline'],
                            delta_db=expected['psnr_delta_vs_baseline']))
    baseline = outputs[variants[0]]
    for name in variants:
        value = float(np.max(np.abs(outputs[name] - baseline)))
        if value != record['summary'][name]['max_abs_vs_baseline']:
            raise ValueError('saved baseline difference mismatch')
    return dict(case=str(root), complete=True, calls=sum(row.get('frames', 1) for row in record['rows']),
                policies=len(variants), archived_source_files_verified=source_count,
                output_pilot_dictionary_repeat_hashes_verified=True, details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--fixtures', nargs='+', required=True)
    parser.add_argument('--raw-cases', nargs='*', default=[])
    parser.add_argument('--dictionaries', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError('verification report already exists')
    fixtures = fixture_index(args.fixtures, args.raw_cases)
    dictionaries = {sha(path): matrix(path) for path in Path(args.dictionaries).glob('dict_n*.mat')}
    report = {'schema': 'nss.native-evidence-verification.v1', 'complete': False,
              'verifier_sha256': sha(__file__), 'cases': []}
    try:
        for path in sorted(Path(args.root).rglob('result.json')):
            record = json.loads(path.read_text())
            if record.get('schema') != 'nss.lssc-native-exploration.v1':
                continue
            result = verify_case(path, fixtures, dictionaries)
            report['cases'].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != 'details'}), flush=True)
        if not report['cases']:
            raise ValueError('no campaign records found')
        report['calls'] = sum(case['calls'] for case in report['cases'])
        report['campaigns'] = len(report['cases'])
        report['complete'] = True
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
