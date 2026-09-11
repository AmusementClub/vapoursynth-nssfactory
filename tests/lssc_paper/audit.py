#!/usr/bin/env python3
"""Recompute saved-stage constraints independently of the pursuit loop."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import chi2

from run import read_image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, input_path, dictionary_path):
    directory, dictionary_path = Path(directory), Path(dictionary_path)
    meta = json.loads((directory/'result.json').read_text())
    assert sha(input_path) == meta['input_sha256']
    assert sha(dictionary_path) == meta['dictionary_sha256']
    for name in ('output','pilot'):
        assert sha(directory/(name+'.f32')) == meta[name+'_sha256']
    assert sha(directory/'trace.npz') == meta['trace_sha256']
    height, width = meta['shape']
    image = read_image(input_path, width, height)
    dictionary = (loadmat(dictionary_path, variable_names=['D'])['D'] if dictionary_path.suffix == '.mat'
                  else np.load(dictionary_path, allow_pickle=False))
    m = dictionary.shape[0]
    block = int(np.sqrt(m))
    sigma = meta['parameters']['sigma_8bit']/255
    tau = meta['parameters']['tau']
    with np.load(directory/'trace.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    positions = arrays['positions']
    expected_positions = [(y,x) for y in range(height-block+1) for x in range(width-block+1)]
    np.testing.assert_array_equal(positions, expected_positions)
    noisy_patches = np.stack([image[y:y+block, x:x+block].ravel(order='F') for y,x in positions])
    checked_groups = 0
    rebuilt = {}
    rounding = {}
    for prefix in ('pilot','final'):
        saved = arrays[prefix+'_patches']
        assert saved.shape == noisy_patches.shape and np.isfinite(saved).all()
        labels = np.arange(len(positions)) if prefix == 'pilot' else arrays['labels']
        unique = np.unique(labels)
        np.testing.assert_array_equal(unique, np.arange(len(unique)))
        offsets = arrays[prefix+'_support_offsets']
        supports = arrays[prefix+'_support_indices']
        assert len(offsets) == len(unique)+1 and offsets[0] == 0 and offsets[-1] == len(supports)
        assert np.all(np.diff(offsets) >= 0)
        for label in unique:
            members = np.flatnonzero(labels == label)
            support = supports[offsets[label]:offsets[label+1]]
            assert len(set(support)) == len(support)
            assert np.all((support >= 0) & (support < dictionary.shape[1]))
            noisy = noisy_patches[members].T
            mean = noisy.mean(axis=0, keepdims=True)
            centered = noisy-mean
            if len(support):
                # A separately evaluated SVD pseudoinverse, not the loop's saved coefficients.
                active = dictionary[:, support]
                fitted = active @ (np.linalg.pinv(active) @ centered)
            else:
                fitted = np.zeros_like(centered)
            np.testing.assert_allclose(saved[members].T, fitted+mean, atol=1e-10, rtol=1e-10)
            energy = np.sum((centered-fitted)**2)
            epsilon = sigma*sigma*chi2.ppf(tau, m*len(members))
            assert energy <= epsilon+1e-10*max(1.,np.sum(centered*centered))
            np.testing.assert_allclose(energy, arrays[prefix+'_residual_squared'][label], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(epsilon, arrays[prefix+'_epsilon'][label], atol=1e-12, rtol=1e-12)
            checked_groups += 1
        numerator, absolute_sum = np.zeros_like(image), np.zeros_like(image)
        counts = np.zeros(image.shape, dtype=np.int64)
        for patch,(y,x) in zip(saved,positions):
            numerator[y:y+block,x:x+block] += patch.reshape(block,block,order='F')
            absolute_sum[y:y+block,x:x+block] += np.abs(patch.reshape(block,block,order='F'))
            counts[y:y+block,x:x+block] += 1
        np.testing.assert_array_equal(counts, arrays['coverage'])
        name = 'pilot' if prefix == 'pilot' else 'output'
        output = np.fromfile(directory/(name+'.f32'),dtype='<f4').reshape(image.shape)
        rebuilt[prefix] = numerator/counts
        # Independent aggregation visits terms in a different order. Its float64
        # rounding can cross one float32 midpoint; that is not a same-code repeat.
        # Bound both summations plus divisions, then add the final half-float32-ULP.
        eps64 = np.finfo(np.float64).eps
        gamma = counts*eps64/(1-counts*eps64)
        allowance = (4*gamma*absolute_sum/counts + 4*eps64*np.abs(rebuilt[prefix])
                     + .5*np.abs(np.spacing(output)).astype(np.float64))
        error = np.abs(rebuilt[prefix]-output.astype(np.float64))
        assert np.all(error <= allowance), 'aggregation exceeds explicit floating rounding bound'
        rounding[prefix] = {'float32_rounding_disagreements':int(np.count_nonzero(
            rebuilt[prefix].astype('<f4') != output)), 'max_absolute_error':float(error.max()),
            'rounding_bound_passed':True}
    if meta['algorithm'] == 'fixed_dictionary_ssc':
        pilot = rebuilt['pilot']
        pilot_patches = np.stack([pilot[y:y+block,x:x+block].ravel(order='F') for y,x in positions])
        window = meta['parameters']['window']
        threshold = (32*sigma)**2/m
        assert threshold == meta['parameters']['matching_raw_ssd_threshold']
        labels = arrays['labels']
        grid = np.arange(len(positions)).reshape(height-block+1,width-block+1)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            seed = members[0]
            y,x = positions[seed]
            candidates = grid[max(0,y-window//2):y+(window-1)//2+1,
                              max(0,x-window//2):x+(window-1)//2+1].ravel()
            available = candidates[labels[candidates] >= label]
            distance = np.sum((pilot_patches[available]-pilot_patches[seed])**2,axis=1)
            # Near-threshold floating summation ties are not silently accepted.
            expected = available[distance <= threshold]
            np.testing.assert_array_equal(expected,members)
    return {'verified':True,'groups_checked':checked_groups,'patches':len(positions),
            'aggregation_rounding':rounding,
            'checks':['file hashes','raster geometry','shared-support LS fit to original noisy patches',
                      'chi-square group residual budget','count-normalized reconstruction',
                      'pilot-guided raw-SSD grouping/window/complete disjoint cover']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--dictionary',required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run,args.input,args.dictionary),indent=2))
