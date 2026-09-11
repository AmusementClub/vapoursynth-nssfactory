#!/usr/bin/env python3
"""Exact-output diagnostic: hoist invariant validation and memoize budgets only.

Single-process serial RESEARCH ADAPTER, not a production API or thread-safe
global patch. No mathematical solver/group/aggregation implementation changes.
Cache lifetime is one denoise call; its construction and first misses are timed.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.io import loadmat

import p2_reference
import reference
from run import read_image, sha


_active = False


@contextmanager
def prepared(dictionary, counters):
    global _active
    if _active:
        raise RuntimeError('nested preparation contexts are unsupported')
    # Copy, validate and mark read-only INSIDE the denoise adapter's timing.
    immutable = np.array(dictionary, dtype=np.float64, order='K', copy=True)
    reference.validate_dictionary(immutable)
    immutable.setflags(write=False)
    saved = [(module, name, getattr(module, name)) for module in (reference, p2_reference)
             for name in ('validate_dictionary', 'noise_budget')]
    original_validate = reference.validate_dictionary
    original_budget = reference.noise_budget

    def validate(value):
        if value is immutable:
            counters['validation_reuses'] = counters.get('validation_reuses', 0) + 1
            return value
        return original_validate(value)

    @lru_cache(maxsize=None)
    def budget(m, size, sigma, tau=.8):
        return original_budget(m, size, sigma, tau)

    _active = True
    try:
        for module in (reference, p2_reference):
            module.validate_dictionary = validate
            module.noise_budget = budget
        yield immutable
    finally:
        counters['budget_cache'] = budget.cache_info()._asdict()
        for module, name, value in saved:
            setattr(module, name, value)
        _active = False


def invoke(image, dictionary, sigma, variant, reuse=False):
    if variant not in ('p1', 'p2'):
        raise ValueError('unknown variant')
    call = (lambda d: reference.denoise(image, d, sigma)) if variant == 'p1' else (
        lambda d: p2_reference.denoise_diagnostic(image, d, sigma, solver='energy_gain', grouping='overlap'))
    counters = {}
    if reuse:
        with prepared(dictionary, counters) as invariant:
            result = call(invariant)
    else:
        result = call(dictionary)
    return result, counters


def double_hash(array):
    return hashlib.sha256(array.astype('<f8').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'dictionary', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1 or any(os.getenv(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')):
        parser.error('positive pairs and single-thread environment required')
    image = read_image(args.input, args.width, args.height)
    dictionary = loadmat(args.dictionary, variable_names=['D'])['D']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    expected, warm, rows = {}, {}, []
    variants = [('p1', False), ('p1', True), ('p2', False), ('p2', True)]
    for variant, reuse in variants:
        result, counters = invoke(image, dictionary, args.sigma / 255, variant, reuse)
        key = variant + ('_reuse' if reuse else '_baseline')
        output, pilot = double_hash(result.output), double_hash(result.pilot)
        if not reuse:
            expected[variant] = (output, pilot)
        if (output, pilot) != expected[variant]:
            raise AssertionError('preparation reuse changed float64 output or pilot')
        warm[key] = {'output_float64_sha256': output, 'pilot_float64_sha256': pilot, 'counters': counters}
        del result
    for pair in range(args.pairs):
        for variant, reuse in (variants if pair % 2 == 0 else list(reversed(variants))):
            gc.collect()
            start = time.perf_counter()
            result, counters = invoke(image, dictionary, args.sigma / 255, variant, reuse)
            seconds = time.perf_counter() - start
            output, pilot = double_hash(result.output), double_hash(result.pilot)
            if (output, pilot) != expected[variant]:
                raise AssertionError('measured output changed')
            row = {'pair': pair, 'variant': variant, 'reuse': reuse, 'seconds': seconds,
                   'output_float64_sha256': output, 'pilot_float64_sha256': pilot, 'counters': counters}
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (out / 'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            del result
    medians = {v + ('_reuse' if reuse else '_baseline'):
               statistics.median(r['seconds'] for r in rows if r['variant'] == v and r['reuse'] == reuse)
               for v, reuse in variants}
    report = {'schema': 'nss.paper-preparation-probe.v1', 'rows': rows, 'warm': warm,
              'median_seconds': medians, 'float64_output_and_pilot_exact': True,
              'formal_c4_gate': False, 'input_sha256': sha(args.input),
              'dictionary_sha256': sha(args.dictionary),
              'timing_boundary': 'whole denoise adapter including dictionary copy, validation and fresh per-call cache',
              'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                                ('reference.py', 'p2_reference.py', 'preparation_probe.py')},
              'scope': 'diagnostic process only; frozen solvers/iterations/groups; no production thread-safety claim'}
    (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(medians, indent=2))


if __name__ == '__main__':
    main()
