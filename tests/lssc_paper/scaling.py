#!/usr/bin/env python3
"""Bounded P1/P2 size/noise stress test; frozen math, no learning or promotion.

Two persistent workers are driven serially in alternating order. Each owns its
RSS high-water mark. Warm instrumentation is excluded from measured runs, and
every measured output must match that worker's warm float32 output exactly.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import platform
import resource
import selectors
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

import p2_reference
import reference
from benchmark import digest
from run import metrics, read_image, sha


@contextmanager
def stage_timers():
    """Non-nested leaf timers; only used on the untimed warm pass."""
    totals = {}
    counts = {}
    originals = []
    state = {'aggregations': 0}

    def wrap(module, name, category):
        original = getattr(module, name)
        originals.append((module, name, original))

        def invoke(*args, **kwargs):
            key = category
            if key == 'solve':
                key = 'pilot_solve' if state['aggregations'] == 0 else 'group_solve'
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[key] = totals.get(key, 0.) + time.perf_counter() - start
                counts[key] = counts.get(key, 0) + 1
                if category == 'aggregate':
                    state['aggregations'] += 1

        setattr(module, name, invoke)

    for module in (reference, p2_reference):
        for name, category in [('noise_budget', 'budget'), ('extract_patches', 'extract'),
                               ('aggregate', 'aggregate'), ('greedy_groups', 'matching')]:
            wrap(module, name, category)
    wrap(reference, 'somp', 'solve')
    wrap(p2_reference, 'simultaneous_ols', 'solve')
    wrap(p2_reference, 'overlapping_groups', 'matching')
    try:
        yield totals, counts
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def statistics_for(result):
    groups = result.final_stage.groups if hasattr(result, 'final_stage') else result.groups
    fits = result.final_stage.pursuits if hasattr(result, 'final_stage') else result.pursuits
    sizes = np.asarray([len(g) for g in groups])
    supports = np.asarray([len(f.support) for f in fits])
    patches = len(result.positions)
    return {'patches': patches, 'groups': len(groups), 'occurrences': int(sizes.sum()),
            'occurrences_per_patch': float(sizes.sum() / patches),
            'group_size_median': float(np.median(sizes)), 'group_size_max': int(sizes.max()),
            'support_size_mean': float(supports.mean()), 'support_size_max': int(supports.max()),
            'all_final_groups_met_budget': all(f.stop_reason == 'noise_budget' for f in fits)}


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def worker(args):
    image = read_image(args.input, args.width, args.height)
    dictionary = loadmat(args.dictionary, variable_names=['D'])['D']
    call = (lambda: reference.denoise(image, dictionary, args.sigma / 255)) if args.variant == 'p1' else (
        lambda: p2_reference.denoise_diagnostic(image, dictionary, args.sigma / 255,
                                               solver='energy_gain', grouping='overlap'))
    expected = None
    for line in sys.stdin:
        command = json.loads(line)['command']
        gc.collect()
        if command == 'warm':
            with stage_timers() as (components, counts):
                start = time.perf_counter()
                result = call()
                seconds = time.perf_counter() - start
            expected = digest(result.output)
            result.output.astype('<f4').tofile(Path(args.out) / (args.variant + '.f32'))
            row = {'phase': 'warm_instrumented', 'seconds': seconds, 'output_sha256': expected,
                   'components_seconds': components, 'component_calls': counts,
                   'other_seconds': seconds - sum(components.values()),
                   'workload': statistics_for(result)}
        elif command == 'run' and expected is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter()
            result = call()
            seconds = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_SELF)
            actual = digest(result.output)
            if actual != expected:
                raise AssertionError('instrumentation or repeat output mismatch')
            row = {'phase': 'measured', 'seconds': seconds, 'output_sha256': actual,
                   'cpu_user_seconds': after.ru_utime - usage.ru_utime,
                   'cpu_system_seconds': after.ru_stime - usage.ru_stime}
        else:
            raise ValueError('unknown command or missing warm pass')
        if not np.isfinite(result.output).all() or np.any(result.coverage <= 0):
            raise AssertionError('nonfinite output or uncovered pixels')
        row['maxrss_bytes'] = rss_bytes()
        row['variant'] = args.variant
        print(json.dumps(row, allow_nan=False), flush=True)
        del result


def fixture(clean, width, height, sigma, seed):
    """Repeat only clean pixels outside native extent; never repeat noise."""
    if min(width, height) < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('positive geometry and finite nonnegative noise required')
    target = np.tile(clean, ((height + len(clean) - 1) // len(clean),
                            (width + clean.shape[1] - 1) // clean.shape[1]))[:height, :width]
    noise = np.random.default_rng(seed).normal(0, sigma / 255, target.shape)
    return target.astype('<f4'), (target + noise).astype('<f4')


def request(process, command, timeout, memory_gib):
    process.stdin.write(json.dumps({'command': command}) + '\n')
    process.stdin.flush()
    started = time.monotonic()
    next_check = started
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            if selector.select(timeout=1):
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'worker exited: {process.poll()} (see stderr log)')
                return json.loads(line)
            now = time.monotonic()
            if now - started > timeout:
                raise TimeoutError(f'worker exceeded {timeout}s per call')
            if now >= next_check:
                probe = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                                       capture_output=True, text=True, check=False)
                if probe.returncode == 0 and probe.stdout.strip():
                    if int(probe.stdout.strip()) * 1024 > memory_gib * 1024**3:
                        raise MemoryError(f'worker exceeded {memory_gib} GiB RSS ceiling')
                next_check = now + 5


def run_case(args, case, destination):
    width, height, sigma = case
    destination.mkdir(exist_ok=False)
    native = np.asarray(Image.open(args.clean).convert('L'), dtype=np.float64) / 255
    clean, noisy = fixture(native, width, height, sigma, args.seed)
    clean.tofile(destination / 'clean.f32')
    noisy.tofile(destination / 'noisy.f32')
    record = {'schema': 'nss.paper-scaling.v1', 'width': width, 'height': height, 'sigma': sigma,
              'dictionary_sha256': sha(args.dictionary), 'source_image_sha256': sha(args.clean),
              'input_sha256': sha(destination / 'noisy.f32'), 'seed': args.seed,
              'fixture_kind': 'native_top_left_crop' if width <= native.shape[1] and height <= native.shape[0]
                              else 'tiled_clean_with_independent_noise_size_stress_only',
              'native_shape': list(native.shape), 'pairs': args.pairs, 'warm': {}, 'rows': [],
              'formal_c4_gate': False, 'cpu_affinity': None, 'idle_sibling_verified': None,
              'timed_source_fills': 0, 'platform': platform.platform(),
              'timing_boundary': 'warm denoise including reference traces; file IO, metrics, GC, hash excluded',
              'memory_boundary': 'separate persistent worker OS peak RSS, including warm instrumented pass',
              'scope': 'fixed dictionary, P1 vs diagnostic energy/overlap P2; no learning or native prediction',
              'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                                ('reference.py', 'p2_reference.py', 'scaling.py')},
              'limits': {'seconds_per_call': args.timeout, 'rss_gib_per_worker': args.memory_gib},
              'complete': False}
    workers, logs = {}, []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    try:
        for variant in ('p1', 'p2'):
            log = (destination / (variant + '.stderr')).open('w')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--variant', variant,
                       '--input', str(destination / 'noisy.f32'), '--dictionary', args.dictionary,
                       '--width', str(width), '--height', str(height), '--sigma', str(sigma),
                       '--out', str(destination)]
            workers[variant] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=log, text=True, env=env)
            warm = request(workers[variant], 'warm', args.timeout, args.memory_gib)
            record['warm'][variant] = warm
            print(json.dumps({'case': destination.name, **warm}), flush=True)
        for pair in range(args.pairs):
            order = ('p1', 'p2') if pair % 2 == 0 else ('p2', 'p1')
            for variant in order:
                row = request(workers[variant], 'run', args.timeout, args.memory_gib)
                row['pair'] = pair
                record['rows'].append(row)
                with (destination / 'samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps({'case': destination.name, **row}), flush=True)
        record['median_seconds'] = {v: statistics.median(r['seconds'] for r in record['rows']
                                                        if r['variant'] == v) for v in workers}
        record['p2_over_p1_paired_ratios'] = [
            next(r['seconds'] for r in record['rows'] if r['pair'] == p and r['variant'] == 'p2') /
            next(r['seconds'] for r in record['rows'] if r['pair'] == p and r['variant'] == 'p1')
            for p in range(args.pairs)]
        record['metrics'] = {v: metrics(clean, np.fromfile(destination / (v + '.f32'), dtype='<f4')
                                       .reshape(height, width)) for v in workers}
        record['complete'] = True
    except (Exception, KeyboardInterrupt) as error:
        record['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        for process in workers.values():
            process.terminate()
        for process in workers.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
        for log in logs:
            log.close()
        (destination / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--variant', choices=['p1', 'p2'])
    parser.add_argument('--input')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--clean')
    parser.add_argument('--cases', nargs='+', help='WIDTHxHEIGHT:SIGMA')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--memory-gib', type=float, default=24)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.pairs < 1 or args.timeout <= 0 or args.memory_gib <= 0 or not args.cases or not args.clean:
        parser.error('positive run limits, clean image and cases required')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in args.cases:
        geometry, sigma = case.split(':')
        width, height = map(int, geometry.split('x'))
        result = run_case(args, (width, height, float(sigma)), out / case.replace(':', '-s'))
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
