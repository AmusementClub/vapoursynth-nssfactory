#!/usr/bin/env python3
"""Bounded, serial, paired exploration; no defaults are promoted automatically.

Each variant has its own persistent process and RSS high-water mark. Clean data
is read only by the coordinator for metrics AFTER native denoising. Saved noisy
input, dictionary, code, build and output hashes identify every comparison.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import numpy as np
from scipy.io import loadmat

from author_budgets import schedule
from benchmark import cpu_delta, cpu_ticks, eligible
from native import Native, NativeError, default_library
from run import metrics, preview, read_image, sha
from scaling import request, rss_bytes


MIXED = dict(precision=1, correlation=1)
FAST_MATCH = MIXED | dict(match_precision=1)
LEARNING = dict(learning_samples=512, learning_groups=24, learning_group_cap=64, learning_iterations=48)
POLICIES = {
    'fp64_direct': {},
    'fp64_gram': dict(correlation=1),
    'mixed_gram': MIXED,
    'refined_gram': dict(precision=2, correlation=1),
    'mixed_ssd': FAST_MATCH,
    'cap64': FAST_MATCH | dict(max_group=64),
    'cap128': FAST_MATCH | dict(max_group=128),
    'support8': FAST_MATCH | dict(max_support=8),
    'epsilon110': FAST_MATCH | dict(epsilon_scale=1.10),
    'learn1': FAST_MATCH | LEARNING | dict(image_passes=1),
    'learn3': FAST_MATCH | LEARNING | dict(image_passes=3),
    'group1': FAST_MATCH | LEARNING | dict(group_passes=1),
    'learn1group1': FAST_MATCH | LEARNING | dict(image_passes=1, group_passes=1),
    'dict256': FAST_MATCH | dict(dictionary_atoms=256),
    'dict256learn1': FAST_MATCH | LEARNING | dict(dictionary_atoms=256, image_passes=1),
}


def digest64(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def matrix(path):
    return loadmat(path, variable_names=['D'])['D'] if Path(path).suffix == '.mat' else np.load(path, allow_pickle=False)


def selected_dictionary(dictionary, atoms):
    if atoms is None or atoms == dictionary.shape[1]:
        return dictionary
    if not 1 <= atoms <= dictionary.shape[1]:
        raise ValueError('dictionary atom budget exceeds supplied dictionary')
    # Explicit stratified atom-subset approximation, NOT retraining or a
    # claimed equivalent 256-atom author model. Optional learning follows it.
    indices = (np.arange(atoms, dtype=np.int64) * dictionary.shape[1] + dictionary.shape[1] // 2) // atoms
    indices = np.minimum(indices, dictionary.shape[1] - 1)
    if len(np.unique(indices)) != atoms:
        raise AssertionError('duplicate dictionary sampling index')
    return dictionary[:, indices]


def worker(args):
    engine = Native(args.library)
    image = read_image(args.input, args.width, args.height)
    policy = json.loads(args.worker_policy) if args.worker_policy else POLICIES[args.worker].copy()
    dictionary = selected_dictionary(matrix(args.dictionary), policy.pop('dictionary_atoms', None))
    policy['memory_limit_bytes'] = int(args.memory_gib * 1024**3)
    expected = None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if args.require_cpu0:
        siblings = Path('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list').read_text().strip()
        if affinity != [0] or siblings not in ('0,1', '0-1') or not Path('/opt/nss-c4/ENVIRONMENT').is_file():
            raise RuntimeError('verified C4 CPU0/idle-sibling lane required')
    for line in sys.stdin:
        command = json.loads(line)['command']
        if command not in ('warm', 'run') or (command == 'run' and expected is None):
            raise ValueError('invalid worker phase')
        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF)
        ticks_before = cpu_ticks()
        times, native_times, frame_hashes = [], [], []
        for frame in range(args.frames):
            start = time.perf_counter()
            try:
                result = engine.denoise(image, dictionary, args.sigma / 255, **policy)
            except NativeError as error:
                print(json.dumps({'failed': True, 'failure': str(error), 'stats': error.stats,
                                  'seconds': time.perf_counter() - start, 'maxrss_bytes': rss_bytes()}, allow_nan=False), flush=True)
                return
            times.append(time.perf_counter() - start)
            native_times.append(result.stats['total_seconds'])
            hashes = {name + '_sha256': digest64(getattr(result, name)) for name in ('output', 'pilot', 'dictionary')}
            frame_hashes.append(hashes)
            if frame + 1 < args.frames:
                del result
        activity = cpu_delta(ticks_before, cpu_ticks())
        after = resource.getrusage(resource.RUSAGE_SELF)
        seconds = sum(times) / len(times)
        if any(value != hashes for value in frame_hashes):
            raise AssertionError('within-row FP64 frame hashes differ')
        if command == 'warm':
            expected = hashes
            for name in ('output', 'pilot', 'dictionary'):
                np.save(Path(args.out) / (args.worker + '-' + name + '.npy'), getattr(result, name), allow_pickle=False)
            result.output.astype('<f4').tofile(Path(args.out) / (args.worker + '.f32'))
        if hashes != expected or not np.isfinite(result.output).all() or np.any(result.coverage == 0):
            raise AssertionError('repeat FP64 hashes/finite/coverage validation failed')
        row = dict(seconds=seconds, native_seconds=sum(native_times) / len(native_times), stats=result.stats,
                   maxrss_bytes=rss_bytes(), cpu_user_seconds=after.ru_utime - before.ru_utime,
                   cpu_system_seconds=after.ru_stime - before.ru_stime,
                   input_dictionary_sha256=digest64(dictionary), atoms=dictionary.shape[1],
                   repeat_fp64_exact=hashes == expected, frame_seconds=times, frames=args.frames,
                   cpu_activity=activity, affinity=affinity,
                   eligible_idle_sample=(eligible(activity) and all(v['ticks'] > 0 for v in activity.values()))
                       if activity is not None else None, **hashes)
        print(json.dumps(row, allow_nan=False), flush=True)
        del result


def summarize(rows, variants, require_c4=False):
    result = {}
    valid_pairs = set()
    if require_c4:
        for pair in {row['pair'] for row in rows if row['pair'] >= 0}:
            group = [row for row in rows if row['pair'] == pair]
            if (len(group) == len(variants) and {row['variant'] for row in group} == set(variants)
                    and all(not row.get('failed') and row.get('eligible_idle_sample') for row in group)):
                valid_pairs.add(pair)
    for name in variants:
        matching = [row for row in rows if row['variant'] == name]
        raw = [row for row in matching if row['pair'] >= 0 and not row.get('failed')]
        measured = [row for row in raw if not require_c4 or row['pair'] in valid_pairs]
        result[name] = {
            'complete_pairs': [row['pair'] for row in measured],
            'median_seconds': statistics.median(row['seconds'] for row in measured) if measured else None,
            'median_native_seconds': statistics.median(row['native_seconds'] for row in measured) if measured else None,
            'peak_rss_bytes': max((row.get('maxrss_bytes', 0) for row in matching), default=0),
            'repeat_fp64_exact': bool(measured) and all(row['repeat_fp64_exact'] for row in measured),
            'failures': [row for row in matching if row.get('failed')],
            'excluded_pairs': [row['pair'] for row in raw if require_c4 and row['pair'] not in valid_pairs],
        }
    baseline = variants[0]
    base_by_pair = {row['pair']: row for row in rows if row['variant'] == baseline and row['pair'] >= 0
                    and not row.get('failed') and (not require_c4 or row['pair'] in valid_pairs)}
    for name in variants:
        paired = [(base_by_pair[row['pair']]['seconds'], row['seconds']) for row in rows
                  if row['variant'] == name and row['pair'] in base_by_pair and not row.get('failed')]
        result[name]['paired_baseline'] = baseline
        result[name]['paired_speedup_median'] = statistics.median(a / b for a, b in paired) if paired else None
    return result


def sources():
    here = Path(__file__).resolve().parent
    files = [here / name for name in ['native.py', 'native_campaign.py', 'reference.py', 'p2_reference.py']]
    files += sorted(path for path in (here / 'native').iterdir() if path.is_file())
    files += [here.parents[1] / 'src/cpu/twsc/math.cpp', here.parents[1] / 'include/nss/cpu_twsc_full.hpp']
    files += [here.parents[1] / name for name in ['include/nss/cpu_lssc.hpp', 'src/cpu/lssc/gemm.hpp',
                                                 'src/cpu/lssc/gemm.cpp', 'src/cpu/lssc/gemm_sme.cpp']]
    files += [here.parents[1] / 'CMakeLists.txt', here.parents[1] / 'tests/CMakeLists.txt']
    return {str(path.relative_to(here.parents[1])): sha(path) for path in files}


def snapshot(out, library):
    root = Path(__file__).resolve().parents[2]
    if (root / '.git').exists():
        listing = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
        names = sorted(set(value.decode('utf8') for value in listing.split(b'\0') if value))
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    else:
        # Guest source archives have no Git metadata. Restrict the snapshot to
        # source directories; never include builds, venvs or external fixtures.
        names = []
        for name in ('CMakeLists.txt', 'README.md', 'LICENSE', 'NOTICE', '.gitignore',
                     'cmake', 'contracts', 'docs', 'include', 'src', 'tests', 'tools', '.github'):
            path = root / name
            files = path.rglob('*') if path.is_dir() else [path]
            names += [str(item.relative_to(root)) for item in files if item.is_file()
                      and '__pycache__' not in item.parts and item.suffix != '.pyc']
        names = sorted(set(names))
        revision = os.environ.get('NSS_REVISION', 'unversioned-source-archive')
    archive = out / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as package:
        for name in names:
            path = root / name
            if path.is_file():
                package.add(path, arcname=name, recursive=False)
    destination = out / library.name
    shutil.copy2(library, destination)
    return {'source_archive': archive.name, 'source_archive_sha256': sha(archive),
            'base_commit': revision,
            'library_file': destination.name}


def load_policies(path=None):
    policies = json.loads(Path(path).read_text()) if path else POLICIES
    if (not isinstance(policies, dict) or not policies or
            any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', name) or not isinstance(value, dict)
                for name, value in policies.items())):
        raise ValueError('invalid named policy set')
    return policies


def coordinator(args):
    if args.pairs < 1 or args.timeout <= 0 or not 0 < args.memory_gib <= 64 or not 1 <= args.frames <= 100:
        raise ValueError('invalid bounded campaign budget')
    if any(os.getenv(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')):
        raise ValueError('single-thread BLAS/OpenMP environment required')
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('duplicate campaign variants')
    policies = load_policies(args.policy_file)
    if any(name not in policies for name in args.variants):
        raise ValueError('variant missing from policy set')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    library = Path(args.library or default_library()).resolve()
    report = {
        'schema': 'nss.lssc-native-exploration.v1', 'complete': False, 'width': args.width,
        'height': args.height, 'sigma': args.sigma, 'pairs': args.pairs, 'timeout_seconds': args.timeout,
        'memory_limit_gib_per_worker': args.memory_gib, 'input_sha256': sha(args.input),
        'dictionary_file_sha256': sha(args.dictionary), 'library_sha256': sha(library),
        'source_sha256': sources(), 'policies': {name: policies[name] for name in args.variants},
        'frames_per_row': args.frames, 'timed_source_fills': 0,
        'platform': platform.platform(), 'python': sys.version,
        'timing_boundary': 'preloaded-input adapter call, including budget table, native preparation, learning, and output allocation; excludes hashes/serialization/metrics',
        'native_timing_boundary': 'native request validation/preparation through final FP64 output; excludes Python adapter',
        'host_gate': 'C4 CPU0, CPU1 SMT idle <=1%, zero steal; reject entire common pair' if args.require_cpu0 else
                     'local serial paired diagnostic; no CPU-affinity/idle-neighbor acceptance proof',
        'scope': 'experimental numerical/algorithm tradeoffs; NOT author/paper equivalence or default promotion',
        'rows': [],
    }
    helper = library.parents[3] / '_deps/highway-build/hwy_list_targets'
    if helper.is_file():
        probe = subprocess.run([str(helper)], text=True, capture_output=True, check=True)
        report['highway_diagnostic'] = probe.stdout
    report.update(snapshot(out, library))
    (out / 'policies.json').write_text(json.dumps(report['policies'], indent=2, allow_nan=False) + '\n')
    frozen_library = out / report['library_file']
    processes, logs, failed = {}, [], set()
    source_before = report['source_sha256']
    try:
        for name in args.variants:
            log = (out / (name + '.stderr.log')).open('w')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                       '--input', str(Path(args.input).resolve()), '--dictionary', str(Path(args.dictionary).resolve()),
                       '--width', str(args.width), '--height', str(args.height), '--sigma', str(args.sigma),
                       '--memory-gib', str(args.memory_gib), '--library', str(frozen_library), '--out', str(out),
                       '--frames', str(args.frames), '--worker-policy', json.dumps(policies[name], allow_nan=False)]
            if args.require_cpu0:
                command.append('--require-cpu0')
            processes[name] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=log, text=True, bufsize=1)
        for pair, index in schedule(args.pairs, len(args.variants)):
            name = args.variants[index]
            if name in failed:
                continue
            try:
                row = request(processes[name], 'warm' if pair == -1 else 'run', args.timeout, args.memory_gib)
            except (Exception, KeyboardInterrupt) as error:
                row = {'failed': True, 'failure': f'{type(error).__name__}: {error}'}
                processes[name].terminate()
                processes[name].wait(timeout=10)
                if isinstance(error, KeyboardInterrupt):
                    raise
            row.update(variant=name, pair=pair)
            report['rows'].append(row)
            if row.get('failed'):
                failed.add(name)
            report['summary'] = summarize(report['rows'], args.variants, args.require_cpu0)
            (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps({'variant': name, 'pair': pair, 'seconds': row.get('seconds'),
                              'failed': row.get('failed', False), 'failure': row.get('failure'),
                              'native_seconds': row.get('native_seconds')}, allow_nan=False), flush=True)
        report['timings_complete'] = not failed
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
        for log in logs:
            log.close()
        report['source_unchanged'] = source_before == sources()
        report['library_unchanged'] = report['library_sha256'] == sha(library)
        report['summary'] = summarize(report['rows'], args.variants, args.require_cpu0)
        (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    outputs = {name: np.load(out / (name + '-output.npy'), allow_pickle=False)
               for name in args.variants if (out / (name + '-output.npy')).exists()}
    baseline = outputs.get(args.variants[0])
    clean = read_image(args.clean, args.width, args.height) if args.clean else None
    for name, pixels in outputs.items():
        row = report['summary'][name]
        row['metrics'] = metrics(clean, pixels) if clean is not None else None
        if baseline is not None:
            row['max_abs_vs_baseline'] = float(np.max(np.abs(pixels - baseline)))
            row['rmse_vs_baseline'] = float(np.sqrt(np.mean((pixels - baseline)**2)))
            if clean is not None:
                row['psnr_delta_vs_baseline'] = row['metrics']['psnr_db'] - metrics(clean, baseline)['psnr_db']
        warm = next(item for item in report['rows'] if item['variant'] == name and item['pair'] == -1)
        if digest64(pixels) != warm.get('output_sha256'):
            raise AssertionError('serialized warm output hash mismatch')
    if clean is not None:
        report['clean_sha256'] = sha(args.clean)
        panels = [('Clean', clean)] + [(name, image) for name, image in outputs.items()]
        for first in range(1, len(panels), 5):
            preview(out / f'comparison-{first:02d}.png', panels[:1] + panels[first:first + 5])
    report['complete'] = bool(report.get('timings_complete') and report['source_unchanged'] and report['library_unchanged'])
    report['all_variants_have_eligible_pairs'] = all(value['complete_pairs'] for value in report['summary'].values())
    (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2, allow_nan=False), flush=True)
    return 0 if report['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'dictionary', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--clean')
    parser.add_argument('--library')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--variants', nargs='+', default=['fp64_direct', 'fp64_gram', 'mixed_gram', 'mixed_ssd'])
    parser.add_argument('--policy-file')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--memory-gib', type=float, default=2)
    parser.add_argument('--worker')
    parser.add_argument('--worker-policy')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--require-cpu0', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    return coordinator(args)


if __name__ == '__main__':
    raise SystemExit(main())
