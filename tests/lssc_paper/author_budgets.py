#!/usr/bin/env python3
"""Single-factor author learning-budget ladder on one preloaded input.

External academic reference only. The same MEX/dictionary/window/thread settings
are held fixed while J1/J2 vary. This does not implement or promote P3 in NSS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from author_probe import quote
from benchmark import cpu_delta, eligible
from run import metrics, read_image, sha


POLICIES = [('fixed_request', 0, 0), ('image5', 5, 0), ('image20', 20, 0), ('full', 20, 5)]


def schedule(pairs, count):
    return [(-1, index) for index in range(count)] + [
        (pair, index) for pair in range(pairs)
        for index in (range(count) if pair % 2 == 0 else reversed(range(count)))]


def parse_cpu(text):
    result = {}
    for line in str(text).splitlines():
        fields = line.split()
        if fields and fields[0] in ('cpu0', 'cpu1'):
            result[fields[0]] = list(map(int, fields[1:9]))
    if set(result) != {'cpu0', 'cpu1'}:
        raise ValueError('missing per-call CPU0/CPU1 telemetry')
    return result


def make_driver(root, temporary, out, *, sigma, block, pairs, policies):
    order = schedule(pairs, len(policies))
    matrix = ';'.join(f'{pair} {index+1} {policies[index][1]} {policies[index][2]}' for pair, index in order)
    return f"""more off;
cd({quote(root)}); addpath({quote(temporary)});
load({quote(Path(temporary)/'inputs.mat')});
Ih=hash('sha256',num2hex(I(:))(:)'); Dh=hash('sha256',num2hex(D(:))(:)');
schedule=[{matrix}];
for job=1:rows(schedule)
  pair=schedule(job,1); variant=schedule(job,2); j1=schedule(job,3); j2=schedule(job,4);
  rand('state',0); randn('state',0);
  fprintf('NSS_BUDGET_BEGIN %d %d %d %d %d\\n',job,pair,variant,j1,j2); fflush(stdout);
  before=fileread('/proc/stat');
  tic;
  [result,Dout]=mexDenoise(I,I,D,{sigma:.17g},{block},80,j1,j2,{(32*sigma)**2:.17g},32,10000,1,1,80);
  seconds=toc;
  after=fileread('/proc/stat');
  assert(strcmp(Ih,hash('sha256',num2hex(I(:))(:)')));
  assert(strcmp(Dh,hash('sha256',num2hex(D(:))(:)')));
  assert(all(size(result)==size(I)) && all(isfinite(result(:))));
  dictionary_change=max(abs(Dout(:)-D(:)));
  output_path=sprintf({quote(str(Path(out)/'job-%02d.f64'))},job);
  f=fopen(output_path,'wb'); fwrite(f,result','double'); fclose(f);
  dictionary_path=sprintf({quote(str(Path(out)/'dictionary-%02d.f64'))},job);
  f=fopen(dictionary_path,'wb'); fwrite(f,Dout,'double'); fclose(f);
  save('-mat7-binary',sprintf({quote(str(Path(out)/'job-%02d.mat'))},job),
       'seconds','dictionary_change','before','after','pair','variant','j1','j2');
  fprintf('NSS_BUDGET_END %d %.17g %.17g\\n',job,seconds,dictionary_change); fflush(stdout);
end
"""


def summarize(out, order, policies, image_shape, clean=None):
    rows = []
    warm = {}
    for job, (pair, index) in enumerate(order, 1):
        path = out / f'job-{job:02d}.mat'
        if not path.exists():
            continue
        data = loadmat(path, simplify_cells=True)
        if int(data['pair']) != pair or int(data['variant']) != index + 1:
            raise AssertionError('driver scheduling mismatch')
        output = out / f'job-{job:02d}.f64'
        dictionary = out / f'dictionary-{job:02d}.f64'
        pixels = np.fromfile(output, dtype='<f8').reshape(image_shape)
        if not np.isfinite(pixels).all():
            raise ValueError('nonfinite author pixels')
        activity = cpu_delta(parse_cpu(data['before']), parse_cpu(data['after']))
        row = {'job': job, 'pair': pair, 'variant': policies[index][0],
               'j1': int(data['j1']), 'j2': int(data['j2']),
               'seconds': float(data['seconds']), 'output_sha256': sha(output),
               'dictionary_sha256': sha(dictionary),
               'dictionary_max_abs_change': float(data['dictionary_change']),
               'cpu_activity': activity, 'eligible_idle_sample': eligible(activity),
               'metrics': metrics(clean, pixels) if clean is not None else None}
        if pair == -1:
            warm[row['variant']] = row
        else:
            baseline = warm[row['variant']]
            row['output_matches_warm'] = row['output_sha256'] == baseline['output_sha256']
            row['dictionary_matches_warm'] = row['dictionary_sha256'] == baseline['dictionary_sha256']
        rows.append(row)
    medians = {}
    for name, _, _ in policies:
        valid = [r for r in rows if r['variant'] == name and r['pair'] >= 0 and r['eligible_idle_sample']]
        medians[name] = statistics.median(r['seconds'] for r in valid) if valid else None
    return rows, medians


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'dictionary', 'author-root', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--clean')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--policies', nargs='+', choices=[p[0] for p in POLICIES],
                        default=[p[0] for p in POLICIES])
    parser.add_argument('--timeout', type=float, default=2400)
    args = parser.parse_args()
    if args.pairs < 1 or not np.isfinite(args.sigma) or args.sigma <= 0 or args.timeout <= 0:
        parser.error('positive pairs, noise and timeout required')
    if not hasattr(os, 'sched_getaffinity') or sorted(os.sched_getaffinity(0)) != [0]:
        parser.error('run with taskset -c 0 on the verified C4 host')
    if any(os.getenv(name) != '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')):
        parser.error('single-thread BLAS/OpenMP required')
    policies = [p for p in POLICIES if p[0] in args.policies]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.author_root).resolve()
    image = read_image(args.input, args.width, args.height)
    dictionary = loadmat(args.dictionary, variable_names=['D'])['D']
    block = int(np.sqrt(dictionary.shape[0]))
    if block * block != dictionary.shape[0] or min(image.shape) < block:
        parser.error('invalid dictionary or input shape')
    binary = root / 'mexDenoise.mexa64'
    order = schedule(args.pairs, len(policies))
    report = {'schema': 'nss.author-learning-budget.v1', 'complete': False,
              'input_sha256': sha(args.input), 'dictionary_sha256': sha(args.dictionary),
              'binary_sha256': sha(binary), 'harness_sha256': sha(__file__),
              'policies': policies, 'input_shape': list(image.shape), 'sigma': args.sigma,
              'block': block, 'window': 32, 'threads': 1, 'pairs': args.pairs,
              'affinity': sorted(os.sched_getaffinity(0)), 'timed_source_fills': 0,
              'timing_boundary': 'warm mexDenoise call; input IO, hashes, telemetry, serialization excluded',
              'telemetry_boundary': '/proc/stat snapshots immediately surrounding each MEX call',
              'scope': 'author binary learning-budget comparison, not new NSS code or MATLAB/paper equivalence'}
    try:
        with tempfile.TemporaryDirectory(prefix='nss-lssc-budget-') as temporary:
            temporary = Path(temporary)
            (temporary / 'mexDenoise.mex').symlink_to(binary)
            savemat(temporary / 'inputs.mat', {'I': image, 'D': dictionary})
            script = make_driver(root, temporary, out, sigma=args.sigma / 255, block=block,
                                 pairs=args.pairs, policies=policies)
            (out / 'driver.m').write_text(script)
            with (out / 'author.log').open('w') as log:
                completed = subprocess.run(['octave', '--no-gui', '--quiet', str(out / 'driver.m')],
                                           stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if completed.returncode:
                raise RuntimeError(f'Octave exit {completed.returncode}; see author.log')
        report['complete'] = True
    except (Exception, KeyboardInterrupt) as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        clean = read_image(args.clean, args.width, args.height) if args.clean else None
        rows, medians = summarize(out, order, policies, image.shape, clean)
        report['rows'], report['eligible_median_seconds'] = rows, medians
        report['all_jobs_recorded'] = len(rows) == len(order)
        report['repeat_outputs_and_dictionaries_exact'] = all(
            r.get('output_matches_warm', True) and r.get('dictionary_matches_warm', True) for r in rows)
        report['zero_learning_verified'] = all(r['dictionary_max_abs_change'] == 0
                                               for r in rows if r['j1'] == r['j2'] == 0)
        report['maxrss_octave_bytes'] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss *
                                           (1 if sys.platform == 'darwin' else 1024))
        (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': report['complete'], 'medians': medians,
                      'repeats_exact': report['repeat_outputs_and_dictionaries_exact']}, indent=2))


if __name__ == '__main__':
    main()
