#!/usr/bin/env python3
"""Replay public BM3D two-stage cases with either backend's frozen pilot.

This identifies propagation from the reference image. It does not by itself
admit pilot errors, matching changes, or quality loss against clean images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare


def worker(args):
    import vapoursynth as vs
    from test_plan01_plugin import clip, evaluate, options
    core = vs.core
    core.num_threads = 1
    core.max_cache_size = 32
    core.std.LoadPlugin(path=str(args.plugin.resolve()))
    case = json.loads(args.case.read_text())
    if case['algorithm'] != 'BM3D':
        raise ValueError('BM3D crossover only')
    source = clip(core, width=36, height=34, frames=3, parent_extra=64)
    mode = case['mode']
    radius = 0 if mode == 'spatial' else 1
    kw = options('BM3D', case['block'], case['group'], radius=radius)
    if mode == 'rolling':
        kw.update(temporal_mode='rolling', rolling_chunk=1, rolling_cache_limit=1)

    def bm(reference=None):
        node = core.nss.BM3D(source, **kw, **({'ref': reference} if reference is not None else {}))
        return core.nss.VAggregate(node, source, radius=radius) if mode == 'legacy' else node

    if args.reference:
        values = np.load(args.reference)
        def fill(n, f):
            out = f.copy()
            np.asarray(out[0])[:] = values[f'n{n}']
            return out
        reference = core.std.ModifyFrame(source, source, fill)
        node = bm(reference)
    elif args.pilot or case['stage'] == 'basic':
        node = bm()
    elif case['stage'] == 'final':
        node = bm(clip(core, width=36, height=34, frames=3, parent_extra=128))
    else:
        node = bm(bm())
    core.std.SetVideoCache(node, mode=0)
    arrays = {f'n{n}': evaluate(node, n)[0][0] for n in range(3)}
    np.savez_compressed(args.out, **arrays)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    left, right = [json.loads((p / 'summary.json').read_text()) for p in (args.left_capture, args.right_capture)]
    original_pixels = [np.load(p / 'pixels.npz') for p in (args.left_capture, args.right_capture)]
    for metadata, plugin in zip((left, right), (args.baseline, args.candidate)):
        if metadata['plugin_sha256'] != hashlib.sha256(plugin.read_bytes()).hexdigest():
            raise RuntimeError('plugin differs from original capture')
    result = dict(schema='nssfactory.public-crossover.v1', numerical_admission=False, cases=[])

    def invoke(plugin, casepath, output, *, pilot=False, reference=None):
        command = [sys.executable, __file__, 'worker', '--plugin', str(plugin.resolve()),
                   '--case', str(casepath), '--out', str(output)]
        if pilot: command.append('--pilot')
        if reference: command += ['--reference', str(reference)]
        process = subprocess.run(command, capture_output=True, text=True)
        output.with_suffix('.log').write_text(process.stdout + process.stderr)
        process.check_returncode()
        values = np.load(output)
        return np.stack([values[f'n{n}'] for n in range(3)])

    for row in left['cases']:
        case = row['parameters']; index = row['index']
        if case['algorithm'] != 'BM3D' or case['stage'] != 'two-stage': continue
        arrays = [np.stack([p[f'c{index}_n{n}_p0'] for n in range(3)]) for p in original_pixels]
        original = compare(*arrays)
        if original['passed']: continue
        directory = args.out / f'c{index}'; directory.mkdir()
        casepath = directory / 'case.json'; casepath.write_text(json.dumps(case, indent=2))
        pilots, repeated = {}, {}
        for side, plugin, expected in zip(('baseline', 'candidate'), (args.baseline, args.candidate), arrays):
            actual = invoke(plugin, casepath, directory / f'{side}-full.npz')
            repeated[side] = bool(np.array_equal(actual, expected))
            pilots[side] = invoke(plugin, casepath, directory / f'{side}-pilot.npz', pilot=True)
        frozen = {}
        for reference in ('baseline', 'candidate'):
            outputs = [invoke(plugin, casepath, directory / f'{side}-using-{reference}.npz',
                              reference=directory / f'{reference}-pilot.npz')
                       for side, plugin in (('baseline', args.baseline), ('candidate', args.candidate))]
            frozen[reference] = compare(*outputs)
        item = dict(index=index, parameters=case, original=original, reproduced=repeated,
                    pilot=compare(pilots['baseline'], pilots['candidate']), fixed_reference=frozen,
                    propagation_isolated=all(repeated.values()) and all(v['passed'] for v in frozen.values()))
        result['cases'].append(item)
        print(json.dumps(dict(index=index, original=original['max_abs'], pilot=item['pilot']['max_abs'],
                              fixed=[v['max_abs'] for v in frozen.values()], reproduced=repeated)), flush=True)
        (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    result['hashes'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [args.baseline, args.candidate, Path(__file__), *(args.out.glob('c*/*.npz'))]}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker')
    for name in ('plugin', 'case', 'out'): w.add_argument('--' + name, type=Path, required=True)
    w.add_argument('--pilot', action='store_true'); w.add_argument('--reference', type=Path)
    r = sub.add_parser('run')
    for name in ('baseline', 'candidate', 'left-capture', 'right-capture', 'out'):
        r.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    worker(args) if args.command == 'worker' else run(args)
