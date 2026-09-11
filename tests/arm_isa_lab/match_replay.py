#!/usr/bin/env python3
"""Causal matcher replay for retained public-matrix numerical differences."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare


def records(path):
    data = path.read_bytes(); cursor = 0; result = []
    while cursor < len(data):
        values = struct.unpack_from('<9iQ', data, cursor); cursor += 44
        header, input_hash = values[:9], values[9]
        if header[8] < 0 or header[8] > 1048576: raise ValueError('invalid trace length')
        payload = np.frombuffer(data, dtype='<u4', count=header[8], offset=cursor).copy()
        cursor += header[8] * 4
        result.append((header, input_hash, payload))
    if cursor != len(data): raise ValueError('truncated trace')
    return result


def trace_delta(left, right):
    a, b = records(left), records(right)
    if len(a) != len(b): return dict(aligned=False, counts=[len(a), len(b)])
    changes = []; different_inputs = 0; distance_only = 0
    for index, ((ha, ia, pa), (hb, ib, pb)) in enumerate(zip(a, b)):
        if ha[:7] != hb[:7]: return dict(aligned=False, first_geometry_mismatch=index)
        different_inputs += ia != ib
        identity = ha[7:] == hb[7:] and (
            np.array_equal(pa, pb) if ha[0] == 6 else np.array_equal(pa.reshape(-1, 5)[:, :4], pb.reshape(-1, 5)[:, :4]))
        if not identity: changes.append(dict(call=index, kind=ha[0], query=list(ha[1:4]), input_hash_equal=ia == ib))
        elif not np.array_equal(pa, pb): distance_only += 1
    return dict(aligned=True, calls=len(a), changed_input_hashes=int(different_inputs),
                changed_match_calls=len(changes), distance_only_calls=distance_only, changes=changes)


def worker(args):
    import vapoursynth as vs
    from test_neon_matrix import graph
    from test_plan01_plugin import evaluate
    core = vs.core; core.num_threads = 1; core.max_cache_size = 32
    core.std.LoadPlugin(path=str(args.plugin.resolve()))
    case = json.loads(args.case.read_text())
    node = graph(core, case)
    arrays = {}
    for n in range(3):
        for p, array in enumerate(evaluate(node, n)[0]): arrays[f'n{n}_p{p}'] = array
    np.savez_compressed(args.out, **arrays)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.comparison.read_text())
    pixels = [np.load(p / 'pixels.npz') for p in (args.left_capture, args.right_capture)]
    report = dict(schema='nssfactory.match-replay.v1', numerical_admission=False, cases=[])

    def invoke(plugin, directory, casepath, name, replay=None):
        output = directory / (name + '.npz'); record = directory / (name + '.bin')
        environment = dict(os.environ, NSS_MATCH_RECORD=str(record))
        environment.pop('NSS_MATCH_REPLAY', None)
        if replay: environment['NSS_MATCH_REPLAY'] = str(replay)
        command = [sys.executable, __file__, 'worker', '--plugin', str(plugin.resolve()),
                   '--case', str(casepath), '--out', str(output)]
        proc = subprocess.run(command, env=environment, text=True, capture_output=True)
        output.with_suffix('.log').write_text(proc.stdout + proc.stderr)
        proc.check_returncode()
        meta = json.loads(Path(str(record) + '.meta.json').read_text())
        if not meta['replay_consumed'] or meta['calls'] == 0: raise RuntimeError('incomplete/unexercised replay')
        return np.load(output), record

    for row in original['results']:
        case = row['parameters']; index = row['index']
        if case['algorithm'] in ('LSSC', 'BM3D', 'NLM'): continue
        if row['max_abs'] <= 1e-5 and math.sqrt(row['mse']) <= 1e-6: continue
        directory = args.out / f'c{index}'; directory.mkdir()
        casepath = directory / 'case.json'; casepath.write_text(json.dumps(case, indent=2))
        actual, logs, reproduced = {}, {}, {}
        for side, plugin, expected in zip(('baseline', 'candidate'), (args.baseline, args.candidate), pixels):
            actual[side], logs[side] = invoke(plugin, directory, casepath, side)
            reproduced[side] = all(np.array_equal(array, expected[f'c{index}_{key}']) for key, array in actual[side].items())
        keys = sorted(actual['baseline'].files)
        # Preserve image axes so the shared diagnostic SSIM uses spatial
        # windows independently in each plane/frame.
        flattened = lambda data: np.stack([data[key] for key in keys])
        fixed = {}
        for reference in ('baseline', 'candidate'):
            values = [invoke(plugin, directory, casepath, side + '-using-' + reference, logs[reference])[0]
                      for side, plugin in (('baseline', args.baseline), ('candidate', args.candidate))]
            fixed[reference] = compare(*map(flattened, values))
        change = trace_delta(logs['baseline'], logs['candidate'])
        item = dict(index=index, parameters=case, reproduced=reproduced, trace=change,
                    original=compare(*[flattened(actual[s]) for s in ('baseline', 'candidate')]),
                    frozen_matching=fixed, propagation_isolated=all(reproduced.values()) and
                    change.get('aligned', False) and all(v['passed'] for v in fixed.values()))
        report['cases'].append(item)
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(index=index, reproduced=reproduced, original=item['original']['max_abs'],
                              changed_match_calls=change.get('changed_match_calls'),
                              fixed=[v['max_abs'] for v in fixed.values()])), flush=True)
    report['hashes'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [args.baseline, args.candidate, Path(__file__), *(args.out.glob('c*/*.bin')), *(args.out.glob('c*/*.npz'))]}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker')
    for n in ('plugin', 'case', 'out'): w.add_argument('--' + n, type=Path, required=True)
    r = sub.add_parser('run')
    for n in ('baseline', 'candidate', 'comparison', 'left-capture', 'right-capture', 'out'):
        r.add_argument('--' + n, type=Path, required=True)
    args = parser.parse_args()
    worker(args) if args.command == 'worker' else run(args)
