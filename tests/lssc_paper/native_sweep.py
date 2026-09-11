#!/usr/bin/env python3
"""Apply a frozen exploration policy set serially to existing hashed fixtures."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
from types import SimpleNamespace

from native_campaign import coordinator, load_policies
from run import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--dictionaries', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy-file')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--memory-gib', type=float, default=2)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--require-cpu0', action='store_true')
    parser.add_argument('--library')
    args = parser.parse_args()
    manifest = Path(args.fixtures).resolve()
    fixtures = json.loads(manifest.read_text())['cases']
    policies = load_policies(args.policy_file)
    variants = args.variants or list(policies)
    selected = [case for case in fixtures if not args.cases or case['id'] in args.cases]
    if not selected or (args.cases and set(args.cases) != {case['id'] for case in selected}):
        raise ValueError('requested cases absent from fixture manifest')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen_policies = out / 'policies.json'
    frozen_policies.write_text(json.dumps({name: policies[name] for name in variants}, indent=2, allow_nan=False) + '\n')
    report = {'schema': 'nss.native-sweep.v1', 'complete': False, 'fixture_manifest_sha256': sha(manifest),
              'policies': {name: policies[name] for name in variants}, 'cases': []}
    for case in selected:
        name = case['id']
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name):
            raise ValueError('unsafe fixture name')
        paths = {}
        for kind in ('noisy', 'clean'):
            path = (manifest.parent / case[kind]).resolve()
            if not path.is_relative_to(manifest.parent) or sha(path) != case[kind + '_sha256']:
                raise ValueError('fixture containment/hash check failed')
            paths[kind] = path
        sigma = case['sigma']
        block = case.get('block', 9 if sigma <= 25 else 12 if sigma <= 50 else 16)
        call = SimpleNamespace(input=str(paths['noisy']), clean=str(paths['clean']), width=case['width'],
                               height=case['height'], sigma=sigma,
                               dictionary=str(Path(args.dictionaries) / f'dict_n{block}.mat'),
                               out=str(out / name), variants=variants, policy_file=str(frozen_policies),
                               pairs=args.pairs, timeout=args.timeout, memory_gib=args.memory_gib,
                               frames=args.frames, require_cpu0=args.require_cpu0, library=args.library)
        print(json.dumps({'case': name, 'state': 'started', 'block': block}), flush=True)
        try:
            with (out / (name + '.log')).open('w') as log, redirect_stdout(log):
                code = coordinator(call)
            result = json.loads((out / name / 'result.json').read_text())
            row = {'case': name, 'complete': code == 0, 'block': block, 'summary': result['summary']}
        except Exception as error:
            row = {'case': name, 'complete': False, 'failure': f'{type(error).__name__}: {error}'}
        report['cases'].append(row)
        (out / 'sweep.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'case': name, 'complete': row['complete'], 'failure': row.get('failure'),
                          'summary': {key: {'seconds': value['median_seconds'],
                                             'psnr_db': value.get('metrics', {}).get('psnr_db'),
                                             'delta_db': value.get('psnr_delta_vs_baseline')}
                                      for key, value in row.get('summary', {}).items()}}, allow_nan=False), flush=True)
    report['complete'] = all(row['complete'] for row in report['cases'])
    (out / 'sweep.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
