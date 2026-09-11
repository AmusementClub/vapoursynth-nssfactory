#!/usr/bin/env python3
"""Stream a frozen NLH cohort through public, warmed filter requests.

Only one original/case is held at once, so 512 crops and full originals do not
accumulate in the VapourSynth input cache. Coefficients are never fitted here.
An empty profile evaluates the actual runtime defaults, including blind preset
selection from the estimated working-space sigma.
"""
import argparse
import json
from pathlib import Path

from nlh_defaults_inputs import file_sha, save_json, validate_manifest
from nlh_defaults_search import (Engine, LANES, evaluation_cases, identity,
                                 lane_cases, summarize_measurements)


def cohort_cases(manifest, lane, split, controls_only=False):
    if not controls_only:
        return lane_cases(manifest, lane, split)
    if split != 'development':
        raise ValueError('legacy control images belong only to development')
    cases = [c for c in manifest['cases'] if c['split'] == split and c['control']]
    if lane == 'rgb':
        return [c for c in cases if c['format'] == 'rgb']
    return [c for c in cases if c['format'] == 'gray' and
            (c['sigma'] <= 50 if lane == 'gray-low' else c['sigma'] > 50)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plugin', 'inputs', 'out', 'profiles'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--lane', choices=LANES, required=True)
    parser.add_argument('--split', choices=('development', 'selection', 'test'), required=True)
    parser.add_argument('--sigma-mode', choices=('explicit', 'blind'), default='explicit')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--score-size', type=int, default=96)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--controls-only', action='store_true',
                        help='Evaluate legacy development controls separately from every selection/test cohort')
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--measurements', type=int, default=3)
    parser.add_argument('--bm3d', action='store_true')
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()
    if args.bm3d and args.sigma_mode == 'blind':
        parser.error('BM3D is evaluated in the explicit same-sigma cohort')
    args.pixels_only = True
    manifest = json.loads((Path(args.inputs) / 'inputs.json').read_text())
    validate_manifest(manifest)
    try:
        cases = cohort_cases(manifest, args.lane, args.split, args.controls_only)
    except ValueError as error:
        parser.error(str(error))
    if args.cases:
        requested = set(args.cases)
        cases = [c for c in cases if c['id'] in requested]
        if {c['id'] for c in cases} != requested:
            parser.error('requested cases do not belong to this lane/split')
    cases = evaluation_cases(cases, args.sigma_mode)
    profiles = json.loads(Path(args.profiles).read_text())['profiles']
    if not profiles or any('sigma' in p for p in profiles.values()):
        parser.error('profiles must be nonempty and cannot tune sigma')
    algorithms = [(label, 'NLH', p) for label, p in profiles.items()]
    if args.bm3d:
        algorithms.append(('BM3D-same-sigma', 'BM3D', {}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procedure = dict(driver_sha256=file_sha(__file__), split=args.split,
                     controls_only=args.controls_only,
                     case_order=[c['id'] for c in cases],
                     parameters='frozen; no refit',
                     timing='public preloaded requests; rotate profile order by case; warmups and repetitions recorded in metadata')
    stamp = out / 'evaluation-procedure.json'
    if stamp.exists() and json.loads(stamp.read_text()) != procedure:
        raise ValueError('evaluation procedure changed; use a new directory')
    save_json(stamp, procedure)
    engine = Engine(args, cases)
    completed = {label: [] for label, _, _ in algorithms}

    def persist():
        rows = []
        for label, algorithm, profile in algorithms:
            measurements = completed[label]
            if not measurements:
                continue
            row = dict(id=identity(dict(profile=profile, algorithm=algorithm)),
                       label=label, algorithm=algorithm, parameters=profile,
                       cases=[r['id'] for r in measurements])
            row.update(summarize_measurements(measurements))
            rows.append(row)
        save_json(out / 'evaluation.json', dict(rows=rows, metadata=engine.metadata,
                  procedure=procedure, complete=all(len(completed[label]) == sum(
                      c['sigma'] is not None or algorithm != 'BM3D' for c in cases)
                      for label, algorithm, _ in algorithms)))

    try:
        for index, case in enumerate(cases):
            order = algorithms[index % len(algorithms):] + algorithms[:index % len(algorithms)]
            for label, algorithm, profile in order:
                if algorithm == 'BM3D' and case['sigma'] is None:
                    continue
                row = engine.evaluate_case(profile, case, algorithm)
                completed[label].append(row)
                print(index + 1, len(cases), case['id'], label,
                      row['quality']['psnr_db'], row['seconds'], flush=True)
            persist()
            engine.close()
    finally:
        engine.close()


if __name__ == '__main__':
    main()
