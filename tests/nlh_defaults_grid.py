#!/usr/bin/env python3
"""Complete the legal block/step grid around a frozen development shortlist.

This is a development-only sensitivity screen. The selected structures are
refitted and evaluated on larger, disjoint image sets before default selection.
"""
import argparse
import copy
import json
from pathlib import Path

from nlh_defaults_inputs import file_sha, save_json, validate_manifest
from nlh_defaults_search import (Engine, LANES, baseline, change, identity,
                                 lane_cases, legal, nondominated, shortlist,
                                 strength_variants, fitted_structures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plugin', 'inputs', 'out', 'profiles'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--native-bridge', required=True)
    parser.add_argument('--lane', choices=LANES, required=True)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--score-size', type=int, default=48)
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()
    profiles = json.loads(Path(args.profiles).read_text())['profiles']
    if 'balanced' not in profiles or any(not legal(p) for p in profiles.values()):
        parser.error('a legal, frozen balanced profile is required')
    anchor = copy.deepcopy(profiles['balanced'])
    # Retain context for the largest tested patch and the full search window.
    # Multi-pass boundary propagation is subsequently checked on larger images.
    required = args.score_size + max(max(p['search_window']) for p in profiles.values()) - 1 + 2 * 15
    if args.size < required:
        parser.error(f'input size must be at least {required} for this anchor')
    manifest = json.loads((Path(args.inputs) / 'inputs.json').read_text())
    validate_manifest(manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procedure = dict(driver_sha256=file_sha(__file__), anchor=anchor,
                     scope='all legal block/step pairs, each stage separately; shared coefficient refit')
    stamp = out / 'grid.json'
    if stamp.exists() and json.loads(stamp.read_text()) != procedure:
        raise ValueError('grid procedure changed; use a new output directory')
    save_json(stamp, procedure)
    engine = Engine(args, lane_cases(manifest, args.lane, 'development', screen=True))
    rows, coverage = {}, []

    def evaluate(p, label):
        key = identity(p)
        if key not in rows:
            rows[key] = engine.evaluate(p, label)
            save_json(out / 'trials.json', list(rows.values()))
        return rows[key]

    try:
        evaluate(baseline(args.lane), 'current-defaults')
        for label, p in profiles.items():
            evaluate(p, 'seed-' + label)
        for stage in (0, 1):
            for block in range(2, 17):
                for step in range(1, block + 1):
                    p = change(change(anchor, 'block_size', block, stage),
                               'block_step', step, stage)
                    if not legal(p):
                        raise AssertionError('legal block/step grid construction failed')
                    fitted = [evaluate(v, f'grid-{stage}-{block}-{step}')
                              for v in strength_variants(p)]
                    best = max(fitted, key=lambda r: (r['mean_psnr'], r['mean_ssim']))
                    coverage.append(dict(stage=stage, block=block, step=step,
                                         trials=[r['id'] for r in fitted], best=best['id']))
                save_json(out / 'coverage.json', coverage)
                print(args.lane, 'grid', stage, block, 'pairs', len(coverage),
                      'trials', len(rows), flush=True)
        expected = {(s, b, t) for s in (0, 1) for b in range(2, 17)
                    for t in range(1, b + 1)}
        actual = {(r['stage'], r['block'], r['step']) for r in coverage}
        if actual != expected:
            raise AssertionError('incomplete block/step coverage')
        chosen = shortlist(list(rows.values()))
        save_json(out / 'summary.json', dict(lane=args.lane, metadata=engine.metadata,
                  selected=chosen, frontier=nondominated(fitted_structures(list(rows.values()))),
                  trials=len(rows), block_step_complete=True, pairs=len(coverage)))
        save_json(out / 'profiles.json', dict(profiles={k: v['parameters'] for k, v in chosen.items()},
                  selection='development screen only; larger-image refit remains required'))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
