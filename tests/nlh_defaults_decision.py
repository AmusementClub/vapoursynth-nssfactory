#!/usr/bin/env python3
"""Compare frozen finalists with paired original-image bootstrap intervals.

All sigma levels and both noise realizations of an original stay together in a
bootstrap draw. Datasets receive equal weight. CC intervals are conditional on
the sampled photographs/camera; they do not estimate between-camera variation.
The proposed simplification still requires warm paired performance confirmation.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from nlh_defaults_inputs import file_sha, save_json
from nlh_defaults_search import fitted_structures, nondominated, shortlist


def paired_interval(candidate, reference, observations, draws=10000):
    def indexed(trial):
        return {observations[key].get('base_case', observations[key]['case']): observations[key]
                for key in trial['cases']}
    actual, expected = indexed(candidate), indexed(reference)
    if actual.keys() != expected.keys():
        raise ValueError('paired candidates have different cohorts')
    strata = {}
    for case, left in actual.items():
        right = expected[case]
        for field in ('input_sha256', 'shape', 'score_size', 'sigma_mode'):
            if left[field] != right[field]:
                raise ValueError('paired candidate mismatch: '+field)
        delta = [left['quality']['psnr_db']-right['quality']['psnr_db'],
                 left['quality']['ssim']-right['quality']['ssim'],
                 math.log(left['seconds']/right['seconds'])]
        strata.setdefault(left['dataset'], {}).setdefault(left['image'], []).append(delta)
    rng = np.random.default_rng(20260909)
    bootstrap, central, counts = [], [], {}
    for dataset, images in sorted(strata.items()):
        values = np.array([np.mean(rows, axis=0) for _, rows in sorted(images.items())])
        counts[dataset] = len(values)
        samples = rng.integers(len(values), size=(draws, len(values)))
        bootstrap.append(np.mean(values[samples], axis=1))
        central.append(np.mean(values, axis=0))
    mean = np.mean(central, axis=0)
    limits = np.quantile(np.mean(bootstrap, axis=0), [.025, .975], axis=0)
    return dict(originals=counts, draws=draws,
                psnr_delta=dict(mean=float(mean[0]), ci95=limits[:, 0].tolist()),
                ssim_delta=dict(mean=float(mean[1]), ci95=limits[:, 1].tolist()),
                latency_ratio=dict(geomean=math.exp(mean[2]), ci95=np.exp(limits[:, 2]).tolist()))


def complexity(row):
    p = row['parameters']
    return (sum(p[k][0] != p[k][1] for k in ('block_size', 'block_step', 'q', 'group_size', 'search_window')),
            p['basic_iters']+p['wiener_iters'], sum(a*b for a, b in zip(p['q'], p['group_size'])))


def propose(rows, observations):
    candidates = nondominated(fitted_structures([r for r in rows if r['algorithm'] == 'NLH']))
    knee = shortlist(candidates)['balanced']
    comparisons = []
    eligible = [knee]
    for row in candidates:
        if row['id'] == knee['id']:
            continue
        result = paired_interval(row, knee, observations)
        compatible = all(result[key]['mean'] >= 0 or result[key]['ci95'][0] <= 0 <= result[key]['ci95'][1]
                         for key in ('psnr_delta', 'ssim_delta'))
        faster = result['latency_ratio']['ci95'][1] < 1
        comparisons.append(dict(candidate=row['id'], reference=knee['id'],
                                quality_compatible=compatible, faster_in_this_cohort=faster, **result))
        if compatible and faster:
            eligible.append(row)
    fastest = min(eligible, key=lambda r: (r['geomean_seconds'], r['id']))
    similar_cost = [fastest]
    for row in eligible:
        if row is fastest:
            continue
        interval = paired_interval(row, fastest, observations)['latency_ratio']['ci95']
        if interval[0] <= 1 <= interval[1]:
            similar_cost.append(row)
    chosen = min(similar_cost, key=lambda r: (complexity(r), r['geomean_seconds'], -r['mean_psnr'], r['id']))
    return dict(knee=knee, proposed=chosen, comparisons=comparisons,
                eligible=[r['id'] for r in eligible], similar_cost=[r['id'] for r in similar_cost],
                status='provisional until representative warm paired timing is confirmed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ranking = json.loads(Path(args.ranking).read_text())
    observations = {r['id']: r for line in (root/'observations.jsonl').read_text().splitlines()
                    if (r := json.loads(line))}
    decision = propose(ranking['rows'], observations)
    decision.update(procedure=__doc__, driver_sha256=file_sha(__file__),
                    ranking_sha256=file_sha(args.ranking),
                    observations_sha256=file_sha(root/'observations.jsonl'))
    save_json(out/'decision.json', decision)
    print(json.dumps(dict(knee=decision['knee']['label'], proposed=decision['proposed']['label'],
                          parameters=decision['proposed']['parameters']), indent=2))


if __name__ == '__main__':
    main()
