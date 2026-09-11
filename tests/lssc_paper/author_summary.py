#!/usr/bin/env python3
"""Derive common-valid-pair statistics without rewriting raw author records."""
import argparse
import json
from pathlib import Path
import statistics


def paired_summary(report):
    names = [policy[0] for policy in report['policies']]
    if len(set(names)) != len(names) or not names:
        raise ValueError('unique nonempty policy list required')
    rows = [row for row in report['rows'] if row['pair'] >= 0]
    index = {}
    for row in rows:
        key = (row['pair'], row['variant'])
        if key in index:
            raise ValueError('duplicate sample identity')
        index[key] = row
    accepted, rejected = [], []
    for pair in range(report['pairs']):
        reasons = []
        for name in names:
            row = index.get((pair, name))
            if row is None:
                reasons.append(name + ': missing')
            elif not row['eligible_idle_sample']:
                reasons.append(name + ': neighbor activity or steal')
            elif not row['output_matches_warm'] or not row['dictionary_matches_warm']:
                reasons.append(name + ': repeat mismatch')
        if reasons:
            rejected.append({'pair': pair, 'reasons': reasons})
        else:
            accepted.append(pair)
    baseline = names[0]
    medians = {name: statistics.median(index[pair, name]['seconds'] for pair in accepted)
               for name in names} if accepted else None
    ratios = {name: [index[pair, name]['seconds'] / index[pair, baseline]['seconds']
                     for pair in accepted] for name in names}
    return {'schema': 'nss.author-common-pairs.v1', 'baseline': baseline,
            'accepted_pairs': accepted, 'rejected_pairs': rejected, 'median_seconds': medians,
            'paired_runtime_ratios_to_baseline': ratios,
            'median_paired_runtime_ratio': {name: statistics.median(values) if values else None
                                          for name, values in ratios.items()},
            'formal_native_nss_gate': False,
            'boundary': 'same-input external MEX budget policies; all policies must be eligible in a pair'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    args = parser.parse_args()
    print(json.dumps(paired_summary(json.loads(Path(args.report).read_text())), indent=2, allow_nan=False))
