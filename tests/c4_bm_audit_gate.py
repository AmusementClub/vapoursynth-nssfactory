#!/usr/bin/env python3
"""Recompute completed paired-gate statistics from the retained raw records."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def interval(ratios):
    rng = random.Random(42)
    values = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return [values[249], values[9749]]


def audit(root):
    raw_path = root / 'raw.jsonl'
    summaries = json.loads((root / 'summary.json').read_text())
    recorded = json.loads((root / 'decision.json').read_text())
    rows = defaultdict(list)
    for line in raw_path.read_text().splitlines():
        row = json.loads(line)
        rows[json.dumps(row['config'], sort_keys=True)].append(row)
    issues, observations, chains = [], [], []
    if len(rows) != len(summaries):
        issues.append('raw and summary configuration counts differ')
    for item in summaries:
        name = item['config']['name']
        records = rows[json.dumps(item['config'], sort_keys=True)]
        pairs = item['pairs']
        variants = {}
        for side in ('baseline', 'candidate'):
            values = sorted((r for r in records if r['variant'] == side), key=lambda r: r['pair'])
            if pairs not in (7, 15) or [r['pair'] for r in values] != list(range(pairs)):
                issues.append(f'{name}: incomplete or duplicate {side} pairs')
            variants[side] = values
            if len({r['sha256'] for r in values}) > 1:
                observations.append(f'{name}: {side} output hashes vary; inspect numerical evidence')
        if any(not math.isfinite(r['ms']) or r['ms'] <= 0 for r in records):
            issues.append(f'{name}: invalid wall time')
            continue
        ratios = [a['ms'] / b['ms'] for a, b in zip(variants['baseline'], variants['candidate'])]
        if not ratios:
            continue
        ci = interval(ratios)
        if ratios != item['ratios'] or statistics.median(ratios) != item['paired_speedup'] or ci != item['ci95']:
            issues.append(f'{name}: stored timing statistics differ from raw records')
        if ci[0] < 1 / 1.01:
            issues.append(f'{name}: regression bound not established')
        if not item['numerical']['passed']:
            issues.append(f'{name}: numerical triage unresolved')
        environment = item['environment']
        before, after = environment['before'], environment['after']
        delta = [b-a for a, b in zip(before['cpu1'], after['cpu1'])]
        idle = delta[3] / sum(delta[:8])
        steal = after['cpu'][7] - before['cpu'][7]
        if idle < .999 or steal != 0:
            issues.append(f'{name}: CPU1/steal contract not met')
        if item['config'].get('stage') == 'two_stage':
            chains.append(ratios)
    geomean = math.exp(statistics.mean(math.log(statistics.median(r)) for r in chains)) if chains else None
    chain_ci = None
    if chains:
        rng = random.Random(42)
        boot = sorted(math.exp(statistics.mean(math.log(statistics.median(rng.choices(r, k=len(r))))
                                             for r in chains)) for _ in range(10000))
        chain_ci = [boot[249], boot[9749]]
    if recorded['kind'] == 'optimization' and (geomean is None or geomean < 1.01 or chain_ci[0] <= 1):
        issues.append('target chain benefit not established')
    global_environment = json.loads((root / 'environment.json').read_text())
    if not global_environment['valid']:
        issues.append('whole-run environment invalid')
    return dict(directory=str(root), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                cases=len(summaries), two_stage_geomean=geomean, two_stage_ci95=chain_ci,
                recorded_passed=recorded['passed'], raw_recomputed_passed=not issues,
                admitted=recorded['passed'] and not issues and recorded['kind'] != 'semantic_change',
                issues=issues, observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(root) for root in args.directories], indent=2))
