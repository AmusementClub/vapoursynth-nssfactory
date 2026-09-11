#!/usr/bin/env python3
"""Rank frozen NLH evaluations without giving a larger dataset more weight.

Synthetic noise levels receive equal weight within DIV2K. DIV2K and CC then
receive equal weight when both are present. The separate dataset and noise-level
results are always retained; the scalar score is a selection rule, not evidence
that either dataset represents all photographs.
"""
import argparse
import csv
import json
from pathlib import Path

from nlh_defaults_inputs import file_sha, save_json
from nlh_defaults_search import nondominated, shortlist, fitted_structures, summarize_measurements as summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (root/'evaluation.json').exists():
        trials = json.loads((root/'evaluation.json').read_text())['rows']
    else:
        trials = json.loads((root/'trials.json').read_text())
    observations = {r['id']: r for line in (root/'observations.jsonl').read_text().splitlines()
                    if (r := json.loads(line))}
    rows, table = [], []
    for trial in trials:
        measurements = [observations[key] for key in trial['cases']]
        row = dict(trial)
        row.update(summarize(measurements))
        rows.append(row)
        for group in row['noise_levels']:
            table.append(dict(id=row['id'], label=row['label'], algorithm=row['algorithm'],
                              parameters=json.dumps(row['parameters'], sort_keys=True), **group))
    candidates = [r for r in rows if r['algorithm'] == 'NLH']
    selected = shortlist(candidates)
    save_json(out/'ranking.json', dict(rows=rows, selected=selected,
              frontier=nondominated(fitted_structures(candidates)), procedure=__doc__,
              script_sha256=file_sha(__file__),
              metadata_sha256=file_sha(root/'metadata.json'),
              observations_sha256=file_sha(root/'observations.jsonl')))
    save_json(out/'profiles.json', dict(profiles={k: v['parameters'] for k, v in selected.items()},
              selection='equal datasets and synthetic-noise levels; normalized PSNR/log-latency knee'))
    with (out/'quality-by-noise.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({k: {field: row[field] for field in ('label', 'mean_psnr', 'mean_ssim', 'geomean_seconds')}
                      for k, row in selected.items()}, indent=2))


if __name__ == '__main__':
    main()
