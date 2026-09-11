#!/usr/bin/env python3
"""Export paired quality changes for frozen, hash-matched NLH cohorts.

These evaluation costs describe the quality cohort. Warm alternating performance
acceptance is recorded by nlh_candidate_campaign.py in a separate experiment.
"""
import argparse
import csv
import json
from pathlib import Path

from nlh_defaults_decision import paired_interval
from nlh_defaults_inputs import file_sha, save_json
from nlh_defaults_search import summarize_measurements


def read_trial(directory, label):
    root = Path(directory)
    evaluation = json.loads((root / 'evaluation.json').read_text())
    if not evaluation['complete']:
        raise ValueError('comparison requires a completed evaluation')
    trial = next(row for row in evaluation['rows'] if row['label'] == label)
    observations = {row['id']: row for line in (root / 'observations.jsonl').read_text().splitlines()
                    if (row := json.loads(line))}
    records = {observations[key].get('base_case', observations[key]['case']): observations[key]
               for key in trial['cases']}
    if len(records) != len(trial['cases']):
        raise ValueError('duplicate case in trial')
    return evaluation, trial, records


def compare(candidate, reference, allow_subset=False):
    common = sorted(candidate.keys() & reference.keys())
    if not common or (not allow_subset and candidate.keys() != reference.keys()):
        raise ValueError('comparison cohorts differ; explicitly allow a recorded subset if intended')
    observations, trial_ids, rows = {}, [[], []], []
    for case in common:
        current, previous = candidate[case], reference[case]
        for field in ('input_sha256', 'shape', 'score_size', 'sigma_mode', 'dataset', 'image', 'sigma'):
            if current[field] != previous[field]:
                raise ValueError('comparison mismatch: ' + field)
        for index, record in enumerate((current, previous)):
            key = f'{index}:{case}'
            observations[key] = record
            trial_ids[index].append(key)
        rows.append(dict(case=case, image=current['image'], dataset=current['dataset'],
                         sigma=current['sigma'], sigma_mode=current['sigma_mode'],
                         input_sha256=current['input_sha256'],
                         candidate_output_sha256=current['output_sha256'],
                         reference_output_sha256=previous['output_sha256'],
                         candidate_psnr=current['quality']['psnr_db'],
                         reference_psnr=previous['quality']['psnr_db'],
                         psnr_delta=current['quality']['psnr_db'] - previous['quality']['psnr_db'],
                         candidate_ssim=current['quality']['ssim'],
                         reference_ssim=previous['quality']['ssim'],
                         ssim_delta=current['quality']['ssim'] - previous['quality']['ssim'],
                         candidate_seconds=current['seconds'], reference_seconds=previous['seconds']))
    intervals = paired_interval(*[dict(cases=keys) for keys in trial_ids], observations)
    # The quality references may run on another host or use a different warmup
    # policy. Their costs must not become a paired speedup claim.
    intervals.pop('latency_ratio')
    summary = dict(paired_cases=len(common),
                   candidate=summarize_measurements([candidate[c] for c in common]),
                   reference=summarize_measurements([reference[c] for c in common]),
                   paired_original_intervals=intervals,
                   interval_scope='equal originals within datasets; retain each original noise levels and seeds together',
                   evaluation_time_scope='quality-cohort costs; not paired performance acceptance',
                   omitted_candidate_cases=sorted(candidate.keys() - set(common)),
                   omitted_reference_cases=sorted(reference.keys() - set(common)),
                   regressions={key: dict(cases=sum(r[key] < 0 for r in rows),
                                          worst=sorted(rows, key=lambda r: r[key])[:8])
                                for key in ('psnr_delta', 'ssim_delta')})
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'candidate-profile', 'reference', 'reference-profile', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--allow-subset', action='store_true')
    args = parser.parse_args()
    ce, ct, candidate = read_trial(args.candidate, args.candidate_profile)
    re, rt, reference = read_trial(args.reference, args.reference_profile)
    summary, rows = compare(candidate, reference, args.allow_subset)
    summary.update(candidate_profile=ct, reference_profile=rt,
                   candidate_metadata=ce['metadata'], reference_metadata=re['metadata'],
                   script_sha256=file_sha(__file__),
                   candidate_records_sha256=file_sha(Path(args.candidate) / 'observations.jsonl'),
                   reference_records_sha256=file_sha(Path(args.reference) / 'observations.jsonl'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'comparison.json', summary)
    with (out / 'case-deltas.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(paired_cases=len(rows),
                          psnr=summary['paired_original_intervals']['psnr_delta'],
                          ssim=summary['paired_original_intervals']['ssim_delta']), indent=2))


if __name__ == '__main__':
    main()
