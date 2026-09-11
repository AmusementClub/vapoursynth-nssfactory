#!/usr/bin/env python3
"""Plot descriptive paired-seed response at unchanged injected sigma."""
import argparse
import json
from pathlib import Path
import statistics

from nlh_defaults_inputs import file_sha, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = config['panels']
    figure, axes = plt.subplots(1, len(panels), figsize=(max(7, 5.5*len(panels)), 5),
                               squeeze=False, layout='constrained')
    exported = []
    for axis, panel in zip(axes[0], panels):
        expected_cohort, score_size = None, None
        levels = set()
        for source in panel['sources']:
            path = Path(source['response'])
            document = json.loads(path.read_text())
            evaluation = json.loads((Path(source['experiment'])/'evaluation.json').read_text())
            if file_sha(Path(source['experiment'])/'evaluation.json') != document['evaluation_sha256']:
                raise ValueError('response belongs to a different evaluation')
            actual_size = evaluation['metadata']['score_size']
            if score_size is not None and score_size != actual_size:
                raise ValueError('response score regions differ')
            score_size = actual_size
            rows = [r for r in document['rows'] if r['label'] == source['profile']]
            cohort = {(r['image'], r['format'], r['sigma']) for r in rows}
            if not cohort or (expected_cohort is not None and cohort != expected_cohort):
                raise ValueError('response cohorts differ')
            expected_cohort = cohort
            values = {}
            for row in rows:
                values.setdefault(row['sigma'], []).append(row['response_score'])
            sigmas = sorted(values)
            levels.update(sigmas)
            mean = [statistics.mean(values[s]) for s in sigmas]
            minimum = [min(values[s]) for s in sigmas]
            maximum = [max(values[s]) for s in sigmas]
            line, = axis.plot(sigmas, mean, marker='o', markersize=4, label=source['label'])
            axis.fill_between(sigmas, minimum, maximum, color=line.get_color(), alpha=.10)
            exported.append(dict(panel=panel['title'], label=source['label'], sigmas=sigmas,
                                 mean=mean, minimum=minimum, maximum=maximum,
                                 score_size=score_size, response_sha256=file_sha(path)))
        axis.set_title(panel['title'], fontsize=11)
        axis.set_xlabel('Injected sigma (8-bit standard deviation)')
        axis.set_ylabel('Paired output-change / input-change norm')
        axis.set_xticks(sorted(levels))
        axis.set_ylim(bottom=0)
        axis.axhline(1, color='#80909D', linestyle=':', linewidth=.8)
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle('Same-sigma paired-seed response\n'
                    'Lines: image means; bands: observed image range\n'
                    'Response is descriptive; detail retention is assessed separately', fontsize=11)
    figure.savefig(out/'paired-response.png', dpi=160)
    figure.savefig(out/'paired-response.svg')
    plt.close(figure)
    save_json(out/'paired-response.json', dict(rows=exported,
              config_sha256=file_sha(args.config), driver_sha256=file_sha(__file__)))
    print('response panels:', len(panels))


if __name__ == '__main__':
    main()
