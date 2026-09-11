#!/usr/bin/env python3
"""Plot frozen selection finalists, keeping reference quality apart from cost.

Reference runs can use a different warmup policy, so their PSNR/SSIM appear only
as horizontal lines. Every candidate within a panel shares one evaluation cohort.
"""
import argparse
import csv
import json
from pathlib import Path
import textwrap

from nlh_defaults_compare import read_trial
from nlh_defaults_inputs import file_sha, save_json
from nlh_defaults_search import fitted_structures, nondominated


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
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    panels = config['panels']
    figure, axes = plt.subplots(2, len(panels), figsize=(max(7., 5.5*len(panels)), 8.5),
                               squeeze=False, layout='constrained')
    table, provenance = [], []
    colors = plt.get_cmap('tab10')
    for column, panel in enumerate(panels):
        root = Path(panel['experiment'])
        evaluation = json.loads((root/'evaluation.json').read_text())
        if not evaluation['complete']:
            raise ValueError('selection plot requires a completed experiment')
        rows = [r for r in evaluation['rows'] if r['algorithm'] == 'NLH']
        front = {r['id'] for r in nondominated(fitted_structures(rows))}
        _, _, cases = read_trial(root, rows[0]['label'])
        for index, row in enumerate(rows):
            _, _, current = read_trial(root, row['label'])
            if current.keys() != cases.keys():
                raise ValueError('candidate cohorts differ')
            for case in cases:
                for field in ('input_sha256', 'shape', 'score_size', 'sigma_mode'):
                    if current[case][field] != cases[case][field]:
                        raise ValueError('candidate inputs differ: '+field)
            chosen = row['label'] == panel.get('selected')
            label = panel.get('labels', {}).get(row['label'], row['label'])
            for axis, metric in zip(axes[:, column], ('mean_psnr', 'mean_ssim')):
                axis.scatter(row['geomean_seconds'], row[metric], color=colors(index % 10),
                             marker='*' if chosen else 'o', s=190 if chosen else 65,
                             edgecolors='#203041' if row['id'] in front else 'none', linewidths=.8,
                             label=label+(' / selected default' if chosen else ''))
            table.append(dict(panel=panel['title'], label=row['label'],
                              mean_psnr=row['mean_psnr'], mean_ssim=row['mean_ssim'],
                              geomean_seconds=row['geomean_seconds'], frontier=row['id'] in front,
                              parameters=json.dumps(row['parameters'], sort_keys=True)))
        for index, reference in enumerate(panel.get('references', [])):
            _, trial, reference_cases = read_trial(reference['experiment'], reference['profile'])
            if cases.keys() != reference_cases.keys():
                raise ValueError('reference quality cohort differs')
            for case in cases:
                for field in ('input_sha256', 'shape', 'score_size', 'sigma_mode'):
                    if cases[case][field] != reference_cases[case][field]:
                        raise ValueError('reference inputs differ: '+field)
            for axis, metric in zip(axes[:, column], ('mean_psnr', 'mean_ssim')):
                axis.axhline(trial[metric], color=('#6D7782', '#965544')[index % 2],
                             linestyle=('--', ':')[index % 2], linewidth=1,
                             label=reference['label']+' / quality reference')
        for axis in axes[:, column]:
            axis.set_xscale('log')
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=7))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel('Geometric mean seconds / frame in candidate evaluation', fontsize=9)
            axis.grid(alpha=.18)
        axes[0, column].set_title(textwrap.fill(panel['title'], 58), fontsize=11)
        axes[0, column].set_ylabel('Mean PSNR (dB)')
        axes[1, column].set_ylabel('Mean SSIM')
        axes[0, column].legend(fontsize=7, loc='best')
        provenance.append(dict(title=panel['title'], metadata=evaluation['metadata'],
                               evaluation_sha256=file_sha(root/'evaluation.json')))
    figure.suptitle('NLH frozen selection finalists\n'
                    'Equal noise-level / dataset weights\n'
                    'Evaluation costs are separate from paired performance acceptance', fontsize=11)
    figure.savefig(out/'selection-pareto.png', dpi=160)
    figure.savefig(out/'selection-pareto.svg')
    plt.close(figure)
    save_json(out/'selection-plot.json', dict(panels=provenance, rows=table,
              config_sha256=file_sha(args.config), script_sha256=file_sha(__file__)))
    with (out/'selection-pareto.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print('selection panels:', len(panels))


if __name__ == '__main__':
    main()
