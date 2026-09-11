#!/usr/bin/env python3
"""Plot measured legal block/step slices with fitted strength coefficients."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'summary.json').read_text())
    coverage = json.loads((root/'coverage.json').read_text())
    anchor = json.loads((root/'grid.json').read_text())['anchor']
    trials = {r['id']: r for r in json.loads((root/'trials.json').read_text())}
    assert len({(r['stage'], r['block'], r['step']) for r in coverage}) == 270
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for stage in (0, 1):
        quality, latency = np.full((15, 16), np.nan), np.full((15, 16), np.nan)
        for cell in coverage:
            if cell['stage'] != stage:
                continue
            row = trials[cell['best']]
            quality[cell['block']-2, cell['step']-1] = row['mean_psnr']
            latency[cell['block']-2, cell['step']-1] = row['geomean_seconds']
        stage_name, other = ('Basic', 'Wiener') if stage == 0 else ('Wiener', 'Basic')
        for column, values in enumerate((quality, latency)):
            axis = axes[stage, column]
            image = axis.imshow(values, origin='lower', extent=(.5, 16.5, 1.5, 16.5),
                                aspect='auto', cmap='viridis' if column == 0 else 'magma',
                                norm=None if column == 0 else LogNorm())
            axis.set_facecolor('#E9EDF0')
            axis.set_xticks((1, 4, 8, 12, 16)); axis.set_yticks((2, 4, 8, 12, 16))
            axis.set_xlabel(stage_name + ' step'); axis.set_ylabel(stage_name + ' block')
            axis.set_title(f'{other} fixed: block {anchor["block_size"][1-stage]}, step {anchor["block_step"][1-stage]}')
            figure.colorbar(image, ax=axis, label='Mean PSNR (dB)' if column == 0 else 'Search seconds / frame')
    figure.suptitle(summary['lane']+' — all 270 legal stage / block / step combinations\n'
                   'Shared strength refit in each cell; development sensitivity slices, not final defaults', fontsize=12)
    figure.savefig(out/'block-step-grid.png', dpi=180)
    figure.savefig(out/'block-step-grid.svg')
    plt.close(figure)


if __name__ == '__main__':
    main()
