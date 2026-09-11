#!/usr/bin/env python3
"""Export complete NLH search records and comparable within-host Pareto plots."""
import argparse
import csv
import json
from pathlib import Path

from nlh_defaults_search import LANES,nondominated,shortlist,fitted_structures
from nlh_defaults_inputs import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(args.results);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    collected=[];summaries={}
    for ax,lane in zip(axes,LANES):
        directory=root/('screen-'+lane)
        if not (directory/'trials.json').exists():
            ax.set_title(lane+' / pending');ax.set_axis_off();continue
        rows=json.loads((directory/'trials.json').read_text())
        metadata=json.loads((directory/'metadata.json').read_text())
        complete=(directory/'summary.json').exists()
        front=nondominated(fitted_structures(rows));chosen=shortlist(rows)
        ax.scatter([r['geomean_seconds'] for r in rows],[r['mean_psnr'] for r in rows],
                   s=10,c='#B4BEC9',alpha=.5,label='All measured configurations')
        ax.scatter([r['geomean_seconds'] for r in front],[r['mean_psnr'] for r in front],
                   s=25,c='#287C8E',label='PSNR / SSIM / time frontier')
        for label,color,marker in (('balanced','#DE7624','D'),('quality','#664C99','^'),('speed','#22815A','s')):
            row=chosen[label]
            ax.scatter(row['geomean_seconds'],row['mean_psnr'],s=80,color=color,marker=marker,label=label)
        for row in rows:
            if row['label']=='current-defaults':
                ax.scatter(row['geomean_seconds'],row['mean_psnr'],s=150,color='#B72D38',marker='*',label='v4 defaults')
        ax.set_xscale('log');ax.set_xlabel('Geometric mean seconds / frame (search only)')
        ax.set_ylabel('Mean PSNR on fixed development cases (dB)')
        ax.set_title(lane+f' | {len(rows)} configurations'+('' if complete else ' | in progress'))
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
        summaries[lane]=dict(complete=complete,trials=len(rows),selected=chosen,metadata=metadata)
        for row in rows:
            collected.append(dict(lane=lane,id=row['id'],label=row['label'],mean_psnr=row['mean_psnr'],
                                  mean_ssim=row['mean_ssim'],geomean_seconds=row['geomean_seconds'],
                                  parameters=json.dumps(row['parameters'],sort_keys=True),
                                  plugin_sha256=metadata['plugin_sha256'],cases=len(row['cases'])))
    figure.suptitle('NLH initial parameter search — before optimization; comparisons stay within each lane and host',fontsize=13)
    figure.savefig(out/'parameter-pareto.png',dpi=170);figure.savefig(out/'parameter-pareto.svg')
    if collected:
        with (out/'trials.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(collected[0]));writer.writeheader();writer.writerows(collected)
    save_json(out/'search-summary.json',summaries)
    print(json.dumps({k:dict(complete=v['complete'],trials=v['trials']) for k,v in summaries.items()}))


if __name__=='__main__':
    main()
