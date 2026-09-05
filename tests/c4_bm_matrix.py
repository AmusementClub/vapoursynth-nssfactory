#!/usr/bin/env python3
"""Emit the maintenance correctness matrix plus full stage coverage for changed DCTs.

The original 89 configurations are retained from maintenance-20260905-a1/gate.py.
Each request is rendered by c4_paired_bm.py in its own plugin process.
"""
import json
import sys

def correctness_configs():
    for block in [1, 2, 4, 8, 12, 16, 32]:
        for group in [1, 2, 4, 8, 16, 32, 64]:
            yield dict(name=f"bm_b{block}_g{group}", algorithm="bm3d", size=[75, 49], frames=2,
                       wiener=group in [2, 8, 32, 64],
                       kwargs=dict(sigma=3, block_size=block, group_size=group,
                                   block_step=block, bm_range=2))
    for block in [4, 8, 12, 16, 32]:
        for radius in [1, 4]:
            for wiener in [False, True]:
                yield dict(name=f"bm_b{block}_r{radius}_w{int(wiener)}", algorithm="bm3d",
                           size=[75, 49], frames=2, wiener=wiener,
                           kwargs=dict(sigma=3, block_size=block, group_size=8,
                                       block_step=block, bm_range=2, radius=radius, ps_range=1))
    for block in [4, 8]:
        for group in [2, 4, 8, 16]:
            for radius in [0, 1]:
                yield dict(name=f"nlh_b{block}_g{group}_r{radius}", algorithm="nlh",
                           size=[75, 49], frames=2,
                           kwargs=dict(sigma=3, block_size=block, group_size=group,
                                       block_step=block, bm_range=3, radius=radius, ps_range=1, q=4))
    for algorithm in ["bm3d", "nlh"]:
        for threads in [1, 2]:
            yield dict(name=f"{algorithm}_short_group_t{threads}", algorithm=algorithm,
                       size=[17, 13], frames=3, threads=threads,
                       kwargs=dict(sigma=3, block_size=8, group_size=64 if algorithm == "bm3d" else 16,
                                   block_step=8, bm_range=1))



def configs():
    rows = list(correctness_configs())
    for row in rows:
        if row.get('kwargs', {}).get('radius', 0): row['aggregate_temporal'] = False
    for block in [4, 8, 12, 16]:
        for group in [1, 2, 4, 8, 16, 32, 64]:
            for stage in ["basic", "wiener", "two_stage"]:
                rows.append(dict(name=f"dct_b{block}_g{group}_{stage}", algorithm="bm3d",
                                 size=[75, 49], frames=2, stage=stage,
                                 kwargs=dict(sigma=3, block_size=block, group_size=group,
                                             block_step=block, bm_range=2)))
    for radius in [1, 4]:
        for stage in ['basic', 'wiener', 'two_stage']:
            rows.append(dict(name=f'aggregate_r{radius}_{stage}', algorithm='bm3d',
                             size=[75, 49], frames=2, stage=stage,
                             kwargs=dict(sigma=3, block_size=8, group_size=8, block_step=8,
                                         bm_range=2, radius=radius, ps_range=1)))
    # Public bm_range starts at 1. Constrain the image to get 1/4 valid
    # candidates, rather than passing the private matcher's zero-range case.
    for edge in [8, 9]:
        for stage in ['basic', 'wiener', 'two_stage']:
            for threads in [1, 2]:
                rows.append(dict(name=f'fused_short_size{edge}_{stage}_t{threads}', algorithm='bm3d',
                                 size=[edge, edge], frames=3, threads=threads, stage=stage,
                                 kwargs=dict(sigma=3, block_size=8, group_size=8, block_step=8,
                                             bm_range=1)))
    # Shared primitives and ordinary builds must remain compatible for all filters.
    for algorithm in ["wnnm", "twsc", "ncsr", "lssc", "nlm", "mcwnnm"]:
        rows.append(dict(name=algorithm, algorithm=algorithm, size=[75, 49], frames=1))
    return rows

if __name__ == "__main__":
    json.dump(configs(), sys.stdout, indent=2)
    print()
