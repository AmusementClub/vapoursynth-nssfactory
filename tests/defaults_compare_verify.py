#!/usr/bin/env python3
"""Verify that delivered native PNG previews match the raw measured pixels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from defaults_compare import ALGORITHMS
from paper_compare import save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--previous')
    args=parser.parse_args();root=Path(args.root).resolve()
    preview_count=0; repeated_rows=0; previous_count=0
    for size in (128,256):
        fixtures=root/f'fixtures{size}'
        result_root=root/f'nss-all-defaults-{size}'
        report_root=root/f'report{size}'
        cases=json.loads((fixtures/'fixtures.json').read_text())['cases']
        rows=json.loads((result_root/'results.json').read_text())['rows']
        summary=json.loads((report_root/'summary.json').read_text())
        assert summary['all_hashes_and_metrics_verified'] and summary['repeat_hashes_equal']
        assert summary['completed_cells']==summary['expected_cells']==len(cases)*8
        assert not summary['failed'] and not summary['missing']
        assert all(c['samples']==3 and c['timing_samples']>=2 for c in summary['cells'])
        repeated_rows+=len(rows)
        by_cell={(r['case'],r['algorithm']):r for r in rows if r['repeat']==0}
        for case in cases:
            shape=(case['height'],case['width'])
            for name in ('clean','noisy',*ALGORITHMS):
                if name in ('clean','noisy'):
                    pixels=np.fromfile(fixtures/case[name],dtype='<f4').reshape(1,*shape)
                else:
                    row=by_cell[(case['id'],name)]
                    assert sha(result_root/row['output'])==row['output_sha256']
                    pixels=np.fromfile(result_root/row['output'],dtype='<f4').reshape(row['channels'],*shape)
                rgb=np.repeat(pixels,3,axis=0) if len(pixels)==1 else pixels
                expected=np.rint(np.clip(rgb.transpose(1,2,0),0,1)*255).astype(np.uint8)
                actual=np.asarray(Image.open(report_root/'native'/f"{case['id']}-{name}.png"))
                assert np.array_equal(expected,actual), f'Preview mismatch {case["id"]} {name}'
                preview_count+=1
        if args.previous:
            previous=Path(args.previous)/('nss-paper-matrix128' if size==128 else 'nss-paper-size256')/'results.json'
            old=json.loads(previous.read_text())['rows']
            for r in old:
                if r['variant']=='default' and r['ok']:
                    current=by_cell[(r['case'],r['algorithm'])]
                    assert current['noisy_sha256']==r['noisy_sha256']
                    assert current['output_sha256']==r['output_sha256']
                    previous_count+=1
    result=dict(measured_rows=repeated_rows,verified_native_pngs=preview_count,
        preview_mapping='exact clip-to-0..1, round*255, RGB display, no other pixel edits',
        all_cells_have_three_identical_repeats=True,min_eligible_timing_repeats=2,
        prior_lssc_ncsr_default_unique_outputs_compared=previous_count,
        prior_output_hashes_equal=True if previous_count else None)
    save_json(root/'delivery-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
