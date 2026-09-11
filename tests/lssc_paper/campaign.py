#!/usr/bin/env python3
"""Bounded P1 campaign on saved fixtures, optionally paired with old evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit import verify
from run import metrics, preview, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--dictionaries', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--previous', help='Previous frozen-plugin/author results directory; comparisons are quality only')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('positive repeat count required')
    fixtures, out = Path(args.fixtures), Path(args.out)
    cases = json.loads((fixtures/'fixtures.json').read_text())['cases']
    if args.cases:
        unknown = set(args.cases)-{c['id'] for c in cases}
        if unknown:
            raise ValueError(f'unknown cases: {unknown}')
        cases = [c for c in cases if c['id'] in args.cases]
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((Path(args.previous)/'results.json').read_text())['rows'] if args.previous else []
    rows, comparisons = [], []
    for case in cases:
        for kind in ('clean','noisy'):
            assert sha(fixtures/case[kind]) == case[kind+'_sha256']
        sigma = case['sigma']
        block = 9 if sigma <= 25 else 12 if sigma <= 50 else 16
        dictionary = Path(args.dictionaries)/f'dict_n{block}.mat'
        for repeat in range(args.repeats):
            destination = out/f"{case['id']}-r{repeat}"
            command = [sys.executable,str(Path(__file__).with_name('run.py')),
                       '--input',str(fixtures/case['noisy']), '--clean',str(fixtures/case['clean']),
                       '--width',str(case['width']), '--height',str(case['height']),
                       '--sigma',str(sigma),'--dictionary',str(dictionary),'--out',str(destination)]
            execution = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
            (out/f"{case['id']}-r{repeat}.log").write_text(execution.stdout)
            if execution.returncode:
                raise RuntimeError(execution.stdout)
            audit = verify(destination,fixtures/case['noisy'],dictionary)
            metadata = json.loads((destination/'result.json').read_text())
            row = dict(case=case['id'], repeat=repeat, result=metadata, audit=audit)
            rows.append(row)
            print(json.dumps({'case':case['id'],'repeat':repeat,'quality':metadata['quality'],
                              'groups':metadata['final']['groups'],'audit':audit['verified']}),flush=True)
            with (out/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        repeated = [r for r in rows if r['case']==case['id']]
        assert len({r['result']['output_sha256'] for r in repeated}) == 1
        assert len({r['result']['pilot_sha256'] for r in repeated}) == 1
        current = repeated[0]['result']
        comparison = {'case':case['id'], 'pilot':current['quality']['pilot'], 'ssc':current['quality']['output']}
        shape = (case['height'],case['width'])
        clean = np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(shape)
        noisy = np.fromfile(fixtures/case['noisy'],dtype='<f4').reshape(shape)
        panels = [('Clean',clean),('Noisy',noisy)]
        selected = [r for r in old if r['case']==case['id'] and r['algorithm']=='LSSC']
        for variant in ('default','dense'):
            match = [r for r in selected if r['variant']==variant]
            if match:
                row = match[0]
                assert row['noisy_sha256']==case['noisy_sha256'] and row['clean_sha256']==case['clean_sha256']
                path = Path(args.previous)/row['output']
                assert sha(path)==row['output_sha256']
                pixels = np.fromfile(path,dtype='<f4').reshape(shape)
                comparison['previous_'+variant] = metrics(clean,pixels)
                panels.append(('Prior '+variant,pixels))
        destination = out/f"{case['id']}-r0"
        for filename,title in [('pilot','Fixed D: SC'),('output','Fixed D: SSC')]:
            pixels = np.fromfile(destination/(filename+'.f32'),dtype='<f4').reshape(shape)
            panels.append((title,pixels))
        full = [r for r in selected if r['variant']=='full']
        if full:
            row=full[0]
            assert row['noisy_sha256']==case['noisy_sha256'] and row['clean_sha256']==case['clean_sha256']
            path=Path(args.previous)/row['output']
            assert sha(path)==row['output_sha256']
            pixels=np.fromfile(path,dtype='<f4').reshape(shape)
            comparison['author_full']=metrics(clean,pixels)
            panels.append(('Author full',pixels))
        preview(out/(case['id']+'-comparison.png'),panels)
        comparisons.append(comparison)
    data = {'schema':'nss.lssc-paper-p1-campaign.v1','rows':rows,'comparisons':comparisons,
            'all_repeat_output_hashes_equal':True,'all_trace_audits_passed':True,
            'complete_lssc':False,'speed_comparison_allowed':False}
    (out/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines=['# 固定字典 P1：SC → 匹配 → SSC','',
           '这是不含字典学习的初步参考，不是完整 LSSC，也不是作者 MEX 数值复现。',
           '所有输入来自保存的同字节 float32 fixture；重复输出哈希一致；每个分组的 LS 重建、残差预算及回填均独立复算。',
           '性能数据仅记录本机 Python 单次调用，不与旧 C4/Octave 计时比较。','',
           '| 图像 | 旧默认 PSNR | 旧 step=1 PSNR | 固定 D SC PSNR | 固定 D SSC PSNR | 作者完整 PSNR |',
           '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        values=[r.get(k,{}).get('psnr_db') for k in ('previous_default','previous_dense','pilot','ssc','author_full')]
        lines.append('| '+r['case']+' | '+' | '.join(f'{v:.3f}' if v is not None else '—' for v in values)+' |')
    for r in comparisons:
        name=r['case']+'-comparison.png'
        lines += ['',f"## {r['case']}",'',f'![comparison]({name})']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
