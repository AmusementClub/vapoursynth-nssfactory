#!/usr/bin/env python3
"""Recompute quality and audit paired HQ optimization evidence."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
from paper_compare import quality, save_json, sha


def eligible(row):
    cpus=row['cpu_activity']
    return cpus['cpu1']['busy_fraction']<=.01 and all(x['steal_ticks']==0 for x in cpus.values())


def report(directory,fixtures):
    manifest=json.loads((directory/'environment.json').read_text())
    rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    cases={c['id']:c for c in json.loads((fixtures/'fixtures.json').read_text())['cases']}
    failures=[r for r in rows if not r.get('ok')]
    good=[r for r in rows if r.get('ok')]
    seen=set();cache={};grouped=defaultdict(list);flags=[];differences=[]
    for row in good:
        key=row['case'],row['variant'],row['repeat']
        if key in seen:raise ValueError(f'Duplicate successful identity: {key}')
        seen.add(key)
        case=cases[row['case']];path=directory/row['output'];digest=sha(path)
        assert digest==row['output_sha256']
        if manifest.get('warmup',1):assert digest==row['warm_output_sha256']
        else:assert row['warm_output_sha256'] is None and row['full_shape_warmup'] is False
        assert row['repeat_hashes_equal'] and row['timed_source_fills']==0 and row['flags']==31
        assert row['plugin_sha256']==manifest['plugin_sha256'][row['variant']]
        assert row['clean_sha256']==case['clean_sha256']==sha(fixtures/case['clean'])
        assert row['noisy_sha256']==case['noisy_sha256']==sha(fixtures/case['noisy'])
        assert all(row['parameters'][key]==value for key,value in manifest['parameters'].items())
        assert row['parameters']['sigma']==case['sigma'] and row['parameters']['radius']==0
        assert len(row['timings'])==row['samples'] and row['seconds']==statistics.median(row['timings'])
        assert all(math.isfinite(value) and value>0 for value in row['timings'])
        cache_key=case['clean_sha256'],digest
        if cache_key not in cache:
            shape=(case['height'],case['width'])
            clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(shape)
            output=np.fromfile(path,dtype='<f4').reshape(shape)
            assert np.isfinite(output).all()
            cache[cache_key]=quality(clean,output)
        for metric,value in cache[cache_key].items():
            if value is None:assert row['quality'][metric] is None,(key,metric)
            else:assert abs(value-row['quality'][metric])<1e-10,(key,metric)
        grouped[row['case']].append(row)
        if not eligible(row):flags.append(dict(case=row['case'],variant=row['variant'],repeat=row['repeat'],cpu=row['cpu_activity']))
    summaries=[]
    variants=list(manifest['plugins'])
    for variant in variants:
        per_case=[]
        for case_id in manifest['cases']:
            items=[r for r in grouped[case_id] if r['variant']==variant]
            reference=[r for r in grouped[case_id] if r['variant']==variants[0]]
            ref_by_repeat={r['repeat']:r for r in reference}
            hashes={r['output_sha256'] for r in items+reference}
            if len(hashes)!=1 or not items or not reference:
                differences.append(dict(case=case_id,variant=variant,hashes=sorted(hashes)))
            pairs=[(r,ref_by_repeat[r['repeat']]) for r in items if r['repeat'] in ref_by_repeat and
                   eligible(r) and eligible(ref_by_repeat[r['repeat']])]
            valid=[r for r in items if eligible(r)]
            per_case.append(dict(case=case_id,exact_reference=len(hashes)==1 and bool(items) and bool(reference),
                psnr_db=items[0]['quality']['psnr_db'] if items else None,
                ssim=items[0]['quality']['ssim'] if items else None,
                seconds=statistics.median(r['seconds'] for r in valid) if valid else None,
                peak_rss_kib=statistics.median(r['peak_rss_kib'] for r in items) if items else None,
                ratio=statistics.median(r['seconds']/b['seconds'] for r,b in pairs) if pairs else None,
                eligible_pairs=len(pairs),runs=len(items)))
        accepted=bool(per_case) and all(x['eligible_pairs']>=2 for x in per_case)
        summaries.append(dict(variant=variant,cases=per_case,timing_supported=accepted,
            geometric_ratio=math.exp(statistics.mean(math.log(x['ratio']) for x in per_case)) if accepted else None,
            mean_seconds=statistics.mean(x['seconds'] for x in per_case) if all(x['seconds'] is not None for x in per_case) else None))
    result=dict(schema='nss.ncsr-hq-audit.v1',rows=len(rows),successful_rows=len(good),failures=failures,
                full_shape_warmup=bool(manifest.get('warmup',1)),
                all_hashes_metrics_and_repeat_checks_verified=True,exact_reference=not differences,
                differences=differences,cpu_flags=flags,summaries=summaries)
    save_json(directory/'audit-summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',action='append',required=True,help='RESULT_DIRECTORY=FIXTURE_DIRECTORY')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();results={}
    for spec in args.dataset:
        directory,fixtures=map(Path,spec.split('=',1));results[directory.name]=report(directory,fixtures)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);save_json(out/'summary.json',results)
    lines=['# NCSR HQ 三批优化验收','',
           '同一高质量模型、同一输入、同一主机。倍率仅使用 CPU1 活动 ≤1% 且无 steal 的配对；每个配置至少两对。',
           '参考输出来自本轮冻结的质量原型，不是旧快速算法。输出逐字节一致才标记 exact。','']
    for name,data in results.items():
        boundary='完整尺寸预热后的帧' if data['full_shape_warmup'] else '首次完整尺寸帧（无完整尺寸预热）'
        lines.extend([f'## {name}','',f"成功 {data['successful_rows']}/{data['rows']}；逐字节参考一致：{data['exact_reference']}；CPU 标记 {len(data['cpu_flags'])} 项。计时：{boundary}。",'',
                      '| 批次 | 图像 | 秒/帧 | 相对 HQ 参考耗时 | PSNR | RSS KiB | 有效配对 |','|---|---|---:|---:|---:|---:|---:|'])
        for summary in data['summaries']:
            for c in summary['cases']:
                timing=f"{c['seconds']:.6f}" if c['seconds'] is not None else 'N/A'
                ratio=f"{c['ratio']:.4f}" if c['eligible_pairs']>=2 else '证据不足'
                psnr=f"{c['psnr_db']:.6f}" if c['psnr_db'] is not None else 'N/A'
                lines.append(f"| {summary['variant']} | {c['case']} | {timing} | {ratio} | {psnr} | {c['peak_rss_kib']} | {c['eligible_pairs']} |")
        lines.append('')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(rows=v['rows'],failures=len(v['failures']),exact=v['exact_reference'],flags=len(v['cpu_flags'])) for k,v in results.items()},indent=2))


if __name__=='__main__':main()
