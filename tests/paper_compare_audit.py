#!/usr/bin/env python3
"""Verify downloaded paper-comparison evidence and summarize paired diagnostics."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from paper_compare import fixture, quality, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sets = {}
    flags = []
    total = 0
    for name in ('matrix128','matched128','size256','confirm128','ista16-confirm','ista64','timing-repair'):
        folder = root / ('nss-paper-' + name)
        rows = json.loads((folder/'results.json').read_text())['rows']
        fixtures = root / ('fixtures256' if name == 'size256' else 'fixtures128')
        for row in rows:
            assert row['ok'], row
            froot, case = fixture(fixtures, row['case'])
            assert row['clean_sha256'] == case['clean_sha256']
            assert row['noisy_sha256'] == case['noisy_sha256']
            assert sha(folder/row['output']) == row['output_sha256']
            shape = (case['height'],case['width'])
            clean = np.fromfile(froot/case['clean'], dtype='<f4').reshape(shape)
            pixels = np.fromfile(folder/row['output'], dtype='<f4').reshape(shape)
            metrics = quality(clean,pixels)
            for metric in ('psnr_db','ssim','mse'):
                assert abs(metrics[metric]-row['quality'][metric]) < 1e-10
            if 'timed_source_fills' in row:
                assert row['timed_source_fills'] == 0
            cpu = row['cpu_activity_during_worker']
            if cpu['cpu1']['busy_fraction'] > .01 or any(c['steal_ticks'] for c in cpu.values()):
                flags.append(dict(dataset=name, case=row['case'], algorithm=row['algorithm'],
                                  variant=row['variant'], repeat=row['repeat'], cpu=cpu))
        sets[name] = rows
        total += len(rows)
    key = lambda r: (r['case'],r['algorithm'],r['variant'])
    baseline = {key(r): r for r in sets['matrix128']}
    repeated = sets['confirm128'] + sets['ista16-confirm'] + sets['timing-repair']
    for row in repeated:
        assert row['output_sha256'] == baseline[key(row)]['output_sha256']
    for name in ('ista16-confirm','ista64','matched128'):
        groups = {}
        for row in sets[name]: groups.setdefault(key(row),[]).append(row)
        for rows in groups.values():
            assert len(rows) == 3
            assert len({r['output_sha256'] for r in rows}) == 1
    ista = []
    for k in dict.fromkeys(key(r) for r in sets['ista64']):
        old = [r for r in sets['ista16-confirm'] if key(r)==k]
        new = [r for r in sets['ista64'] if key(r)==k]
        def eligible(r):
            cpu = r['cpu_activity_during_worker']
            return cpu['cpu1']['busy_fraction'] <= .01 and not any(c['steal_ticks'] for c in cpu.values())
        good_repeats = {r['repeat'] for r in old if eligible(r)} & {r['repeat'] for r in new if eligible(r)}
        assert len(good_repeats) >= 2
        old_seconds = statistics.median(r['seconds'] for r in old if r['repeat'] in good_repeats)
        new_seconds = statistics.median(r['seconds'] for r in new if r['repeat'] in good_repeats)
        ista.append(dict(case=k[0],variant=k[2],old_psnr=old[0]['quality']['psnr_db'],
            new_psnr=new[0]['quality']['psnr_db'],gain_db=new[0]['quality']['psnr_db']-old[0]['quality']['psnr_db'],
            seconds16=old_seconds,seconds64=new_seconds,ratio=new_seconds/old_seconds,
            eligible_timing_pairs=len(good_repeats)))
    matched = []
    for case in dict.fromkeys(r['case'] for r in sets['matched128']):
        rows = [r for r in sets['matched128'] if r['case']==case]
        matched.append(dict(case=case,default_psnr=baseline[(case,'NCSR','default')]['quality']['psnr_db'],
            matched_psnr=rows[0]['quality']['psnr_db'],full_psnr=baseline[(case,'NCSR','full')]['quality']['psnr_db'],
            matched_seconds=statistics.median(r['seconds'] for r in rows)))
    check = root/'nss-paper-evidence/nss-paper-input-check.f32'
    check_meta = json.loads(Path(str(check)+'.json').read_text())
    assert check_meta['author_input_dictionary_immutability_checked']
    assert sha(check) == check_meta['output_sha256'] == baseline[('house-128-s25','LSSC','fast')]['output_sha256']
    summary = dict(total_rows=total, all_output_hashes_and_metrics_verified=True,
        baseline_repeat_comparisons=len(repeated), baseline_repeat_hashes_equal=True,
        triplicate_hashes_equal=True, lssc_input_dictionary_immutability_checked=True,
        cpu_activity_flags=flags, ista=ista, ncsr_matched=matched)
    save_json(root/'supplemental-audit.json',summary)
    lines = ['# 补充诊断与证据审计','',
        f'{total} 项输出的 SHA256 和全部 PSNR/SSIM/MSE 重算通过；{len(repeated)} 项基线复测与主矩阵字节一致；三重复组内输出全部一致。',
        'LSSC 作者 MEX 单独检查确认 warmup 和计时调用均不修改输入 I 或载入字典 D，结果与主矩阵相同。','',
        '## LSSC：只改 ISTA 16 → 64','',
        '画质使用全部三对输出；耗时只取双方 CPU1 活动均不超过 1% 且无 steal 的配对，剩余每项 2–3 对。','',
        '| 图像 | 配置 | 16次 PSNR | 64次 PSNR | 增益 dB | 16次秒 | 64次秒 | 耗时倍率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in ista:
        lines.append(f"| {r['case']} | {r['variant']} | {r['old_psnr']:.3f} | {r['new_psnr']:.3f} | {r['gain_db']:+.3f} | {r['seconds16']:.5f} | {r['seconds64']:.5f} | {r['ratio']:.2f} |")
    lines += ['', '## NCSR：接近作者公共参数，仍用当前核','',
        '| 图像 | 当前默认 PSNR | 调参 PSNR | 完整参考 PSNR | 调参秒 |', '|---|---:|---:|---:|---:|']
    for r in matched:
        lines.append(f"| {r['case']} | {r['default_psnr']:.3f} | {r['matched_psnr']:.3f} | {r['full_psnr']:.3f} | {r['matched_seconds']:.5f} |")
    lines += ['', '## CPU 活动标记','',
        '以下为原始观测，不能把被标记的单次耗时当作干净测量；主矩阵计时替换见 report128/summary.json。活动窗口覆盖整个 worker，短任务会受 tick 粒度影响。','']
    for r in flags:
        lines.append(f"- {r['dataset']} {r['case']} {r['algorithm']} {r['variant']} r{r['repeat']}: CPU1={r['cpu']['cpu1']['busy_fraction']:.6f}")
    (root/'supplemental-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
