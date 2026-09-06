#!/usr/bin/env python3
"""Audit integrated-filter A/B records and render their comparison tables."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from c4_paired_bm import environment_delta, interval


def audit(directory):
    summaries = json.loads((directory / 'summary.json').read_text())
    records = defaultdict(lambda: defaultdict(dict))
    for line in (directory / 'raw.jsonl').read_text().splitlines():
        r = json.loads(line)
        side = records[r['config']['name']][r['variant']]
        assert r['pair'] not in side, 'duplicate pair'
        side[r['pair']] = r
    output = []
    for item in summaries:
        name = item['config']['name']
        sides = records[name]
        indices = set(range(item['pairs']))
        assert set(sides['baseline']) == set(sides['candidate']) == indices
        ratios = []
        for i in sorted(indices):
            a, b = sides['baseline'][i], sides['candidate'][i]
            for field in ('timed_first', 'timed_frames', 'input_sha256'):
                assert a[field] == b[field], (name, field)
            assert all(math.isfinite(r['ms']) and r['ms'] > 0 for r in (a, b))
            ratios.append(a['ms'] / b['ms'])
        assert ratios == item['ratios']
        assert statistics.median(ratios) == item['paired_speedup']
        assert interval(ratios) == item['ci95']
        env = environment_delta(item['environment']['before'], item['environment']['after'])
        assert env == item['environment']
        value = dict(item)
        for side in ('baseline', 'candidate'):
            values = list(sides[side].values())
            value[side + '_ms'] = statistics.median(r['ms'] for r in values)
            value[side + '_rss_mib'] = max(r['peak_rss_kib'] for r in values) / 1024
            value[side + '_hash_stable'] = len({r['sha256'] for r in values}) == 1
        output.append(value)
    assert len(records) == len(output)
    return output


def timing_table(rows):
    lines = ['| 配置 | 尺寸 | A ms/frame | B ms/frame | A/B | 95% CI | 对数 | 环境 |',
             '|---|---|---:|---:|---:|---|---:|---|']
    for r in rows:
        config = r['config']
        lines.append(f"| {config['name']} | {'×'.join(map(str, config['size']))} | "
                     f"{r['baseline_ms']:.3f} | {r['candidate_ms']:.3f} | {r['paired_speedup']:.5f}× | "
                     f"[{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}] | {r['pairs']} | "
                     f"{'通过' if r['environment']['valid'] else '无效'} |")
    return '\n'.join(lines)


def error_table(original, isolation):
    isolated = {r['config']['name']: r for r in isolation}
    lines = ['| 配置 | 原版→新版最大误差 | RMSE | PSNR(A,B) dB | 优化关闭→开启最大误差 | 全时段哈希相同 |',
             '|---|---:|---:|---:|---:|---|']
    for r in original:
        name = r['config']['name']
        n = r['numerical']
        peer = isolated.get(name)
        err = f"{peer['numerical'].get('max_abs', float('nan')):.8g}" if peer else '未完成'
        exact = ('是' if peer['exact'] else '否') if peer else '未完成'
        lines.append(f"| {name} | {n.get('max_abs', float('nan')):.8g} | "
                     f"{n.get('rmse', float('nan')):.8g} | {n.get('psnr_between', float('nan')):.3f} | {err} | {exact} |")
    return '\n'.join(lines)


def pmu_table(root):
    metrics = ('tma_retiring', 'tma_bad_speculation', 'tma_frontend_bound', 'tma_backend_bound',
               'tma_core_bound', 'tma_memory_bound')
    lines = ['| 配置 | Retiring A→B | Bad speculation A→B | Frontend A→B | Backend A→B | Core A→B | Memory A→B |',
             '|---|---:|---:|---:|---:|---:|---:|']
    values = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        row = dict(name=directory.name)
        for side in ('baseline', 'candidate'):
            side_root = directory / side
            m = {}
            for level in ('TopdownL1', 'TopdownL2', 'TopdownL3'):
                for value, key in re.findall(r'#\s+([0-9.]+) %\s+(tma_[a-z0-9_]+)', (side_root / (level + '.txt')).read_text()):
                    m.setdefault(key, float(value))
            assert all(key in m for key in metrics), (directory, m)
            row[side] = m
        lines.append('| ' + directory.name + ' | ' + ' | '.join(
            f"{row['baseline'][key]:.1f}%→{row['candidate'][key]:.1f}%" for key in metrics) + ' |')
        values.append(row)
    return '\n'.join(lines), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab', type=Path, required=True)
    parser.add_argument('--pmu', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = audit(args.ab / 'original-vs-selected')
    isolation = audit(args.ab / 'correctness-vs-selected')
    tables = ['## 原主仓库 A → 合并后 B', timing_table(original),
              '## 正确性版本 A（mask=0）→ 默认优化 B（mask=2305）', timing_table(isolation),
              '## 输出误差', error_table(original, isolation)]
    value = dict(original=original, isolation=isolation)
    if args.pmu:
        table, pmu = pmu_table(args.pmu)
        tables += ['## 同一帧请求区间的 Top-down', table]
        value['pmu'] = pmu
    args.out.write_text('\n\n'.join(tables) + '\n')
    args.out.with_suffix('.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(dict(original=len(original), isolation=len(isolation),
                          invalid_environments=sum(not r['environment']['valid'] for r in original + isolation))))


if __name__ == '__main__':
    main()
