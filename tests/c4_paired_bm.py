#!/usr/bin/env python3
"""Same-host BM3D A/B gate; initialization/PMU are outside frame timing.

Seven alternating pairs, optionally extended to fifteen. Speedups are paired
baseline/candidate times; regression bounds use their reciprocal. Bootstrap is
seeded and preserves pairing. Raw results are append-only within a fresh run.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def worker(plugin, config):
    import numpy as np
    import vapoursynth as vs
    from profile_cpu_all import make_source, make_filter
    core = vs.core
    core.num_threads = config.get('threads', 1)
    core.std.LoadPlugin(path=plugin)
    frames = config.get('frames', 12)
    w, h = config.get('size', [1920, 1080])
    algorithm = config.get('algorithm', 'bm3d')
    source = make_source(core, algorithm, frames, w, h, 42, config.get('sample'))
    if 'kwargs' not in config and algorithm != 'bm3d':
        output = make_filter(core, algorithm, source)
    else:
        kw = dict(config.get('kwargs', {}))
        if not kw:
            b = config.get('block', 8)
            kw = dict(sigma=3, block_size=b, block_step=min(8, b), group_size=config.get('group', 8),
                      bm_range=7, radius=config.get('radius', 0))
        radius = kw.get('radius', 0)
        stage = config.get('stage', 'wiener' if config.get('wiener') else 'basic')
        fn = getattr(core.nss, {'bm3d': 'BM3D', 'nlh': 'NLH'}[algorithm])
        if stage == 'two_stage':
            pilot = fn(source, **kw)
            if radius: pilot = core.nss.VAggregate(pilot, source, radius=radius)
            output = fn(source, ref=pilot, **kw)
        elif stage == 'wiener': output = fn(source, ref=source, **kw)
        else: output = fn(source, **kw)
        if radius and algorithm == 'bm3d' and config.get('aggregate_temporal', True): output = core.nss.VAggregate(output, source, radius=radius)
    output.get_frame(0)
    start = time.perf_counter()
    if core.num_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(core.num_threads) as pool:
            results = list(pool.map(output.get_frame, range(1, frames + 1)))
    else: results = [output.get_frame(i) for i in range(1, frames + 1)]
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    for frame in results:
        for plane in range(frame.format.num_planes):
            values = np.ascontiguousarray(np.asarray(frame[plane]), dtype=np.float32)
            if not np.isfinite(values).all(): raise RuntimeError('nonfinite output')
            digest.update(values.tobytes())
    return dict(ms=elapsed * 1000 / frames, sha256=digest.hexdigest())


def interval(ratios):
    rng = random.Random(42)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return [samples[249], samples[9749]]


def cpu_stat():
    return {row.split()[0]: list(map(int, row.split()[1:]))
            for row in Path('/proc/stat').read_text().splitlines() if row.startswith('cpu')}


def environment_delta(before, after):
    delta = [b - a for a, b in zip(before['cpu1'], after['cpu1'])]
    idle = delta[3] / sum(delta[:8])
    steal = after['cpu'][7] - before['cpu'][7]
    return dict(cpu1_idle=idle, steal_ticks=steal,
                valid=idle >= .999 and steal == 0, before=before, after=after)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    configs = json.loads(Path(args.configs).read_text())
    manifest = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [args.baseline, args.candidate, __file__, args.configs,
                          Path(__file__).with_name('profile_cpu_all.py')]}
    for config in configs:
        if config.get('sample'):
            p = config['sample']; manifest[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    before = cpu_stat()
    (out / 'environment-before.json').write_text(json.dumps(before, indent=2))
    # Recheck code and inputs after the run as well: mutable campaign paths
    # must never silently change midway through a comparison.

    summary = []
    with (out / 'raw.jsonl').open('w') as raw:
        for config in configs:
            case_before = cpu_stat()
            rows = {'baseline': [], 'candidate': []}
            target = args.pairs
            pair = 0
            while pair < target:
                for name in (['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']):
                    cmd = ['taskset', '-c', '0', sys.executable, __file__, 'worker',
                           '--plugin', getattr(args, name), '--config', json.dumps(config)]
                    row = json.loads(subprocess.check_output(cmd, text=True))
                    rows[name].append(row)
                    raw.write(json.dumps(dict(config=config, variant=name, pair=pair, **row)) + '\n'); raw.flush()
                pair += 1
                if pair == target:
                    ratios = [a['ms'] / b['ms'] for a, b in zip(rows['baseline'], rows['candidate'])]
                    ci = interval(ratios)
                    # Resolve uncertain benefit or the 1% regression boundary once.
                    if args.extend and target == 7 and ((not args.regression_only and ci[0] <= 1 <= ci[1])
                                                      or (ci[0] <= 1 / 1.01 <= ci[1])):
                        target = 15
            hashes = {r['sha256'] for values in rows.values() for r in values}
            item = dict(config=config, pairs=pair, exact=len(hashes) == 1,
                        paired_speedup=statistics.median(ratios), ci95=ci, ratios=ratios,
                        no_regression_confirmed=ci[0] >= 1 / 1.01,
                        benefit_confirmed=ci[0] > 1,
                        environment=environment_delta(case_before, cpu_stat()))
            summary.append(item)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(item), flush=True)
            if not item['exact']: raise RuntimeError('output mismatch')
    for name, digest in manifest.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('input changed during paired run: ' + name)
    after = cpu_stat()
    environment = environment_delta(before, after)
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    stages = [r for r in summary if r['config'].get('stage') == 'two_stage']
    geometric = math.exp(statistics.mean(math.log(r['paired_speedup']) for r in stages)) if stages else None
    rng = random.Random(42)
    boot = sorted(math.exp(statistics.mean(math.log(statistics.median(rng.choices(r['ratios'], k=len(r['ratios']))))
                                         for r in stages)) for _ in range(10000)) if stages else []
    aggregate_ci = [boot[249], boot[9749]] if boot else None
    benefit = bool(stages) and geometric >= 1.01 and aggregate_ci[0] > 1
    decision = dict(kind='regression' if args.regression_only else 'optimization',
                    two_stage_geomean=geometric, two_stage_ci95=aggregate_ci,
                    passed=(args.regression_only or benefit) and bool(summary)
                    and args.pairs >= 7 and all(r['exact'] and r['no_regression_confirmed'] for r in summary)
                    and environment['valid'])
    (out / 'decision.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['worker', 'run'])
    for name in ['plugin', 'config', 'baseline', 'candidate', 'configs', 'out']: p.add_argument('--' + name)
    p.add_argument('--pairs', type=int, choices=[1, 3, 7, 15], default=7)
    p.add_argument('--extend', action='store_true')
    p.add_argument('--regression-only', action='store_true',
                   help='neutral cleanup/control gate; no speedup requirement')
    a = p.parse_args()
    if a.mode == 'worker': print(json.dumps(worker(a.plugin, json.loads(a.config))))
    else: run(a)

if __name__ == '__main__': main()
