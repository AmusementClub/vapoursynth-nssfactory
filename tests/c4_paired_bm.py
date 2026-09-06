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
import resource
import tempfile
import shutil
from bm_numerics import compare


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
    radius=config.get('kwargs',{}).get('radius',config.get('radius',0))
    warmup=config.get('warmup',3)
    first=warmup+4*radius
    rolling_chunk=0
    if radius and config.get('kwargs',{}).get('temporal_mode')=='rolling':
        rolling_chunk=config['kwargs'].get('rolling_chunk',4)
        first=((first+rolling_chunk-1)//rolling_chunk)*rolling_chunk
        frames=((max(frames,rolling_chunk)+rolling_chunk-1)//rolling_chunk)*rolling_chunk
    source = make_source(core, algorithm, first+frames+4*radius, w, h, 42, config.get('sample'))
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
        fn = getattr(core.nss, {'bm3d': 'BM3D', 'nlh': 'NLH', 'wnnm': 'WNNM',
                                'twsc': 'TWSC', 'ncsr': 'NCSR', 'mcwnnm': 'MCWNNM'}[algorithm])
        if stage == 'two_stage':
            pilot = fn(source, **kw)
            if radius and kw.get('temporal_mode','legacy') != 'rolling': pilot = core.nss.VAggregate(pilot, source, radius=radius)
            output = fn(source, ref=pilot, **kw)
        elif stage == 'wiener': output = fn(source, ref=source, **kw)
        else: output = fn(source, **kw)
        if radius and kw.get('temporal_mode','legacy') != 'rolling' and config.get('aggregate_temporal', True): output = core.nss.VAggregate(output, source, radius=radius)
    # Materialize raw input before timing; retain references through the timed window.
    inputs=[source.get_frame(i) for i in range(source.num_frames)]
    for n in range(max(0,first-warmup),first): output.get_frame(n)
    order=list(range(first, first+frames))
    if config.get('access')=='random': random.Random(42).shuffle(order)
    start = time.perf_counter()
    if core.num_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(core.num_threads) as pool:
            results = list(pool.map(output.get_frame, order))
    else: results = [output.get_frame(i) for i in order]
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    for frame in results:
        for plane in range(frame.format.num_planes):
            values = np.ascontiguousarray(np.asarray(frame[plane]), dtype=np.float32)
            if not np.isfinite(values).all(): raise RuntimeError('nonfinite output')
            digest.update(values.tobytes())
    if config.get('_dump'):
        selected=results if w<256 else results[:1]
        np.save(config['_dump'],np.stack([np.stack([np.array(f[p]) for p in range(f.format.num_planes)]) for f in selected]))
    return dict(ms=elapsed * 1000 / frames, timed_first=first, timed_frames=frames, rolling_chunk=rolling_chunk, access=config.get('access','sequential'), sha256=digest.hexdigest(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


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


def invoke_worker(args, name, config, timeout=None):
    cmd = ['taskset', '-c', '0', sys.executable, __file__, 'worker',
           '--plugin', getattr(args, name), '--config', json.dumps(config)]
    start = time.monotonic()
    row = json.loads(subprocess.check_output(cmd, text=True, timeout=timeout))
    row['worker_wall_seconds'] = time.monotonic() - start
    return row


def calibrate_group(args, config, deadline, dumps=None, case_id=0):
    probe = dict(config, frames=1)
    kw = config.get('kwargs', {})
    if config.get('access') == 'random' and kw.get('temporal_mode') == 'rolling':
        probe['frames'] = 4 * kw.get('rolling_chunk', 4)
    rows = []
    for name in ('baseline', 'candidate'):
        worker_config = dict(probe)
        if dumps is not None:
            worker_config['_dump'] = str(dumps/f'{case_id}-{name}.npy')
        rows.append(invoke_worker(args, name, worker_config, max(.001, deadline-time.monotonic())))
    quantum = max(r['rolling_chunk'] for r in rows) or 1
    per_frame = sum(r['ms']/1000 for r in rows)
    overhead = sum(max(0, r['worker_wall_seconds']-r['ms']/1000*r['timed_frames']) for r in rows)
    if config.get('access') == 'random':
        # Larger random windows miss the bounded chunk cache more often than
        # the four-chunk probe. Reserve headroom while still scaling frame count.
        per_frame *= 1.5
    remaining = max(0, deadline-time.monotonic()-2)
    minimum_frames = probe['frames'] if config.get('access') == 'random' else quantum
    minimum_pair = overhead + minimum_frames*per_frame
    pairs = next((n for n in (15, 7, 3, 1) if n <= args.pairs and n*minimum_pair <= remaining*.9), 0)
    frames = minimum_frames if not pairs else max(minimum_frames, min(128, int((remaining*.9/pairs-overhead)/per_frame))//quantum*quantum)
    return dict(config, frames=frames), pairs, rows


def run(args):
    os.sched_setaffinity(0,{0})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    configs = json.loads(Path(args.configs).read_text())
    manifest = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [args.baseline, args.candidate, __file__, args.configs,
                          Path(__file__).with_name('profile_cpu_all.py'),Path(__file__).with_name('bm_numerics.py')]}
    for config in configs:
        if config.get('sample'):
            p = config['sample']; manifest[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    dump_tmp=tempfile.TemporaryDirectory(prefix='nss-paired-',dir='/dev/shm')
    dumps=Path(dump_tmp.name)
    os.sync()
    time.sleep(2)
    before = cpu_stat()
    (out / 'environment-before.json').write_text(json.dumps(before, indent=2))
    # Recheck code and inputs after the run as well: mutable campaign paths
    # must never silently change midway through a comparison.

    summary = []
    (out/'summary.json').write_text('[]\n')
    budgets = []
    with (out / 'raw.jsonl').open('w') as raw:
        for case_id, config in enumerate(configs):
            group_start = time.monotonic()
            deadline = group_start + args.group_seconds if args.group_seconds else None
            case_before = cpu_stat()
            rows = {'baseline': [], 'candidate': []}
            target = args.pairs
            effective_config = dict(config)
            budget = dict(config=config, target_seconds=args.group_seconds)
            pair = 0
            if deadline:
                calibration = None
                try:
                    effective_config, target, calibration = calibrate_group(args, config, deadline, dumps, case_id)
                    budget.update(calibration=calibration, frames=effective_config['frames'], planned_pairs=target)
                except subprocess.TimeoutExpired:
                    target = 0
                    budget['reason'] = 'calibration exceeded group budget'
                if target == 0 and calibration:
                    # Preserve the complete minimal A/B probe when another pair
                    # cannot fit. It remains a one-pair diagnostic, never a gate.
                    target = pair = 1
                    budget['calibration_used_as_pair'] = True
                    for name, row in zip(('baseline', 'candidate'), calibration):
                        rows[name].append(row)
                        raw.write(json.dumps(dict(config=config, variant=name, pair=0, **row))+'\n'); raw.flush()
                if target == 0:
                    budget.update(reason=budget.get('reason', 'no complete pair fits group budget'), completed_pairs=0,
                                  wall_seconds=time.monotonic()-group_start)
                    budgets.append(budget)
                    (out/'budgets.json').write_text(json.dumps(budgets, indent=2))
                    print(json.dumps(budget), flush=True)
                    continue
            while pair < target:
                pair_rows = {}
                for name in (['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']):
                    worker_config=dict(effective_config)
                    if pair==0: worker_config['_dump']=str(dumps / f'{case_id}-{name}.npy')
                    try:
                        row = invoke_worker(args, name, worker_config, max(.001, deadline-time.monotonic()) if deadline else None)
                    except subprocess.TimeoutExpired:
                        budget['reason'] = 'worker exceeded group budget; unfinished pair excluded'
                        break
                    pair_rows[name] = row
                if len(pair_rows) != 2:
                    break
                for name, row in pair_rows.items():
                    rows[name].append(row)
                    raw.write(json.dumps(dict(config=config, variant=name, pair=pair, **row)) + '\n'); raw.flush()
                pair += 1
                if pair == target:
                    ratios = [a['ms'] / b['ms'] for a, b in zip(rows['baseline'], rows['candidate'])]
                    ci = interval(ratios)
                    # Resolve uncertain benefit or the 1% regression boundary once.
                    if not deadline and args.extend and target == 7 and ((not args.regression_only and ci[0] <= 1 <= ci[1])
                                                      or (ci[0] <= 1 / 1.01 <= ci[1])):
                        target = 15
            if deadline:
                budget.update(completed_pairs=pair, wall_seconds=time.monotonic()-group_start)
                budgets.append(budget)
                (out/'budgets.json').write_text(json.dumps(budgets, indent=2))
            if not pair:
                continue
            ratios = [a['ms'] / b['ms'] for a, b in zip(rows['baseline'], rows['candidate'])]
            ci = interval(ratios)
            hashes = {r['sha256'] for values in rows.values() for r in values}
            import numpy as np
            numerical=compare(np.load(dumps/f'{case_id}-baseline.npy'),np.load(dumps/f'{case_id}-candidate.npy'))
            item = dict(numerical=numerical,config=config, effective_config=effective_config, budget=budget if deadline else None,
                        pairs=pair, exact=len(hashes) == 1,
                        paired_speedup=statistics.median(ratios), ci95=ci, ratios=ratios,
                        no_regression_confirmed=ci[0] >= 1 / 1.01,
                        benefit_confirmed=ci[0] > 1,
                        environment=environment_delta(case_before, cpu_stat()))
            if getattr(args, 'selection_threshold', None) is not None:
                item['selected'] = numerical['passed'] and item['environment']['valid'] and item['paired_speedup'] > args.selection_threshold
                item['selection_evidence'] = 'single_pair' if pair == 1 else 'repeated_pairs'
            summary.append(item)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(item), flush=True)
            if not numerical['passed']:
                for side in ['baseline','candidate']:shutil.copy2(dumps/f'{case_id}-{side}.npy',out/f'{case_id}-{side}.npy')
                if not args.semantic_change:raise RuntimeError('numerical deviation requires stage replay; arrays preserved')
            if numerical['passed']:
                (dumps/f'{case_id}-baseline.npy').unlink()
                (dumps/f'{case_id}-candidate.npy').unlink()
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
    decision = dict(kind='semantic_change' if args.semantic_change else 'regression' if args.regression_only else 'optimization',
                    two_stage_geomean=geometric, two_stage_ci95=aggregate_ci,
                    passed=not args.semantic_change and (args.regression_only or benefit) and bool(summary)
                    and len(summary) == len(configs)
                    and all(r['pairs'] >= 7 and r['numerical']['passed'] and r['no_regression_confirmed'] and r['environment']['valid'] for r in summary)
                    and environment['valid'])
    (out / 'decision.json').write_text(json.dumps(decision, indent=2))
    if getattr(args, 'selection_threshold', None) is not None:
        selection = dict(kind='per_configuration_selection', threshold=args.selection_threshold,
                         selected=[r['config']['name'] for r in summary if r['selected']],
                         missing=[c['name'] for c in configs if c['name'] not in {r['config']['name'] for r in summary}],
                         policy='median speedup exceeds threshold; numerical and per-case environment checks required; CI and pair count reported')
        (out/'selection.json').write_text(json.dumps(selection, indent=2))
    dump_tmp.cleanup()
    print(json.dumps(decision), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['worker', 'run'])
    for name in ['plugin', 'config', 'baseline', 'candidate', 'configs', 'out']: p.add_argument('--' + name)
    p.add_argument('--pairs', type=int, choices=[1, 3, 7, 15], default=7)
    p.add_argument('--extend', action='store_true')
    p.add_argument('--selection-threshold', type=float, help='separate per-case selection policy; does not rewrite historical formal gate')
    p.add_argument('--group-seconds', type=float, default=0,
                   help='calibrate each entire A/B group including startup/warmup; disables extension; fewer than 7 pairs cannot pass')
    p.add_argument('--semantic-change',action='store_true',help='B0-to-B1 diagnostic only; cannot pass an optimization gate')
    p.add_argument('--regression-only', action='store_true',
                   help='neutral cleanup/control gate; no speedup requirement')
    a = p.parse_args()
    if a.group_seconds and not 30 <= a.group_seconds <= 60:
        p.error('--group-seconds must be between 30 and 60')
    if a.selection_threshold is not None and (not math.isfinite(a.selection_threshold) or a.selection_threshold <= 1):
        p.error('--selection-threshold must be finite and greater than one')
    if a.mode == 'worker': print(json.dumps(worker(a.plugin, json.loads(a.config))))
    else: run(a)

if __name__ == '__main__': main()
