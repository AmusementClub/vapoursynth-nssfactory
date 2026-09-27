#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Bounded current-public-API parameter screen; never an automatic default change.

prepare freezes scene splits before any outputs; run uses isolated timed workers,
append-only raw records and a wall budget; report retains incomplete/timeout cells.
Only explicitly frozen finalists may open the sealed split. All scenes in the
initial archive-backed pilot were used historically: its sealed split is campaign
held-out, not a fresh external test. Native YUV/motion/blind-noise are out of scope.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from defaults_compare import ALGORITHMS, plane_quality
from paper_compare import cpu_activity, cpu_ticks, fixture, save_json, sha

SCENES = {'house': 'search', 'DIV2K-0805': 'search',
          'barbara': 'validation', 'DIV2K-0822': 'validation',
          'peppers256': 'sealed', 'DIV2K-0840': 'sealed'}


def candidates():
    def c(name, **kwargs):
        return dict(id=name, parameters=kwargs)
    return {
        'NLM': [c('default'), c('h06-spatial', d=0, a=2, s=4, h_sigma=.6),
                c('h09-spatial', d=0, a=2, s=4, h_sigma=.9),
                c('h06-smallpatch', d=0, a=2, s=2, h_sigma=.6)],
        'BM3D': [c('default'), c('g16-step8', group_size=16, block_step=8),
                 c('g16-step6', group_size=16, block_step=6),
                 dict(id='basic-final', parameters={}, pipeline='basic-final')],
        'WNNM': [c('default'), c('b4-g16', block_size=4, block_step=4, group_size=16),
                 c('g16', group_size=16), c('g16-residual', group_size=16, residual=1)],
        'MCWNNM': [c('default'), c('i1-admm5', iters=1, admm_iter=5),
                   c('b4-g16-i2', block_size=4, block_step=4, group_size=16, iters=2),
                   c('b4-g16-i1-admm5', block_size=4, block_step=4, group_size=16,
                     iters=1, admm_iter=5)],
        'TWSC': [c('default'), c('b4-g48-i4', block_size=4, block_step=4,
                 group_size=48, iters=4, admm_iter=5),
                 c('b4-g24-i6', block_size=4, block_step=4, group_size=24,
                   iters=6, admm_iter=5)],
        'NLH': [c('default'), c('fewer-rounds', basic_iters=1, wiener_iters=1),
                dict(id='v4', parameters={}, preset='v4'),
                dict(id='v4-step4', parameters={'block_step': [4, 4]}, preset='v4')],
        'NCSR': [c('default'), c('i1', iters=1), c('g16', group_size=16),
                 c('b4-g16', block_size=4, block_step=4, group_size=16)],
        'LSSC': [c('default'), c('b4-step4', block_size=4, block_step=4),
                 c('b16-step16', block_size=16, block_step=16)],
    }


def prepare(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    scenes = {}
    for directory in args.source:
        root = Path(directory)
        for row in json.loads((root/'fixtures.json').read_text())['cases']:
            if row['image'] in SCENES:
                previous = scenes.get(row['image'])
                if previous and previous[1]['clean_sha256'] != row['clean_sha256']:
                    raise ValueError('multiple source crops for scene '+row['image'])
                scenes[row['image']] = (root, row)
    if set(scenes) != set(SCENES):
        raise ValueError('missing scenes: '+repr(set(SCENES)-set(scenes)))
    cases = []
    for scene, (root, row) in sorted(scenes.items()):
        if sha(root/row['clean']) != row['clean_sha256']:
            raise ValueError('source clean hash mismatch')
        c = row.get('channels', 1); h, w = row['height'], row['width']
        clean = np.fromfile(root/row['clean'], dtype='<f4').reshape(c, h, w)
        size = args.size
        if size < 16 or min(h, w) < size: raise ValueError('invalid crop size')
        y, x = (h-size)//2, (w-size)//2
        clean = np.ascontiguousarray(clean[:, y:y+size, x:x+size])
        clean_path = out/f'{scene}-clean.f32'; clean.tofile(clean_path)
        for sigma in args.sigmas:
            for repeat in range(args.seeds):
                seed = int.from_bytes(hashlib.sha256(
                    f'balanced-20260919:{scene}:{sigma:g}:{repeat}'.encode()).digest()[:8], 'little')
                noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape)
                noisy = (clean.astype(np.float64)+noise*(sigma/255)).astype('<f4')
                name = f'{scene}-{size}-s{sigma:g}-n{repeat}'
                path = out/(name+'-noisy.f32'); noisy.tofile(path)
                cases.append(dict(id=name, image=scene, split=SCENES[scene], width=size,
                    height=size, channels=c, sigma=sigma, seed=seed, noise_repeat=repeat,
                    clean=clean_path.name, noisy=path.name, clean_sha256=sha(clean_path),
                    noisy_sha256=sha(path), parent_clean_sha256=row['clean_sha256'],
                    parent_original_sha256=row.get('original_sha256'),
                    crop_in_parent=[x,y,size,size], parent_crop=row.get('crop')))
    save_json(out/'fixtures.json', dict(schema='nss.balanced-fixtures.v1', cases=cases,
        scene_split=SCENES, historically_used_scenes=True,
        scope='small previously-used-image pilot; campaign holdout only, not fresh external validation',
        input_policy='native clean center crop; no resize; new independent channel AWGN; unclipped float32',
        unsupported=['native YUV', 'moving clips', 'real noise', 'unequal-channel noise', 'blind estimation']))
    save_json(out/'candidates.json', dict(schema='nss.balanced-candidates.v1', algorithms=candidates()))
    save_json(out/'nlh-v4.json', json.loads((Path(__file__).parent/'data/nlh_presets_v4.json').read_text()))
    print(json.dumps(dict(fixtures=str(out), cases=len(cases), scenes=len(scenes))))


def resolve_parameters(candidate, case, fixtures):
    extra = dict(candidate['parameters'])
    if candidate.get('preset') == 'v4':
        lane = 'rgb' if case['channels'] == 3 else 'gray-low' if case['sigma'] <= 50 else 'gray-high'
        extra = dict(json.loads((Path(fixtures)/'nlh-v4.json').read_text())['profiles'][lane], **extra)
    if 'h_sigma' in extra:
        extra['h'] = extra.pop('h_sigma')*case['sigma']
    return extra


def host():
    result = dict(hostname=platform.node(), kernel=platform.release(), machine=platform.machine())
    boot = Path('/proc/sys/kernel/random/boot_id')
    if boot.exists(): result['boot_id'] = boot.read_text().strip()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        result['cpu_model'] = next((s.split(':', 1)[1].strip() for s in cpuinfo.read_text().splitlines()
                                    if s.startswith('model name')), 'unknown')
    return result


def worker(args):
    import vapoursynth as vs
    root, case = fixture(args.fixtures, args.case)
    candidate = json.loads(args.candidate_json)
    shape = (case['channels'], case['height'], case['width'])
    noisy = np.fromfile(root/case['noisy'], dtype='<f4').reshape(shape)
    clean = np.fromfile(root/case['clean'], dtype='<f4').reshape(shape)
    if args.algorithm == 'MCWNNM' and shape[0] != 3: raise ValueError('MCWNNM requires genuine RGB')
    core = vs.core; core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    # Hold only six underlying frames; Loop provides unlimited unique requests.
    blank = core.std.BlankClip(width=shape[2], height=shape[1], length=6,
                              format=vs.RGBS if shape[0] == 3 else vs.GRAYS)
    fills = 0
    def fill(n, f):
        nonlocal fills
        fills += 1; frame = f.copy()
        for p in range(shape[0]): np.copyto(np.asarray(frame[p]), noisy[p])
        return frame
    cached = core.std.ModifyFrame(blank, blank, fill)
    core.std.SetVideoCache(cached, mode=1, fixedsize=6, maxsize=6)
    held = [cached.get_frame(n) for n in range(6)]
    source = core.std.Loop(cached, times=args.max_frames+8)
    parameters = {} if args.algorithm == 'NLM' else {'sigma': case['sigma']}
    parameters.update(resolve_parameters(candidate, case, args.fixtures))
    node = getattr(core.nss, args.algorithm)(source, **parameters)
    core.std.SetVideoCache(node, mode=0)
    if candidate.get('pipeline') == 'basic-final':
        node = core.nss.BM3D(source, ref=node, **parameters)
        core.std.SetVideoCache(node, mode=0)
    if parameters.get('radius', 0): raise ValueError('this pilot is spatial-only')
    start = time.perf_counter(); warm = node.get_frame(1); warm_seconds = time.perf_counter()-start
    warm_pixels = np.stack([np.asarray(warm[p]).copy() for p in range(shape[0])])
    frames = min(args.max_frames, max(1, math.ceil(args.target_seconds/max(warm_seconds, 1e-6))))
    before_fills = fills
    before = cpu_ticks() if Path('/proc/stat').exists() else None
    start = time.perf_counter()
    for n in range(3, 3+frames): frame = node.get_frame(n)
    seconds = time.perf_counter()-start
    activity = cpu_activity(before, cpu_ticks()) if before else {}
    if fills != before_fills: raise AssertionError('source fill in timing boundary')
    pixels = np.stack([np.asarray(frame[p]).copy() for p in range(shape[0])]).astype('<f4')
    if not np.isfinite(pixels).all(): raise AssertionError('nonfinite output')
    if not np.array_equal(warm_pixels, pixels): raise AssertionError('warm/repeated static output mismatch')
    props = {k:v for k,v in frame.props.items() if k.startswith('_NSS') and isinstance(v, (int,float,list))}
    noise = noisy.astype(np.float64)-clean; error = pixels.astype(np.float64)-clean
    energy = float(np.sum(noise*noise))
    output = Path(args.output); pixels.tofile(output)
    sibling = Path('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list')
    topology = sibling.read_text().strip() if sibling.exists() else None
    # This is the repository's verified CPU0/idle CPU1 C4 lane. A different
    # topology keeps quality but cannot silently qualify these timings.
    timing_eligible = (topology == '0-1' or topology == '0,1') and bool(activity) and (
        activity['cpu1']['total_ticks'] >= 10 and activity['cpu1']['busy_fraction'] <= .01 and
        not any(c['steal_ticks'] for c in activity.values()))
    save_json(str(output)+'.json', dict(case=case['id'], image=case['image'], split=case['split'],
        algorithm=args.algorithm, candidate=candidate['id'], supplied_parameters=parameters,
        pipeline=candidate.get('pipeline', 'single'), shape=list(shape), sigma=case['sigma'],
        noisy_sha256=case['noisy_sha256'], clean_sha256=case['clean_sha256'],
        plugin_sha256=sha(args.plugin), harness_sha256=sha(__file__), output=output.name,
        output_sha256=sha(output), quality=plane_quality(clean, pixels), noisy_quality=plane_quality(clean, noisy),
        noise_projection_retention=float(np.sum(error*noise)/energy),
        removed_signal_rms_over_input_noise=float(np.sqrt(np.sum((pixels-noisy)**2)/energy)),
        seconds=seconds/frames, elapsed_seconds=seconds, frames=frames, warmup_seconds=warm_seconds,
        output_cache_disabled=True, source_frames_preloaded=len(held), timed_source_fills=0,
        cpu_activity=activity, cpu0_thread_siblings=topology, timing_eligible=timing_eligible,
        timing_scope='warm unique uncached requests; preloaded source; serialization excluded',
        temporal_policy='repeated static noisy frame; NLM default d=1 has no independent temporal information',
        resource_props=props, backend={k:v.decode() if isinstance(v,bytes) else v
            for k,v in core.nss.Backend().items()}, host=host()))


def selected_cases(args):
    manifest = json.loads((Path(args.fixtures)/'fixtures.json').read_text())
    all_cases = manifest['cases']
    scene_splits = {}
    for row in all_cases:
        if row['image'] in scene_splits and scene_splits[row['image']] != row['split']:
            raise ValueError('scene leaks across splits')
        scene_splits[row['image']] = row['split']
    return [r for r in all_cases if r['split'] == args.split and
            (not args.sigmas or r['sigma'] in args.sigmas) and
            (args.noise_repeat is None or r['noise_repeat'] == args.noise_repeat)]


def run(args):
    if not hasattr(os, 'sched_getaffinity') or os.sched_getaffinity(0) != {0}:
        raise ValueError('campaign must run under taskset -c 0 on verified C4')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    config = Path(args.candidates or Path(args.fixtures)/'candidates.json')
    recipes = json.loads(config.read_text())['algorithms']
    finalists = json.loads(Path(args.finalists).read_text()) if args.finalists else None
    if args.split == 'sealed' and not finalists: raise ValueError('sealed split requires frozen --finalists')
    if finalists and finalists['candidates_sha256'] != sha(config): raise ValueError('finalist config changed')
    cases = selected_cases(args)
    if not cases: raise ValueError('no cases selected')
    identity = dict(plugin_sha256=sha(args.plugin), fixtures_sha256=sha(Path(args.fixtures)/'fixtures.json'),
        candidates_sha256=sha(config), nlh_v4_sha256=sha(Path(args.fixtures)/'nlh-v4.json'),
        harness_sha256=sha(__file__),
        metrics_sha256=sha(Path(__file__).with_name('paper_compare.py')),
        defaults_helper_sha256=sha(Path(__file__).with_name('defaults_compare.py')),
        finalists_sha256=sha(args.finalists) if args.finalists else None,
        split=args.split, cases=[c['id'] for c in cases], algorithms=args.algorithms, repeats=args.repeats,
        timeout=args.timeout, twsc_timeout=args.twsc_timeout, target_seconds=args.target_seconds,
        max_frames=args.max_frames, candidate_ids=args.candidate_ids)
    identity_path = out/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity: raise ValueError('resume identity differs')
    else: save_json(identity_path, identity)
    save_json(out/'host.json', host())
    raw = out/'results.jsonl'
    rows = [json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['case'], r['algorithm'], r['candidate'], r['repeat']) for r in rows}
    jobs = []
    for case in cases:
        for algorithm in args.algorithms:
            selected = recipes[algorithm]
            if args.candidate_ids: selected = [c for c in selected if c['id'] in args.candidate_ids]
            if finalists:
                names = ['default', *finalists['algorithms'].get(algorithm, [])]
                selected = [c for c in selected if c['id'] in names]
            if not selected: raise ValueError('no candidates for '+algorithm)
            for repeat in range(args.repeats):
                for candidate in (selected if repeat%2 == 0 else selected[::-1]):
                    jobs.append((case, algorithm, candidate, repeat))
    started = time.monotonic(); deadline = started+args.budget_seconds
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for case, algorithm, candidate, repeat in jobs:
        key = (case['id'], algorithm, candidate['id'], repeat)
        if key in done: continue
        if time.monotonic() >= deadline: break
        stem = '-'.join(map(str,key)); output = out/(stem+'.f32')
        row = dict(case=key[0], algorithm=algorithm, candidate=candidate['id'], repeat=repeat,
                   split=args.split, status='pending', ok=False)
        if algorithm == 'MCWNNM' and case['channels'] != 3:
            row.update(status='unsupported', error='genuine RGB required; gray duplication forbidden')
        else:
            timeout = min(args.twsc_timeout if algorithm == 'TWSC' else args.timeout,
                          max(.01, deadline-time.monotonic()))
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--fixtures',
                str(Path(args.fixtures).resolve()), '--case', case['id'], '--algorithm', algorithm,
                '--candidate-json', json.dumps(candidate), '--plugin', str(Path(args.plugin).resolve()),
                '--output', str(output.resolve()), '--target-seconds', str(args.target_seconds),
                '--max-frames', str(args.max_frames)]
            launched = time.monotonic()
            try:
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                output.with_suffix('.log').write_bytes(result.stdout)
                if result.returncode: raise RuntimeError(result.stdout[-3500:].decode(errors='replace'))
                row.update(json.loads(Path(str(output)+'.json').read_text()), status='complete', ok=True)
            except subprocess.TimeoutExpired as error:
                output.with_suffix('.log').write_bytes(error.stdout or b'')
                row.update(status='timeout', timeout_seconds=timeout,
                           error='bounded worker killed; no quality or runtime conclusion')
            except Exception as error:
                row.update(status='failed', error=str(error))
            row['worker_wall_seconds'] = time.monotonic()-launched
        with raw.open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
        rows.append(row); done.add(key)
        print(f"{stem}: {row['status']}"+(f" {row['seconds']:.5f}s {row['quality']['psnr_db']:.3f}dB" if row['ok'] else ''), flush=True)
    remaining = [dict(case=c['id'], algorithm=a, candidate=p['id'], repeat=r)
                 for c,a,p,r in jobs if (c['id'],a,p['id'],r) not in done]
    save_json(out/'completion.json', dict(complete=not remaining, scheduled=len(jobs),
        recorded=len(rows), successful=sum(r['ok'] for r in rows), remaining=remaining,
        wall_seconds=time.monotonic()-started, budget_seconds=args.budget_seconds,
        timeouts=sum(r['status']=='timeout' for r in rows),
        resume_policy='exact identity only; terminal timeouts/failures retained, not retried silently'))
    report(argparse.Namespace(results=str(out)))


def report(args):
    out = Path(args.results)
    rows = [json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    identity = json.loads((out/'identity.json').read_text())
    cells = {}
    for r in rows:
        if r['ok']: cells.setdefault((r['case'],r['algorithm'],r['candidate']), []).append(r)
    comparisons = []
    for (case, algorithm, candidate), values in cells.items():
        if candidate == 'default' or (case,algorithm,'default') not in cells: continue
        baseline = cells[(case,algorithm,'default')]
        if len({json.dumps(v['supplied_parameters'], sort_keys=True) for v in values}) != 1:
            raise ValueError('candidate repeat parameters changed')
        if len({json.dumps(v['supplied_parameters'], sort_keys=True) for v in baseline}) != 1:
            raise ValueError('default repeat parameters changed')
        if len({v['output_sha256'] for v in values}) != 1: raise ValueError('candidate repeat output changed')
        if len({v['output_sha256'] for v in baseline}) != 1: raise ValueError('default repeat output changed')
        pairs = [(b,v) for b in baseline for v in values if b['repeat'] == v['repeat']]
        eligible = [(b,v) for b,v in pairs if b['timing_eligible'] and v['timing_eligible'] and
                    b['host'] == v['host'] and b['noisy_sha256'] == v['noisy_sha256']]
        speedup = statistics.median(b['seconds']/v['seconds'] for b,v in eligible) if eligible else None
        b,v = baseline[0], values[0]
        comparisons.append(dict(case=case, algorithm=algorithm, candidate=candidate, speedup=speedup,
            valid_pairs=len(eligible), psnr_delta_db=v['quality']['psnr_db']-b['quality']['psnr_db'],
            ssim_delta=v['quality']['ssim']-b['quality']['ssim'],
            psnr_vs_noisy_db=v['quality']['psnr_db']-v['noisy_quality']['psnr_db'],
            seconds=v['seconds'], default_seconds=b['seconds']))
    decisions = []
    for algorithm in ALGORITHMS:
        items = [r for r in comparisons if r['algorithm'] == algorithm]
        decisions.append(dict(algorithm=algorithm, comparisons=len(items),
            decision='no qualified replacement: current recipe retained pending complete validation and sealed gate',
            candidates=sorted({r['candidate'] for r in items})))
    save_json(out/'summary.json', dict(schema='nss.balanced-screen.v1', identity=identity,
        comparisons=comparisons, decisions=decisions,
        failures=[r for r in rows if not r['ok']],
        qualification='screen evidence only; requires scene-diverse larger validation, sealed quality, visual and temporal checks',
        policy=dict(median_speedup=1.2,median_psnr_loss_db=.2,worst_psnr_loss_db=.5,worst_ssim_loss=.005)))
    lines = ['# Balanced parameter screen', '',
        'Pilot evidence only. All initial scenes were used historically. No production defaults changed.', '',
        '| Case | Algorithm | Candidate | PSNR delta dB | SSIM delta | Speedup | Eligible pairs |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        speed = f"{r['speedup']:.3f}x" if r['speedup'] else 'unqualified'
        lines.append(f"| {r['case']} | {r['algorithm']} | {r['candidate']} | {r['psnr_delta_db']:+.3f} | {r['ssim_delta']:+.5f} | {speed} | {r['valid_pairs']} |")
    lines += ['', 'Timeouts and failed/unsupported cells are retained in results.jsonl; pending cells are listed in completion.json.',
              '', 'Every algorithm retains its current recipe pending complete validation and sealed quality gates.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--source', nargs='+', required=True); s.add_argument('--out', required=True)
    s.add_argument('--size', type=int, default=128); s.add_argument('--sigmas', nargs='+', type=float, default=[5,10,25,50,75])
    s.add_argument('--seeds', type=int, default=2)
    s = sub.add_parser('run')
    for k in ('plugin','fixtures','out'): s.add_argument('--'+k, required=True)
    s.add_argument('--split', choices=['search','validation','sealed'], default='search')
    s.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=list(ALGORITHMS))
    s.add_argument('--candidates'); s.add_argument('--finalists'); s.add_argument('--candidate-ids', nargs='+')
    s.add_argument('--sigmas', nargs='+', type=float); s.add_argument('--noise-repeat', type=int)
    s.add_argument('--repeats', type=int, default=1); s.add_argument('--budget-seconds', type=float, default=1800)
    s.add_argument('--timeout', type=float, default=180); s.add_argument('--twsc-timeout', type=float, default=120)
    s.add_argument('--target-seconds', type=float, default=.3); s.add_argument('--max-frames', type=int, default=256)
    s = sub.add_parser('worker')
    for k in ('plugin','fixtures','case','algorithm','candidate-json','output'): s.add_argument('--'+k, required=True)
    s.add_argument('--target-seconds', type=float, default=.3); s.add_argument('--max-frames', type=int, default=256)
    s = sub.add_parser('report'); s.add_argument('--results', required=True)
    args = p.parse_args()
    for name in ('budget_seconds','timeout','twsc_timeout','target_seconds','max_frames','repeats','seeds'):
        if hasattr(args,name) and getattr(args,name) <= 0: p.error(name+' must be positive')
    globals()[args.mode](args)


if __name__ == '__main__': main()
