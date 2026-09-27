#!/usr/bin/env python3
"""Bounded moving-crop quality check for frozen public-API recipes.

Use balanced_campaign's {algorithms: {NAME: [{id, parameters, pipeline?}]}}
recipe schema. Explicit --cases prevent silently opening every fixture split.
Temporal filters use explicit radius1/2, including NLM d; LSSC is a spatial control.
This synthetic translated-scene check is not natural-video or timing admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from balanced_campaign import resolve_parameters
from defaults_compare import ALGORITHMS, plane_quality
from paper_compare import fixture, save_json, sha


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f4').tobytes()).hexdigest()


def temporal_layout(algorithm, pipeline, radius, frames):
    if radius not in (1, 2): raise ValueError('temporal radius must be1 or2')
    if frames < 7 or frames % 2 != 1: raise ValueError('at least7 odd source frames required')
    if pipeline not in ('single', 'basic-final') or (pipeline == 'basic-final' and algorithm != 'BM3D'):
        raise ValueError('only BM3D supports the basic-final recipe pipeline')
    effective = 0 if algorithm == 'LSSC' else radius
    # VAggregate reads centers +/-r; each center reads source +/-r. A BM3D
    # final stage also requests its aggregated Basic guide over that window.
    halo = effective if algorithm in ('NLM', 'LSSC') else 2*effective
    if pipeline == 'basic-final': halo *= 2
    if frames < 2*halo+1:
        raise ValueError(f'{algorithm}/{pipeline} radius{radius} requires at least{2*halo+1} source frames for a full-halo interior output')
    interior = list(range(halo, frames-halo))
    return dict(requested_radius=radius, effective_radius=effective, source_halo=halo,
        interior_frames=interior, boundary_frames=[n for n in range(frames) if n not in interior],
        source_halo_policy='Interior outputs have the full composed source window; clip boundaries use the public temporal contract')


def moving_input(root, case, frames):
    manifest = json.loads((root/'fixtures.json').read_text())
    if any(metadata.get('channel_sigma') is not None or metadata.get('do_not_pool_with_homogeneous_noise')
           for metadata in (case, manifest)):
        raise ValueError('channel_sigma/unequal-channel fixtures are unsupported: temporal noise generation is homogeneous')
    if frames < 7 or frames % 2 != 1: raise ValueError('at least7 odd source frames required')
    parent = np.fromfile(root/case['clean'], dtype='<f4').reshape(
        case['channels'], case['height'], case['width'])
    margin = frames//2
    height, width = parent.shape[1]-2*margin, parent.shape[2]-2*margin
    if min(height, width) < 16 or not np.isfinite(parent).all() or not case['sigma'] > 0:
        raise ValueError('finite clean fixture, positive sigma and >=16-pixel moving crops required')
    offsets = [(margin+(n % 3)-1, n) for n in range(frames)]
    clean = np.stack([parent[:, y:y+height, x:x+width] for y, x in offsets])
    seed = int.from_bytes(hashlib.sha256(
        f"temporal-20260919:{case['id']}:{case['seed']}".encode()).digest()[:8], 'little')
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape)
    noisy = (clean.astype(np.float64)+noise*(case['sigma']/255)).astype('<f4')
    return clean, noisy, offsets, seed


def residual_variation(clean, noisy, output, offsets):
    height, width = clean.shape[-2:]
    top, left = max(y for y, x in offsets), max(x for y, x in offsets)
    bottom = min(y+height for y, x in offsets)
    right = min(x+width for y, x in offsets)
    def aligned(values):
        return np.stack([values[n, :, top-y:bottom-y, left-x:right-x]
                         for n, (y, x) in enumerate(offsets)])
    # Every aligned pixel refers to exactly the same clean parent coordinate.
    reference = aligned(clean)
    if not np.array_equal(reference, np.broadcast_to(reference[0], reference.shape)):
        raise AssertionError('motion alignment does not preserve clean-scene identity')
    residual = aligned(output.astype(np.float64)-clean)
    noise = aligned(noisy.astype(np.float64)-clean)
    return dict(aligned_shape=list(residual.shape),
        output_adjacent_rms=[float(np.sqrt(np.mean(x*x))) for x in np.diff(residual, axis=0)],
        noisy_adjacent_rms=[float(np.sqrt(np.mean(x*x))) for x in np.diff(noise, axis=0)],
        output_temporal_std_rms=float(np.sqrt(np.mean(np.var(residual, axis=0)))),
        noisy_temporal_std_rms=float(np.sqrt(np.mean(np.var(noise, axis=0)))),
        interpretation='motion-aligned residual variation; natural-video flicker is not established')


def quality(clean, pixels):
    value = plane_quality(clean, pixels)
    if not np.isfinite(value['psnr_db']): value['psnr_db'] = None
    return value


def worker(args):
    import vapoursynth as vs
    root, case = fixture(args.fixtures, args.case)
    recipe = json.loads(args.recipe_json)
    pipeline = recipe.get('pipeline', 'single')
    layout = temporal_layout(args.algorithm, pipeline, args.radius, args.frames)
    clean, noisy, offsets, seed = moving_input(root, case, args.frames)
    frames, channels, height, width = noisy.shape
    if args.algorithm == 'MCWNNM' and channels != 3:
        raise ValueError('MCWNNM requires genuine RGB')
    if channels not in (1, 3): raise ValueError('Gray/RGB fixtures required')
    parameters = {} if args.algorithm == 'NLM' else dict(sigma=case['sigma'])
    parameters.update(resolve_parameters(recipe, case, args.fixtures))
    parameters['d' if args.algorithm == 'NLM' else 'radius'] = layout['effective_radius']
    core = vs.core; core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    blank = core.std.BlankClip(width=width, height=height, length=frames,
                              format=vs.GRAYS if channels == 1 else vs.RGBS)
    fills = 0
    def fill(n, f):
        nonlocal fills
        fills += 1; frame = f.copy()
        for p in range(channels): np.copyto(np.asarray(frame[p]), noisy[n, p])
        return frame
    source = core.std.ModifyFrame(blank, blank, fill)
    core.std.SetVideoCache(source, mode=1, fixedsize=frames, maxsize=frames)
    held = [source.get_frame(n) for n in range(frames)]
    source_before = [array_sha(np.stack([np.asarray(f[p]) for p in range(channels)])) for f in held]
    if source_before != [array_sha(a) for a in noisy]: raise AssertionError('source payload differs')
    def build():
        def stage(reference=None):
            options = dict(parameters)
            if reference is not None: options['ref'] = reference
            raw = getattr(core.nss, args.algorithm)(source, **options)
            core.std.SetVideoCache(raw, mode=0)
            node = core.nss.VAggregate(raw, source, radius=layout['effective_radius']) if args.algorithm not in ('NLM', 'LSSC') else raw
            core.std.SetVideoCache(node, mode=0)
            return node
        basic = stage()
        return stage(basic) if pipeline == 'basic-final' else basic
    odd_order = [frames-1, 0, frames//2]
    odd_order += [n for n in range(frames-1, -1, -1) if n not in odd_order]
    outputs = []; properties = {}; started = time.monotonic(); before = fills
    for repeat, order in enumerate((list(range(frames)), odd_order)):
        node = build(); captured = {}
        for n in order:
            with node.get_frame(n) as frame:
                captured[n] = np.stack([np.array(frame[p], copy=True) for p in range(channels)])
                if repeat == 0:
                    properties[str(n)] = {k:v for k,v in frame.props.items()
                        if k.startswith('_NSS') and isinstance(v, (int, float, list))}
        outputs.append(np.stack([captured[n] for n in range(frames)]))
        del node
    render_seconds = time.monotonic()-started
    repeat_exact = np.array_equal(outputs[0], outputs[1])
    output = Path(args.output)
    # Keep discrepant arrays if the repeat/order gate fails, rather than only
    # retaining a worker error string.
    np.savez_compressed(output, clean=clean, noisy=noisy, output=outputs[0], offsets=np.array(offsets),
                        **({} if repeat_exact else {'repeat_output': outputs[1]}))
    if fills != before: raise AssertionError('preloaded source was recomputed')
    if not np.isfinite(outputs[0]).all(): raise AssertionError('nonfinite final output')
    if not repeat_exact: raise AssertionError('fresh-node/request-order output mismatch')
    source_after = [array_sha(np.stack([np.asarray(f[p]) for p in range(channels)])) for f in held]
    if source_after != source_before: raise AssertionError('source frame was mutated')
    save_json(str(output)+'.json', dict(case=case['id'], split=case['split'], algorithm=args.algorithm,
        recipe=recipe['id'], supplied_parameters=parameters, pipeline=pipeline, frames=frames, radius=args.radius,
        temporal_layout=layout,
        shape=list(noisy.shape), motion_offsets_yx=offsets, noise_seed=seed, sigma=case['sigma'],
        temporal_supported=args.algorithm != 'LSSC',
        role='spatial-per-frame control; radius>0 unsupported' if args.algorithm == 'LSSC' else f'temporal radius{args.radius}',
        independent_noise_per_frame=True, unclipped=True, source_preloaded=frames,
        source_sha256=array_sha(noisy), clean_sha256=array_sha(clean), parent_clean_sha256=case['clean_sha256'],
        source_frame_sha256=source_before, final_frame_sha256=[array_sha(a) for a in outputs[0]],
        final_sha256=array_sha(outputs[0]), repeat_final_sha256=array_sha(outputs[1]),
        repeat_exact=True, source_unchanged=True,
        request_orders=[list(range(frames)), odd_order], timed_source_fills=0,
        quality=[dict(frame=n, output=quality(clean[n], outputs[0][n]), noisy=quality(clean[n], noisy[n]))
                 for n in range(frames)], residual_variation=residual_variation(clean, noisy, outputs[0], offsets),
        resource_props=properties, resource_scope='final node; VAggregate owns these props for aggregated filters',
        worker_render_seconds=render_seconds, timing_role='diagnostic elapsed time, not a performance gate',
        plugin_sha256=sha(args.plugin), harness_sha256=sha(__file__), output=output.name, output_sha256=sha(output),
        backend={k:v.decode() if isinstance(v,bytes) else v for k,v in core.nss.Backend().items()},
        platform=platform.platform(), vapoursynth=str(vs.__version__)))


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recipes = json.loads(Path(args.recipes).read_text())['algorithms']
    cases = [fixture(args.fixtures, name)[1] for name in args.cases]
    if any(c['split'] == 'sealed' for c in cases) and not args.allow_sealed:
        raise ValueError('sealed cases require explicitly frozen recipes and --allow-sealed')
    identity = dict(plugin=sha(args.plugin), recipes=sha(args.recipes), harness=sha(__file__),
        helpers={name:sha(Path(__file__).with_name(name)) for name in
                 ('balanced_campaign.py', 'defaults_compare.py', 'paper_compare.py')},
        fixtures=sha(Path(args.fixtures)/'fixtures.json'), nlh_v4=sha(Path(args.fixtures)/'nlh-v4.json'),
        cases=args.cases, frames=args.frames, radius=args.radius, timeout=args.timeout, twsc_timeout=args.twsc_timeout)
    path = out/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity: raise ValueError('resume identity changed')
    save_json(path, identity)
    raw = out/'results.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['case'], r['algorithm'], r['recipe']) for r in rows}
    jobs = [(case, algorithm, recipe) for case in cases for algorithm, entries in recipes.items() for recipe in entries]
    keys = [(case['id'], algorithm, recipe['id']) for case, algorithm, recipe in jobs]
    if len(keys) != len(set(keys)): raise ValueError('duplicate recipe/case identities')
    if any(a not in ALGORITHMS for c,a,r in jobs): raise ValueError('unknown algorithm')
    for case, algorithm, recipe in jobs:
        temporal_layout(algorithm, recipe.get('pipeline', 'single'), args.radius, args.frames)
    deadline = time.monotonic()+args.budget_seconds
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for index, ((case, algorithm, recipe), key) in enumerate(zip(jobs, keys)):
        if key in done: continue
        remaining = deadline-time.monotonic()
        if remaining <= 0: break
        output = out/f'{index:04d}-{algorithm}.npz'
        row = dict(case=case['id'], algorithm=algorithm, recipe=recipe['id'], radius=args.radius, status='pending')
        if algorithm == 'MCWNNM' and case['channels'] != 3:
            row.update(status='unsupported', reason='genuine RGB required')
        else:
            timeout = min(args.twsc_timeout if algorithm == 'TWSC' else args.timeout, remaining)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--plugin', str(Path(args.plugin).resolve()),
                '--fixtures', str(Path(args.fixtures).resolve()), '--case', case['id'], '--algorithm', algorithm,
                '--recipe-json', json.dumps(recipe), '--frames', str(args.frames), '--radius', str(args.radius),
                '--output', str(output.resolve())]
            started = time.monotonic()
            try:
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
                output.with_suffix('.log').write_bytes(result.stdout)
                if result.returncode: raise RuntimeError(result.stdout[-3000:].decode(errors='replace'))
                row.update(json.loads(Path(str(output)+'.json').read_text()), status='complete')
            except subprocess.TimeoutExpired as error:
                output.with_suffix('.log').write_bytes(error.stdout or b'')
                row.update(status='timeout', timeout_seconds=timeout, reason='incomplete temporal evidence')
            except Exception as error: row.update(status='failed', reason=str(error))
            row['worker_wall_seconds'] = time.monotonic()-started
        with raw.open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
        rows.append(row); done.add(key); print(key, row['status'], flush=True)
    save_json(out/'summary.json', dict(identity=identity, records=rows,
        pending=[dict(case=c, algorithm=a, recipe=r) for c,a,r in keys if (c,a,r) not in done],
        counts={status:sum(r['status']==status for r in rows) for status in ('complete','unsupported','timeout','failed')},
        all_completed=all(r['status']=='complete' for r in rows) and len(done)==len(keys),
        qualification='synthetic moving-crop check only; no automatic recipe or temporal-quality admission',
        cumulative_budget_note='budget applies per invocation; coordinator accounts for resumed total wall time'))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('run', 'worker'):
        p = sub.add_parser(mode)
        for key in ('plugin', 'fixtures'): p.add_argument('--'+key, required=True)
        p.add_argument('--frames', type=int, default=7)
        p.add_argument('--radius', type=int, choices=(1, 2), default=1)
        if mode == 'run':
            for key in ('recipes', 'out'): p.add_argument('--'+key, required=True)
            p.add_argument('--cases', nargs='+', required=True); p.add_argument('--allow-sealed', action='store_true')
            p.add_argument('--budget-seconds', type=float, default=900)
            p.add_argument('--timeout', type=float, default=180); p.add_argument('--twsc-timeout', type=float, default=60)
        else:
            for key in ('case', 'algorithm', 'recipe-json', 'output'): p.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if args.frames < 7 or args.frames % 2 != 1: parser.error('--frames requires an odd count>=7')
    for key in ('budget_seconds', 'timeout', 'twsc_timeout'):
        if hasattr(args, key) and getattr(args, key) <= 0: parser.error(key+' must be positive')
    globals()[args.mode](args)


if __name__ == '__main__': main()
