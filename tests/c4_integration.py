#!/usr/bin/env python3
"""All-filter integration workloads using the bounded paired runner.

Source construction and hashing are outside frame timing. Temporal fixtures
translate a real image and regenerate noise per frame; they are synthetic motion,
not natural video. Both plugin processes receive identical source arrays.
"""
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

import c4_paired_bm as paired


def worker(plugin, config):
    import numpy as np
    import vapoursynth as vs
    core = vs.core
    core.num_threads = config.get('threads', 1)
    core.std.LoadPlugin(path=plugin)
    algorithm = config['algorithm']
    kw = dict(config['kwargs'])
    radius = kw.get('radius', kw.get('d', 0))
    chunk = kw.get('rolling_chunk', 4) if kw.get('temporal_mode') == 'rolling' else 0
    first = config.get('warmup', 1) + 4 * radius
    frames = config.get('frames', 2)
    if chunk:
        first = (first + chunk - 1) // chunk * chunk
        frames = (max(frames, chunk) + chunk - 1) // chunk * chunk
    length = first + frames + 4 * radius + 1
    w, h = config['size']
    raw = np.fromfile(config['sample'], np.uint8).reshape(1080, 1920)
    # Explicit nearest-neighbor sampling keeps the source dependency-free.
    base = raw[np.arange(h) * 1080 // h][:, np.arange(w) * 1920 // w].astype(np.float32) / np.float32(255)
    planes = 3 if algorithm == 'mcwnnm' else 1
    motion = config.get('motion', False)
    def payload(n):
        clean = base[:, np.clip(np.arange(w) - (n % 7 - 3) * 2, 0, w - 1)] if motion else base
        rng = np.random.RandomState(42 + n if motion else 42)
        return np.stack([clean + rng.randn(h, w).astype(np.float32) * np.float32(3 / 255) for _ in range(planes)])
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGBS if planes == 3 else vs.GRAYS, length=length)
    source_fills = 0
    def fill(n, f):
        nonlocal source_fills
        source_fills += 1
        out = f.copy()
        for p, values in enumerate(payload(n)):
            np.asarray(out[p])[:] = values
        return out
    source = core.std.ModifyFrame(blank, blank, fill)
    # Retaining Python frame references alone does not pin the node's cache.
    # Spatial consumers can make VS disable its automatic source cache.
    core.std.SetVideoCache(source, mode=1, fixedsize=1, maxsize=length)
    fn = getattr(core.nss, dict(nlm='NLM', bm3d='BM3D', wnnm='WNNM', twsc='TWSC', ncsr='NCSR', lssc='LSSC', nlh='NLH', mcwnnm='MCWNNM')[algorithm])
    def filtered(ref=None):
        args = dict(kw)
        if ref is not None:
            args['ref' if algorithm == 'bm3d' else 'rclip'] = ref
        out = fn(source, **args)
        if radius and algorithm not in ('nlm', 'lssc') and not chunk:
            out = core.nss.VAggregate(out, source, radius=radius)
        return out
    stage = config.get('stage', 'basic')
    output = filtered(filtered()) if stage == 'two_stage' else filtered(source) if stage == 'wiener' else filtered()
    # Pin/materialize input before timing, including the entire dependency halo.
    inputs = [source.get_frame(n) for n in range(length)]
    for n in range(max(0, first - config.get('warmup', 1)), first):
        output.get_frame(n)
    order = list(range(first, first + frames))
    if config.get('access') == 'random':
        random.Random(42).shuffle(order)
    fills_before_timing = source_fills
    start = time.perf_counter()
    outputs = [output.get_frame(n) for n in order]
    elapsed = time.perf_counter() - start
    timed_source_fills = source_fills - fills_before_timing
    if timed_source_fills:
        raise RuntimeError(f"source was recomputed inside frame timing: {timed_source_fills}")
    arrays = [np.stack([np.array(f[p]) for p in range(planes)]) for f in outputs]
    digest = hashlib.sha256()
    for a in arrays:
        if not np.isfinite(a).all():
            raise RuntimeError('nonfinite output')
        digest.update(a.tobytes())
    if config.get('_dump'):
        # Seven small motion frames, or two full-size spatial frames, bound the
        # temporary float64 clean/SSIM diagnostics outside the timing interval.
        np.save(config['_dump'], np.stack(arrays[:7 if w * h < 1920 * 1080 else 2]))
    return dict(ms=elapsed * 1000 / frames, timed_first=first, timed_frames=frames,
                source_fills_before_timing=fills_before_timing, timed_source_fills=timed_source_fills,
                rolling_chunk=chunk, access=config.get('access', 'sequential'),
                sha256=digest.hexdigest(), input_sha256=hashlib.sha256(payload(first).tobytes()).hexdigest(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def configs():
    sample = '/opt/nss-c4/samples/gray8/MAPPA.gray8'
    common = dict(sigma=3, block_size=8, block_step=8, group_size=8, bm_range=7)
    args = dict(nlm=dict(d=1, a=2, s=4, h=1.2, channels='Y'),
                bm3d=dict(common), wnnm=dict(common), twsc=dict(common), ncsr=dict(common),
                lssc=dict(sigma=3, block_size=8, block_step=8),
                nlh=dict(common, group_size=16, bm_range=20, q=4),
                mcwnnm=dict(common, sigma=[3, 3, 3]))
    rows = []
    for algorithm, kw in args.items():
        rows.append(dict(name=algorithm + '_1080', algorithm=algorithm, kwargs=kw,
                         size=[1920, 1080], sample=sample, frames=2))
    for group in (16, 32):
        rows.append(dict(name=f'bm3d_g{group}_chain', algorithm='bm3d', kwargs=dict(common, group_size=group),
                         stage='two_stage', size=[1920, 1080], sample=sample, frames=2))
    for algorithm in ('bm3d', 'wnnm', 'twsc', 'ncsr', 'nlh', 'mcwnnm'):
        kw = dict(args[algorithm], radius=1, ps_num=2, ps_range=2)
        rows.append(dict(name=algorithm + '_motion_r1', algorithm=algorithm, kwargs=kw,
                         size=[320, 180], sample=sample, frames=7, motion=True,
                         stage='two_stage' if algorithm == 'bm3d' else 'basic'))
    for access in ('sequential', 'random'):
        rows.append(dict(name='bm3d_rolling_' + access, algorithm='bm3d',
                         kwargs=dict(common, radius=1, temporal_mode='rolling', rolling_chunk=2,
                                     rolling_cache_limit=1, ps_num=2, ps_range=2),
                         stage='two_stage', size=[320, 180], sample=sample, frames=8,
                         motion=True, access=access))
    for algorithm in ('wnnm', 'twsc', 'ncsr'):
        rows.append(dict(name=algorithm + '_g32', algorithm=algorithm,
                         kwargs=dict(args[algorithm], group_size=32), size=[640, 360],
                         sample=sample, frames=2))
    return rows


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'configs':
        print(json.dumps(configs(), indent=2))
    else:
        # Reuse calibration, complete-pair timeout handling and raw evidence.
        paired.worker = worker
        paired.__file__ = str(Path(__file__).resolve())
        paired.main()
