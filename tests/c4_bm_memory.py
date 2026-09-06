#!/usr/bin/env python3
"""Separate bounded memory/revisit diagnostic; never a paired timing gate."""
import argparse
import gc
import ctypes
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs
from profile_cpu_all import make_source

process = ctypes.CDLL(None)
for symbol in ('nss_probe_requests', 'nss_probe_bytes'):
    if hasattr(process, symbol):
        getattr(process, symbol).restype = ctypes.c_uint64


def rss():
    rows = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    result = {key: int(rows[key].split()[0]) for key in ('VmRSS', 'VmHWM')}
    if hasattr(process, 'nss_probe_requests'):
        result.update(allocation_requests=process.nss_probe_requests(), requested_bytes=process.nss_probe_bytes())
    return result


def run(a):
    os.sched_setaffinity(0, {0})
    core = vs.core
    core.num_threads = 1
    core.max_cache_size = 64
    core.std.LoadPlugin(path=a.plugin)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    sizes = [(3840, 2160, 8)] if a.four_k else [(640, 360, 32), (1920, 1080, 8)]
    for w, h, count in sizes:
        src = make_source(core, 'bm3d', count, w, h)
        for mode in ('legacy', 'rolling'):
            kw = dict(sigma=3, block_size=8, group_size=8, block_step=8, bm_range=7, radius=1)
            if mode == 'rolling':
                kw.update(temporal_mode='rolling', rolling_chunk=2, rolling_cache_limit=1)
            node = core.nss.BM3D(src, **kw)
            if mode == 'legacy':
                node = core.nss.VAggregate(node, src, radius=1)
            core.std.SetVideoCache(node, mode=0)
            hashes = {}
            observations = [dict(phase='created', **rss())]
            for epoch in range(3):
                order = list(range(count))
                if epoch:
                    random.Random(42 + epoch).shuffle(order)
                start = time.perf_counter()
                for n in order:
                    frame = node.get_frame(n)
                    array = np.asarray(frame[0])
                    assert np.isfinite(array).all()
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    if n in hashes:
                        assert digest == hashes[n], f'revisit changed output: {mode} {n}'
                    hashes[n] = digest
                    del array, frame
                observations.append(dict(phase=f'epoch_{epoch}', seconds=time.perf_counter()-start, **rss()))
            del node
            gc.collect()
            observations.append(dict(phase='node_released', **rss()))
            rows.append(dict(size=[w, h], frames=count, mode=mode, cache_limit_mib=64,
                             observations=observations, output_hashes=hashes))
            (out / 'memory.json').write_text(json.dumps(rows, indent=2))
            print(f'{w}x{h} {mode} complete', flush=True)
        del src
        gc.collect()
    (out / 'complete.json').write_text(json.dumps(dict(
        plugin_sha256=hashlib.sha256(Path(a.plugin).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cases=len(rows), completed=True, timing_kind='diagnostic_includes_hashing'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--four-k', action='store_true')
    run(parser.parse_args())
