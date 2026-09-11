#!/usr/bin/env python3
"""Capture native full-filter output and metrics against frozen clean images."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys

import numpy as np
import vapoursynth as vs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bm_numerics import psnr, ssim
from test_plan01_plugin import evaluate

parser = argparse.ArgumentParser(description=__doc__)
for name in ('plugin', 'pack', 'out'): parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--algorithms', nargs='+')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
metadata = json.loads((args.pack / 'manifest.json').read_text())
packfile = args.pack / 'fixtures.npz'
if hashlib.sha256(packfile.read_bytes()).hexdigest() != metadata['data_sha256']: raise RuntimeError('fixture hash mismatch')
data = np.load(packfile)
core = vs.core; core.num_threads = 1; core.max_cache_size = 64
core.std.LoadPlugin(path=str(args.plugin.resolve()))
report = dict(schema='nssfactory.arm-quality-capture.v1', passed=False, platform=platform.platform(),
              plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
              fixture_sha256=metadata['data_sha256'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              numpy=np.__version__, vapoursynth=str(vs.__version__), cases=[])
outputs = {}
try:
    for index, case in enumerate(metadata['cases']):
        name = case['algorithm']
        if args.algorithms and name not in args.algorithms: continue
        fixture, sigma, color, radius = case['fixture'], case['sigma'], case['color'], case['radius']
        values = data[f'{fixture}_s{sigma:g}_{color}']; clean = data[f'{fixture}_clean_{color}']
        length, planes, height, width = values.shape
        blank = core.std.BlankClip(width=width, height=height, length=length, format=vs.RGBS if color == 'rgb' else vs.GRAYS)
        def fill(n, f, values=values):
            out = f.copy()
            for p in range(out.format.num_planes): np.asarray(out[p])[:] = values[n, p]
            return out
        source = core.std.ModifyFrame(blank, blank, fill)
        core.std.SetVideoCache(source, mode=1, fixedsize=1, maxsize=length)
        if name == 'NLM': kw = dict(d=radius, a=2, s=4, h=max(.1, sigma / 2.5), channels='Y')
        elif name == 'LSSC': kw = dict(sigma=sigma, block_size=8, block_step=8)
        else:
            kw = dict(sigma=[sigma] * planes if name in ('MCWNNM', 'TWSC') else sigma,
                      block_size=8, block_step=8, group_size=16 if name == 'NLH' else 8, bm_range=7, radius=radius)
        def filtered(reference=None):
            extra = {} if reference is None else {'ref' if name == 'BM3D' else 'rclip': reference}
            node = getattr(core.nss, name)(source, **kw, **extra)
            return core.nss.VAggregate(node, source, radius=radius) if radius and name not in ('NLM', 'LSSC') else node
        node = filtered(filtered()) if case['stage'] == 'two-stage' else filtered()
        requested = (0, 2, 4)
        actual = np.stack([np.stack(evaluate(node, n)[0]) for n in requested])
        truth = clean[list(requested)]
        metrics = dict(psnr_against_clean=psnr(actual, truth), ssim_against_clean=ssim(actual, truth))
        outputs[f'c{index}'] = actual
        report['cases'].append(dict(index=index, parameters=case, kwargs=kw, metrics=metrics,
                                   output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        del node, source, blank, values, clean
        if len(report['cases']) % 16 == 0: print(f'{len(report["cases"])} quality cases captured', flush=True)
    report['passed'] = True
except Exception as error:
    report['error'] = str(error)
finally:
    output = args.out / 'pixels.npz'; np.savez_compressed(output, **outputs)
    report['pixels_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['max_child_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024))
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']), error=report.get('error'))))
raise SystemExit(0 if report['passed'] else 1)
