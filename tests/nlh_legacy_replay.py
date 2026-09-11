#!/usr/bin/env python3
"""Record v4 automatic outputs and replay its explicit recipes on the new model."""
import argparse
import json
from pathlib import Path

import numpy as np
import vapoursynth as vs

from nlh_defaults_inputs import file_sha, save_json
from test_full_image_plugin import make_clip, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plugin', 'out', 'recipes'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--reference')
    parser.add_argument('--cases', nargs='+', help='Run a recorded subset for failure diagnosis')
    args = parser.parse_args()
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    recipes = json.loads(Path(args.recipes).read_text())['profiles']
    previous = np.load(args.reference) if args.reference else None
    saved, rows = {}, []
    fixtures = [
        ('gray-low', vs.GRAYS, 25, 0, 1.),
        ('gray-high-temporal', vs.GRAYS, 75, 1, 1.),
        ('gray-blind-low', vs.GRAYS, None, 0, 1.),
        ('gray-blind-high-temporal', vs.GRAYS, None, 1, 8.),
        ('rgb-explicit', vs.RGBS, 25, 0, 1.),
        ('rgb-blind-temporal', vs.RGBS, None, 1, 1.),
        ('rgb-zero-planes', vs.RGBS, [0, 3, 0], 1, 1.),
        ('yuv420-explicit-temporal', vs.YUV420PS, 75, 1, 1.),
        ('yuv420-blind', vs.YUV420PS, None, 0, 1.),
    ]
    if args.cases and set(args.cases) - {row[0] for row in fixtures}:
        parser.error('unknown fixture name')
    for index, (label, fmt, sigma, radius, scale) in enumerate(fixtures):
        if args.cases and label not in args.cases:
            continue
        print(label, 'START', flush=True)
        source, _ = make_clip(core, fmt, frames=3, width=32, height=30, seed=601+index)
        if scale != 1:
            # Keep fixture generation independent of Expr's JIT and plugin
            # symbol interactions. Multiplication by eight is exact in float32.
            def scale_frame(n, f, scale=scale):
                frame = f.copy()
                for plane in range(frame.format.num_planes):
                    np.multiply(np.asarray(f[plane]), np.float32(scale),
                                out=np.asarray(frame[plane]))
                return frame
            source = core.std.ModifyFrame(source, source, scale_frame)
        fixed = dict(radius=radius)
        if sigma is not None:
            fixed['sigma'] = sigma
        if previous is None:
            raw = core.nss.NLH(source, **fixed)
        else:
            nodes = {lane: core.nss.NLH(source, **dict(profile, **fixed))
                     for lane, profile in recipes.items()}
            def select(n, label=label, nodes=nodes, fmt=fmt):
                units = previous[f'{label}-{n}-sigma']
                lane = 'rgb' if fmt != vs.GRAYS else 'gray-low' if np.max(units) <= 50 else 'gray-high'
                return nodes[lane]
            template = core.std.BlankClip(source, height=source.height*2*(2*radius+1)) if radius else source
            raw = core.std.FrameEval(template, eval=select)
        output = core.nss.VAggregate(raw, source, radius=radius) if radius else raw
        for n in (0, 1, 2):
            raw_planes, props = values(raw, n)
            normal, _ = values(output, n)
            key = f'{label}-{n}'
            arrays = {key+'-sigma': np.atleast_1d(np.asarray(props['_NSSSigma'], dtype=np.float64))}
            arrays.update({key+f'-raw-{c}': a for c, a in enumerate(raw_planes)})
            arrays.update({key+f'-out-{c}': a for c, a in enumerate(normal)})
            if previous is not None:
                for name, array in arrays.items():
                    np.testing.assert_array_equal(array, previous[name], err_msg=name)
            saved.update(arrays)
        rows.append(dict(case=label, passed=True, radius=radius, sigma=sigma, scale=scale))
        print(label, 'PASS', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'outputs.npz', **saved)
    save_json(out/'verification.json', dict(passed=True, fixtures=rows, frames=len(rows)*3,
              compared_arrays=len(saved) if previous is not None else 0,
              plugin_sha256=file_sha(args.plugin), recipes_sha256=file_sha(args.recipes),
              reference_sha256=file_sha(args.reference) if args.reference else None,
              driver_sha256=file_sha(__file__),
              scope='Exact sigma, raw temporal contributions and final pixels; v4 auto versus explicit old recipes'))


if __name__ == '__main__':
    main()
