#!/usr/bin/env python3
"""Exact NLH guide-packing A/B gate, with one plugin loaded per process.

This is a same-model differential regression gate, not an independent
mathematical oracle or a timing benchmark. Inputs and work are deliberately
small. The existing full-image oracle gate remains required.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cases():
    common = dict(block_size=[4, 4], block_step=[3, 4], group_size=[8, 16],
                  q=[2, 4], search_window=[5, 7], basic_iters=2, wiener_iters=2,
                  lambda_basic=.6, hard_strength=1., wiener_sigma_scale=.32,
                  ps_num=2, ps_range=1)
    rows = []

    def add(name, fmt='GRAYS', size=(19, 17), sigma=25, radius=0,
            reference=False, constant=None, overrides=None, defaults=False):
        params = {} if defaults else dict(common)
        params.update(radius=radius)
        if sigma is not None:
            params['sigma'] = sigma
        params.update(overrides or {})
        rows.append(dict(name=name, format=fmt, size=list(size), parameters=params,
                         reference=reference, constant=constant, seed=1701 + len(rows),
                         frames=5 if radius else 3))

    add('gray-v5-low', sigma=3, defaults=True)
    add('gray-v5-high', sigma=75, defaults=True)
    add('gray-v5-blind', sigma=None, defaults=True)
    add('gray-r1-shape-pair', radius=1,
        overrides=dict(block_size=[3, 4], q=[2, 8], group_size=[4, 8]))
    add('gray-r2-external', radius=2, reference=True)
    add('rgb-r2', fmt='RGBS', size=(18, 16), radius=2, sigma=[3, 25, 75])
    add('rgb-r1-external', fmt='RGBS', size=(18, 16), radius=1, reference=True)
    add('rgb-blind-external', fmt='RGBS', size=(16, 16), sigma=None, reference=True)
    add('rgb-zero-luma', fmt='RGBS', size=(18, 16), sigma=[0, 3, 0], radius=1)
    add('yuv444-r1', fmt='YUV444PS', size=(18, 16), radius=1, sigma=[3, 4, 5])
    add('yuv420-r2', fmt='YUV420PS', size=(20, 18), radius=2, sigma=[3, 4, 5])
    add('yuv420-blind', fmt='YUV420PS', size=(18, 16), sigma=None)
    add('yuv420-r1-external', fmt='YUV420PS', size=(20, 18), radius=1, reference=True)
    add('constant-ties', constant=.25, overrides=dict(q=[4, 4], group_size=[16, 16]))
    add('zero-wiener-noise', overrides=dict(wiener_sigma_scale=0.))
    add('zero-sigma-tiny', size=(2, 3), sigma=0, radius=2,
        overrides=dict(block_size=[2, 2], block_step=[2, 2], q=[2, 2],
                       group_size=[2, 2], ps_num=1, search_window=[1, 1]))
    for group, q in ((2, 2), (4, 4), (8, 8), (16, 16), (32, 4), (64, 4)):
        add(f'shape-g{group}-q{q}', size=(12, 10),
            overrides=dict(group_size=[group, group], q=[q, q], ps_num=1,
                           basic_iters=1, search_window=[9, 9]))
    # A one-position spatial search gives n=1 despite the requested group64.
    add('short-group-n1', size=(7, 6),
        overrides=dict(group_size=[64, 64], q=[16, 16], ps_num=1,
                       basic_iters=1, search_window=[1, 1]))
    return rows


def normalized(value):
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


LIVE_RESOURCE_PROPERTIES = {'_NSSResourceBytes', '_NSSResourcePeak'}


def property_differences(left, right, include_live_resources=True):
    """Budget bytes/peak are shared per-node snapshots, affected by concurrency."""
    differences = []
    for frame in sorted(set(left) | set(right)):
        a, b = left.get(frame, {}), right.get(frame, {})
        for key in sorted(set(a) | set(b)):
            if not include_live_resources and key in LIVE_RESOURCE_PROPERTIES:
                continue
            if a.get(key) != b.get(key):
                differences.append(dict(frame=frame, property=key, baseline=a.get(key), candidate=b.get(key)))
    return differences


def validate_resources(properties):
    counts = properties['_NSSResourceBytes']
    peak, limit = properties['_NSSResourcePeak'], properties['_NSSResourceLimit']
    if not isinstance(counts, list) or len(counts) != 6 or any(not isinstance(n, int) or n < 0 for n in counts):
        raise AssertionError('invalid resource byte counters')
    if not isinstance(peak, int) or not isinstance(limit, int) or not 0 <= peak <= limit or limit <= 0:
        raise AssertionError('resource peak exceeds the recorded budget')
    if sum(counts[:-1]) > peak:  # Framework references are reported but not owned.
        raise AssertionError('owned resource bytes exceed the recorded peak')


def assert_pixels(left, right, label):
    import numpy as np
    if left.dtype != right.dtype or left.shape != right.shape:
        raise AssertionError(f'{label}: dtype/shape changed')
    if left.tobytes() != right.tobytes():
        difference = float(np.max(np.abs(left.astype(np.float64) - right)))
        raise AssertionError(f'{label}: pixel bits differ; max_abs={difference}')


def worker(plugin, out, selected):
    import numpy as np
    import vapoursynth as vs
    import test_full_image_plugin as fixtures
    make_clip, values = fixtures.make_clip, fixtures.values

    out.mkdir(parents=True, exist_ok=False)
    core = vs.core
    core.max_cache_size = 32
    core.std.LoadPlugin(path=str(plugin))
    rows = []
    identities = {str(plugin): sha(plugin), str(Path(__file__).resolve()): sha(__file__)}
    helper = Path(fixtures.__file__).resolve()
    identities[str(helper)] = sha(helper)
    save(out / 'inputs.json', identities)
    for case in selected:
        row = dict(case=case, passed=False, records=[])
        try:
            source, original = make_clip(core, getattr(vs, case['format']),
                                         frames=case['frames'], width=case['size'][0],
                                         height=case['size'][1], seed=case['seed'],
                                         constant=case['constant'])
            guide, guide_pixels = make_clip(core, getattr(vs, case['format']),
                                           frames=case['frames'], width=case['size'][0],
                                           height=case['size'][1], seed=case['seed'] + 500)
            digest = hashlib.sha256()
            for sequence in (original, guide_pixels):
                for frame in sequence:
                    for plane in frame:
                        digest.update(plane.tobytes())
            row['input_sha256'] = digest.hexdigest()
            order = [case['frames'] - 1, 0, case['frames'] // 2]
            expected = None
            expected_props = None
            records = row['records']
            for threads in (1, 2):
                core.num_threads = threads
                params = dict(case['parameters'])
                if case['reference']:
                    params['rclip'] = guide
                raw = core.nss.NLH(source, **params)
                radius = params['radius']
                final = core.nss.VAggregate(raw, source, radius=radius) if radius else raw
                core.std.SetVideoCache(raw, mode=0)
                if radius:
                    core.std.SetVideoCache(final, mode=0)

                def capture(n):
                    raw_planes, props = values(raw, n)
                    output, _ = values(final, n) if radius else (raw_planes, props)
                    arrays = {f'n{n}_raw_p{p}': a for p, a in enumerate(raw_planes)}
                    arrays.update({f'n{n}_out_p{p}': a for p, a in enumerate(output)})
                    for key, array in arrays.items():
                        if not np.isfinite(array).all():
                            raise AssertionError(f'{key}: nonfinite pixels')
                    sigma = params.get('sigma')
                    if sigma is not None:
                        sigmas = sigma if isinstance(sigma, list) else [sigma]
                        for p, array in enumerate(output):
                            if sigmas[min(p, len(sigmas) - 1)] == 0:
                                assert_pixels(array, original[n][p], f'zero-plane-{n}-{p}')
                    properties = {k: normalized(v) for k, v in props.items() if k.startswith('_NSS')}
                    validate_resources(properties)
                    return n, arrays, properties

                with ThreadPoolExecutor(max_workers=threads) as pool:
                    captured = list(pool.map(capture, order))
                arrays = {key: value for _, values_, _ in captured for key, value in values_.items()}
                props = {str(n): value for n, _, value in captured}
                # Persist both observations before checking; failures must retain
                # the actual arrays/properties that explain their verdict.
                filename = f"{case['name']}-threads{threads}.npz"
                np.savez_compressed(out / filename, **arrays)
                save(out / f"{case['name']}-threads{threads}.properties.json", props)
                records.append(dict(threads=threads, arrays=filename, sha256=sha(out / filename),
                                    properties=props, requested_frame_order=order))
                if expected is not None:
                    differences = property_differences(expected_props, props)
                    if differences:
                        row['repeat_property_differences'] = differences
                    semantic_changes = property_differences(expected_props, props, include_live_resources=False)
                    if arrays.keys() != expected.keys() or semantic_changes:
                        raise AssertionError('thread/repeat output schema or semantic properties changed: ' + repr(semantic_changes))
                    for key in arrays:
                        assert_pixels(arrays[key], expected[key], 'threads-' + key)
                else:
                    expected, expected_props = arrays, props
            row.update(passed=True, records=records, compared_arrays=len(expected))
        except Exception:
            row['error'] = traceback.format_exc()
        rows.append(row)
        save(out / 'summary.json', dict(passed=all(r['passed'] for r in rows), cases=rows))
        print(case['name'], 'PASS' if row['passed'] else 'FAIL', flush=True)
    if any(sha(path) != digest for path, digest in identities.items()):
        raise AssertionError('worker input file changed')
    return all(row['passed'] for row in rows)


def run_worker(plugin, out, selected, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--plugin', str(plugin),
               '--out', str(out), '--cases', *[row['name'] for row in selected]]
    with out.with_suffix('.stdout').open('w') as stdout, out.with_suffix('.stderr').open('w') as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            return dict(returncode=proc.wait(timeout=timeout), timed_out=False)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return dict(returncode=proc.returncode, timed_out=True)
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise


def compare(out, selected):
    import numpy as np
    manifests = [json.loads((out / side / 'summary.json').read_text()) for side in ('baseline', 'candidate')]
    if not all(data['passed'] for data in manifests):
        raise AssertionError('worker case failed; see individual summaries')
    indexed = [{r['case']['name']: r for r in data['cases']} for data in manifests]
    rows = []
    for case in selected:
        left, right = [data[case['name']] for data in indexed]
        if left['case'] != right['case'] or left['input_sha256'] != right['input_sha256']:
            raise AssertionError('A/B input mismatch: ' + case['name'])
        compared = 0
        resource_variations = []
        for a, b in zip(left['records'], right['records'], strict=True):
            differences = property_differences(a['properties'], b['properties'], include_live_resources=a['threads'] == 1)
            if a['threads'] != b['threads'] or differences:
                raise AssertionError('A/B scheduling or checked properties changed: ' + case['name'] + repr(differences))
            if a['threads'] > 1:
                resource_variations.extend(property_differences(a['properties'], b['properties']))
            with np.load(out / 'baseline' / a['arrays']) as one, np.load(out / 'candidate' / b['arrays']) as two:
                if set(one.files) != set(two.files):
                    raise AssertionError('A/B array schema changed: ' + case['name'])
                for key in one.files:
                    assert_pixels(one[key], two[key], case['name'] + '/' + key)
                    compared += 1
        rows.append(dict(case=case['name'], passed=True, compared_arrays=compared,
                         concurrent_resource_variations=resource_variations))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--timeout', type=float, default=300., help='Maximum seconds per plugin worker')
    parser.add_argument('--cases', nargs='+', choices=[row['name'] for row in cases()])
    parser.add_argument('--list', action='store_true', help='Print bounded case definitions without importing VapourSynth')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plugin', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = [row for row in cases() if not args.cases or row['name'] in args.cases]
    if args.list:
        print(json.dumps(selected, indent=2))
        return
    if not args.out or args.timeout <= 0:
        parser.error('--out and positive --timeout are required')
    args.out = args.out.resolve()
    if args.worker:
        if not args.plugin:
            parser.error('worker requires --plugin')
        raise SystemExit(0 if worker(args.plugin.resolve(), args.out, selected) else 1)
    if not args.baseline or not args.candidate:
        parser.error('--baseline and --candidate are required')
    plugins = dict(baseline=args.baseline.resolve(), candidate=args.candidate.resolve())
    for plugin in plugins.values():
        if not plugin.is_file():
            parser.error('plugin does not exist: ' + str(plugin))
    args.out.mkdir(parents=True, exist_ok=False)
    summary = dict(schema='nss.nlh-pack-differential.v1', passed=False,
                   scope='Same-model exact pixels/raw temporal contributions; independent mathematical gate still required',
                   resource_policy='Exact A/B serial resources; concurrent byte/peak snapshots retained with bounds, not equality; all semantic properties exact',
                   plugin_sha256={side: sha(plugin) for side, plugin in plugins.items()},
                   driver_sha256=sha(__file__), workers={})
    save(args.out / 'cases.json', selected)
    try:
        for side, plugin in plugins.items():
            summary['workers'][side] = run_worker(plugin, args.out / side, selected, args.timeout)
            save(args.out / 'comparison.json', summary)
        if any(row['returncode'] != 0 for row in summary['workers'].values()):
            raise AssertionError('plugin worker failed or timed out; logs and partial summaries retained')
        summary['cases'] = compare(args.out, selected)
        if summary['plugin_sha256'] != {side: sha(plugin) for side, plugin in plugins.items()}:
            raise AssertionError('plugin changed during comparison')
        if summary['driver_sha256'] != sha(__file__):
            raise AssertionError('driver changed during comparison')
        summary['passed'] = True
    except Exception:
        summary['error'] = traceback.format_exc()
    finally:
        save(args.out / 'comparison.json', summary)
    print(json.dumps(dict(passed=summary['passed'], cases=len(selected),
                          report=str(args.out / 'comparison.json'))))
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
