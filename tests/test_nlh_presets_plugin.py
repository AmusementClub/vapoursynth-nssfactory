#!/usr/bin/env python3
"""Check public preset resolution against a frozen, explicit profile contract.

The contract comes from the selected experiment profiles, independently of the
C++ resolver. This gate checks default/explicit pixel identity, diagnostic
values, sigma boundaries, small planes, overrides and temporal contributions.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import vapoursynth as vs

from nlh_defaults_inputs import file_sha, save_json
from test_full_image_plugin import make_clip, values


def pair(value):
    return list(value) if isinstance(value, (tuple, list)) else [value, value]


def sigma_planes(value, planes):
    values = np.atleast_1d(value).astype(float)
    return np.pad(values, (0, planes-len(values)), mode='edge')


def expected_parameters(profiles, fmt, sigma, noise_model, width, height, sigma_is_working=False):
    units = sigma_planes(sigma, fmt.num_planes)
    if fmt.color_family == vs.RGB and not sigma_is_working:
        transform = np.array([[.299, .587, .114],
                              [-.168736607142857, -.331263392857143, .5],
                              [.5, -.4186875, -.0813125]])
        units = np.sqrt((transform*transform) @ (units*units))
    lane = ('rgb' if noise_model == 'real' or (noise_model == 'auto' and fmt.color_family != vs.GRAY)
            else 'gray-low' if max(units) <= 50 else 'gray-high')
    profile = copy.deepcopy(profiles[lane])
    active = [min(width >> (fmt.subsampling_w if p else 0),
                  height >> (fmt.subsampling_h if p else 0))
              for p in range(fmt.num_planes) if units[p] > 0]
    available = min([16, *active])
    for stage in (0, 1):
        profile['block_size'][stage] = min(profile['block_size'][stage], available)
        block = profile['block_size'][stage]
        profile['block_step'][stage] = min(profile['block_step'][stage], block)
        while profile['q'][stage] > block*block:
            profile['q'][stage] //= 2
    profile['noise_model'] = noise_model
    return profile, units


def check_diagnostics(props, expected, version):
    assert props['_NSSModelVersion'] == version
    for key, field in (('_NSSBlockSize', 'block_size'), ('_NSSBlockStep', 'block_step'),
                       ('_NSSGroupSize', 'group_size'), ('_NSSQ', 'q'),
                       ('_NSSSearchWindow', 'search_window')):
        assert pair(props[key]) == pair(expected[field]), (key, props[key], expected[field])
    assert pair(props['_NSSIterations']) == [expected['basic_iters'], expected['wiener_iters']]
    for key, field in (('_NSSLambdaBasic', 'lambda_basic'), ('_NSSHardStrength', 'hard_strength'),
                       ('_NSSWienerSigmaScale', 'wiener_sigma_scale')):
        assert props[key] == expected[field], (key, props[key], expected[field])
    assert props['_NSSHardCoefficient'] == 2.025*expected['hard_strength']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plugin', 'profiles', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--model-version', type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = json.loads(Path(args.profiles).read_text())
    profiles = contract['profiles']
    if set(profiles) != {'gray-low', 'gray-high', 'rgb'}:
        raise ValueError('three complete preset profiles are required')
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    records = []

    def run(label, callback):
        try:
            detail = callback() or {}
            records.append(dict(case=label, passed=True, **detail))
            print(label, 'PASS', flush=True)
        except Exception:
            error = traceback.format_exc()
            records.append(dict(case=label, passed=False, error=error))
            print(label, 'FAIL', error, flush=True)

    def compare(fmt, sigma, noise_model='auto', width=32, height=30, radius=0, override=None):
        source, original = make_clip(core, fmt, frames=3, width=width, height=height)
        supplied = dict(noise_model=noise_model, radius=radius)
        if sigma is not None:
            supplied['sigma'] = sigma
        if override:
            supplied.update(override)
        automatic = core.nss.NLH(source, **supplied)
        hashes = []
        for n in (0, 1, 2) if radius else (1,):
            actual, props = values(automatic, n)
            # Independent noise-estimator gates cover the estimate itself. Here
            # blind Gray checks that this runtime estimate selects the preset.
            known_sigma = props['_NSSSigma'] if sigma is None else sigma
            expected, units = expected_parameters(profiles, source.format, known_sigma,
                                                   noise_model, width, height, sigma_is_working=sigma is None)
            if override:
                expected.update(override)
            explicit = dict(expected, radius=radius)
            if sigma is not None:
                explicit['sigma'] = sigma
            reference = core.nss.NLH(source, **explicit)
            pixels, explicit_props = values(reference, n)
            check_diagnostics(props, expected, args.model_version)
            check_diagnostics(explicit_props, expected, args.model_version)
            if sigma is not None:
                np.testing.assert_allclose(np.atleast_1d(props['_NSSSigma']), units, rtol=1e-12, atol=1e-12)
            for p, (left, right) in enumerate(zip(actual, pixels)):
                np.testing.assert_array_equal(left, right)
                if sigma is not None and sigma_planes(sigma, source.format.num_planes)[p] == 0:
                    if not radius:
                        np.testing.assert_array_equal(left, original[n][p])
                hashes.append(hashlib.sha256(left.tobytes()).hexdigest())
            if radius:
                left, _ = values(core.nss.VAggregate(automatic, source, radius=radius), n)
                right, _ = values(core.nss.VAggregate(reference, source, radius=radius), n)
                for a, b in zip(left, right):
                    np.testing.assert_array_equal(a, b)
        return dict(output_sha256=hashes, parameters=expected)

    for sigma in (0, 5, 50, 50+1e-9, 75):
        run('gray-boundary-' + str(sigma), lambda sigma=sigma: compare(vs.GRAYS, sigma))
    run('gray-real-preset', lambda: compare(vs.GRAYS, 25, 'real'))
    run('rgb-auto', lambda: compare(vs.RGBS, 25))
    run('rgb-short-sigma-zero-planes', lambda: compare(vs.RGBS, [25, 0]))
    run('rgb-working-sigma-awgn', lambda: compare(vs.RGBS, 80, 'awgn'))
    run('yuv444-zero-planes', lambda: compare(vs.YUV444PS, [25, 0, 0]))
    run('yuv420-small-active-chroma', lambda: compare(vs.YUV420PS, [25, 25, 25]))
    run('yuv420-disabled-chroma', lambda: compare(vs.YUV420PS, [25, 0, 0]))
    run('two-pixel-active-plane', lambda: compare(vs.GRAYS, 25, width=2, height=2))
    run('one-pixel-zero-identity', lambda: compare(vs.GRAYS, 0, width=1, height=1))
    run('gray-blind-runtime-preset', lambda: compare(vs.GRAYS, None))
    run('rgb-blind-real-preset', lambda: compare(vs.RGBS, None))
    run('yuv-blind-real-preset', lambda: compare(vs.YUV420PS, None))
    run('temporal-boundaries', lambda: compare(vs.GRAYS, 75, radius=1))
    run('temporal-yuv-zero-planes', lambda: compare(vs.YUV420PS, [25, 0, 0], radius=1))

    def dynamic_blind_temporal():
        source, original = make_clip(core, vs.GRAYS, frames=3, width=32, height=30)
        original[1][0] *= 16  # Power-of-two scale forces a distinct estimated-noise preset.
        probe = core.nss.NLH(source, block_size=4, block_step=4, group_size=8,
                             q=2, search_window=5, basic_iters=1)
        estimates = [values(probe, n)[1]['_NSSSigma'] for n in range(3)]
        bins = ['gray-low' if sigma <= 50 else 'gray-high' for sigma in estimates]
        assert bins == ['gray-low', 'gray-high', 'gray-low'], estimates
        explicit = {lane: core.nss.NLH(source, radius=1, **profiles[lane]) for lane in set(bins)}
        selected = core.std.FrameEval(explicit['gray-low'], lambda n: explicit[bins[n]])
        automatic = core.nss.NLH(source, radius=1)
        for n in range(3):
            expected = profiles[bins[n]]
            left, props = values(automatic, n)
            right, _ = values(selected, n)
            check_diagnostics(props, expected, args.model_version)
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)
        a = core.nss.VAggregate(automatic, source, radius=1)
        b = core.nss.VAggregate(selected, source, radius=1)
        for n in range(3):
            left, _ = values(a, n); right, _ = values(b, n)
            for x, y in zip(left, right):
                np.testing.assert_array_equal(x, y)
        return dict(estimated_sigma=estimates, presets=bins)

    run('blind-temporal-changing-preset', dynamic_blind_temporal)
    for field in ('hard_strength', 'wiener_sigma_scale', 'lambda_basic'):
        run('explicit-zero-' + field, lambda field=field: compare(vs.GRAYS, 25, override={field: 0.}))
    run('explicit-scalar-broadcast', lambda: compare(vs.GRAYS, 25,
        override=dict(block_size=4, block_step=3, group_size=8, q=2, search_window=5,
                      basic_iters=1, wiener_iters=1)))

    def reject_sentinels():
        source, _ = make_clip(core, vs.GRAYS)
        for field, value in (('block_size', 0), ('block_step', 0), ('group_size', 0),
                             ('q', 0), ('search_window', 0), ('basic_iters', 0),
                             ('wiener_iters', 0), ('lambda_basic', -1.),
                             ('hard_strength', -1.), ('wiener_sigma_scale', -1.)):
            try:
                values(core.nss.NLH(source, sigma=25, **{field: value}), 0)
            except vs.Error:
                continue
            raise AssertionError('explicit internal sentinel accepted: ' + field)
        return dict(rejected=10)

    run('private-sentinels-rejected', reject_sentinels)
    save_json(out/'verification.json', dict(rows=records, passed=all(r['passed'] for r in records),
              plugin_sha256=file_sha(args.plugin), profiles_sha256=file_sha(args.profiles),
              script_sha256=file_sha(__file__), model_version=args.model_version))
    if not all(r['passed'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
