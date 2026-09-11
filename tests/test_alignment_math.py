#!/usr/bin/env python3
"""Independent FP64 references for the TWSC v3 / NLH v4 contracts.

Expected values use NumPy matrices and SciPy's Schur-based Sylvester solve,
not any production transform, finisher, or eigensolver. Every probe input,
output and failure is retained in --out.
"""
import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester

# Independent contract constant, not imported from the production implementation.
NLH_HARD_COEFFICIENT = 2.025
NLH_WIENER_SIGMA_SCALE = 0.08


def haar(n):
    h = np.zeros((n, n), dtype=np.float64)
    h[0] = 1 / np.sqrt(n)
    row, width = 1, n
    while width > 1:
        for start in range(0, n, width):
            h[row, start:start + width // 2] = 1 / np.sqrt(width)
            h[row, start + width // 2:start + width] = -1 / np.sqrt(width)
            row += 1
        width //= 2
    return h


def row_neighbors(guide, q):
    m, _ = guide.shape
    # The dyadic ranking fixtures make the production FP32 scores exact.
    distances = np.sum((guide[:, None, :] - guide[None, :, :]) ** 2, axis=2)
    return np.array([[i] + sorted((j for j in range(m) if j != i), key=lambda j: (distances[i, j], j))[:q - 1]
                     for i in range(m)], dtype=np.int32)


def solve(y, d, s, precision, sigma, iterations, rho=.5, mu=1.1, tol=1e-6):
    p = d * s
    a = p.T @ (precision[:, None] * p)
    a = (a + a.T) / 2
    data = p.T @ (precision[:, None] * y)
    sigma = np.maximum(sigma.astype(np.float32).astype(float), 1e-6)
    c = np.zeros_like(data)
    z, dual, trace = c.copy(), c.copy(), []
    converged = False
    for iteration in range(iterations):
        old_c, old_z = c.copy(), z.copy()
        b = np.diag(.5 * rho * sigma)
        e = data + .5 * (rho * z - dual) * sigma[None, :]
        c = solve_sylvester(a, b, e)
        temp = c + dual / rho
        z = np.sign(temp) * np.maximum(np.abs(temp) - 1 / rho, 0)
        residuals = [np.linalg.norm(c - z), np.linalg.norm(c - old_c), np.linalg.norm(z - old_z)]
        converged = bool(max(residuals) <= tol)
        if not converged:
            dual += rho * (c - z)
        trace.extend([c.copy(), z.copy(), dual.copy()])
        if converged:
            break
        rho *= mu
    return p @ c, trace, dict(iterations=iteration + 1, converged=converged, residuals=residuals), c


class Suite:
    def __init__(self, probe, out):
        self.probe, self.out, self.cases = Path(probe).resolve(), Path(out).resolve(), []
        self.out.mkdir(parents=True, exist_ok=False)
        self.rng = np.random.default_rng(20260908)

    def call(self, name, mode, m, n, r, iterations, values):
        prefix = self.out / name
        payload = np.concatenate([np.asarray(v, dtype=np.float64).ravel(order='F') for v in values])
        src, dst = prefix.with_suffix('.input.f64'), prefix.with_suffix('.actual.f64')
        payload.tofile(src)
        run = subprocess.run([str(self.probe), mode, str(m), str(n), str(r), str(iterations), str(src), str(dst)],
                             capture_output=True, text=True, timeout=180)
        prefix.with_suffix('.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            raise AssertionError(run.stderr)
        return np.fromfile(dst, dtype=np.float64)

    def case(self, name, function):
        try:
            detail = function() or {}
            self.cases.append(dict(name=name, passed=True, **detail))
            print(name, 'PASS', flush=True)
        except Exception:
            self.cases.append(dict(name=name, passed=False, error=traceback.format_exc()))
            print(name, 'FAIL', self.cases[-1]['error'], flush=True)

    def fixed(self, name, iterations, zero=False, unequal=True):
        m, n, r = 9, 5, 5
        d = np.linalg.qr(self.rng.normal(size=(m, r)))[0]
        s = np.linspace(.3, 2, r)
        if zero:
            s[-2:] = 0
        y = self.rng.normal(size=(m, n)) * .2
        precision = np.linspace(2, 8, m) if unequal else np.ones(m) * 3
        sigma = np.linspace(.3, .7, n).astype(np.float32).astype(float)
        rho, mu, tol = (2., 1.01, 1e-10) if iterations > 100 else (.5, 1.1, 1e-6)
        actual = self.call(name, 'twsc-fixed', m, n, r, iterations, [y, d, s, precision, sigma, [rho, mu, tol]])
        expected, trace, status, c = solve(y, d, s, precision, sigma, iterations, rho, mu, tol)
        assert actual[0] == status['iterations'] and bool(actual[1]) == status['converged']
        assert actual[5] <= 1e-10
        result = actual[7:7 + m * n].reshape((m, n), order='F')
        np.testing.assert_allclose(result, expected, rtol=2e-7, atol=2e-9)
        expected_trace = np.concatenate([v.ravel(order='F') for v in trace])
        np.testing.assert_allclose(actual[7 + m * n:], expected_trace, rtol=2e-6, atol=2e-8)
        detail = dict(max_abs=float(np.max(np.abs(result - expected))), sylvester_residual=float(actual[5]), **status)
        if iterations > 100:
            assert status['converged']
            p = d * s
            gradient = 2 * p.T @ (precision[:, None] * (p @ c - y)) / sigma[None, :]
            active = np.abs(c) > 1e-7
            kkt = np.where(active, np.abs(gradient + np.sign(c)), np.maximum(np.abs(gradient) - 1, 0))
            detail['kkt_max'] = float(np.max(kkt))
            assert detail['kkt_max'] < 2e-5, detail
        return detail

    def group(self, name, m, n, unequal):
        a = (self.rng.normal(size=(m, n)) * .03 + self.rng.uniform(.1, .7, (m, 1))).astype(np.float32).astype(float)
        rows = np.linspace(.01, .06, m) if unequal else np.full(m, .02)
        cols = np.linspace(.005, .02, n)
        rows, cols = rows.astype(np.float32).astype(float), cols.astype(np.float32).astype(float)
        actual = self.call(name, 'twsc-group', m, n, 0, 10, [a, rows, cols, [.5, 1.1, 1e-6]])
        r, offset = min(m, n), 7 + m * n
        d = actual[offset:offset + m * r].reshape((m, r), order='F'); offset += m * r
        spectrum = actual[offset:offset + r]; offset += r
        mean = actual[offset:offset + m]; offset += m
        weights = actual[offset:offset + n]
        # Independent centering/spectrum checks, followed by the fixed-D solver.
        np.testing.assert_allclose(mean, np.mean(a, axis=1), atol=1e-14, rtol=0)
        centered = (a - mean[:, None]).astype(np.float32).astype(float)
        singular = np.linalg.svd(centered, compute_uv=False)
        expected_s = np.sqrt(np.maximum(singular ** 2 - n * cols[0] ** 2, 0))
        np.testing.assert_allclose(spectrum, expected_s, atol=2e-5, rtol=2e-4)
        result, _, _, _ = solve(centered, d, spectrum, 1 / np.maximum(rows, 1e-6), cols, 10)
        result += mean[:, None]
        got = actual[7:7 + m * n].reshape((m, n), order='F')
        np.testing.assert_allclose(got, result, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(weights, 1 / np.maximum(cols, 1e-6), atol=1e-5, rtol=2e-7)
        assert actual[5] <= 1e-10
        return dict(max_abs=float(np.max(np.abs(got - result))), double_svd=bool(actual[6]))

    def svd(self, name, m, n, rank=None, scale=1):
        r = min(m, n)
        if rank is None:
            a = self.rng.normal(size=(m, n)) * scale
        elif rank == 0:
            a = np.zeros((m, n))
        else:
            u = np.linalg.qr(self.rng.normal(size=(m, rank)))[0]
            v = np.linalg.qr(self.rng.normal(size=(n, rank)))[0]
            a = (u * np.geomspace(1, .01, rank)) @ v.T * scale
        a = a.astype(np.float32).astype(float)
        actual = self.call(name, 'svd', m, n, 0, 0, [a])
        d = actual[1:1 + m * r].reshape((m, r), order='F')
        s = actual[1 + m * r:1 + m * r + r]
        vt = actual[1 + m * r + r:].reshape((r, n), order='F')
        rebuilt = (d * s) @ vt
        error = np.linalg.norm(rebuilt - a) / max(np.linalg.norm(a), 1e-300)
        assert error <= 5e-5, error
        assert np.all(s >= 0) and np.all(s[:-1] >= s[1:])
        active = s > 0
        orthogonal = np.max(np.abs(d[:, active].T @ d[:, active] - np.eye(np.sum(active)))) if active.any() else 0
        assert orthogonal <= 5e-4, orthogonal
        truth = np.linalg.svd(a, compute_uv=False)
        spectral_error = np.max(np.abs(s - truth)) / max(np.max(truth), 1e-300)
        assert spectral_error <= 5e-5, spectral_error
        return dict(relative_reconstruction=float(error), orthogonality=float(orthogonal), spectrum_error=float(spectral_error), double_svd=bool(actual[0]))

    def nlh(self, name, m, n, q, sigma, wiener, strength=1., wiener_scale=.5):
        y = self.rng.normal(.1, .3, (m, n)).astype(np.float32).astype(float)
        ref = (.8 * y + self.rng.normal(0, .01, y.shape)).astype(np.float32).astype(float)
        guide = self.rng.integers(-32, 33, (m, n)) / 128
        idx = row_neighbors(guide, q)
        actual = self.call(name, 'nlh-wiener' if wiener else 'nlh-hard', m, n, q, 2, [y, ref, guide, [sigma, strength, wiener_scale]])
        num, den = np.zeros_like(y), np.zeros_like(y)
        hq, hn = haar(q), haar(n)
        for row in range(m):
            matrix = y[idx[row]]
            if sigma != 0:
                coefficients = hq @ matrix @ hn.T
                if wiener:
                    r = hq @ ref[idx[row]] @ hn.T
                    gain = r ** 2 / (r ** 2 + (wiener_scale * sigma) ** 2)
                    coefficients *= gain * gain
                else:
                    coefficients[np.abs(coefficients) < NLH_HARD_COEFFICIENT * strength * sigma] = 0
                    coefficients[max(0, q - 2):, 1:] = 0
                matrix = hq.T @ coefficients @ hn
            for k, destination in enumerate(idx[row]):
                num[destination] += matrix[k]
                den[destination] += 1
        got_num = actual[:m * n].reshape((m, n), order='F')
        got_den = actual[m * n:2 * m * n].reshape((m, n), order='F')
        np.testing.assert_array_equal(actual[2 * m * n:].astype(int).reshape(m, q), idx)
        np.testing.assert_array_equal(got_den, den)
        np.testing.assert_allclose(got_num / den, num / den, atol=2e-5, rtol=2e-4)
        return dict(max_abs=float(np.max(np.abs(got_num / den - num / den))), contribution_count=int(den.sum()))

    def transform(self, name, q, n):
        y = self.rng.normal(size=(q, n)).astype(np.float32).astype(float)
        actual = self.call(name, 'haar', q, n, 0, 0, [y]).reshape((q, n), order='F')
        expected = haar(q) @ y @ haar(n).T
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        restored = self.call(name + '_inverse', 'ihaar', q, n, 0, 0, [actual]).reshape((q, n), order='F')
        np.testing.assert_allclose(restored, y, atol=2e-5, rtol=2e-4)
        return dict(max_abs=float(np.max(np.abs(actual - expected))))

    def noise(self, name):
        height, width = 11, 12
        y = self.rng.integers(0, 64, (height, width)) / 128
        guide = self.rng.integers(0, 64, (height, width)) / 128
        patches = {(x, z): guide[z:z+8, x:x+8].ravel() for z in range(height-7) for x in range(width-7)}
        estimates = []
        for pos, patch in patches.items():
            neighbors = [pos] + sorted((key for key in patches if key != pos),
                                      key=lambda key: (np.sum((patch-patches[key])**2), key[1], key[0]))[:15]
            g = np.array([patches[key] for key in neighbors]).T
            pixels = np.array([y[z:z+8, x:x+8].ravel() for x,z in neighbors]).T
            indices = row_neighbors(g, 4)
            estimates.append(np.mean([np.sqrt(np.mean((pixels[i]-pixels[j])**2)) for i in range(64) for j in indices[i,1:]]))
        actual = self.call(name, 'noise', width, height, 0, 0, [y.ravel(), guide.ravel()])[0]
        expected = np.mean(estimates)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        return dict(actual=float(actual), expected=float(expected))

    def run(self):
        for name, iterations, zero, unequal in [('twsc_finite10',10,False,True), ('twsc_zero_spectrum',10,True,True),
                                                ('twsc_equal_noise',10,False,False), ('twsc_converged_kkt',1000,False,True)]:
            self.case(name, lambda name=name, iterations=iterations, zero=zero, unequal=unequal: self.fixed(name, iterations, zero, unequal))
        for m,n in [(16,4),(64,8),(49,70),(192,90),(81,140)]:
            name=f'twsc_group_{m}_{n}'
            self.case(name, lambda name=name,m=m,n=n: self.group(name,m,n,True))
        for m,n,rank,scale in [(1,1,0,1),(16,8,None,1),(64,32,None,1),(49,70,None,1),(192,90,None,1),
                               (81,140,None,1),(768,256,12,1),(64,8,None,1e20),(64,8,None,1e-20)]:
            name=f'svd_{m}_{n}_{rank}_{scale}'
            self.case(name, lambda name=name,m=m,n=n,rank=rank,scale=scale: self.svd(name,m,n,rank,scale))
        for q,n in [(2,1),(4,16),(8,32),(16,64)]:
            name=f'haar_{q}_{n}'
            self.case(name, lambda name=name,q=q,n=n: self.transform(name,q,n))
        for m,n,q in [(9,1,2),(19,16,4),(32,32,8),(64,64,16)]:
            for sigma in [0,.02,.2,2.]:
                for wiener in [False,True]:
                    name=f'nlh_{m}_{n}_{q}_{sigma}_{wiener}'
                    self.case(name, lambda name=name,m=m,n=n,q=q,sigma=sigma,wiener=wiener: self.nlh(name,m,n,q,sigma,wiener))
        self.case('nlh_hard_strength', lambda: self.nlh('nlh_hard_strength',19,16,4,.02,False,strength=1.5))
        self.case('nlh_calibrated_wiener', lambda: self.nlh('nlh_calibrated_wiener',19,16,4,.05,True,wiener_scale=NLH_WIENER_SIGMA_SCALE))
        self.case('blind_noise', lambda: self.noise('blind_noise'))
        report=dict(passed=all(c['passed'] for c in self.cases), cases=self.cases,
                    probe_sha256=hashlib.sha256(self.probe.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        return report['passed']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    raise SystemExit(0 if Suite(args.probe,args.out).run() else 1)
