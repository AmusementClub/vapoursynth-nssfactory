#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""BM4D prototype (NumPy), written from Maggioni, Katkovnik, Egiazarian and
Foi, "Nonlocal Transform-Domain Filter for Volumetric Data Denoising and
Reconstruction", IEEE TIP 22(1), 2013, with the authors' published default
parameters (profile "np" of their v4 software: cubes 4^3 / 5^3, step 3,
groups 16 / 32, search half-width 7, matching thresholds 2.9527 / 0.7689 in
units of sigma^2 per voxel, Kaiser window beta 2). No code of the authors'
software is used. Departures, to be decided for the plugin: the hard stage's
cube transform is the DCT (the paper's profile uses bior1.5), and the
matching distance is ||C - C'||^2 / (L^3 sigma^2).

A video (frames as the third dimension) is one volume. Input and output are
float arrays (T, H, W) in 8-bit units.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HT = dict(L=4, step=3, M=16, sw=7, tau=2.9527)
WIE = dict(L=5, step=3, M=32, sw=7, tau=0.7689)
LAMBDA = 2.7
BETA = 2.0


def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


def haar_matrix(m):
    """Orthonormal full Haar decomposition of length m (a power of two)."""
    h = np.array([[1.0]])
    while h.shape[0] < m:
        n = h.shape[0]
        top = np.kron(h, [1.0, 1.0])
        bottom = np.kron(np.eye(n), [1.0, -1.0])
        h = np.vstack([top, bottom]) / np.sqrt(2.0)
    return h


def positions(n, L, step):
    p = list(range(0, n - L + 1, step))
    if p[-1] != n - L:
        p.append(n - L)
    return np.array(p)


def box_sum3(a, L):
    """Sums of a over every L x L x L window (valid positions)."""
    c = np.cumsum(np.cumsum(np.cumsum(a, 0), 1), 2)
    c = np.pad(c, ((1, 0), (1, 0), (1, 0)))
    return (c[L:, L:, L:] - c[:-L, L:, L:] - c[L:, :-L, L:] - c[L:, L:, :-L]
            + c[:-L, :-L, L:] + c[:-L, L:, :-L] + c[L:, :-L, :-L] - c[:-L, :-L, :-L])


def match(V, refs, L, M, sw, tau, sigma):
    """Top-M cubes (squared distance / (L^3 sigma^2) <= tau, the reference first)."""
    T, H, W = V.shape
    rz, ry, rx = refs
    R = len(rz)
    limit = tau * L ** 3 * sigma ** 2
    best_d = np.full((R, M), np.inf)
    best_p = np.zeros((R, M, 3), dtype=np.int32)
    best_d[:, 0] = -1.0  # the reference itself, always first
    best_p[:, 0] = np.stack(refs, 1)
    Vd = V.astype(np.float64)
    for dz in range(-sw, sw + 1):
        cand_d, cand_p = [], []
        for dy in range(-sw, sw + 1):
            for dx in range(-sw, sw + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                # cube at p against cube at p + off, for p where both fit
                z0, z1 = max(0, -dz), min(T, T - dz)
                y0, y1 = max(0, -dy), min(H, H - dy)
                x0, x1 = max(0, -dx), min(W, W - dx)
                if z1 - z0 < L or y1 - y0 < L or x1 - x0 < L:
                    continue
                diff = Vd[z0:z1, y0:y1, x0:x1] - Vd[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                D = box_sum3(diff * diff, L)
                iz, iy, ix = rz - z0, ry - y0, rx - x0
                ok = ((iz >= 0) & (iz < D.shape[0]) & (iy >= 0) & (iy < D.shape[1]) & (ix >= 0) & (ix < D.shape[2]))
                d = np.full(R, np.inf)
                d[ok] = D[iz[ok], iy[ok], ix[ok]]
                d[d > limit] = np.inf
                cand_d.append(d)
                cand_p.append((dz, dy, dx))
        if not cand_d:
            continue
        cd = np.stack(cand_d, 1)
        offs = np.array(cand_p, dtype=np.int32)
        alld = np.concatenate([best_d[:, 1:], cd], 1)
        k = M - 1
        sel = np.argpartition(alld, k - 1, axis=1)[:, :k] if alld.shape[1] > k else np.argsort(alld, 1)
        newd = np.take_along_axis(alld, sel, 1)
        oldp = best_p[:, 1:]
        from_old = sel < k
        candp = np.stack(refs, 1)[:, None, :] + offs[np.clip(sel - k, 0, None)]
        newp = np.where(from_old[..., None], np.take_along_axis(oldp, np.clip(sel, 0, k - 1)[..., None], 1), candp)
        best_d[:, 1:] = newd
        best_p[:, 1:] = newp
    order = np.argsort(best_d, 1, kind="stable")
    best_d = np.take_along_axis(best_d, order, 1)
    best_p = np.take_along_axis(best_p, order[..., None], 1)
    count = np.isfinite(best_d).sum(1)
    size = 2 ** np.floor(np.log2(np.maximum(count, 1))).astype(int)
    return best_p, size


def transform3(c, D):
    c = np.einsum("ij,...jkl->...ikl", D, c)
    c = np.einsum("ij,...kjl->...kil", D, c)
    return np.einsum("ij,...klj->...kli", D, c)


def inverse3(c, D):
    return transform3(c, D.T)


def aggregate(num, den, est, w, pos, L, window):
    """num/den += w window est over the cubes at pos (R, m, 3)."""
    T, H, W = num.shape
    off = np.stack(np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 3)
    p = pos.reshape(-1, 1, 3) + off[None]
    lin = (p[..., 0] * H + p[..., 1]) * W + p[..., 2]
    ww = (w[:, None, None] * np.ones(pos.shape[1])[None, :, None] * window.reshape(1, 1, -1)).reshape(-1)
    num += np.bincount(lin.reshape(-1), weights=(ww * est.reshape(-1)), minlength=num.size).reshape(num.shape)
    den += np.bincount(lin.reshape(-1), weights=ww, minlength=num.size).reshape(num.shape)


def stage(V, sigma, basic=None, chunk=2000):
    p = WIE if basic is not None else HT
    L, M, sw, tau = p["L"], p["M"], p["sw"], p["tau"]
    T, H, W = V.shape
    zz, yy, xx = np.meshgrid(positions(T, L, p["step"]), positions(H, L, p["step"]), positions(W, L, p["step"]),
                             indexing="ij")
    refs = (zz.ravel(), yy.ravel(), xx.ravel())
    pos, size = match(V if basic is None else basic, refs, L, M, sw, tau, sigma)
    D = dct_matrix(L)
    k1 = np.kaiser(L, BETA)
    window = (k1[:, None, None] * k1[None, :, None] * k1[None, None, :])
    num = np.zeros(V.shape)
    den = np.zeros(V.shape)
    cubes_n = sliding_window_view(V.astype(np.float64), (L, L, L))
    cubes_b = None if basic is None else sliding_window_view(basic.astype(np.float64), (L, L, L))
    for m in np.unique(size):
        idx = np.nonzero(size == m)[0]
        Hm = haar_matrix(m)
        for s in range(0, len(idx), chunk):
            r = idx[s:s + chunk]
            pp = pos[r, :m]
            g = cubes_n[pp[..., 0], pp[..., 1], pp[..., 2]]           # (R, m, L, L, L)
            c = np.einsum("ij,rjklm->riklm", Hm, transform3(g, D))
            if basic is None:
                keep = np.abs(c) >= LAMBDA * sigma
                c = c * keep
                n = keep.reshape(len(r), -1).sum(1)
                w = 1.0 / (sigma ** 2 * np.maximum(n, 1))
            else:
                gb = cubes_b[pp[..., 0], pp[..., 1], pp[..., 2]]
                cb = np.einsum("ij,rjklm->riklm", Hm, transform3(gb, D))
                Wc = cb * cb / (cb * cb + sigma ** 2)
                c = c * Wc
                w = 1.0 / (sigma ** 2 * np.maximum((Wc * Wc).reshape(len(r), -1).sum(1), 1e-12))
            est = inverse3(np.einsum("ji,rjklm->riklm", Hm, c), D)
            aggregate(num, den, est, w, pp, L, window)
    return num / den


def bm4d(V, sigma):
    basic = stage(V, sigma)
    return stage(V, sigma, basic=basic), basic
