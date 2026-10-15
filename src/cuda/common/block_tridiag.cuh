// SPDX-License-Identifier: GPL-2.0-only
// Symmetric eigendecomposition for one thread block: Householder
// tridiagonalization, then either the accumulated reflectors and implicit QL
// on the tridiagonal matrix, or divide and conquer on it with the reflectors
// applied to its vectors (block_eigen_dc, given scratch). QL is about a
// quarter of the instructions of block_jacobi.cuh at k = 64 (which took 10.8
// sweeps a matrix there), but its rotations are a serial chain on thread 0
// (the other threads apply them one iteration behind): a quarter of a 64 x 64
// block's samples were the other warps waiting for it. Divide and conquer has
// no such chain. Every sum has a fixed order: the result does not depend on
// scheduling.
#pragma once

#include <cuda_runtime.h>

#include "cuda/common/block_jacobi.cuh"

namespace nss_cuda {

// 1 / x to about two units in the last place (the secular equation).
template <class Real>
__device__ __forceinline__ Real block_eigen_rcp(Real x) {
    if constexpr (sizeof(Real) == 4) return __fdividef(1.0f, x);
    else return Real(1) / x;
}

// Four reals loaded as one (the reflectors of a step of block_eigen_dc).
template <class Real>
struct alignas(4 * sizeof(Real)) BlockEigenQuad {
    Real x[4];
};

// Reals of the scratch of block_eigen_dc (any memory, read by every thread).
__host__ __device__ constexpr int block_eigen_dc_reals(int k) { return 3 * k * k + 20 * k + 8; }

// Divide and conquer on the symmetric tridiagonal matrix (d, e) of order k
// (Cuppen; the deflation of LAPACK's dlaed2, the vectors of Gu and
// Eisenstat): the matrix is torn into leaves of at most 16 by rank-one
// pieces, each leaf is solved by warp_eigen on one warp, and the nodes of a
// level are merged together up the tree. The vectors in the tridiagonal
// basis are kept as rows of a (k x k, its contents are not read); at the end
// the reflectors take them to the eigenvectors. d becomes the eigenvalues, in
// no order. The block has at least k threads, a multiple of 32; scratch:
// block_eigen_dc_reals(k) reals of any memory.
template <class Real>
__device__ void block_eigen_dc(Real* a, const Real* reflectors, int k, Real* d, const Real* e, Real* scratch) {
    const int lane = threadIdx.x, lanes = blockDim.x, warp = lane >> 5, warps = lanes >> 5;
    const Real eps = sizeof(Real) == 8 ? Real(2.2e-16) : Real(1.2e-7);
    Real* z = a;  // the vectors (in the block's shared memory)
    Real* u = scratch + k * k;  // the secular vectors of a merge (K x K); first the leaves' matrices
    Real* tmp = u + k * k;    // a merge's new rows (n x n)
    Real* lam = tmp + k * k;  // eigenvalues of the rows of z
    Real* ds = lam + k;       // sorted values, weights and rows of a merge
    Real* zs = ds + k;
    Real* kd = zs + k;  // kept: values, weights, rows
    Real* kz = kd + k;
    Real* dd = kz + k;  // deflated: values
    Real* tau = dd + k;
    Real* zh = tau + k;
    Real* inv = zh + k;
    Real* rc = inv + k;  // deflating rotations: c, s
    Real* newlam = rc + 2 * k;
    int* src = reinterpret_cast<int*>(newlam + k);
    int* ksrc = src + k;
    int* dsrc = ksrc + k;
    int* org = dsrc + k;
    int* rr = org + k;  // deflating rotations: rows
    int* counts = rr + 2 * k;
    int shift = 0;  // leaves = 1 << shift
    while (((k + (1 << shift) - 1) >> shift) > kWarpEigen) ++shift;
    const int leaves = 1 << shift;
    const auto bound = [=](int p) { return (p * k) >> shift; };
    __syncthreads();
    // The tears: T = diag(T1 - |b| e_last e_last', T2 - |b| e_0 e_0') + |b| v v'.
    for (int p = 1 + lane; p < leaves; p += lanes) {
        const int s = bound(p);
        const Real b = fabs(e[s - 1]);
        d[s - 1] -= b;
        d[s] -= b;
    }
    for (int i = lane; i < k * k; i += lanes) z[i] = 0;
    __syncthreads();
    for (int p = warp; p < leaves; p += warps) {
        const int lo = bound(p), n = bound(p + 1) - lo, l = lane & 31;
        Real* g = u + lo * kWarpEigen;
        for (int i = l; i < n * n; i += 32) {
            const int r = i / n, c = i % n;
            g[i] = r == c ? d[lo + r] : c == r + 1 ? e[lo + r] : r == c + 1 ? e[lo + c] : Real(0);
        }
        __syncwarp();
        warp_eigen(g, n, lam + lo);
        for (int i = l; i < n * n; i += 32) z[(lo + i / n) * k + lo + i % n] = g[i];
    }
    __syncthreads();
    for (int width = 1; width < leaves; width *= 2) {
        // The nodes of a level are merged together: node p spans leaves
        // [2 p width, 2 (p + 1) width), its arrays are at its first row.
        const int nodes = leaves / (2 * width);
        const auto node_of = [=](int i) {
            int p = 0;
            while (bound((p + 1) * 2 * width) <= i) ++p;
            return p;
        };
        // z' v in the children's vectors, over sqrt 2 (|v| = sqrt 2), by value.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int n = bound((p + 1) * 2 * width) - lo, t = j - lo;
            const Real beta = e[mid - 1];
            const Real w = (j < mid ? z[j * k + mid - 1] : copysign(Real(1), beta) * z[j * k + mid]) * Real(0.70710678118654752);
            const Real mine = lam[j];
            int rank = 0;
            for (int q = 0; q < n; ++q) {
                const Real other = lam[lo + q];
                rank += other < mine || (other == mine && q < t);
            }
            ds[lo + rank] = mine;
            zs[lo + rank] = w;
            src[lo + rank] = j;
        }
        __syncthreads();
        // Deflation, one thread a node (on different warps when there are).
        const int stride = nodes * 32 <= lanes ? 32 : 1;
        if (lane % stride == 0 && lane / stride < nodes) {
            const int p = lane / stride, lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int n = bound((p + 1) * 2 * width) - lo;
            const Real rho = 2 * fabs(e[mid - 1]);
            Real* ds_ = ds + lo; Real* zs_ = zs + lo; const int* src_ = src + lo;
            Real dmax = 0, zmax = 0;
            for (int i = 0; i < n; ++i) dmax = fmax(dmax, fabs(ds_[i])), zmax = fmax(zmax, fabs(zs_[i]));
            // dlaed2's tolerance, on the scale of D + rho z z' (LAPACK
            // scales the matrix to norm 1 first; |z| alone deflated groups
            // of small norm).
            const Real tol = 8 * eps * fmax(dmax, rho * zmax);
            int kept = 0, dropped = 0, rotations = 0, prev = -1;
            for (int i = 0; i < n; ++i) {
                if (rho * fabs(zs_[i]) <= tol) {
                    dd[lo + dropped] = ds_[i], dsrc[lo + dropped++] = src_[i];
                    continue;
                }
                if (prev >= 0) {
                    const Real h = hypot(zs_[i], zs_[prev]);
                    const Real c = zs_[i] / h, s = -zs_[prev] / h;
                    if (fabs((ds_[i] - ds_[prev]) * c * s) <= tol) {
                        zs_[i] = h;
                        rc[2 * (lo + rotations)] = c, rc[2 * (lo + rotations) + 1] = s;
                        rr[2 * (lo + rotations)] = src_[prev], rr[2 * (lo + rotations) + 1] = src_[i];
                        ++rotations;
                        const Real low = ds_[prev] * c * c + ds_[i] * s * s;
                        ds_[i] = ds_[prev] * s * s + ds_[i] * c * c;
                        dd[lo + dropped] = low, dsrc[lo + dropped++] = src_[prev];
                    } else {
                        kd[lo + kept] = ds_[prev], kz[lo + kept] = zs_[prev], ksrc[lo + kept++] = src_[prev];
                    }
                }
                prev = i;
            }
            if (prev >= 0) kd[lo + kept] = ds_[prev], kz[lo + kept] = zs_[prev], ksrc[lo + kept++] = src_[prev];
            Real total = 0;
            // From here zs holds kz^2, and inv[lo] their sum until the norms.
            for (int i = 0; i < kept; ++i) zs_[i] = kz[lo + i] * kz[lo + i], total += zs_[i];
            counts[2 * p] = kept, counts[2 * p + 1] = rotations;
            inv[lo] = total;
        }
        __syncthreads();
        // The roots of 1 + rho sum kz^2 / (kd - x), one a thread, as tau
        // from the nearer pole kd[o]; Newton on tau f(tau), which has no pole
        // at the origin, inside a bracket.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int t = j - lo, kept = counts[2 * p];
            if (t >= kept) {
                newlam[j] = dd[lo + t - kept];
                continue;
            }
            const Real rho = 2 * fabs(e[mid - 1]);
            const Real* kd_ = kd + lo; const Real* k2 = zs + lo;
            int o = t;
            Real low = 0, high = rho * inv[lo];
            if (t + 1 < kept) {
                const Real half = (kd_[t + 1] - kd_[t]) / 2;
                Real f = 1;
                for (int l = 0; l < kept; ++l) f += rho * k2[l] * block_eigen_rcp((kd_[l] - kd_[t]) - half);
                if (f >= 0) high = half;
                else o = t + 1, low = (kd_[t] - kd_[t + 1]) / 2, high = 0;
            }
            Real x = (low + high) / 2;
            const Real wo = rho * k2[o], origin = kd_[o];
            for (int iteration = 0; iteration < 60; ++iteration) {
                Real s1 = 0, s2 = 0, s3 = 0;
                for (int l = 0; l < kept; ++l) {
                    if (l == o) continue;
                    const Real r = block_eigen_rcp((kd_[l] - origin) - x), q = k2[l] * r;
                    s1 += q, s2 += q * r, s3 += fabs(q);
                }
                const Real h = x * (1 + rho * s1) - wo;
                if (fabs(h) <= 4 * eps * (fabs(x) * (1 + rho * s3) + wo)) break;
                if ((h > 0) == (x > 0)) high = x;
                else low = x;
                Real next = x - h / (1 + rho * s1 + x * rho * s2);
                if (!(next > low && next < high)) next = (low + high) / 2;
                if (next == x) break;
                x = next;
            }
            org[j] = o, tau[j] = x;
            newlam[j] = origin + x;
        }
        __syncthreads();
        // Weights consistent with the roots (Gu and Eisenstat); the deflating
        // rotations of the vectors, one column a thread.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int t = j - lo, kept = counts[2 * p];
            if (t < kept) {
                const Real* kd_ = kd + lo;
                const Real rho = 2 * fabs(e[mid - 1]);
                Real prod = -((kd_[t] - kd_[org[j]]) - tau[j]) / rho;
                for (int q = 0; q < kept; ++q)
                    if (q != t) prod *= ((kd_[t] - kd_[org[lo + q]]) - tau[lo + q]) * block_eigen_rcp(kd_[t] - kd_[q]);
                zh[j] = copysign(sqrt(fmax(prod, Real(0))), kz[j]);
            }
            const int rotations = counts[2 * p + 1];
            for (int r = 0; r < rotations; ++r) {
                const Real cc = rc[2 * (lo + r)], ss = rc[2 * (lo + r) + 1];
                Real* x = z + rr[2 * (lo + r)] * k + j;
                Real* y = z + rr[2 * (lo + r) + 1] * k + j;
                const Real px = *x, py = *y;
                *x = cc * px + ss * py;
                *y = cc * py - ss * px;
            }
        }
        __syncthreads();
        // The secular vectors, scaled to unit length in the product.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j), lo = bound(p * 2 * width);
            const int t = j - lo, kept = counts[2 * p];
            if (t >= kept) continue;
            const Real* kd_ = kd + lo;
            Real* row = u + lo * k + t * kept;
            const Real origin = kd_[org[j]], x = tau[j];
            Real norm = 0;
            for (int i = 0; i < kept; ++i) {
                const Real v = zh[lo + i] * block_eigen_rcp((kd_[i] - origin) - x);
                row[i] = v;
                norm += v * v;
            }
            inv[j] = rsqrt(norm);
        }
        __syncthreads();
        // New rows: the kept vectors times the secular vectors, then the
        // deflated ones; four columns an item.
        const int width_quads = (bound(2 * width) + 3) / 4 + 1;
        for (int item = lane; item < k * width_quads; item += lanes) {
            const int j = item / width_quads, c0 = (item % width_quads) * 4;
            const int p = node_of(j), lo = bound(p * 2 * width), n = bound((p + 1) * 2 * width) - lo;
            const int t = j - lo, kept = counts[2 * p];
            if (c0 >= n) continue;
            Real acc[4] = {0, 0, 0, 0};
            if (t < kept) {
                const Real* row = u + lo * k + t * kept;
                for (int i = 0; i < kept; ++i) {
                    const Real w = row[i];
                    const Real* from = z + ksrc[lo + i] * k + lo + c0;
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        if (c0 + q < n) acc[q] += w * from[q];
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[q] *= inv[j];
            } else {
                const Real* from = z + dsrc[lo + t - kept] * k + lo + c0;
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (c0 + q < n) acc[q] = from[q];
            }
            Real* to = tmp + lo * k + t * n + c0;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                if (c0 + q < n) to[q] = acc[q];
        }
        __syncthreads();
        // A row a warp (neighbouring threads write neighbouring columns).
        for (int j = warp; j < k; j += warps) {
            const int p = node_of(j), lo = bound(p * 2 * width), n = bound((p + 1) * 2 * width) - lo;
            for (int c = lane & 31; c < n; c += 32) z[j * k + lo + c] = tmp[lo * k + (j - lo) * n + c];
        }
        for (int j = lane; j < k; j += lanes) lam[j] = newlam[j];
        __syncthreads();
    }
    // a <- z Q^T = z H_{k - 3} ... H_0 (reflectors: row j holds tau at j and
    // u beyond), four reflectors a step. The scratch is free from here: the
    // reflectors of step b, H_{4b + 3} ... H_{4b}, go there as one quad of
    // their components a column (zero up to the reflector's own row; a
    // missing or empty reflector is all zeros), with their tau and the
    // products of their vectors (v_m . v_i, m > i) beside (the quads take at
    // most k (k + 1) reals and the alignment, before tmp).
    using Quad = BlockEigenQuad<Real>;
    const int steps = (k - 2 + 3) / 4;
    Quad* quads = reinterpret_cast<Quad*>((reinterpret_cast<unsigned long long>(scratch) + sizeof(Quad) - 1) / sizeof(Quad) * sizeof(Quad));
    Real* coefficients = tmp;  // 10 a step: tau of 4b .. 4b + 3, then the products
    for (int i = lane; i < steps * k; i += lanes) {
        const int b = i / k, c = i % k;
        Quad q;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int j = 4 * b + m;
            q.x[m] = j < k - 2 && c > j && reflectors[j * k + j] != Real(0) ? reflectors[j * k + c] : Real(0);
        }
        quads[i] = q;
        if (c < 4) {
            const int j = 4 * b + c;
            coefficients[10 * b + c] = j < k - 2 ? reflectors[j * k + j] : Real(0);
        }
    }
    __syncthreads();
    for (int i = lane; i < steps * 6; i += lanes) {
        // The pairs (m, i): (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2).
        const int b = i / 6, pair = i % 6;
        const int m = pair < 1 ? 1 : pair < 3 ? 2 : 3, o = pair - (m * (m - 1)) / 2;
        Real sum = 0;
        for (int c = 4 * b + 1; c < k; ++c) {
            const Quad q = quads[b * k + c];
            sum += q.x[m] * q.x[o];
        }
        coefficients[10 * b + 4 + pair] = sum;
    }
    __syncthreads();
    {
        // Row by row without barriers: two threads a row, each a half of the
        // columns, from a column skewed by the row so that the rows of a warp
        // read different banks. Every lane of a warp runs every step (the
        // shuffles, and the __syncwarp that orders a step's writes of one
        // half of a row before the next step's reads of the other); a lane
        // past the rows has no columns.
        const int per = lanes >= 2 * k ? 2 : 1, r = lane / per, part = lane % per;
        if (lane < 32 * ((per * k + 31) / 32)) {
            Real* row = a + min(r, k - 1) * k;
            for (int b = steps - 1; b >= 0; --b) {
                __syncwarp();
                const Quad* v = quads + b * k;
                const Real* f = coefficients + 10 * b;
                const int j = 4 * b, len = k - j - 1, half = (len + per - 1) / per;
                const int from = r < k ? part * half : len, to = min(len, from + half);
                const int skew = r % len;
                Real dot[4] = {0, 0, 0, 0};
                for (int t = from, c = j + 1 + (from + skew) % len; t < to; ++t) {
                    const Quad q = v[c];
                    const Real x = row[c];
#pragma unroll
                    for (int m = 0; m < 4; ++m) dot[m] += x * q.x[m];
                    if (++c == k) c = j + 1;
                }
                if (per == 2) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const Real other = __shfl_xor_sync(0xffffffffu, dot[m], 1);
                        dot[m] = part == 0 ? dot[m] + other : other + dot[m];
                    }
                }
                // H_{j + 3} first: each scale takes off what the ones before
                // it moved along its vector.
                const Real s3 = f[3] * dot[3];
                const Real s2 = f[2] * (dot[2] - s3 * f[9]);
                const Real s1 = f[1] * (dot[1] - s3 * f[8] - s2 * f[6]);
                const Real s0 = f[0] * (dot[0] - s3 * f[7] - s2 * f[5] - s1 * f[4]);
                if (r >= k) continue;
                for (int t = from, c = j + 1 + (from + skew) % len; t < to; ++t) {
                    const Quad q = v[c];
                    Real x = row[c];
                    x = fma(-s3, q.x[3], x);
                    x = fma(-s2, q.x[2], x);
                    x = fma(-s1, q.x[1], x);
                    x = fma(-s0, q.x[0], x);
                    row[c] = x;
                    if (++c == k) c = j + 1;
                }
            }
        }
    }
    __syncthreads();
    for (int i = lane; i < k; i += lanes) d[i] = lam[i];
    __syncthreads();
}

// Reals of scratch block_eigen needs: `fast` is read by every thread for
// every element (shared memory), `slow` holds the tridiagonal matrix and the
// rotations (thread 0 writes them; any memory).
__host__ __device__ constexpr int block_eigen_fast_reals(int k) { return k + 20; }
__host__ __device__ constexpr int block_eigen_slow_reals(int k) { return 6 * k; }

// Rounded operations the compiler may not contract.
__device__ __forceinline__ float block_eigen_mul(float x, float y) { return __fmul_rn(x, y); }
__device__ __forceinline__ double block_eigen_mul(double x, double y) { return __dmul_rn(x, y); }
__device__ __forceinline__ float block_eigen_add(float x, float y) { return __fadd_rn(x, y); }
__device__ __forceinline__ double block_eigen_add(double x, double y) { return __dadd_rn(x, y); }
__device__ __forceinline__ float block_eigen_sub(float x, float y) { return __fsub_rn(x, y); }
__device__ __forceinline__ double block_eigen_sub(double x, double y) { return __dsub_rn(x, y); }

template <class Real>
__device__ __forceinline__ Real block_eigen_warp_sum(Real x) {
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) x += __shfl_xor_sync(0xffffffffu, x, step);
    return x;
}

// The rotations of the next QL iteration that has any: (c, s) of the planes
// (i, i + 1), i in [range[0], range[1]), to be applied downwards from
// range[1] - 1. range[0] < 0 when the matrix is diagonal. Thread 0 only.
template <class Real>
__device__ void block_eigen_ql(Real* d, Real* e, int k, Real floor, int& l, int& iteration, int& split, Real* rot, int* range) {
    const Real tolerance = sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7);
    const auto negligible = [=](Real off, Real left, Real right) {
        return fabs(off) <= tolerance * (fabs(left) + fabs(right)) || fabs(off) <= floor;
    };
    while (l < k) {
        // m: the first negligible off-diagonal element from l. The iteration
        // before left it in `split` (met on its way, the same test);
        // otherwise search, four at a time so that the loads do not wait
        // for one another.
        int m = split;
        if (m < l) {
            for (m = l; m + 4 < k; m += 4) {
                const Real e0 = e[m], e1 = e[m + 1], e2 = e[m + 2], e3 = e[m + 3];
                const Real d0 = d[m], d1 = d[m + 1], d2 = d[m + 2], d3 = d[m + 3], d4 = d[m + 4];
                const int hit = negligible(e0, d0, d1) ? 0 : negligible(e1, d1, d2) ? 1 : negligible(e2, d2, d3) ? 2 : negligible(e3, d3, d4) ? 3 : 4;
                if (hit < 4) {
                    m += hit;
                    goto found;
                }
            }
            for (; m < k - 1; ++m) {
                if (negligible(e[m], d[m], d[m + 1])) break;
            }
        found:
            split = m;
        }
        if (m == l || iteration == 30) {
            if (m == l) split = -1;
            ++l;
            iteration = 0;
            continue;
        }
        ++iteration;
        Real g = (d[l + 1] - d[l]) / (Real(2) * e[l]);
        Real r = sqrt(g * g + Real(1));
        g = d[m] - d[l] + e[l] / (g + copysign(r, g));
        // The chain g -> (c, s) -> g of one plane after another bounds the
        // whole decomposition: a reciprocal square root with one Newton
        // step instead of a square root and a division, the next plane's
        // elements loaded a plane ahead, d[i + 1] kept in a register.
        Real s = 1, c = 1, p = 0;
        int i = m - 1;
        Real here = d[i], off = e[i], above = d[m], settled = 0;
        bool vanished = false;
        for (; i >= l; --i) {
            const Real f = s * off, b = c * off;
            const Real di = here;
            if (i > l) here = d[i - 1], off = e[i - 1];
            const Real square = f * f + g * g;
            if (square == Real(0)) {
                e[i + 1] = 0;
                d[i + 1] = above - p;
                e[m] = 0;
                vanished = true;
                break;
            }
            Real inverse = rsqrt(square);
            inverse = inverse * (Real(1.5) - Real(0.5) * square * inverse * inverse);
            const Real length = square * inverse;
            e[i + 1] = length;
            s = f * inverse;
            c = g * inverse;
            g = above - p;
            r = (di - g) * s + Real(2) * c * b;
            p = s * r;
            const Real diagonal = g + p;
            d[i + 1] = diagonal;
            if (i + 1 < m && negligible(length, diagonal, settled)) split = i + 1;
            settled = diagonal;
            g = c * r - b;
            above = di;
            rot[2 * i] = c;
            rot[2 * i + 1] = s;
        }
        int low = l;
        if (vanished) {
            low = i + 1;
            split = -1;
        } else {
            const Real diagonal = above - p;
            d[l] = diagonal;
            e[l] = g;
            e[m] = 0;
            if (negligible(g, diagonal, settled)) split = l;
        }
        if (m > low) {
            range[0] = low;
            range[1] = m;
            return;
        }
    }
    range[0] = -1;
    range[1] = -1;
}

// a: symmetric k x k (row-major, both triangles), on return its
// eigenvectors, one per row. The eigenvalues are slow[0 .. k), in no order.
// reflectors: k x k reals of any memory (written once, read once). Keep a
// and fast in shared memory. The block has at least max(k, 32) threads, a
// multiple of 32. With kDc and dc (block_eigen_dc_reals(k) reals of any
// memory) the tridiagonal matrix goes to block_eigen_dc; that path also
// takes a leaner rank-2 update (only the columns beyond j, products that
// cannot be contracted so that both triangles get the same bits); the QL
// path keeps the original one.
//
// Threads are (row, segment of columns); a thread walks its columns from
// its row's offset when k is even, so that the threads of a warp read
// different banks (a column of a row-major matrix is one bank when k is a
// multiple of 32).
template <class Real, bool kDc = false>
__device__ void block_eigen(Real* a, Real* reflectors, int k, Real* fast, Real* slow, Real* dc = nullptr) {
    const bool divide = kDc && dc != nullptr;
    const int lane = threadIdx.x, lanes = blockDim.x;
    Real* d = slow;
    Real* e = slow + k;
    Real* rot = slow + 2 * k;  // two batches of k rotations
    Real* w = fast;
    Real* part = fast + k;  // one sum per warp (16)
    int* range = reinterpret_cast<int*>(part + 16);  // two batches
    const int row = lane % k, segment = lane / k;
    const int span = (k + lanes / k - 1) / (lanes / k);
    const int begin = segment * span;
    const int first = begin + ((k & 1) ? 0 : row % span);
    const bool owner = lane < k;
    const int owner_warps = (k + 31) / 32;

    // A <- H A H for the reflector H = I - tau u u^T that clears row j
    // beyond j + 1; u stays in row j, tau on the diagonal.
    for (int j = 0; j + 2 < k; ++j) {
        const Real* x = a + j * k;
        Real tau = 0, alpha = 0, u0 = 0, mine = 0, p = 0;
        if (owner && row >= j) {
            Real tail = 0;
            for (int c = j + 2; c < k; ++c) tail += x[c] * x[c];  // the same on every thread
            const Real x0 = x[j + 1];
            const Real norm = sqrt(tail + x0 * x0);
            const Real h = tail + x0 * x0 + fabs(x0) * norm;
            alpha = u0 = x0;
            if (tail > Real(0) && h > Real(1e-30)) {
                alpha = x0 > Real(0) ? -norm : norm;
                u0 = x0 - alpha;
                tau = Real(1) / h;
            }
            if (row > j && tau != Real(0)) {
                const Real* mine_row = a + row * k;
                mine = row == j + 1 ? u0 : x[row];
                Real sum = 0;
                if (divide) {
                    // Over x (x[j + 1] is still x0), then the correction to u0.
                    for (int t = j + 1, c = (k & 1) ? j + 1 : row; t < k; ++t) {
                        sum += mine_row[c] * x[c];
                        if (++c == k) c = j + 1;
                    }
                    sum += mine_row[j + 1] * (u0 - x0);
                } else {
                    for (int t = j + 1, c = (k & 1) ? j + 1 : row; t < k; ++t) {
                        sum += mine_row[c] * (c == j + 1 ? u0 : x[c]);
                        if (++c == k) c = j + 1;
                    }
                }
                p = tau * sum;
            }
        }
        if (lane < 32 * owner_warps) {
            const Real sum = block_eigen_warp_sum(p * mine);
            if (lane % 32 == 0) part[lane / 32] = sum;
        }
        __syncthreads();
        if (owner && row >= j) {
            Real dot = 0;
            for (int warp = 0; warp < owner_warps; ++warp) dot += part[warp];
            if (row > j) w[row] = p - Real(0.5) * tau * dot * mine;
            if (row == j + 1) {
                a[j * k + j + 1] = u0;
                e[j] = alpha;
            }
            if (row == j) {
                d[j] = a[j * k + j];
                a[j * k + j] = tau;
            }
        }
        __syncthreads();
        // The columns beyond j of the thread's segment, from a skewed one.
        // (row, c) and (c, row) get the same rounded products and sum
        // (commutative), so the matrix stays symmetric bit for bit.
        const int low = max(begin, j + 1), high = min(begin + span, k);
        if (!divide && row > j && begin + span > j + 1 && x[j] != Real(0)) {
            const Real ui = x[row], wi = w[row];
            for (int t = 0, c = first; t < span; ++t) {
                if (c > j && c < k) {
                    // The same operations for (row, c) and (c, row).
                    const bool upper = row <= c;
                    const Real uc = x[c], wc = w[c];
                    Real value = a[row * k + c];
                    value = fma(-(upper ? ui : uc), upper ? wc : wi, value);
                    value = fma(-(upper ? wi : wc), upper ? uc : ui, value);
                    a[row * k + c] = value;
                }
                if (++c == begin + span) c = begin;
            }
        }
        if (divide && row > j && high > low && x[j] != Real(0)) {
            const Real ui = x[row], wi = w[row];
            for (int t = low, c = low + ((k & 1) ? 0 : row % (high - low)); t < high; ++t) {
                const Real uc = x[c], wc = w[c];
                Real* at = a + row * k + c;
                *at = block_eigen_sub(*at, block_eigen_add(block_eigen_mul(ui, wc), block_eigen_mul(wi, uc)));
                if (++c == high) c = low;
            }
        }
        __syncthreads();
    }
    // The reflectors move out, with the last two rows the tridiagonal
    // matrix is complete; a becomes the identity.
    for (int t = 0, c = first; t < span; ++t) {
        if (c < k) {
            const Real value = a[row * k + c];
            if (row + 2 < k) {
                if (c >= row) reflectors[row * k + c] = value;
            } else if (c == row) {
                d[row] = value;
                if (row + 1 == k) e[row] = 0;
            } else if (c == row + 1) {
                e[row] = value;
            }
            a[row * k + c] = row == c ? Real(1) : Real(0);
        }
        if (++c == begin + span) c = begin;
    }
    __syncthreads();

    if constexpr (kDc) {
        if (divide) {
            block_eigen_dc(a, reflectors, k, d, e, dc);
            return;
        }
    }
    // Q = H_0 ... H_{k - 3}, from the last reflector: Q <- H_j Q only
    // changes the rows and columns beyond j.
    for (int j = k - 3; j >= 0; --j) {
        const Real* u = reflectors + j * k;
        const Real tau = u[j];
        if (tau == Real(0)) continue;  // the same on every thread
        if (owner && row > j) {
            Real sum = 0;
            for (int i = j + 1; i < k; ++i) sum += u[i] * a[i * k + row];
            w[row] = tau * sum;
        }
        __syncthreads();
        if (row > j && begin + span > j + 1) {
            const Real ui = u[row];
            for (int t = 0, c = first; t < span; ++t) {
                if (c > j && c < k) a[row * k + c] = fma(-ui, w[c], a[row * k + c]);
                if (++c == begin + span) c = begin;
            }
        }
        __syncthreads();
    }
    // Transposed, a thread's vector components are one bank apart.
    for (int t = 0, c = first; t < span; ++t) {
        if (c > row && c < k) {
            const Real upper = a[row * k + c];
            a[row * k + c] = a[c * k + row];
            a[c * k + row] = upper;
        }
        if (++c == begin + span) c = begin;
    }

    // Implicit QL. Thread 0 computes the rotations of an iteration while
    // the others apply the iteration before.
    int l = 0, iteration = 0, split = -1;
    Real floor = 0;
    if (lane == 0) {
        for (int i = 0; i < k; ++i) floor += fabs(d[i]);
        floor *= (sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7)) / k;
        block_eigen_ql(d, e, k, floor, l, iteration, split, rot, range);
    }
    __syncthreads();
    for (int batch = 0;; batch ^= 1) {
        const int low = range[2 * batch], high = range[2 * batch + 1];
        if (low < 0) break;
        if (lane == 0) {
            block_eigen_ql(d, e, k, floor, l, iteration, split, rot + 2 * k * (batch ^ 1), range + 2 * (batch ^ 1));
        } else {
            const Real* cs = rot + 2 * k * batch;
            for (int r = lane - 1; r < k; r += lanes - 1) {
                Real upper = a[high * k + r];
                for (int i = high - 1; i >= low; --i) {
                    const Real c = cs[2 * i], s = cs[2 * i + 1];
                    const Real lower = a[i * k + r];
                    a[(i + 1) * k + r] = s * lower + c * upper;
                    upper = c * lower - s * upper;
                }
                a[low * k + r] = upper;
            }
        }
        __syncthreads();
    }
}

}  // namespace nss_cuda
