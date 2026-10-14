// SPDX-License-Identifier: GPL-2.0-only
// Symmetric eigendecomposition for one thread block: Householder
// tridiagonalization, the accumulated reflectors, then implicit QL on the
// tridiagonal matrix. About a quarter of the instructions of
// block_jacobi.cuh at k = 64 (which took 10.8 sweeps a matrix there). The QL
// rotations are a serial chain on thread 0 (the other threads apply them to
// the vectors one iteration behind), so a block is bound by that chain and
// an SM should hold as many blocks as its shared memory allows. Every sum
// has a fixed order: the result does not depend on scheduling.
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

// Reals of scratch block_eigen needs: `fast` is read by every thread for
// every element (shared memory), `slow` holds the tridiagonal matrix and the
// rotations (thread 0 writes them; any memory).
__host__ __device__ constexpr int block_eigen_fast_reals(int k) { return k + 20; }
__host__ __device__ constexpr int block_eigen_slow_reals(int k) { return 6 * k; }

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
// multiple of 32.
//
// Threads are (row, segment of columns); a thread walks its columns from
// its row's offset when k is even, so that the threads of a warp read
// different banks (a column of a row-major matrix is one bank when k is a
// multiple of 32).
template <class Real>
__device__ void block_eigen(Real* a, Real* reflectors, int k, Real* fast, Real* slow) {
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
                for (int t = j + 1, c = (k & 1) ? j + 1 : row; t < k; ++t) {
                    sum += mine_row[c] * (c == j + 1 ? u0 : x[c]);
                    if (++c == k) c = j + 1;
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
        if (row > j && begin + span > j + 1 && x[j] != Real(0)) {
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
