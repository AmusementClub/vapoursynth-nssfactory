// SPDX-License-Identifier: GPL-2.0-only
// Parallel cyclic Jacobi eigendecomposition for one thread block: each round
// rotates the disjoint index pairs of a round-robin schedule, so the
// rotations of a round commute and the result does not depend on thread
// scheduling. Rotations are computed from the matrix at the start of the
// round; g then takes J^T g J in one pass (columns, then rows, per element)
// and v takes v J.
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

// Pair `pair` of round `round` in a round-robin schedule over K (even)
// indices: round < K - 1, pair < K / 2.
__device__ __forceinline__ void jacobi_schedule(int round, int pair, int K, int& p, int& q) {
    int a = round + pair;
    if (a >= K - 1) a -= K - 1;
    int b = round - pair;
    if (b < 0) b += K - 1;
    if (pair == 0) b = K - 1;
    p = min(a, b);
    q = max(a, b);
}

// g: symmetric k x k (row-major), v: its eigenvectors on return. cs holds the
// (k + 1) / 2 rotations of a round and is read by every thread for every
// element: keep it in shared memory. *flag is shared scratch (the last round
// of the sweep that rotated, plus one). Without
// `identity` the rotations go on from the v passed in (g then being in that
// basis).
//
// `lanes` threads work on the matrix, this one being `lane` of them. A block
// may hold several matrices, each with its own g, v and cs and all with the
// same k and *flag: every thread of the block then calls this, and the
// sweeps go on until no matrix rotates. A matrix that has converged is not
// changed by the further sweeps (no element passes the test again), so its
// result does not depend on what shares the block.
//
// A round's work items are (row, pair): row `row` of v under the rotation of
// `pair`, and for row < k / 2 the 2 x 2 block of g that the pairs `row` and
// `pair` span, which no other item touches. Neighbouring threads take
// neighbouring pairs of one row, that is different columns of one matrix row
// (the same column of neighbouring rows is a bank conflict when k is a
// multiple of 32).
template <class Real>
__device__ void block_jacobi(Real* g, Real* v, int k, Real* cs, int* flag, bool identity, int lane, int lanes) {
    for (int e = lane; identity && e < k * k; e += lanes) v[e] = (e / k == e % k) ? Real(1) : Real(0);
    const Real tolerance = sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7);
    __syncthreads();
    if (k < 2) return;
    Real trace = 0;
    for (int i = 0; i < k; ++i) trace += fabs(g[i * k + i]);  // identical on every thread
    const Real floor = tolerance * trace / k;
    const int K = k + (k & 1), half = K / 2;
    // Item `lane`, then every `lanes`-th: (row, pair) advances without a division.
    const int row_first = lane / half, pair_first = lane % half;
    const int row_step = lanes / half, pair_step = lanes % half;
    for (int sweep = 0; sweep < 30; ++sweep) {
        if (lane == 0) *flag = 0;  // the same value from every matrix's lane 0
        __syncthreads();
        for (int round = 0; round < K - 1; ++round) {
            for (int pair = lane; pair < half; pair += lanes) {
                int p, q;
                jacobi_schedule(round, pair, K, p, q);
                Real c = 1, sn = 0;
                if (q < k) {
                    const Real apq = g[p * k + q], app = g[p * k + p], aqq = g[q * k + q];
                    if (fabs(apq) > tolerance * sqrt(fabs(app * aqq)) && fabs(apq) > floor) {
                        const Real theta = (aqq - app) / (Real(2) * apq);
                        const Real t = copysign(Real(1), theta) / (fabs(theta) + sqrt(theta * theta + Real(1)));
                        c = Real(1) / sqrt(t * t + Real(1));
                        sn = t * c;
                        *flag = round + 1;
                    }
                }
                cs[2 * pair] = c;
                cs[2 * pair + 1] = sn;
            }
            __syncthreads();
            // No pair of this round rotates (late sweeps): nothing to apply. The
            // barrier stays: the next round writes what this test reads.
            if (*flag != round + 1) {
                __syncthreads();
                continue;
            }
            Real c = 1, sn = 0;
            int p = 0, q = 0;
            for (int row = row_first, pair = pair_first, loaded = -1; row < k;) {
                if (pair != loaded) {  // a thread keeps its pair when lanes is a multiple of k / 2
                    c = cs[2 * pair], sn = cs[2 * pair + 1];  // sn != 0 only with q < k
                    jacobi_schedule(round, pair, K, p, q);
                    loaded = pair;
                }
                if (sn != Real(0)) {
                    const Real vip = v[row * k + p], viq = v[row * k + q];
                    v[row * k + p] = c * vip - sn * viq;
                    v[row * k + q] = sn * vip + c * viq;
                }
                if (row < half) {
                    // Rows (pr, qr) of the pair `row`, columns (p, q).
                    const Real cr = cs[2 * row], sr = cs[2 * row + 1];
                    if (sn != Real(0) || sr != Real(0)) {
                        int pr, qr;
                        jacobi_schedule(round, row, K, pr, qr);
                        const bool wide = q < k, tall = qr < k;
                        Real a = g[pr * k + p], b = wide ? g[pr * k + q] : Real(0);
                        Real d = tall ? g[qr * k + p] : Real(0), e = wide && tall ? g[qr * k + q] : Real(0);
                        if (sn != Real(0)) {
                            const Real ap = c * a - sn * b, aq = sn * a + c * b;
                            const Real dp = c * d - sn * e, dq = sn * d + c * e;
                            a = ap, b = aq, d = dp, e = dq;
                        }
                        if (sr != Real(0)) {
                            const Real ap = cr * a - sr * d, dp = sr * a + cr * d;
                            const Real bp = cr * b - sr * e, ep = sr * b + cr * e;
                            a = ap, d = dp, b = bp, e = ep;
                        }
                        g[pr * k + p] = a;
                        if (wide) g[pr * k + q] = b;
                        if (tall) g[qr * k + p] = d;
                        if (wide && tall) g[qr * k + q] = e;
                    }
                }
                pair += pair_step;
                row += row_step;
                if (pair >= half) {
                    pair -= half;
                    ++row;
                }
            }
            __syncthreads();
        }
        // Every thread takes the sweep's verdict before thread 0 may reset it.
        const int rotated = *flag;
        __syncthreads();
        if (!rotated) break;
    }
}

// One matrix per block, on all its threads.
template <class Real>
__device__ void block_jacobi(Real* g, Real* v, int k, Real* cs, int* flag, bool identity = true) {
    block_jacobi(g, v, k, cs, flag, identity, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
}

}  // namespace nss_cuda
