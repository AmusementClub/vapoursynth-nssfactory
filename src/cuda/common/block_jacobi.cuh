// SPDX-License-Identifier: GPL-2.0-only
// Parallel cyclic Jacobi eigendecomposition for one thread block: each round
// rotates the disjoint index pairs of a round-robin schedule, so the
// rotations of a round commute and the result does not depend on thread
// scheduling. Rotations are computed from the matrix at the start of the
// round, applied to all columns, then to all rows.
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda {

// Pair `pair` of round `round` in a round-robin schedule over K (even) indices.
__device__ __forceinline__ void jacobi_schedule(int round, int pair, int K, int& p, int& q) {
    const int a = (round + pair) % (K - 1);
    const int b = pair == 0 ? K - 1 : (round - pair + (K - 1)) % (K - 1);
    p = min(a, b);
    q = max(a, b);
}

// g: symmetric k x k (row-major), v: its eigenvectors on return. cs holds the
// k / 2 + 1 rotations of a round; *flag is shared scratch. Without `identity`
// the rotations go on from the v passed in (g then being in that basis).
//
// `lanes` threads work on the matrix, this one being `lane` of them. A block
// may hold several matrices, each with its own g, v and cs and all with the
// same k and *flag: every thread of the block then calls this, and the
// sweeps go on until no matrix rotates. A matrix that has converged is not
// changed by the further sweeps (no element passes the test again), so its
// result does not depend on what shares the block.
template <class Real>
__device__ void block_jacobi(Real* g, Real* v, int k, Real* cs, int* flag, bool identity, int lane, int lanes) {
    for (int e = lane; identity && e < k * k; e += lanes) v[e] = (e / k == e % k) ? Real(1) : Real(0);
    const Real tolerance = sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7);
    __syncthreads();
    Real trace = 0;
    for (int i = 0; i < k; ++i) trace += fabs(g[i * k + i]);  // identical on every thread
    const Real floor = tolerance * trace / k;
    const int K = k + (k & 1), half = K / 2;
    for (int sweep = 0; sweep < 30 && k > 1; ++sweep) {
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
                        *flag = 1;
                    }
                }
                cs[2 * pair] = c;
                cs[2 * pair + 1] = sn;
            }
            __syncthreads();
            // Columns of g and v.
            for (int e = lane; e < half * k; e += lanes) {
                const int pair = e / k, i = e % k;
                const Real c = cs[2 * pair], sn = cs[2 * pair + 1];
                if (sn == Real(0)) continue;
                int p, q;
                jacobi_schedule(round, pair, K, p, q);
                const Real gip = g[i * k + p], giq = g[i * k + q];
                g[i * k + p] = c * gip - sn * giq;
                g[i * k + q] = sn * gip + c * giq;
                const Real vip = v[i * k + p], viq = v[i * k + q];
                v[i * k + p] = c * vip - sn * viq;
                v[i * k + q] = sn * vip + c * viq;
            }
            __syncthreads();
            // Rows of g.
            for (int e = lane; e < half * k; e += lanes) {
                const int pair = e / k, i = e % k;
                const Real c = cs[2 * pair], sn = cs[2 * pair + 1];
                if (sn == Real(0)) continue;
                int p, q;
                jacobi_schedule(round, pair, K, p, q);
                const Real gpi = g[p * k + i], gqi = g[q * k + i];
                g[p * k + i] = c * gpi - sn * gqi;
                g[q * k + i] = sn * gpi + c * gqi;
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
