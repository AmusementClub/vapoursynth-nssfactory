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

// Cyclic Jacobi for a k x k matrix, k <= 16, on one warp (blockDim.x == 32),
// in registers. The pairs of a round are fixed, (2j, 2j + 1), and the indices
// move between rounds as in the circle method (0 stays, the other 15 step
// along one ring: 1 -> 2 -> 4 -> ... -> 14 -> 15 -> 13 -> ... -> 3 -> 1), so
// every pair meets once in 15 rounds and the indices are back in place after
// each sweep. Lane (ja, jb) = (lane % 8, lane / 8) holds the rows of pair ja
// at the columns of pairs jb and jb + 4, of g and of v; a move is a shuffle
// along the lane's ja (rows of g) and jb (columns of g and v). The matrix is
// padded with zeros to 16, which no rotation touches.
//
// a: symmetric k x k (row-major); on return its eigenvectors, one per row.
// values: the k eigenvalues, in no order. Real is float (__fdividef, the
// tolerances).
constexpr int kWarpEigen = 16;
constexpr float kWarpEigenSettled = 1e-8f;  // apq^2 / |app aqq| after which one sweep converges

// The columns of pairs (jb, jb + 4) of one row move one step along the ring:
// column 2P takes 2P - 2 (P >= 2) or 1 (P = 1), 0 stays; 2P + 1 takes 2P + 3
// (P <= 6), 15 takes 14. u[b]: column 2 (jb + 4 (b / 2)) + b % 2.
template <class Real>
__device__ __forceinline__ void warp_eigen_move_columns(Real* u, int jb, int down, int up) {
    const unsigned all = 0xffffffffu;
    const Real even0 = __shfl_sync(all, jb == 0 ? u[1] : u[0], down);
    const Real even1 = __shfl_sync(all, jb == 3 ? u[0] : u[2], down);
    const Real odd0 = __shfl_sync(all, jb == 0 ? u[3] : u[1], up);
    const Real odd1 = __shfl_sync(all, u[3], up);
    u[0] = jb == 0 ? u[0] : even0;
    u[1] = odd0;
    u[3] = jb == 3 ? u[2] : odd1;
    u[2] = even1;
}

template <class Real>
__device__ void warp_eigen(Real* a, int k, Real* values) {
    const unsigned all = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    if (k < 2) {
        if (lane < k) values[0] = a[0], a[0] = Real(1);
        __syncwarp();
        return;
    }
    const int ja = lane & 7, jb = lane >> 3;
    const int r0 = 2 * ja;
    // Cell (dr, b): row r0 + dr, column col(b).
    const auto col = [=](int b) { return 2 * (jb + 4 * (b >> 1)) + (b & 1); };
    Real x[2][4], y[2][4];
    Real trace = 0;
    for (int i = 0; i < k; ++i) trace += fabs(a[i * k + i]);  // identical on every lane
#pragma unroll
    for (int dr = 0; dr < 2; ++dr) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int r = r0 + dr, c = col(b);
            x[dr][b] = r < k && c < k ? a[r * k + c] : Real(0);
            y[dr][b] = r == c ? Real(1) : Real(0);
        }
    }
    __syncwarp();
    const Real tolerance = sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7);
    const Real floor = tolerance * trace / k;
    // The diagonal block of pair j is on lane j + 8 (j & 3), in its block j >> 2.
    const bool diagonal = jb == (ja & 3);
    const int slot = 2 * (ja >> 2);
    const int from_a = ja + 8 * (ja & 3), from_b = 9 * jb, from_d = 9 * jb + 4;
    const int down = ja + 8 * ((jb + 3) & 3), up = ja + 8 * ((jb + 1) & 3);  // along jb
    const int before = (lane & ~7) | ((ja + 7) & 7), after = (lane & ~7) | ((ja + 1) & 7);  // along ja
    for (int sweep = 0; sweep < 30; ++sweep) {
        bool large = false;
        for (int round = 0; round < kWarpEigen - 1; ++round) {
            Real c = 1, sn = 0;
            if (diagonal) {
                // The rotation that clears apq: t = tan, from x = aqq - app and y = 2 apq
                // (t = sign(x) y / (|x| + hypot(x, y))); c by a reciprocal square root
                // with one Newton step, so that c^2 + s^2 = 1 to rounding.
                const Real app = x[0][slot], apq = x[0][slot + 1], aqq = x[1][slot + 1];
                if (apq * apq > tolerance * tolerance * fabs(app * aqq) && fabs(apq) > floor) {
                    const Real dx = aqq - app, dy = Real(2) * apq;
                    const Real square = fma(dx, dx, dy * dy);
                    const Real hypot = square * rsqrt(square);
                    const Real t = __fdividef(copysign(dy, dx) * copysign(Real(1), dy), fabs(dx) + hypot);
                    const Real one = fma(t, t, Real(1));
                    Real inverse = rsqrt(one);
                    inverse = inverse * (Real(1.5) - Real(0.5) * one * inverse * inverse);
                    c = inverse;
                    sn = t * inverse;
                    large = large || apq * apq > kWarpEigenSettled * fabs(app * aqq);
                }
            }
            const Real ca = __shfl_sync(all, c, from_a), sa = __shfl_sync(all, sn, from_a);
            const Real cb = __shfl_sync(all, c, from_b), sb = __shfl_sync(all, sn, from_b);
            const Real cd = __shfl_sync(all, c, from_d), sd = __shfl_sync(all, sn, from_d);
            // g: columns, then rows; v: columns.
#pragma unroll
            for (int dr = 0; dr < 2; ++dr) {
                Real p = x[dr][0], q = x[dr][1];
                x[dr][0] = cb * p - sb * q, x[dr][1] = sb * p + cb * q;
                p = x[dr][2], q = x[dr][3];
                x[dr][2] = cd * p - sd * q, x[dr][3] = sd * p + cd * q;
                p = y[dr][0], q = y[dr][1];
                y[dr][0] = cb * p - sb * q, y[dr][1] = sb * p + cb * q;
                p = y[dr][2], q = y[dr][3];
                y[dr][2] = cd * p - sd * q, y[dr][3] = sd * p + cd * q;
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const Real p = x[0][b], q = x[1][b];
                x[0][b] = ca * p - sa * q, x[1][b] = sa * p + ca * q;
            }
            // The move: columns of g and v, then rows of g (row 2P takes
            // 2P - 2 or 1, 0 stays; 2P + 1 takes 2P + 3, 15 takes 14).
#pragma unroll
            for (int dr = 0; dr < 2; ++dr) {
                warp_eigen_move_columns(x[dr], jb, down, up);
                warp_eigen_move_columns(y[dr], jb, down, up);
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const Real even = __shfl_sync(all, ja == 0 ? x[1][b] : x[0][b], before);
                const Real odd = __shfl_sync(all, x[1][b], after);
                const Real stay = x[0][b];
                x[0][b] = ja == 0 ? stay : even;
                x[1][b] = ja == 7 ? stay : odd;
            }
        }
        // Cyclic Jacobi converges quadratically: after a sweep whose
        // rotations were all small, the next would be below the tolerance.
        if (!__any_sync(all, large)) break;
    }
    if (diagonal) {
        if (r0 < k) values[r0] = x[0][slot];
        if (r0 + 1 < k) values[r0 + 1] = x[1][slot + 1];
    }
#pragma unroll
    for (int dr = 0; dr < 2; ++dr) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int r = r0 + dr, c = col(b);
            if (r < k && c < k) a[c * k + r] = y[dr][b];
        }
    }
    __syncwarp();
}

}  // namespace nss_cuda
