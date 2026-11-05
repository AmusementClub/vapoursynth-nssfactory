// SPDX-License-Identifier: GPL-2.0-only
// Parallel cyclic Jacobi eigendecomposition for the matrices of one workgroup
// (nss_cuda's block_jacobi<float, KN>, block_jacobi.cuh): each round rotates
// the disjoint index pairs of a round-robin schedule, so the rotations of a
// round commute and the result does not depend on scheduling; g then takes
// J^T g J and v takes v J. Every matrix of the workgroup has KN = N and its
// own g, v and cs (s_g, s_v, s_cs at gb, gb, cb); they share s_flag (the
// last round of the sweep that rotated, plus one; set by atomic max), and the
// sweeps go on until no matrix rotates (a converged matrix is not changed by
// further sweeps). Called by every invocation of the workgroup.
//
// Needs: N (even), shared float s_g[], s_v[], s_cs[], shared int s_flag.
#ifndef NSS_VK_WNNM_BLOCK_JACOBI_GLSL
#define NSS_VK_WNNM_BLOCK_JACOBI_GLSL

// Pair `pair` of round `round` in a round-robin schedule over K indices.
void jacobi_schedule(int round, int pair, int K, out int p, out int q) {
    int a = round + pair;
    if (a >= K - 1) a -= K - 1;
    int b = round - pair;
    if (b < 0) b += K - 1;
    if (pair == 0) b = K - 1;
    p = min(a, b);
    q = max(a, b);
}

float copysign_one(float x) { return uintBitsToFloat(0x3f800000u | (floatBitsToUint(x) & 0x80000000u)); }

// g (row-major N x N at s_g[gb]) to its eigenvalues on the diagonal, v
// (s_v[gb]) to the eigenvectors (column i for eigenvalue i), from v = I;
// lane: this invocation's index among the matrix's N.
void block_jacobi(int gb, int cb, int lane) {
    const int k = N, hk = N / 2;
    for (int e = lane; e < k * k; e += k) s_v[gb + e] = (e / k == e % k) ? 1.0f : 0.0f;
    const float tolerance = 1e-7f;
    barrier();
    float trace = 0.0f;
    for (int i = 0; i < k; ++i) trace += abs(s_g[gb + i * k + i]);  // identical on every invocation
    const float floor_ = tolerance * trace / float(k);
    const int row_first = lane / hk, mine = lane % hk;
    for (int sweep = 0; sweep < 30; ++sweep) {
        if (gl_LocalInvocationIndex == 0u) s_flag = 0;  // one writer (nss_cuda: every matrix's lane 0)
        barrier();
        for (int round = 0; round < k - 1; ++round) {
            if (lane < hk) {
                int p, q;
                jacobi_schedule(round, lane, k, p, q);
                float c = 1.0f, sn = 0.0f;
                const float apq = s_g[gb + p * k + q], app = s_g[gb + p * k + p], aqq = s_g[gb + q * k + q];
                if (abs(apq) > tolerance * sqrt(abs(app * aqq)) && abs(apq) > floor_) {
                    const float theta = (aqq - app) / (2.0f * apq);
                    const float t = copysign_one(theta) / (abs(theta) + sqrt(theta * theta + 1.0f));
                    c = 1.0f / sqrt(t * t + 1.0f);
                    sn = t * c;
                    // The round rotated: atomically (lanes of a round store at
                    // once; plain stores of one value would still race).
                    atomicMax(s_flag, round + 1);
                }
                s_cs[cb + 2 * lane] = c;
                s_cs[cb + 2 * lane + 1] = sn;
            }
            barrier();
            // No pair of this round rotates: nothing to apply (the barrier
            // stays: the next round writes what this test reads).
            if (s_flag != round + 1) {
                barrier();
                continue;
            }
            const float c = s_cs[cb + 2 * mine], sn = s_cs[cb + 2 * mine + 1];
            int p, q;
            jacobi_schedule(round, mine, k, p, q);
            [[unroll]] for (int i = 0; i < hk; ++i) {
                const int row = row_first + 2 * i;
                if (sn != 0.0f) {
                    const float vip = s_v[gb + row * k + p], viq = s_v[gb + row * k + q];
                    s_v[gb + row * k + p] = c * vip - sn * viq;
                    s_v[gb + row * k + q] = sn * vip + c * viq;
                }
                if (row < hk) {
                    // Rows (pr, qr) of the pair `row`, columns (p, q).
                    const float cr = s_cs[cb + 2 * row], sr = s_cs[cb + 2 * row + 1];
                    if (sn != 0.0f || sr != 0.0f) {
                        int pr, qr;
                        jacobi_schedule(round, row, k, pr, qr);
                        float a = s_g[gb + pr * k + p], b = s_g[gb + pr * k + q];
                        float d = s_g[gb + qr * k + p], e = s_g[gb + qr * k + q];
                        if (sn != 0.0f) {
                            const float ap = c * a - sn * b, aq = sn * a + c * b;
                            const float dp = c * d - sn * e, dq = sn * d + c * e;
                            a = ap;
                            b = aq;
                            d = dp;
                            e = dq;
                        }
                        if (sr != 0.0f) {
                            const float ap = cr * a - sr * d, dp = sr * a + cr * d;
                            const float bp = cr * b - sr * e, ep = sr * b + cr * e;
                            a = ap;
                            d = dp;
                            b = bp;
                            e = ep;
                        }
                        s_g[gb + pr * k + p] = a;
                        s_g[gb + pr * k + q] = b;
                        s_g[gb + qr * k + p] = d;
                        s_g[gb + qr * k + q] = e;
                    }
                }
            }
            barrier();
        }
        // Every invocation takes the sweep's verdict before lane 0 may reset it.
        const int rotated = s_flag;
        barrier();
        if (rotated == 0) break;
    }
}

#endif  // NSS_VK_WNNM_BLOCK_JACOBI_GLSL
