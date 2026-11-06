// SPDX-License-Identifier: GPL-2.0-only
// Symmetric eigendecomposition of a 32 x 32 matrix on one 32-invocation
// workgroup, the GLSL port of nss_cuda's block_eigen<float, true> with divide
// and conquer (block_tridiag.cuh; only the divide-and-conquer branch, which
// is what the Gram kernels take at 17 to 32 patches): Householder
// tridiagonalization, divide and conquer on the tridiagonal matrix (Cuppen;
// dlaed2's deflation, the vectors of Gu and Eisenstat; leaves of at most 16
// solved by warp_eigen on the warp), then the reflectors applied to the
// vectors. Every sum has a fixed order: run-to-run identical.
//
// The workgroup is one 32-lane warp (WARP_L 32; native or emulated), so
// nss_cuda's __syncthreads and __syncwarp are both sync() here. Storage: a
// (the matrix, then the vectors, one per row) s_g, the reflectors s_v, fast
// s_fast (N + 20), slow s_slow (6 N: d, e), all shared; the divide and
// conquer's scratch (dc_floats() a group) in device memory at `dc`, its
// integers stored as float bits.
//
// Needs: N == 32, common/warp.glsl (WARP_L 32), wnnm/common.glsl (Scratch),
// shared float s_g[N * N], s_v[N * N], s_fast[N + 20], s_slow[6 * N].
#ifndef NSS_VK_WNNM_BLOCK_EIGEN_GLSL
#define NSS_VK_WNNM_BLOCK_EIGEN_GLSL

#if N != 32
#error block_eigen.glsl serves N = 32
#endif

const int kWarpEigen = 16;
const float kWarpEigenSettled = 1e-8f;
const float kEps = 1.2e-7f;

// The device scratch's floats a group, vulkan/wnnm/kernels.cpp's kDcFloats.
int dc_floats() { return 2 * N * N; }

void sync() {
    memoryBarrierBuffer();
    barrier();
}

float copysign_f(float x, float y) {
    return uintBitsToFloat((floatBitsToUint(x) & 0x7fffffffu) | (floatBitsToUint(y) & 0x80000000u));
}

// The divide and conquer's scratch: its two k x k matrices (the leaves' and
// secular vectors u, a merge's new rows tmp) in device memory (DM, this
// workgroup's at g_dc), its vectors in shared memory (DV; nss_cuda keeps all
// of it in device memory, read through L1 in a block, where a coherent Vulkan
// buffer is read past L1 on NVIDIA: the secular equation's loops waited on
// L2), integers as float bits.
uint64_t g_dc;
#define DM(i) Scratch(g_dc).v[(i)]
shared float s_dcv[20 * N + 8 + 10 * 8];
#define DV(i) s_dcv[(i)]
#define DVI(i) floatBitsToInt(s_dcv[(i)])
#define SET_DVI(i, value) s_dcv[(i)] = intBitsToFloat(value)

// ---------------------------------------------------------------- warp_eigen
// Cyclic Jacobi for a k x k matrix (k <= 16) in the scratch at `ab` on the
// warp, in registers (nss_cuda's warp_eigen, block_jacobi.cuh): the pairs of
// a round are (2j, 2j + 1), the indices move between rounds as in the circle
// method, lane (ja, jb) holds the rows of pair ja at the columns of pairs jb
// and jb + 4 of g and v; on return the eigenvectors, one per row, at ab and
// the eigenvalues at vb (in no order).

// The columns of pairs (jb, jb + 4) of one row move one step along the ring.
void warp_eigen_move_columns(inout float u0, inout float u1, inout float u2, inout float u3, int jb, int down, int up) {
    const float even0 = w_shfl_f(jb == 0 ? u1 : u0, down);
    const float even1 = w_shfl_f(jb == 3 ? u0 : u2, down);
    const float odd0 = w_shfl_f(jb == 0 ? u3 : u1, up);
    const float odd1 = w_shfl_f(u3, up);
    u0 = jb == 0 ? u0 : even0;
    u1 = odd0;
    u3 = jb == 3 ? u2 : odd1;
    u2 = even1;
}

void warp_eigen(int ab, int k, int vb) {
    const int lane = int(gl_LocalInvocationID.x) & 31;
    if (k < 2) {
        if (lane < k) {
            DV(vb) = DM(ab);
            DM(ab) = 1.0f;
        }
        sync();
        return;
    }
    const int ja = lane & 7, jb = lane >> 3;
    const int r0 = 2 * ja;
    // Cell (dr, b), index dr * 4 + b: row r0 + dr, column 2 (jb + 4 (b >> 1)) + (b & 1).
    float x[8], y[8];
    float trace = 0.0f;
    for (int i = 0; i < k; ++i) trace += abs(DM(ab + i * k + i));  // identical on every lane
    [[unroll]] for (int dr = 0; dr < 2; ++dr) {
        [[unroll]] for (int b = 0; b < 4; ++b) {
            const int r = r0 + dr, c = 2 * (jb + 4 * (b >> 1)) + (b & 1);
            x[dr * 4 + b] = r < k && c < k ? DM(ab + r * k + c) : 0.0f;
            y[dr * 4 + b] = r == c ? 1.0f : 0.0f;
        }
    }
    sync();
    const float tolerance = 1e-7f;
    const float floor_ = tolerance * trace / float(k);
    // The diagonal block of pair j is on lane j + 8 (j & 3), in its block j >> 2.
    const bool diagonal = jb == (ja & 3);
    const bool slot2 = (ja >> 2) != 0;  // slot = 2 (ja >> 2)
    const int from_a = ja + 8 * (ja & 3), from_b = 9 * jb, from_d = 9 * jb + 4;
    const int down = ja + 8 * ((jb + 3) & 3), up = ja + 8 * ((jb + 1) & 3);  // along jb
    const int before = (lane & ~7) | ((ja + 7) & 7), after = (lane & ~7) | ((ja + 1) & 7);  // along ja
    for (int sweep = 0; sweep < 30; ++sweep) {
        bool large = false;
        [[unroll]] for (int round = 0; round < kWarpEigen - 1; ++round) {
            float c = 1.0f, sn = 0.0f;
            if (diagonal) {
                const float app = slot2 ? x[2] : x[0], apq = slot2 ? x[3] : x[1], aqq = slot2 ? x[7] : x[5];
                if (apq * apq > tolerance * tolerance * abs(app * aqq) && abs(apq) > floor_) {
                    const float dx = aqq - app, dy = 2.0f * apq;
                    const float square = fma(dx, dx, dy * dy);
                    const float hyp = square * inversesqrt(square);
                    const float t = (copysign_f(dy, dx) * copysign_f(1.0f, dy)) / (abs(dx) + hyp);
                    const float one = fma(t, t, 1.0f);
                    float inverse = inversesqrt(one);
                    inverse = inverse * (1.5f - 0.5f * one * inverse * inverse);
                    c = inverse;
                    sn = t * inverse;
                    large = large || apq * apq > kWarpEigenSettled * abs(app * aqq);
                }
            }
            const float ca = w_shfl_f(c, from_a), sa = w_shfl_f(sn, from_a);
            const float cb = w_shfl_f(c, from_b), sb = w_shfl_f(sn, from_b);
            const float cd = w_shfl_f(c, from_d), sd = w_shfl_f(sn, from_d);
            // g: columns, then rows; v: columns.
            [[unroll]] for (int dr = 0; dr < 2; ++dr) {
                float p = x[dr * 4], q = x[dr * 4 + 1];
                x[dr * 4] = cb * p - sb * q;
                x[dr * 4 + 1] = sb * p + cb * q;
                p = x[dr * 4 + 2];
                q = x[dr * 4 + 3];
                x[dr * 4 + 2] = cd * p - sd * q;
                x[dr * 4 + 3] = sd * p + cd * q;
                p = y[dr * 4];
                q = y[dr * 4 + 1];
                y[dr * 4] = cb * p - sb * q;
                y[dr * 4 + 1] = sb * p + cb * q;
                p = y[dr * 4 + 2];
                q = y[dr * 4 + 3];
                y[dr * 4 + 2] = cd * p - sd * q;
                y[dr * 4 + 3] = sd * p + cd * q;
            }
            [[unroll]] for (int b = 0; b < 4; ++b) {
                const float p = x[b], q = x[4 + b];
                x[b] = ca * p - sa * q;
                x[4 + b] = sa * p + ca * q;
            }
            // The move: columns of g and v, then rows of g.
            warp_eigen_move_columns(x[0], x[1], x[2], x[3], jb, down, up);
            warp_eigen_move_columns(x[4], x[5], x[6], x[7], jb, down, up);
            warp_eigen_move_columns(y[0], y[1], y[2], y[3], jb, down, up);
            warp_eigen_move_columns(y[4], y[5], y[6], y[7], jb, down, up);
            [[unroll]] for (int b = 0; b < 4; ++b) {
                const float even = w_shfl_f(ja == 0 ? x[4 + b] : x[b], before);
                const float odd = w_shfl_f(x[4 + b], after);
                const float stay = x[b];
                x[b] = ja == 0 ? stay : even;
                x[4 + b] = ja == 7 ? stay : odd;
            }
        }
        // Converged after a sweep whose rotations were all small. An exact
        // count (not w_any, which the emulated layer answers true): the
        // sweeps taken change the bits.
        if (w_ballot(large) == 0u) break;
    }
    if (diagonal) {
        if (r0 < k) DV(vb + r0) = slot2 ? x[2] : x[0];
        if (r0 + 1 < k) DV(vb + r0 + 1) = slot2 ? x[7] : x[5];
    }
    [[unroll]] for (int dr = 0; dr < 2; ++dr) {
        [[unroll]] for (int b = 0; b < 4; ++b) {
            const int r = r0 + dr, c = 2 * (jb + 4 * (b >> 1)) + (b & 1);
            if (r < k && c < k) DM(ab + c * k + r) = y[dr * 4 + b];
        }
    }
    sync();
}

// ------------------------------------------------------------ block_eigen_dc
// The scratch's arrays (offsets in floats), as nss_cuda's.
const int kK = N, kK2 = N * N;
const int kU = 0, kTmp = kK2;  // DM
const int kLam = 0;            // DV from here
const int kDs = kLam + kK, kZs = kDs + kK, kKd = kZs + kK, kKz = kKd + kK, kDd = kKz + kK, kTau = kDd + kK;
const int kZh = kTau + kK, kInv = kZh + kK, kRc = kInv + kK, kNewlam = kRc + 2 * kK;
const int kSrc = kNewlam + kK, kKsrc = kSrc + kK, kDsrc = kKsrc + kK, kOrg = kDsrc + kK, kRr = kOrg + kK;
const int kCounts = kRr + 2 * kK;
const int kCoef = 20 * kK + 8;  // the reflectors' coefficients: 10 a step

// Reflector j's component at column c (nss_cuda's quads): zero up to its
// own row, and for a missing (j >= k - 2) or empty (tau 0) reflector.
float reflector(int j, int c) { return j < kK - 2 && c > j && s_v[j * kK + j] != 0.0f ? s_v[j * kK + c] : 0.0f; }

int g_shift;
int bound(int p) { return (p * kK) >> g_shift; }
int node_of(int i, int width) {
    int p = 0;
    while (bound((p + 1) * 2 * width) <= i) ++p;
    return p;
}

// d (s_slow[0 .. k)), e (s_slow[k .. 2k)): the tridiagonal matrix; on return d
// holds the eigenvalues (no order) and s_g the vectors, one per row.
void block_eigen_dc() {
    const int k = kK, lane = int(gl_LocalInvocationID.x), lanes = 32;
    // s_g's contents are not read: the vectors in the tridiagonal basis, rows.
    g_shift = 0;
    while (((k + (1 << g_shift) - 1) >> g_shift) > kWarpEigen) ++g_shift;
    const int leaves = 1 << g_shift;
    sync();
    // The tears: T = diag(T1 - |b| e_last e_last', T2 - |b| e_0 e_0') + |b| v v'.
    for (int p = 1 + lane; p < leaves; p += lanes) {
        const int s = bound(p);
        const float b = abs(s_slow[k + s - 1]);
        s_slow[s - 1] -= b;
        s_slow[s] -= b;
    }
    for (int i = lane; i < k * k; i += lanes) s_g[i] = 0.0f;
    sync();
    for (int p = 0; p < leaves; ++p) {  // one warp: every leaf in turn
        const int lo = bound(p), n = bound(p + 1) - lo, l = lane & 31;
        const int g = kU + lo * kWarpEigen;
        for (int i = l; i < n * n; i += 32) {
            const int r = i / n, c = i % n;
            DM(g + i) = r == c ? s_slow[lo + r] : c == r + 1 ? s_slow[k + lo + r] : r == c + 1 ? s_slow[k + lo + c] : 0.0f;
        }
        sync();
        warp_eigen(g, n, kLam + lo);
        for (int i = l; i < n * n; i += 32) s_g[(lo + i / n) * k + lo + i % n] = DM(g + i);
    }
    sync();
    for (int width = 1; width < leaves; width *= 2) {
        // The nodes of a level are merged together: node p spans leaves
        // [2 p width, 2 (p + 1) width), its arrays are at its first row.
        const int nodes = leaves / (2 * width);
        // z' v in the children's vectors, over sqrt 2 (|v| = sqrt 2), by value.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j, width), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int n = bound((p + 1) * 2 * width) - lo, t = j - lo;
            const float beta = s_slow[k + mid - 1];
            const float w = (j < mid ? s_g[j * k + mid - 1] : copysign_f(1.0f, beta) * s_g[j * k + mid]) * 0.70710678118654752f;
            const float mine = DV(kLam + j);
            int rank = 0;
            for (int q = 0; q < n; ++q) {
                const float other = DV(kLam + lo + q);
                rank += (other < mine || (other == mine && q < t)) ? 1 : 0;
            }
            DV(kDs + lo + rank) = mine;
            DV(kZs + lo + rank) = w;
            SET_DVI(kSrc + lo + rank, j);
        }
        sync();
        // Deflation, one invocation a node.
        const int stride = nodes * 32 <= lanes ? 32 : 1;
        if (lane % stride == 0 && lane / stride < nodes) {
            const int p = lane / stride, lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int n = bound((p + 1) * 2 * width) - lo;
            const float rho = 2.0f * abs(s_slow[k + mid - 1]);
            float dmax = 0.0f, zmax = 0.0f;
            for (int i = 0; i < n; ++i) dmax = max(dmax, abs(DV(kDs + lo + i))), zmax = max(zmax, abs(DV(kZs + lo + i)));
            // dlaed2's tolerance, on the scale of D + rho z z'.
            const float tol = 8.0f * kEps * max(dmax, rho * zmax);
            int kept = 0, dropped = 0, rotations = 0, prev = -1;
            for (int i = 0; i < n; ++i) {
                if (rho * abs(DV(kZs + lo + i)) <= tol) {
                    DV(kDd + lo + dropped) = DV(kDs + lo + i);
                    SET_DVI(kDsrc + lo + dropped, DVI(kSrc + lo + i));
                    ++dropped;
                    continue;
                }
                if (prev >= 0) {
                    const float zi = DV(kZs + lo + i), zp = DV(kZs + lo + prev);
                    const float h = sqrt(zi * zi + zp * zp);
                    const float c = zi / h, s = -zp / h;
                    const float di = DV(kDs + lo + i), dp = DV(kDs + lo + prev);
                    if (abs((di - dp) * c * s) <= tol) {
                        DV(kZs + lo + i) = h;
                        DV(kRc + 2 * (lo + rotations)) = c;
                        DV(kRc + 2 * (lo + rotations) + 1) = s;
                        SET_DVI(kRr + 2 * (lo + rotations), DVI(kSrc + lo + prev));
                        SET_DVI(kRr + 2 * (lo + rotations) + 1, DVI(kSrc + lo + i));
                        ++rotations;
                        const float low = dp * c * c + di * s * s;
                        DV(kDs + lo + i) = dp * s * s + di * c * c;
                        DV(kDd + lo + dropped) = low;
                        SET_DVI(kDsrc + lo + dropped, DVI(kSrc + lo + prev));
                        ++dropped;
                    } else {
                        DV(kKd + lo + kept) = dp;
                        DV(kKz + lo + kept) = zp;
                        SET_DVI(kKsrc + lo + kept, DVI(kSrc + lo + prev));
                        ++kept;
                    }
                }
                prev = i;
            }
            if (prev >= 0) {
                DV(kKd + lo + kept) = DV(kDs + lo + prev);
                DV(kKz + lo + kept) = DV(kZs + lo + prev);
                SET_DVI(kKsrc + lo + kept, DVI(kSrc + lo + prev));
                ++kept;
            }
            float total = 0.0f;
            // From here zs holds kz^2, and inv[lo] their sum until the norms.
            for (int i = 0; i < kept; ++i) {
                const float kz = DV(kKz + lo + i);
                DV(kZs + lo + i) = kz * kz;
                total += kz * kz;
            }
            SET_DVI(kCounts + 2 * p, kept);
            SET_DVI(kCounts + 2 * p + 1, rotations);
            DV(kInv + lo) = total;
        }
        sync();
        // The roots of 1 + rho sum kz^2 / (kd - x), one an invocation, as tau
        // from the nearer pole kd[o]; Newton on tau f(tau) inside a bracket.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j, width), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int t = j - lo, kept = DVI(kCounts + 2 * p);
            if (t >= kept) {
                DV(kNewlam + j) = DV(kDd + lo + t - kept);
                continue;
            }
            const float rho = 2.0f * abs(s_slow[k + mid - 1]);
            int o = t;
            float low = 0.0f, high = rho * DV(kInv + lo);
            if (t + 1 < kept) {
                const float half_ = (DV(kKd + lo + t + 1) - DV(kKd + lo + t)) / 2.0f;
                float f = 1.0f;
                for (int l = 0; l < kept; ++l) f += rho * DV(kZs + lo + l) * (1.0f / ((DV(kKd + lo + l) - DV(kKd + lo + t)) - half_));
                if (f >= 0.0f) {
                    high = half_;
                } else {
                    o = t + 1;
                    low = (DV(kKd + lo + t) - DV(kKd + lo + t + 1)) / 2.0f;
                    high = 0.0f;
                }
            }
            float x = (low + high) / 2.0f;
            const float wo = rho * DV(kZs + lo + o), origin = DV(kKd + lo + o);
            for (int iteration = 0; iteration < 60; ++iteration) {
                float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int l = 0; l < kept; ++l) {
                    if (l == o) continue;
                    const float rr = 1.0f / ((DV(kKd + lo + l) - origin) - x), q = DV(kZs + lo + l) * rr;
                    s1 += q;
                    s2 += q * rr;
                    s3 += abs(q);
                }
                const float h = x * (1.0f + rho * s1) - wo;
                if (abs(h) <= 4.0f * kEps * (abs(x) * (1.0f + rho * s3) + wo)) break;
                if ((h > 0.0f) == (x > 0.0f)) {
                    high = x;
                } else {
                    low = x;
                }
                float next = x - h / (1.0f + rho * s1 + x * rho * s2);
                if (!(next > low && next < high)) next = (low + high) / 2.0f;
                if (next == x) break;
                x = next;
            }
            SET_DVI(kOrg + j, o);
            DV(kTau + j) = x;
            DV(kNewlam + j) = origin + x;
        }
        sync();
        // Weights consistent with the roots (Gu and Eisenstat); the deflating
        // rotations of the vectors, one column an invocation.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j, width), lo = bound(p * 2 * width), mid = bound((2 * p + 1) * width);
            const int t = j - lo, kept = DVI(kCounts + 2 * p);
            if (t < kept) {
                const float rho = 2.0f * abs(s_slow[k + mid - 1]);
                const float kdt = DV(kKd + lo + t);
                float prod = -((kdt - DV(kKd + lo + DVI(kOrg + j))) - DV(kTau + j)) / rho;
                for (int q = 0; q < kept; ++q) {
                    if (q != t)
                        prod *= ((kdt - DV(kKd + lo + DVI(kOrg + lo + q))) - DV(kTau + lo + q)) * (1.0f / (kdt - DV(kKd + lo + q)));
                }
                DV(kZh + j) = copysign_f(sqrt(max(prod, 0.0f)), DV(kKz + j));
            }
            const int rotations = DVI(kCounts + 2 * p + 1);
            for (int r = 0; r < rotations; ++r) {
                const float cc = DV(kRc + 2 * (lo + r)), ss = DV(kRc + 2 * (lo + r) + 1);
                const int xi = DVI(kRr + 2 * (lo + r)) * k + j, yi = DVI(kRr + 2 * (lo + r) + 1) * k + j;
                const float px = s_g[xi], py = s_g[yi];
                s_g[xi] = cc * px + ss * py;
                s_g[yi] = cc * py - ss * px;
            }
        }
        sync();
        // The secular vectors, scaled to unit length in the product.
        for (int j = lane; j < k; j += lanes) {
            const int p = node_of(j, width), lo = bound(p * 2 * width);
            const int t = j - lo, kept = DVI(kCounts + 2 * p);
            if (t >= kept) continue;
            const int row = kU + lo * k + t * kept;
            const float origin = DV(kKd + lo + DVI(kOrg + j)), x = DV(kTau + j);
            float norm = 0.0f;
            for (int i = 0; i < kept; ++i) {
                const float v = DV(kZh + lo + i) * (1.0f / ((DV(kKd + lo + i) - origin) - x));
                DM(row + i) = v;
                norm += v * v;
            }
            DV(kInv + j) = inversesqrt(norm);
        }
        sync();
        // New rows: the kept vectors times the secular vectors, then the
        // deflated ones; four columns an item.
        const int width_quads = (bound(2 * width) + 3) / 4 + 1;
        for (int item = lane; item < k * width_quads; item += lanes) {
            const int j = item / width_quads, c0 = (item % width_quads) * 4;
            const int p = node_of(j, width), lo = bound(p * 2 * width), n = bound((p + 1) * 2 * width) - lo;
            const int t = j - lo, kept = DVI(kCounts + 2 * p);
            if (c0 >= n) continue;
            float acc[4] = float[4](0.0f, 0.0f, 0.0f, 0.0f);
            if (t < kept) {
                const int row = kU + lo * k + t * kept;
                for (int i = 0; i < kept; ++i) {
                    const float w = DM(row + i);
                    const int from = DVI(kKsrc + lo + i) * k + lo + c0;
                    [[unroll]] for (int q = 0; q < 4; ++q) {
                        if (c0 + q < n) acc[q] += w * s_g[from + q];
                    }
                }
                const float scale = DV(kInv + j);
                [[unroll]] for (int q = 0; q < 4; ++q) acc[q] *= scale;
            } else {
                const int from = DVI(kDsrc + lo + t - kept) * k + lo + c0;
                [[unroll]] for (int q = 0; q < 4; ++q) {
                    if (c0 + q < n) acc[q] = s_g[from + q];
                }
            }
            const int to = kTmp + lo * k + t * n + c0;
            [[unroll]] for (int q = 0; q < 4; ++q) {
                if (c0 + q < n) DM(to + q) = acc[q];
            }
        }
        sync();
        for (int j = 0; j < k; ++j) {  // a row a warp: every row in turn
            const int p = node_of(j, width), lo = bound(p * 2 * width), n = bound((p + 1) * 2 * width) - lo;
            for (int c = lane & 31; c < n; c += 32) s_g[j * k + lo + c] = DM(kTmp + lo * k + (j - lo) * n + c);
        }
        for (int j = lane; j < k; j += lanes) DV(kLam + j) = DV(kNewlam + j);
        sync();
    }
    // a <- z Q^T = z H_{k - 3} ... H_0 (reflectors: row j of s_v holds tau at j
    // and u beyond), four reflectors a step: reflector 4 b + m's component at
    // column c (zero up to its own row; a missing or empty reflector is all
    // zeros), read from s_v (nss_cuda copies them into quads first), with
    // their tau and the products of their vectors (v_m . v_i, m > i) at kCoef.
    const int steps = (k - 2 + 3) / 4;
    for (int i = lane; i < steps * 4; i += lanes) {
        const int b = i / 4, c = i % 4, j = 4 * b + c;
        DV(kCoef + 10 * b + c) = j < k - 2 ? s_v[j * k + j] : 0.0f;
    }
    for (int i = lane; i < steps * 6; i += lanes) {
        // The pairs (m, o): (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2).
        const int b = i / 6, pair = i % 6;
        const int m = pair < 1 ? 1 : pair < 3 ? 2 : 3, o = pair - (m * (m - 1)) / 2;
        float sum = 0.0f;
        for (int c = 4 * b + 1; c < k; ++c) sum += reflector(4 * b + m, c) * reflector(4 * b + o, c);
        DV(kCoef + 10 * b + 4 + pair) = sum;
    }
    sync();
    {
        // Row by row: one invocation a row (k = 32 lanes), from a column
        // skewed by the row so that the rows read different banks.
        const int r = lane;
        const int rowb = r * k;
        for (int b = steps - 1; b >= 0; --b) {
            const int j = 4 * b, len = k - j - 1;
            const int skew = r % len;
            float dot[4] = float[4](0.0f, 0.0f, 0.0f, 0.0f);
            for (int t = 0, c = j + 1 + skew % len; t < len; ++t) {
                const float x = s_g[rowb + c];
                [[unroll]] for (int m = 0; m < 4; ++m) dot[m] += x * reflector(j + m, c);
                if (++c == k) c = j + 1;
            }
            // H_{j + 3} first: each scale takes off what the ones before it
            // moved along its vector.
            const int f = kCoef + 10 * b;
            const float s3 = DV(f + 3) * dot[3];
            const float s2 = DV(f + 2) * (dot[2] - s3 * DV(f + 9));
            const float s1 = DV(f + 1) * (dot[1] - s3 * DV(f + 8) - s2 * DV(f + 6));
            const float s0 = DV(f + 0) * (dot[0] - s3 * DV(f + 7) - s2 * DV(f + 5) - s1 * DV(f + 4));
            for (int t = 0, c = j + 1 + skew % len; t < len; ++t) {
                float x = s_g[rowb + c];
                x = fma(-s3, reflector(j + 3, c), x);
                x = fma(-s2, reflector(j + 2, c), x);
                x = fma(-s1, reflector(j + 1, c), x);
                x = fma(-s0, reflector(j, c), x);
                s_g[rowb + c] = x;
                if (++c == k) c = j + 1;
            }
        }
    }
    sync();
    for (int i = lane; i < k; i += lanes) s_slow[i] = DV(kLam + i);
    sync();
}

// -------------------------------------------------------------- block_eigen
// s_g: symmetric 32 x 32 (row-major, both triangles); on return its
// eigenvectors, one per row, and the eigenvalues in s_slow[0 .. 32) (no
// order). s_v takes the reflectors. The divide-and-conquer branch of
// nss_cuda's block_eigen<float, true>: the rank-2 update only over the columns
// beyond j, with products that are not contracted, so that both triangles
// keep the same bits.
void block_eigen(uint64_t dc) {
    g_dc = dc;
    const int k = N, lane = int(gl_LocalInvocationID.x);
    const int row = lane;  // one invocation a row (32 lanes)
    for (int j = 0; j + 2 < k; ++j) {
        const int x = j * k;  // row j of s_g
        float tau = 0.0f, alpha = 0.0f, u0 = 0.0f, mine = 0.0f, p = 0.0f;
        if (row >= j) {
            float tail = 0.0f;
            for (int c = j + 2; c < k; ++c) tail += s_g[x + c] * s_g[x + c];  // the same on every invocation
            const float x0 = s_g[x + j + 1];
            const float norm = sqrt(tail + x0 * x0);
            const float h = tail + x0 * x0 + abs(x0) * norm;
            alpha = x0;
            u0 = x0;
            if (tail > 0.0f && h > 1e-30f) {
                alpha = x0 > 0.0f ? -norm : norm;
                u0 = x0 - alpha;
                tau = 1.0f / h;
            }
            if (row > j && tau != 0.0f) {
                const int mine_row = row * k;
                mine = row == j + 1 ? u0 : s_g[x + row];
                float sum = 0.0f;
                // Over x (x[j + 1] is still x0), then the correction to u0.
                for (int t = j + 1, c = row; t < k; ++t) {
                    sum += s_g[mine_row + c] * s_g[x + c];
                    if (++c == k) c = j + 1;
                }
                sum += s_g[mine_row + j + 1] * (u0 - x0);
                p = tau * sum;
            }
        }
        const float dot = w_sum_f(p * mine);
        sync();
        if (row >= j) {
            if (row > j) s_fast[row] = p - 0.5f * tau * dot * mine;
            if (row == j + 1) {
                s_g[x + j + 1] = u0;
                s_slow[k + j] = alpha;
            }
            if (row == j) {
                s_slow[j] = s_g[x + j];
                s_g[x + j] = tau;
            }
        }
        sync();
        // The columns beyond j, from a skewed one; (row, c) and (c, row) get
        // the same rounded products and sum, so the matrix stays symmetric.
        const int low = j + 1, high = k;
        if (row > j && s_g[x + j] != 0.0f) {
            const float ui = s_g[x + row], wi = s_fast[row];
            for (int t = low, c = low + row % (high - low); t < high; ++t) {
                const float uc = s_g[x + c], wc = s_fast[c];
                precise float m1 = ui * wc;
                precise float m2 = wi * uc;
                precise float update = m1 + m2;
                precise float value = s_g[row * k + c] - update;
                s_g[row * k + c] = value;
                if (++c == high) c = low;
            }
        }
        sync();
    }
    // The reflectors move out; with the last two rows the tridiagonal matrix
    // is complete; s_g becomes the identity.
    for (int t = 0, c = row; t < k; ++t) {
        const float value = s_g[row * k + c];
        if (row + 2 < k) {
            if (c >= row) s_v[row * k + c] = value;
        } else if (c == row) {
            s_slow[row] = value;
            if (row + 1 == k) s_slow[k + row] = 0.0f;
        } else if (c == row + 1) {
            s_slow[k + row] = value;
        }
        s_g[row * k + c] = row == c ? 1.0f : 0.0f;
        if (++c == k) c = 0;
    }
    sync();
    block_eigen_dc();
}

#endif  // NSS_VK_WNNM_BLOCK_EIGEN_GLSL
