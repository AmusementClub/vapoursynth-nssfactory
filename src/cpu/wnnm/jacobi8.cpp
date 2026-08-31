#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/wnnm/jacobi8.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void JacobiSvd8(const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V) {
    alignas(32) float U8[64];
    alignas(32) float V8[64];
    if (lda == 8) {
        std::memcpy(U8, A, 64 * sizeof(float));
    } else {
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
                U8[i + j * 8] = A[i + j * lda];
            }
        }
    }
    std::memset(V8, 0, 64 * sizeof(float));
    for (int i = 0; i < 8; ++i) {
        V8[i + i * 8] = 1.f;
    }

#if HWY_MAX_BYTES >= 32
    {
        const hn::FixedTag<float, 8> d;
        using VW = hn::Vec<hn::FixedTag<float, 8>>;
        VW u[8];
        VW vv[8];
        float nrm[8];
        const hn::FixedTag<float, 4> d4;
        auto hsum8 = [&](VW vec) {
            auto x = hn::Add(hn::LowerHalf(d4, vec), hn::UpperHalf(d4, vec));
            x = hn::Add(x, hn::Reverse2(d4, x));
            x = hn::Add(x, hn::Reverse(d4, x));
            return hn::GetLane(x);
        };
        for (int j = 0; j < 8; ++j) {
            u[j] = hn::Load(d, U8 + j * 8);
            vv[j] = hn::Load(d, V8 + j * 8);
            nrm[j] = hsum8(hn::Mul(u[j], u[j]));
        }
        static constexpr int kPair[7][4][2] = {
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, {{0, 2}, {1, 3}, {4, 6}, {5, 7}}, {{0, 3}, {1, 2}, {4, 7}, {5, 6}},
            {{0, 4}, {1, 5}, {2, 6}, {3, 7}}, {{0, 5}, {1, 4}, {2, 7}, {3, 6}}, {{0, 6}, {1, 7}, {2, 4}, {3, 5}},
            {{0, 7}, {1, 6}, {2, 5}, {3, 4}},
        };
        for (int sweep = 0; sweep < 8; ++sweep) {
            float max_off = 0.f;
            for (int rnd = 0; rnd < 7; ++rnd) {
                VW upv[4];
                VW uqv[4];
                float apq[4];
                float app[4];
                float aqq[4];
                int pidx[4];
                int qidx[4];
                for (int k = 0; k < 4; ++k) {
                    const int p = kPair[rnd][k][0];
                    const int q = kPair[rnd][k][1];
                    pidx[k] = p;
                    qidx[k] = q;
                    upv[k] = u[p];
                    uqv[k] = u[q];
                    app[k] = nrm[p];
                    aqq[k] = nrm[q];
                    apq[k] = hsum8(hn::Mul(upv[k], uqv[k]));
                    max_off = std::max(max_off, std::fabs(apq[k]));
                }
                for (int k = 0; k < 4; ++k) {
                    if (std::fabs(apq[k]) <= 1e-14f * (std::sqrt(app[k] * aqq[k]) + 1e-20f)) {
                        continue;
                    }
                    const int p = pidx[k];
                    const int q = qidx[k];
                    const float zeta = (aqq[k] - app[k]) / (2.f * apq[k]);
                    const float t = std::copysign(1.f, zeta) / (std::fabs(zeta) + std::sqrt(1.f + zeta * zeta));
                    const float cs = 1.f / std::sqrt(1.f + t * t);
                    const float sn = cs * t;
                    const auto vcs = hn::Set(d, cs);
                    const auto vsn = hn::Set(d, sn);
                    u[p] = hn::Sub(hn::Mul(vcs, upv[k]), hn::Mul(vsn, uqv[k]));
                    u[q] = hn::Add(hn::Mul(vsn, upv[k]), hn::Mul(vcs, uqv[k]));
                    const VW vp = vv[p];
                    const VW vq = vv[q];
                    vv[p] = hn::Sub(hn::Mul(vcs, vp), hn::Mul(vsn, vq));
                    vv[q] = hn::Add(hn::Mul(vsn, vp), hn::Mul(vcs, vq));
                    const float css = cs * cs;
                    const float sns = sn * sn;
                    const float two = 2.f * cs * sn * apq[k];
                    nrm[p] = css * app[k] + sns * aqq[k] - two;
                    nrm[q] = sns * app[k] + css * aqq[k] + two;
                }
                if (max_off < 1e-8f) {
                    break;
                }
            }
            if (max_off < 1e-8f) {
                break;
            }
        }
        for (int j = 0; j < 8; ++j) {
            const float nrmj = std::sqrt(std::max(nrm[j], 0.f));
            S[j] = nrmj;
            if (nrmj > 1e-20f) {
                u[j] = hn::Mul(u[j], hn::Set(d, 1.f / nrmj));
            }
            hn::Store(u[j], d, U8 + j * 8);
            hn::Store(vv[j], d, V8 + j * 8);
        }
    }
#else
    for (int sweep = 0; sweep < 12; ++sweep) {
        float max_off = 0.f;
        for (int p = 0; p < 7; ++p) {
            for (int q = p + 1; q < 8; ++q) {
                float app = 0.f, aqq = 0.f, apq = 0.f;
                for (int i = 0; i < 8; ++i) {
                    const float up = U8[i + p * 8];
                    const float uq = U8[i + q * 8];
                    app += up * up;
                    aqq += uq * uq;
                    apq += up * uq;
                }
                max_off = std::max(max_off, std::fabs(apq));
                if (std::fabs(apq) <= 1e-14f * (std::sqrt(app * aqq) + 1e-20f)) {
                    continue;
                }
                const float zeta = (aqq - app) / (2.f * apq);
                const float t = std::copysign(1.f, zeta) / (std::fabs(zeta) + std::sqrt(1.f + zeta * zeta));
                const float cs = 1.f / std::sqrt(1.f + t * t);
                const float sn = cs * t;
                for (int i = 0; i < 8; ++i) {
                    const float up = U8[i + p * 8];
                    const float uq = U8[i + q * 8];
                    U8[i + p * 8] = cs * up - sn * uq;
                    U8[i + q * 8] = sn * up + cs * uq;
                    const float vp = V8[i + p * 8];
                    const float vq = V8[i + q * 8];
                    V8[i + p * 8] = cs * vp - sn * vq;
                    V8[i + q * 8] = sn * vp + cs * vq;
                }
            }
        }
        if (max_off < 1e-8f) {
            break;
        }
    }
    for (int j = 0; j < 8; ++j) {
        float nrm = 0.f;
        for (int i = 0; i < 8; ++i) {
            nrm += U8[i + j * 8] * U8[i + j * 8];
        }
        nrm = std::sqrt(nrm);
        S[j] = nrm;
        if (nrm > 1e-20f) {
            const float inv = 1.f / nrm;
            for (int i = 0; i < 8; ++i) {
                U8[i + j * 8] *= inv;
            }
        }
    }
#endif

    for (int a = 0; a < 8; ++a) {
        int best = a;
        for (int b = a + 1; b < 8; ++b) {
            if (S[b] > S[best]) {
                best = b;
            }
        }
        if (best != a) {
            std::swap(S[a], S[best]);
            float tmp[8];
            std::memcpy(tmp, U8 + a * 8, 8 * sizeof(float));
            std::memcpy(U8 + a * 8, U8 + best * 8, 8 * sizeof(float));
            std::memcpy(U8 + best * 8, tmp, 8 * sizeof(float));
            std::memcpy(tmp, V8 + a * 8, 8 * sizeof(float));
            std::memcpy(V8 + a * 8, V8 + best * 8, 8 * sizeof(float));
            std::memcpy(V8 + best * 8, tmp, 8 * sizeof(float));
        }
    }

    if (ldu == 8) {
        std::memcpy(U, U8, 64 * sizeof(float));
    } else {
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
                U[i + j * ldu] = U8[i + j * 8];
            }
        }
    }
    std::memcpy(V, V8, 64 * sizeof(float));
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            Vt[i + j * ldvt] = V8[j + i * 8];
        }
    }
}

static float DotN(const float* a, const float* b, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc0 = hn::Zero(d);
    auto acc1 = hn::Zero(d);
    auto acc2 = hn::Zero(d);
    auto acc3 = hn::Zero(d);
    int i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        acc0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), acc0);
        acc1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), acc1);
        acc2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N), acc2);
        acc3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N), acc3);
    }
    auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
    for (; i + N <= n; i += N) {
        acc = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), acc);
    }
    float s = hn::ReduceSum(d, acc);
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

static void AxpyN(float* y, const float* x, float a, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto va = hn::Set(d, a);
    int i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d, y + i);
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i + N), hn::LoadU(d, y + i + N)), d, y + i + N);
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i + 2 * N), hn::LoadU(d, y + i + 2 * N)), d, y + i + 2 * N);
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i + 3 * N), hn::LoadU(d, y + i + 3 * N)), d, y + i + 3 * N);
    }
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::MulAdd(va, hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d, y + i);
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
}

static void ApplyHouseholderCols(float* col0, int ld, int ncols, const float* v, int len, float b) {
    if (ncols < 1 || len < 1) {
        return;
    }
    const hn::ScalableTag<float> dd;
    const int NN = static_cast<int>(hn::Lanes(dd));
    int j0 = 0;
    while (j0 < ncols) {
        const int nc = ncols - j0 < 8 ? ncols - j0 : 8;
        if (len >= 8) {
            hn::Vec<hn::ScalableTag<float>> acc[8];
            float dots[8];
            for (int j = 0; j < nc; ++j) {
                acc[j] = hn::Zero(dd);
            }
            int i = 0;
            for (; i + NN <= len; i += NN) {
                const auto vv = hn::LoadU(dd, v + i);
                for (int j = 0; j < nc; ++j) {
                    acc[j] = hn::MulAdd(vv, hn::LoadU(dd, col0 + (j0 + j) * ld + i), acc[j]);
                }
            }
            for (int j = 0; j < nc; ++j) {
                float s = hn::ReduceSum(dd, acc[j]);
                for (int t = i; t < len; ++t) {
                    s += v[t] * col0[(j0 + j) * ld + t];
                }
                dots[j] = b * s;
            }
            i = 0;
            for (; i + NN <= len; i += NN) {
                const auto vv = hn::LoadU(dd, v + i);
                for (int j = 0; j < nc; ++j) {
                    float* wj = col0 + (j0 + j) * ld + i;
                    hn::StoreU(hn::MulAdd(hn::Set(dd, -dots[j]), vv, hn::LoadU(dd, wj)), dd, wj);
                }
            }
            for (int t = i; t < len; ++t) {
                for (int j = 0; j < nc; ++j) {
                    col0[(j0 + j) * ld + t] -= dots[j] * v[t];
                }
            }
        } else {
            for (int j = 0; j < nc; ++j) {
                float* wj = col0 + (j0 + j) * ld;
                const float f = b * DotN(v, wj, len);
                AxpyN(wj, v, -f, len);
            }
        }
        j0 += nc;
    }
}

int HouseholderQR(int m, int n, const float* A, int lda, float* Q, int ldq, float* R, int ldr, float* W, float* Vh,
                  float* beta) {
    if (m < n || n <= 0) {
        return -1;
    }
    const int ldw = m;
    for (int j = 0; j < n; ++j) {
        if (lda == m) {
            std::memcpy(W + j * ldw, A + j * lda, static_cast<size_t>(m) * sizeof(float));
        } else {
            for (int i = 0; i < m; ++i) {
                W[i + j * ldw] = A[i + j * lda];
            }
        }
    }
    std::memset(Vh, 0, static_cast<size_t>(m * n) * sizeof(float));
    std::memset(beta, 0, static_cast<size_t>(n) * sizeof(float));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            R[i + j * ldr] = 0.f;
        }
    }

    for (int k = 0; k < n; ++k) {
        const int len = m - k;
        float* wcol = W + k + k * ldw;
        const float norm2 = DotN(wcol, wcol, len);
        const float norm = std::sqrt(norm2);
        if (norm < 1e-20f) {
            continue;
        }
        const float x0 = wcol[0];
        const float sign = (x0 >= 0.f) ? 1.f : -1.f;
        float* vcol = Vh + k + k * m;
        vcol[0] = x0 + sign * norm;
        if (len > 1) {
            std::memcpy(vcol + 1, wcol + 1, static_cast<size_t>(len - 1) * sizeof(float));
        }
        const float vtv = DotN(vcol, vcol, len);
        if (vtv < 1e-30f) {
            continue;
        }
        beta[k] = 2.f / vtv;
        const float b = beta[k];
        ApplyHouseholderCols(W + k + k * ldw, ldw, n - k, vcol, len, b);
        R[k + k * ldr] = -sign * norm;
        for (int j = k + 1; j < n; ++j) {
            R[k + j * ldr] = W[k + j * ldw];
        }
    }

    if (ldq == m) {
        std::memset(Q, 0, static_cast<size_t>(m * n) * sizeof(float));
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Q[i + j * ldq] = 0.f;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        Q[i + i * ldq] = 1.f;
    }
    for (int k = n - 1; k >= 0; --k) {
        const float b = beta[k];
        if (b == 0.f) {
            continue;
        }
        const int len = m - k;
        float* vcol = Vh + k + k * m;
        ApplyHouseholderCols(Q + k + k * ldq, ldq, n - k, vcol, len, b);
    }
    return 0;
}

void GemmNN(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    if (m < 1 || n < 1 || k < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const int mr = 4 * N;
    int j0 = 0;
    for (; j0 + 4 <= n && mr <= m; j0 += 4) {
        for (int i0 = 0; i0 + mr <= m; i0 += mr) {
            auto c00 = hn::Zero(d), c01 = hn::Zero(d), c02 = hn::Zero(d), c03 = hn::Zero(d);
            auto c10 = hn::Zero(d), c11 = hn::Zero(d), c12 = hn::Zero(d), c13 = hn::Zero(d);
            auto c20 = hn::Zero(d), c21 = hn::Zero(d), c22 = hn::Zero(d), c23 = hn::Zero(d);
            auto c30 = hn::Zero(d), c31 = hn::Zero(d), c32 = hn::Zero(d), c33 = hn::Zero(d);
            for (int t = 0; t < k; ++t) {
                const float* at = A + t * lda + i0;
                const auto a0 = hn::LoadU(d, at);
                const auto a1 = hn::LoadU(d, at + N);
                const auto a2 = hn::LoadU(d, at + 2 * N);
                const auto a3 = hn::LoadU(d, at + 3 * N);
                const auto b0 = hn::Set(d, B[t + j0 * ldb]);
                const auto b1 = hn::Set(d, B[t + (j0 + 1) * ldb]);
                const auto b2 = hn::Set(d, B[t + (j0 + 2) * ldb]);
                const auto b3 = hn::Set(d, B[t + (j0 + 3) * ldb]);
                c00 = hn::MulAdd(a0, b0, c00);
                c01 = hn::MulAdd(a1, b0, c01);
                c02 = hn::MulAdd(a2, b0, c02);
                c03 = hn::MulAdd(a3, b0, c03);
                c10 = hn::MulAdd(a0, b1, c10);
                c11 = hn::MulAdd(a1, b1, c11);
                c12 = hn::MulAdd(a2, b1, c12);
                c13 = hn::MulAdd(a3, b1, c13);
                c20 = hn::MulAdd(a0, b2, c20);
                c21 = hn::MulAdd(a1, b2, c21);
                c22 = hn::MulAdd(a2, b2, c22);
                c23 = hn::MulAdd(a3, b2, c23);
                c30 = hn::MulAdd(a0, b3, c30);
                c31 = hn::MulAdd(a1, b3, c31);
                c32 = hn::MulAdd(a2, b3, c32);
                c33 = hn::MulAdd(a3, b3, c33);
            }
            float* c0 = C + j0 * ldc + i0;
            float* c1 = C + (j0 + 1) * ldc + i0;
            float* c2 = C + (j0 + 2) * ldc + i0;
            float* c3 = C + (j0 + 3) * ldc + i0;
            hn::StoreU(c00, d, c0);
            hn::StoreU(c01, d, c0 + N);
            hn::StoreU(c02, d, c0 + 2 * N);
            hn::StoreU(c03, d, c0 + 3 * N);
            hn::StoreU(c10, d, c1);
            hn::StoreU(c11, d, c1 + N);
            hn::StoreU(c12, d, c1 + 2 * N);
            hn::StoreU(c13, d, c1 + 3 * N);
            hn::StoreU(c20, d, c2);
            hn::StoreU(c21, d, c2 + N);
            hn::StoreU(c22, d, c2 + 2 * N);
            hn::StoreU(c23, d, c2 + 3 * N);
            hn::StoreU(c30, d, c3);
            hn::StoreU(c31, d, c3 + N);
            hn::StoreU(c32, d, c3 + 2 * N);
            hn::StoreU(c33, d, c3 + 3 * N);
        }
    }
    for (int j = j0; j < n; ++j) {
        const float* bj = B + j * ldb;
        float* cj = C + j * ldc;
        int i0 = 0;
        for (; i0 + mr <= m; i0 += mr) {
            auto a0 = hn::Zero(d);
            auto a1 = hn::Zero(d);
            auto a2 = hn::Zero(d);
            auto a3 = hn::Zero(d);
            for (int t = 0; t < k; ++t) {
                const auto bt = hn::Set(d, bj[t]);
                const float* at = A + t * lda + i0;
                a0 = hn::MulAdd(bt, hn::LoadU(d, at), a0);
                a1 = hn::MulAdd(bt, hn::LoadU(d, at + N), a1);
                a2 = hn::MulAdd(bt, hn::LoadU(d, at + 2 * N), a2);
                a3 = hn::MulAdd(bt, hn::LoadU(d, at + 3 * N), a3);
            }
            hn::StoreU(a0, d, cj + i0);
            hn::StoreU(a1, d, cj + i0 + N);
            hn::StoreU(a2, d, cj + i0 + 2 * N);
            hn::StoreU(a3, d, cj + i0 + 3 * N);
        }
        for (; i0 + N <= m; i0 += N) {
            auto acc = hn::Zero(d);
            for (int t = 0; t < k; ++t) {
                acc = hn::MulAdd(hn::Set(d, bj[t]), hn::LoadU(d, A + t * lda + i0), acc);
            }
            hn::StoreU(acc, d, cj + i0);
        }
        for (; i0 < m; ++i0) {
            float s = 0.f;
            for (int t = 0; t < k; ++t) {
                s += A[i0 + t * lda] * bj[t];
            }
            cj[i0] = s;
        }
    }
}

void GemmTN(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    if (m < 1 || n < 1 || k < 1) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        const float* bj = B + j * ldb;
        float* cj = C + j * ldc;
        int t = 0;
        for (; t + 4 <= k; t += 4) {
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            const float* a0 = A + t * lda;
            const float* a1 = A + (t + 1) * lda;
            const float* a2 = A + (t + 2) * lda;
            const float* a3 = A + (t + 3) * lda;
            int i = 0;
            for (; i + N <= m; i += N) {
                const auto bv = hn::LoadU(d, bj + i);
                acc0 = hn::MulAdd(hn::LoadU(d, a0 + i), bv, acc0);
                acc1 = hn::MulAdd(hn::LoadU(d, a1 + i), bv, acc1);
                acc2 = hn::MulAdd(hn::LoadU(d, a2 + i), bv, acc2);
                acc3 = hn::MulAdd(hn::LoadU(d, a3 + i), bv, acc3);
            }
            float s0 = hn::ReduceSum(d, acc0);
            float s1 = hn::ReduceSum(d, acc1);
            float s2 = hn::ReduceSum(d, acc2);
            float s3 = hn::ReduceSum(d, acc3);
            for (; i < m; ++i) {
                const float bv = bj[i];
                s0 += a0[i] * bv;
                s1 += a1[i] * bv;
                s2 += a2[i] * bv;
                s3 += a3[i] * bv;
            }
            cj[t] = s0;
            cj[t + 1] = s1;
            cj[t + 2] = s2;
            cj[t + 3] = s3;
        }
        for (; t < k; ++t) {
            cj[t] = DotN(A + t * lda, bj, m);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(JacobiSvd8);
HWY_EXPORT(HouseholderQR);
HWY_EXPORT(GemmNN);
HWY_EXPORT(GemmTN);

void jacobi_svd_8(const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V) {
    HWY_DYNAMIC_DISPATCH(JacobiSvd8)(A, lda, U, ldu, S, Vt, ldvt, V);
}

int householder_qr_hwy(int m, int n, const float* A, int lda, float* Q, int ldq, float* R, int ldr, float* W, float* Vh,
                       float* beta) {
    return HWY_DYNAMIC_DISPATCH(HouseholderQR)(m, n, A, lda, Q, ldq, R, ldr, W, Vh, beta);
}

void gemm_nn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    HWY_DYNAMIC_DISPATCH(GemmNN)(m, n, k, A, lda, B, ldb, C, ldc);
}

void gemm_tn_hwy(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    HWY_DYNAMIC_DISPATCH(GemmTN)(m, n, k, A, lda, B, ldb, C, ldc);
}

}  // namespace nss
#endif
