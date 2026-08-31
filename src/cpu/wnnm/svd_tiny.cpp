#include "nss/cpu_api.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>

namespace nss {
namespace {

struct Bump {
    float* base = nullptr;
    int used = 0;
    int cap = 0;
    float* take(int k) {
        if (k <= 0) {
            return base + used;
        }
        if (!base || used + k > cap) {
            return nullptr;
        }
        float* p = base + used;
        used += k;
        return p;
    }
};

float* tls_pool(int* cap_out) {
    constexpr int kCap = kSvdMaxM * kSvdMaxN * 6 + kSvdMaxN * kSvdMaxN * 8 + 1024;
    static thread_local float buf[kCap];
    if (cap_out) {
        *cap_out = kCap;
    }
    return buf;
}

void copy_mat(const float* src, int lds, float* dst, int ldd, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            dst[i + j * ldd] = src[i + j * lds];
        }
    }
}

int householder_qr(int m, int n, const float* A, int lda, float* Q, int ldq, float* R, int ldr, Bump& bump) {
    float* W = bump.take(m * n);
    float* V = bump.take(m * n);
    float* beta = bump.take(n);
    if (!W || !V || !beta) {
        return -1;
    }
    return householder_qr_hwy(m, n, A, lda, Q, ldq, R, ldr, W, V, beta);
}

void jacobi_svd_n(int n, const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V) {
    copy_mat(A, lda, U, ldu, n, n);
    for (int i = 0; i < n * n; ++i) {
        V[i] = 0.f;
    }
    for (int i = 0; i < n; ++i) {
        V[i + i * n] = 1.f;
    }

    for (int sweep = 0; sweep < 32; ++sweep) {
        float max_off = 0.f;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                float app = 0.f, aqq = 0.f, apq = 0.f;
                for (int i = 0; i < n; ++i) {
                    const float up = U[i + p * ldu];
                    const float uq = U[i + q * ldu];
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
                for (int i = 0; i < n; ++i) {
                    const float up = U[i + p * ldu];
                    const float uq = U[i + q * ldu];
                    U[i + p * ldu] = cs * up - sn * uq;
                    U[i + q * ldu] = sn * up + cs * uq;
                }
                for (int i = 0; i < n; ++i) {
                    const float vp = V[i + p * n];
                    const float vq = V[i + q * n];
                    V[i + p * n] = cs * vp - sn * vq;
                    V[i + q * n] = sn * vp + cs * vq;
                }
            }
        }
        if (max_off < 1e-8f) {
            break;
        }
    }

    for (int j = 0; j < n; ++j) {
        float nrm = 0.f;
        for (int i = 0; i < n; ++i) {
            nrm += U[i + j * ldu] * U[i + j * ldu];
        }
        nrm = std::sqrt(nrm);
        S[j] = nrm;
        if (nrm > 1e-20f) {
            const float inv = 1.f / nrm;
            for (int i = 0; i < n; ++i) {
                U[i + j * ldu] *= inv;
            }
        }
    }

    for (int a = 0; a < n; ++a) {
        int best = a;
        for (int b = a + 1; b < n; ++b) {
            if (S[b] > S[best]) {
                best = b;
            }
        }
        if (best != a) {
            std::swap(S[a], S[best]);
            for (int i = 0; i < n; ++i) {
                std::swap(U[i + a * ldu], U[i + best * ldu]);
                std::swap(V[i + a * n], V[i + best * n]);
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Vt[i + j * ldvt] = V[j + i * n];
        }
    }
}

int svd_mn(int m, int n, const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, Bump& bump) {
    float* Q = bump.take(m * n);
    float* R = bump.take(n * n);
    float* Ur = bump.take(n * n);
    float* Vj = bump.take(n * n);
    if (!Q || !R || !Ur || !Vj) {
        return -1;
    }
    if (householder_qr(m, n, A, lda, Q, m, R, n, bump) != 0) {
        return -1;
    }
    if (n == 8) {
        jacobi_svd_8(R, n, Ur, n, S, Vt, ldvt, Vj);
    } else {
        jacobi_svd_n(n, R, n, Ur, n, S, Vt, ldvt, Vj);
    }
    gemm_nn_hwy(m, n, n, Q, m, Ur, n, U, ldu);
    return 0;
}

}  // namespace

int svd_economy(int m, int n, const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt,
                float* work, int work_floats) {
    if (m <= 0 || n <= 0 || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    int cap = work_floats;
    float* pool = work;
    if (!pool || cap < m * n * 4 + n * n * 4 + n) {
        pool = tls_pool(&cap);
    }
    Bump bump{pool, 0, cap};

    if (m < n) {
        float* AT = bump.take(n * m);
        float* U2 = bump.take(n * std::max(n, m));
        float* Vt2 = bump.take(m * n);
        if (!AT || !U2 || !Vt2) {
            return -1;
        }
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                AT[i + j * n] = A[j + i * lda];
            }
        }
        const int rc = svd_mn(n, m, AT, n, U2, n, S, Vt2, m, bump);
        if (rc != 0) {
            return rc;
        }
        const int r = m;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                U[i + j * ldu] = 0.f;
            }
        }
        for (int j = 0; j < r; ++j) {
            for (int i = 0; i < m; ++i) {
                U[i + j * ldu] = Vt2[j + i * m];
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Vt[i + j * ldvt] = 0.f;
            }
        }
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) {
                Vt[i + j * ldvt] = U2[j + i * n];
            }
        }
        return 0;
    }
    return svd_mn(m, n, A, lda, U, ldu, S, Vt, ldvt, bump);
}

}  // namespace nss
