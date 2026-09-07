#include "nss/cpu_api.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "cpu/wnnm/numerics.hpp"

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

int householder_qr(int m, int n, const float* A, int lda, float* R, int ldr, float* Vh, float* beta, Bump& bump) {
    float* W = bump.take(m * n);
    if (!W) {
        return -1;
    }
    return householder_qr_hwy(m, n, A, lda, nullptr, 0, R, ldr, W, Vh, beta, false);
}

void jacobi_svd_n(int n, const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt, float* V) {
    copy_mat(A, lda, U, ldu, n, n);
    for (int i = 0; i < n * n; ++i) {
        V[i] = 0.f;
    }
    for (int i = 0; i < n; ++i) {
        V[i + i * n] = 1.f;
    }

    float max_norm = 0.f;
    for (int col = 0; col < n; ++col) {
        float norm = 0.f;
        for (int row = 0; row < n; ++row) norm += U[row + col * ldu] * U[row + col * ldu];
        max_norm = std::max(max_norm, norm);
    }
    const float rank_floor = max_norm * detail::kSvdRankFloorSquared;
    for (int sweep = 0; sweep < 32; ++sweep) {
        bool rotated = false;
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
                if (app <= rank_floor || aqq <= rank_floor || std::fabs(apq) <= detail::kJacobiCorrelationTolerance * std::sqrt(app * aqq)) {
                    continue;
                }
                rotated = true;
                const float t = detail::jacobi_tangent(app, aqq, apq);
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
        if (!rotated) {
            break;
        }
    }

    for (int j = 0; j < n; ++j) {
        float nrm = 0.f;
        for (int i = 0; i < n; ++i) {
            nrm += U[i + j * ldu] * U[i + j * ldu];
        }
        nrm = nrm > rank_floor ? std::sqrt(nrm) : 0.f;
        S[j] = nrm;
        if (nrm > 1e-20f) {
            const float inv = 1.f / nrm;
            for (int i = 0; i < n; ++i) {
                U[i + j * ldu] *= inv;
            }
        } else {
            for (int i = 0; i < n; ++i) U[i + j * ldu] = 0.f;
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
    float* R = bump.take(n * n);
    float* Ur = bump.take(n * n);
    float* Vj = bump.take(n * n);
    float* Vh = bump.take(m * n);
    float* beta = bump.take(n);
    if (!R || !Ur || !Vj || !Vh || !beta) {
        return -1;
    }
    if (householder_qr(m, n, A, lda, R, n, Vh, beta, bump) != 0) {
        return -1;
    }
    if (n == 8) {
        jacobi_svd_8(R, n, Ur, n, S, Vt, ldvt, Vj);
    } else {
        jacobi_svd_n(n, R, n, Ur, n, S, Vt, ldvt, Vj);
    }
    // U starts as [U_R; 0]. Applying the stored reflectors directly avoids
    // materializing Q and a second matrix multiplication.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            U[i + j * ldu] = (i < n) ? Ur[i + j * n] : 0.f;
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        if (beta[k] == 0.f) {
            continue;
        }
        apply_householder_hwy(U + k, ldu, n, Vh + k + k * m, m - k, beta[k]);
    }
    return 0;
}

}  // namespace

int svd_economy(int m, int n, const float* A, int lda, float* U, int ldu, float* S, float* Vt, int ldvt,
                float* work, int work_floats) {
    if (m <= 0 || n <= 0 || m > kSvdMaxM || n > kSvdMaxN || !A || !U || !S || !Vt ||
        lda < m || ldu < m || ldvt < n) {
        return -1;
    }
    int cap = work_floats;
    float* pool = work;
    if (!pool || cap < m * n * 4 + n * n * 4 + n) {
        pool = tls_pool(&cap);
    }
    Bump bump{pool, 0, cap};
    std::uint32_t maximum = 0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            maximum = std::max(maximum, detail::svd_magnitude_bits(A[row + col * lda]));
        }
    }
    if (maximum >= 0x7f800000u) return -1;
    const float scale = detail::svd_input_scale(maximum);
    if (scale != 1.f) {
        float* scaled = bump.take(m * n);
        if (!scaled) return -1;
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) scaled[row + col * m] = A[row + col * lda] * scale;
        }
        A = scaled;
        lda = m;
    }
    auto finish = [&](int rc) {
        if (rc != 0) return rc;
        for (int i = 0; i < std::min(m, n); ++i) {
            S[i] /= scale;
            if (detail::svd_magnitude_bits(S[i]) >= 0x7f800000u) return -1;
        }
        return 0;
    };

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
        return finish(0);
    }
    return finish(svd_mn(m, n, A, lda, U, ldu, S, Vt, ldvt, bump));
}

}  // namespace nss
