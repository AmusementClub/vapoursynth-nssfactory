#include "nss/cpu_twsc_full.hpp"
#include "nss/cpu_api.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace nss {
namespace {

template<class T>
bool qr_svd(const float* a, int m, int n, int lda, TwscWorkspace& work,
            ResourceVector<T>& qr, ResourceVector<T>& aux) {
    const int rows = std::max(m, n), r = std::min(m, n);
    const bool transposed = m < n;
    qr.resize(std::size_t(rows) * r);
    aux.assign(std::size_t(2) * r * r + std::size_t(rows) * r + 2 * r, T(0));
    T* b = aux.data();
    T* v = b + r * r;
    T* u = v + r * r;
    T* tau = u + rows * r;
    T* singular = tau + r;
    double maximum = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        const double x = a[i + j * lda];
        if (!std::isfinite(x)) return false;
        maximum = std::max(maximum, std::abs(x));
    }
    int exponent = 0;
    if (maximum > 0) std::frexp(maximum, &exponent);
    const double scale = std::ldexp(1.0, -exponent);
    for (int j = 0; j < r; ++j) for (int i = 0; i < rows; ++i)
        qr[i + j * rows] = T(double(a[transposed ? j + i * lda : i + j * lda]) * scale);
    for (int k = 0; k < r; ++k) {
        double norm2 = 0;
        for (int i = k; i < rows; ++i) norm2 += double(qr[i + k * rows]) * qr[i + k * rows];
        if (norm2 == 0) continue;
        const double x = qr[k + k * rows];
        const double alpha = -std::copysign(std::sqrt(norm2), x);
        const double inverse = 1.0 / (x - alpha);
        tau[k] = T((alpha - x) / alpha);
        for (int i = k + 1; i < rows; ++i) qr[i + k * rows] = T(qr[i + k * rows] * inverse);
        qr[k + k * rows] = T(alpha);
        for (int j = k + 1; j < r; ++j) {
            double dot = qr[k + j * rows];
            for (int i = k + 1; i < rows; ++i) dot += double(qr[i + k * rows]) * qr[i + j * rows];
            dot *= tau[k];
            qr[k + j * rows] = T(qr[k + j * rows] - dot);
            for (int i = k + 1; i < rows; ++i)
                qr[i + j * rows] = T(qr[i + j * rows] - double(qr[i + k * rows]) * dot);
        }
    }
    double initial_norm = 0;
    for (int j = 0; j < r; ++j) {
        v[j + j * r] = T(1);
        double norm = 0;
        for (int i = 0; i <= j; ++i) { b[i + j * r] = qr[i + j * rows]; norm += double(b[i + j * r]) * b[i + j * r]; }
        initial_norm = std::max(initial_norm, norm);
    }
    const double rank_floor = initial_norm * 1e-14;
    // FP64 protects accumulated rotations while retaining the same relative
    // correlation and effective-rank policy as the ordinary decomposition.
    constexpr double tolerance = 1e-6;
    for (int sweep = 0; sweep < 128; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < r - 1; ++p) for (int q = p + 1; q < r; ++q) {
            double app = 0, aqq = 0, apq = 0;
            for (int i = 0; i < r; ++i) {
                const double x = b[i + p * r], y = b[i + q * r];
                app += x * x; aqq += y * y; apq += x * y;
            }
            if (app <= rank_floor || aqq <= rank_floor || std::abs(apq) <= tolerance * std::sqrt(app * aqq)) continue;
            rotated = true;
            const double z = (aqq - app) / (2 * apq);
            // Rescaled QR, the relative rank floor and the correlation test
            // bound |z| far below sqrt(DBL_MAX); a scaled hypot is unnecessary.
            const double t = std::copysign(1.0, z) / (std::abs(z) + std::sqrt(1.0 + z * z));
            const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
            for (int i = 0; i < r; ++i) {
                const double x = b[i + p * r], y = b[i + q * r];
                b[i + p * r] = T(c * x - s * y); b[i + q * r] = T(s * x + c * y);
                const double vx = v[i + p * r], vy = v[i + q * r];
                v[i + p * r] = T(c * vx - s * vy); v[i + q * r] = T(s * vx + c * vy);
            }
        }
        if (!rotated) break;
    }
    for (int j = 0; j < r; ++j) {
        double norm = 0;
        for (int i = 0; i < r; ++i) norm += double(b[i + j * r]) * b[i + j * r];
        singular[j] = norm > rank_floor ? T(std::sqrt(norm)) : T(0);
    }
    for (int j = 0; j < r; ++j) {
        int best = j;
        for (int k = j + 1; k < r; ++k) if (singular[k] > singular[best]) best = k;
        if (best == j) continue;
        std::swap(singular[j], singular[best]);
        for (int i = 0; i < r; ++i) { std::swap(b[i + j * r], b[i + best * r]); std::swap(v[i + j * r], v[i + best * r]); }
    }
    for (int j = 0; j < r; ++j) if (singular[j] > 0)
        for (int i = 0; i < r; ++i) u[i + j * rows] = b[i + j * r] / singular[j];
    for (int k = r - 1; k >= 0; --k) {
        if (tau[k] == 0) continue;
        for (int j = 0; j < r; ++j) {
            double dot = u[k + j * rows];
            for (int i = k + 1; i < rows; ++i) dot += double(qr[i + k * rows]) * u[i + j * rows];
            dot *= tau[k];
            u[k + j * rows] = T(u[k + j * rows] - dot);
            for (int i = k + 1; i < rows; ++i) u[i + j * rows] = T(u[i + j * rows] - double(qr[i + k * rows]) * dot);
        }
    }
    for (int k = 0; k < r; ++k) {
        work.singular[k] = double(singular[k]) / scale;
        for (int i = 0; i < m; ++i) work.dictionary[i + k * m] = singular[k] == 0 ? 0.0 : double(transposed ? v[i + k * r] : u[i + k * rows]);
        for (int j = 0; j < n; ++j) work.vt[k + j * r] = singular[k] == 0 ? 0.0 : double(transposed ? u[j + k * rows] : v[j + k * r]);
    }
    return true;
}

bool valid_svd(const float* a, int m, int n, int lda, const TwscWorkspace& w) {
#if !NSS_ALIGNMENT_GENERIC && !defined(NSS_TWSC_SCALAR_VALIDATION)
    if (n > 32 && std::min(m, n) >= 32 && twsc_svd_validation_lanes_available())
        return twsc_valid_svd_lanes(a, m, n, lda, w);
#endif
    const int r = std::min(m, n);
    double error = 0, norm = 0, orthogonal = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        double x = 0;
        for (int k = 0; k < r; ++k) x += w.dictionary[i + k * m] * w.singular[k] * w.vt[k + j * r];
        const double y = a[i + j * lda], d = x - y;
        if (!std::isfinite(x)) return false;
        error += d * d; norm += y * y;
    }
    for (int j = 0; j < r; ++j) if (w.singular[j] > 0) for (int k = 0; k <= j; ++k) if (w.singular[k] > 0) {
        double dot = 0;
        for (int i = 0; i < m; ++i) dot += w.dictionary[i + j * m] * w.dictionary[i + k * m];
        orthogonal = std::max(orthogonal, std::abs(dot - (j == k ? 1.0 : 0.0)));
        dot = 0;
        for (int i = 0; i < n; ++i) dot += w.vt[j + i * r] * w.vt[k + i * r];
        orthogonal = std::max(orthogonal, std::abs(dot - (j == k ? 1.0 : 0.0)));
    }
    return error <= norm * 2.5e-9 && orthogonal <= 5e-4;
}
} // namespace

bool twsc_valid_svd(const float* a, int m, int n, int lda, const TwscWorkspace& work) {
    return valid_svd(a, m, n, lda, work);
}
bool twsc_svd_clustered(const TwscWorkspace& work) {
    if (work.singular.empty()) return false;
    const double gap = work.singular.front() * 1e-4;
    for (std::size_t i = 1; i < work.singular.size(); ++i)
        if (work.singular[i] > 0 && work.singular[i - 1] - work.singular[i] <= gap) return true;
    return false;
}

bool twsc_svd(const float* a, int m, int n, int lda, TwscWorkspace& work, bool& double_fallback, bool orthogonal_double) {
    if (!a || m < 1 || m > kTwscMaxRows || n < 1 || n > kTwscMaxColumns || lda < m) return false;
    const int r = std::min(m, n);
    work.dictionary.resize(std::size_t(m) * r); work.singular.resize(r); work.vt.resize(std::size_t(r) * n);
    double_fallback = false;
    if (orthogonal_double) {
        double_fallback = true;
        return qr_svd(a, m, n, lda, work, work.qr64, work.aux64) && valid_svd(a, m, n, lda, work);
    }
    bool ok = false;
#if !NSS_ALIGNMENT_GENERIC
    if (m <= kSvdMaxM && n <= kSvdMaxN) {
        const int prefix = m * n + n + n * n;
        const int need = m * n * 6 + n * n * 8 + n + 256;
        work.small.resize(prefix + need);
        float* u = work.small.data(); float* s = u + m * n; float* vt = s + n;
        ok = svd_economy(m, n, a, lda, u, m, s, vt, n, work.small.data() + prefix, need) == 0;
        if (ok) for (int k = 0; k < r; ++k) {
            work.singular[k] = s[k];
            for (int i = 0; i < m; ++i) work.dictionary[i + k * m] = u[i + k * m];
            for (int j = 0; j < n; ++j) work.vt[k + j * r] = vt[k + j * n];
        }
    } else
#endif
        ok = qr_svd(a, m, n, lda, work, work.qr32, work.aux32);
    // Near-repeated spectra admit arbitrary bases. Route both scalar and lane
    // kernels through the same FP64 construction before an L1 coding model can
    // amplify their otherwise small, valid SVD differences.
    if (ok && valid_svd(a, m, n, lda, work) && !twsc_svd_clustered(work)) return true;
    double_fallback = true;
    return qr_svd(a, m, n, lda, work, work.qr64, work.aux64) && valid_svd(a, m, n, lda, work);
}

bool twsc_symmetric_eigen(const double* matrix, int n, double* values, double* vectors,
                          ResourceVector<double>& scratch) {
    if (!matrix || !values || !vectors || n < 1 || n > kTwscMaxColumns) return false;
    scratch.assign(matrix, matrix + std::size_t(n) * n);
    double norm = 0;
    for (double value : scratch) { if (!std::isfinite(value)) return false; norm = std::max(norm, std::abs(value)); }
    std::fill_n(vectors, std::size_t(n) * n, 0.0);
    for (int i = 0; i < n; ++i) vectors[i + i * n] = 1;
    if (norm == 0) { std::fill_n(values, n, 0.0); return true; }
    for (double& value : scratch) value /= norm;
    for (int sweep = 0; sweep < 128; ++sweep) {
        double off = 0;
        for (int p = 0; p < n - 1; ++p) for (int q = p + 1; q < n; ++q) {
            const double apq = scratch[p + q * n];
            off = std::max(off, std::abs(apq));
            if (std::abs(apq) < 1e-15) continue;
            const double app = scratch[p + p * n], aqq = scratch[q + q * n];
            const double z = (aqq - app) / (2 * apq);
            const double t = std::copysign(1.0, z) / (std::abs(z) + std::sqrt(1.0 + z * z));
            const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
            for (int i = 0; i < n; ++i) if (i != p && i != q) {
                const double x = scratch[i + p * n], y = scratch[i + q * n];
                scratch[i + p * n] = scratch[p + i * n] = c * x - s * y;
                scratch[i + q * n] = scratch[q + i * n] = s * x + c * y;
            }
            scratch[p + p * n] = app - t * apq;
            scratch[q + q * n] = aqq + t * apq;
            scratch[p + q * n] = scratch[q + p * n] = 0;
            for (int i = 0; i < n; ++i) {
                const double x = vectors[i + p * n], y = vectors[i + q * n];
                vectors[i + p * n] = c * x - s * y; vectors[i + q * n] = s * x + c * y;
            }
        }
        if (off <= 1e-14) break;
    }
    for (int i = 0; i < n; ++i) values[i] = scratch[i + i * n] * norm;
    double residual = 0, denominator = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
        double x = 0;
        for (int k = 0; k < n; ++k) x += matrix[i + k * n] * vectors[k + j * n];
        const double e = x - vectors[i + j * n] * values[j];
        residual += e * e; denominator += matrix[i + j * n] * matrix[i + j * n];
    }
    return std::isfinite(residual) && residual <= denominator * 1e-22;
}
} // namespace nss
