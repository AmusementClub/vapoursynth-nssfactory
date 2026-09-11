#include "nss/cpu_twsc_full.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nss {
namespace {
void check_options(const TwscSolverOptions& o) {
    if (o.iterations < 1 || o.iterations > 1000 || !(o.rho > 0) || !std::isfinite(o.rho) ||
        o.mu < 1 || !std::isfinite(o.mu) || !(o.tolerance > 0) || !std::isfinite(o.tolerance))
        throw std::invalid_argument("nss.TWSC: invalid ADMM iterations/rho/mu/tol");
}
double norm2(const ResourceVector<double>& x) {
    double result = 0;
    for (double value : x) result += value * value;
    return result;
}
// Direct SPD column solve is used only if the eigen-based solve misses the
// independent Sylvester residual gate. It changes no weights or objective.
bool cholesky_column(const double* a, double shift, int r, const double* b, double* x,
                      ResourceVector<double>& scratch) {
    scratch.assign(a, a + std::size_t(r) * r);
    for (int j = 0; j < r; ++j) {
        double diagonal = scratch[j + j * r] + shift;
        for (int k = 0; k < j; ++k) diagonal -= scratch[j + k * r] * scratch[j + k * r];
        if (!(diagonal > 0) || !std::isfinite(diagonal)) return false;
        scratch[j + j * r] = std::sqrt(diagonal);
        for (int i = j + 1; i < r; ++i) {
            double value = scratch[i + j * r];
            for (int k = 0; k < j; ++k) value -= scratch[i + k * r] * scratch[j + k * r];
            scratch[i + j * r] = value / scratch[j + j * r];
        }
    }
    for (int i = 0; i < r; ++i) {
        double value = b[i];
        for (int k = 0; k < i; ++k) value -= scratch[i + k * r] * x[k];
        x[i] = value / scratch[i + i * r];
    }
    for (int i = r - 1; i >= 0; --i) {
        double value = x[i];
        for (int k = i + 1; k < r; ++k) value -= scratch[k + i * r] * x[k];
        x[i] = value / scratch[i + i * r];
    }
    return true;
}
} // namespace

TwscSolverStats twsc_solve(const double* y, const double* dictionary, const double* spectrum,
                          int m, int n, int r, const double* row_precision,
                          const float* column_sigma, const TwscSolverOptions& options,
                          double* output, TwscWorkspace& w, ResourceVector<double>* trace) {
    check_options(options);
    if (!y || !dictionary || !spectrum || !row_precision || !column_sigma || !output ||
        m < 1 || m > kTwscMaxRows || n < 1 || n > kTwscMaxColumns || r < 1 || r > std::min(m, n))
        throw std::invalid_argument("nss.TWSC: unsupported full solver shape");
    const std::size_t mr = std::size_t(m) * r, rn = std::size_t(r) * n;
    w.atoms.resize(mr);
    w.data.resize(mr);
    for (int k = 0; k < r; ++k) {
        if (!(spectrum[k] >= 0) || !std::isfinite(spectrum[k])) throw std::runtime_error("nss.TWSC: invalid spectrum");
        for (int i = 0; i < m; ++i) {
            if (!(row_precision[i] > 0) || !std::isfinite(row_precision[i])) throw std::runtime_error("nss.TWSC: invalid row precision");
            const auto index = i + k * m;
            w.atoms[index] = dictionary[index] * spectrum[k];
            w.data[index] = w.atoms[index] * row_precision[i];
        }
    }
    w.gram.resize(std::size_t(r) * r);
    twsc_gemm_tn(m, r, r, w.atoms.data(), w.data.data(), w.gram.data());
    for (int j = 0; j < r; ++j) for (int i = 0; i < j; ++i) {
        const double average = 0.5 * (w.gram[i + j * r] + w.gram[j + i * r]);
        w.gram[i + j * r] = w.gram[j + i * r] = average;
    }
    w.eigenvectors.resize(std::size_t(r) * r); w.eigenvalues.resize(r);
    const double gram_norm = std::sqrt(norm2(w.gram));
    double off_norm2 = 0;
    for (int j = 0; j < r; ++j) for (int i = 0; i < r; ++i) if (i != j) off_norm2 += w.gram[i + j * r] * w.gram[i + j * r];
    const double off_norm = std::sqrt(off_norm2);
    const bool diagonal = !NSS_ALIGNMENT_GENERIC && off_norm <= 1e-12 * gram_norm;
    if (diagonal) {
        for (int i = 0; i < r; ++i) w.eigenvalues[i] = w.gram[i + i * r];
    } else if (!twsc_symmetric_eigen(w.gram.data(), r, w.eigenvalues.data(), w.eigenvectors.data(), w.scratch))
        throw std::runtime_error("nss.TWSC: symmetric eigendecomposition failed residual gate");
    for (double& value : w.eigenvalues) {
        if (value < -64 * std::numeric_limits<double>::epsilon() * r * gram_norm)
            throw std::runtime_error("nss.TWSC: indefinite weighted Gram matrix");
        value = std::max(value, 0.0);
    }
    w.projected.resize(std::size_t(m) * n);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) w.projected[i + j * m] = y[i + j * m] * row_precision[i];
    w.data.resize(rn);
    twsc_gemm_tn(m, n, r, w.atoms.data(), w.projected.data(), w.data.data());
    w.coefficients.assign(rn, 0); w.auxiliary.assign(rn, 0); w.dual.assign(rn, 0);
    w.previous.resize(rn); w.rhs.resize(rn); w.projected.resize(rn);
    ResourceVector<double> columns(n);
    for (int j = 0; j < n; ++j) {
        if (!(column_sigma[j] >= 0) || !std::isfinite(column_sigma[j])) throw std::runtime_error("nss.TWSC: invalid column noise");
        columns[j] = std::max(double(column_sigma[j]), kTwscNoiseFloor);
    }
    if (trace) trace->clear();
    double rho = options.rho;
    TwscSolverStats stats;
    auto residual = [&]() {
        if (diagonal) {
            // Proven upper bound for the original (not diagonalized) equation:
            // ||offdiag(A) C|| <= ||offdiag(A)||_F ||C||_F, plus division error.
            const double cnorm = std::sqrt(norm2(w.coefficients)), enorm = std::sqrt(norm2(w.rhs));
            double bnorm2 = 0;
            for (double column : columns) { const double b = 0.5 * rho * column; bnorm2 += b * b; }
            const double scale = (gram_norm + std::sqrt(bnorm2)) * cnorm + enorm;
            const double bound = off_norm * cnorm + 16 * std::numeric_limits<double>::epsilon() * enorm;
            return scale == 0 ? bound : bound / scale;
        }
        twsc_gemm_nn(r, n, r, w.gram.data(), w.coefficients.data(), w.projected.data());
        double error = 0, bnorm = 0;
        for (int j = 0; j < n; ++j) {
            const double shift = 0.5 * rho * columns[j];
            bnorm += shift * shift;
            for (int i = 0; i < r; ++i) {
                const int index = i + j * r;
                const double e = w.projected[index] + shift * w.coefficients[index] - w.rhs[index];
                error += e * e;
            }
        }
        const double denominator = (gram_norm + std::sqrt(bnorm)) * std::sqrt(norm2(w.coefficients)) + std::sqrt(norm2(w.rhs));
        return denominator == 0 ? std::sqrt(error) : std::sqrt(error) / denominator;
    };
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        w.previous = w.coefficients;
        for (int j = 0; j < n; ++j) for (int i = 0; i < r; ++i) {
            const int index = i + j * r;
            w.rhs[index] = w.data[index] + 0.5 * (rho * w.auxiliary[index] - w.dual[index]) * columns[j];
        }
        if (diagonal) {
            for (int j = 0; j < n; ++j) for (int i = 0; i < r; ++i)
                w.coefficients[i + j * r] = w.rhs[i + j * r] / (w.eigenvalues[i] + 0.5 * rho * columns[j]);
        } else {
            twsc_gemm_tn(r, n, r, w.eigenvectors.data(), w.rhs.data(), w.projected.data());
            for (int j = 0; j < n; ++j) for (int i = 0; i < r; ++i)
                w.projected[i + j * r] /= w.eigenvalues[i] + 0.5 * rho * columns[j];
            twsc_gemm_nn(r, n, r, w.eigenvectors.data(), w.projected.data(), w.coefficients.data());
        }
        double error = residual();
        if (!std::isfinite(error) || error > 1e-10) {
            for (int j = 0; j < n; ++j)
                if (!cholesky_column(w.gram.data(), 0.5 * rho * columns[j], r, w.rhs.data() + j * r,
                                      w.coefficients.data() + j * r, w.scratch))
                    throw std::runtime_error("nss.TWSC: Sylvester fallback failed");
            error = residual();
            if (!std::isfinite(error) || error > 1e-10) throw std::runtime_error("nss.TWSC: Sylvester residual exceeds 1e-10");
        }
        stats.sylvester_residual = std::max(stats.sylvester_residual, error);
        double primal = 0, dc = 0, dz = 0;
        for (std::size_t i = 0; i < rn; ++i) {
            const double temp = w.coefficients[i] + w.dual[i] / rho;
            const double z = std::copysign(std::max(std::abs(temp) - 1.0 / rho, 0.0), temp);
            const double p = w.coefficients[i] - z;
            const double cchange = w.coefficients[i] - w.previous[i];
            const double zchange = z - w.auxiliary[i];
            primal += p * p; dc += cchange * cchange; dz += zchange * zchange;
            w.auxiliary[i] = z;
        }
        stats.iterations = iteration + 1; stats.primal = std::sqrt(primal);
        stats.coefficient_change = std::sqrt(dc); stats.auxiliary_change = std::sqrt(dz);
        if (!std::isfinite(primal + dc + dz)) throw std::runtime_error("nss.TWSC: nonfinite ADMM state");
        stats.converged = stats.primal <= options.tolerance && stats.coefficient_change <= options.tolerance &&
                          stats.auxiliary_change <= options.tolerance;
        if (!stats.converged) for (std::size_t i = 0; i < rn; ++i) w.dual[i] += rho * (w.coefficients[i] - w.auxiliary[i]);
        if (trace) for (const auto* matrix : {&w.coefficients, &w.auxiliary, &w.dual}) trace->insert(trace->end(), matrix->begin(), matrix->end());
        if (stats.converged) break;
        if (iteration + 1 < options.iterations) {
            rho *= options.mu;
            if (!(rho > 0) || !std::isfinite(rho)) throw std::runtime_error("nss.TWSC: ADMM rho overflow");
        }
    }
    twsc_gemm_nn(m, n, r, w.atoms.data(), w.coefficients.data(), output);
    return stats;
}

void twsc_prepare_group(const float* group, int m, int n, int lda, TwscWorkspace& w) {
    if (!group || m < 1 || m > kTwscMaxRows || n < 1 || n > kTwscMaxColumns || lda < m)
        throw std::invalid_argument("nss.TWSC: unsupported group shape");
    w.mean.assign(m, 0); w.centered.resize(std::size_t(m) * n); w.input.resize(std::size_t(m) * n);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        const float x = group[i + j * lda];
        if (!std::isfinite(x)) throw std::invalid_argument("nss.TWSC: nonfinite group input");
        w.mean[i] += double(x) / n;
    }
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        // Store the FP32-centered matrix consistently for SVD and coding.
        w.input[i + j * m] = float(double(group[i + j * lda]) - w.mean[i]);
        w.centered[i + j * m] = w.input[i + j * m];
    }
}

TwscSolverStats twsc_filter_full(float* group, int m, int n, int lda, const float* row_sigma,
                                const float* column_sigma, float* column_weight,
                                const TwscSolverOptions& options, TwscWorkspace& w) {
    check_options(options);
    if (!row_sigma || !column_sigma) throw std::invalid_argument("nss.TWSC: missing noise weights");
    twsc_prepare_group(group, m, n, lda, w);
    bool double_fallback = false;
    bool uniform = true;
    for (int i = 1; i < m; ++i) uniform = uniform && row_sigma[i] == row_sigma[0];
    // A more orthogonal FP64 dictionary makes W1-uniform Sylvester systems
    // diagonal within a rigorous residual bound. No ADMM iteration is skipped.
    if (!twsc_svd(w.input.data(), m, n, m, w, double_fallback, uniform && n > 32)) throw std::runtime_error("nss.TWSC: SVD failed residual gate");
    return twsc_finish_full(group, m, n, lda, row_sigma, column_sigma, column_weight, options, w, double_fallback);
}

TwscSolverStats twsc_finish_full(float* group, int m, int n, int lda, const float* row_sigma,
                                const float* column_sigma, float* column_weight,
                                const TwscSolverOptions& options, TwscWorkspace& w, bool double_fallback) {
    check_options(options);
    const int r = std::min(m, n);
    if (!row_sigma || !column_sigma) throw std::invalid_argument("nss.TWSC: missing noise weights");
    for (int j = 0; j < n; ++j) if (!(column_sigma[j] >= 0) || !std::isfinite(column_sigma[j]))
        throw std::invalid_argument("nss.TWSC: invalid column noise");
    const double noise = n * double(column_sigma[0]) * column_sigma[0];
    for (int k = 0; k < r; ++k) w.singular[k] = std::sqrt(std::max(w.singular[k] * w.singular[k] - noise, 0.0));
    int active = r;
    while (active > 0 && w.singular[active - 1] == 0) --active;
    ResourceVector<double> precision(m);
    for (int i = 0; i < m; ++i) {
        if (!(row_sigma[i] >= 0) || !std::isfinite(row_sigma[i])) throw std::invalid_argument("nss.TWSC: invalid row noise");
        precision[i] = 1.0 / std::max(double(row_sigma[i]), kTwscNoiseFloor);
    }
    w.solution.resize(std::size_t(m) * n);
    TwscSolverStats stats;
    if (active) stats = twsc_solve(w.centered.data(), w.dictionary.data(), w.singular.data(), m, n, active, precision.data(),
                                  column_sigma, options, w.solution.data(), w);
    else { std::fill(w.solution.begin(), w.solution.end(), 0.0); stats.converged = true; stats.iterations = 1; }
    stats.double_svd = double_fallback;
    for (int j = 0; j < n; ++j) {
        if (column_weight) column_weight[j] = float(1.0 / std::max(double(column_sigma[j]), kTwscNoiseFloor));
        for (int i = 0; i < m; ++i) {
            const double value = w.solution[i + j * m] + w.mean[i];
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                throw std::runtime_error("nss.TWSC: unrepresentable reconstruction");
            group[i + j * lda] = float(value);
        }
    }
    return stats;
}
} // namespace nss
