#include "nss/cpu_nlh_full.hpp"
#include "nss/cpu_nlh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nss {
void nlh_filter_full(const float* input, const float* reference, int m, int n, int lda,
                     const NlhGroupOptions& o, const int* row_indices, NlhWorkspace& w) {
    const int q = o.q;
    if (!input || m < 1 || m > 256 || n < 1 || n > 64 || (n & (n - 1)) || lda < m ||
        q < 1 || q > 16 || q > m || (q & (q - 1)) || !(o.sigma >= 0) || !std::isfinite(o.sigma) ||
        !(o.hard_strength >= 0) || !std::isfinite(o.hard_strength) || o.wiener_iterations < 1 || o.wiener_iterations > 64 ||
        !(o.wiener_sigma_scale >= 0) || !std::isfinite(o.wiener_sigma_scale) || (o.wiener && !reference))
        throw std::invalid_argument("nss.NLH: invalid full group parameters");
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i)
        if (!std::isfinite(input[i + j * lda]) || (o.wiener && !std::isfinite(reference[i + j * lda])))
            throw std::invalid_argument("nss.NLH: nonfinite group input");
    if (!row_indices) {
        w.indices.resize(std::size_t(m) * q);
        pixel_match(o.wiener ? reference : input, m, n, lda, q, w.indices.data());
        row_indices = w.indices.data();
    }
    w.numerator.assign(std::size_t(m) * n, 0);
    w.denominator.assign(std::size_t(m) * n, 0);
    w.matrix.resize(std::size_t(q) * n); w.reference.resize(std::size_t(q) * n);
    const double threshold = kNlhHardCoefficient * o.hard_strength * o.sigma;
    const double noise = (o.wiener_sigma_scale * o.sigma) * (o.wiener_sigma_scale * o.sigma);
    for (int row = 0; row < m; ++row) {
        const int* indices = row_indices + row * q;
        for (int k = 0; k < q; ++k) if (indices[k] < 0 || indices[k] >= m)
            throw std::invalid_argument("nss.NLH: invalid shared pixel index");
        if (o.sigma == 0) {
            for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
                const int destination = indices[k] + j * m;
                w.numerator[destination] += input[indices[k] + j * lda];
                w.denominator[destination] += 1;
            }
            continue;
        }
        for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
            w.matrix[k + j * q] = input[indices[k] + j * lda];
            if (o.wiener) w.reference[k + j * q] = reference[indices[k] + j * lda];
        }
        nlh_haar2d(w.matrix.data(), q, n, false);
        if (o.wiener) nlh_haar2d(w.reference.data(), q, n, false);
        for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
            const int index = k + j * q;
            if (o.wiener) {
                const double r = w.reference[index];
                const double r2 = r * r;
                // The zero-noise limit is identity, including 0/0 coefficients.
                const double gain = noise == 0 ? 1.0 : r2 / (r2 + noise);
                double value = w.matrix[index];
                for (int iteration = 0; iteration < o.wiener_iterations; ++iteration) value *= gain;
                w.matrix[index] = float(value);
            } else if (std::abs(double(w.matrix[index])) < threshold ||
                       (j > 0 && k >= std::max(0, q - 2))) {
                w.matrix[index] = 0;
            }
        }
        nlh_haar2d(w.matrix.data(), q, n, true);
        for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
            const int destination = indices[k] + j * m;
            w.numerator[destination] += w.matrix[k + j * q];
            w.denominator[destination] += 1;
        }
    }
}

void nlh_filter_full_batch(NlhFullBatchItem* items, int count) {
    if (count == 0) return;
    if (!items || count < 0 || count > 4096) throw std::invalid_argument("nss.NLH: invalid batch");
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i];
        if (!item.options || !item.work) throw std::invalid_argument("nss.NLH: incomplete batch item");
        nlh_filter_full(item.input, item.reference, item.m, item.n, item.lda, *item.options, item.row_indices, *item.work);
    }
}
} // namespace nss
