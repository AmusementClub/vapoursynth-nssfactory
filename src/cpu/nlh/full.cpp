// SPDX-License-Identifier: GPL-2.0-only
#include "nss/cpu_nlh_full.hpp"
#include "nss/cpu_nlh.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
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
    // Branch-free exponent test (vectorizable); equivalent to std::isfinite.
    std::uint32_t nonfinite = 0;
    for (int j = 0; j < n; ++j) {
        const float* in = input + j * lda;
        for (int i = 0; i < m; ++i)
            nonfinite |= static_cast<std::uint32_t>((std::bit_cast<std::uint32_t>(in[i]) & 0x7f800000u) == 0x7f800000u);
        if (o.wiener) {
            const float* rf = reference + j * lda;
            for (int i = 0; i < m; ++i)
                nonfinite |= static_cast<std::uint32_t>((std::bit_cast<std::uint32_t>(rf[i]) & 0x7f800000u) == 0x7f800000u);
        }
    }
    if (nonfinite) throw std::invalid_argument("nss.NLH: nonfinite group input");
    if (!row_indices) {
        w.indices.resize(std::size_t(m) * q);
        pixel_match(o.wiener ? reference : input, m, n, lda, q, w.indices.data());
        row_indices = w.indices.data();
    }
    w.numerator.assign(std::size_t(m) * n, 0);
    w.denominator.assign(std::size_t(m) * n, 0);
    const double threshold = kNlhHardCoefficient * o.hard_strength * o.sigma;
    const double noise = (o.wiener_sigma_scale * o.sigma) * (o.wiener_sigma_scale * o.sigma);
    for (int i = 0; i < m * q; ++i)
        if (row_indices[i] < 0 || row_indices[i] >= m)
            throw std::invalid_argument("nss.NLH: invalid shared pixel index");
    if (o.sigma == 0) {
        for (int row = 0; row < m; ++row) {
            const int* indices = row_indices + row * q;
            for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
                const int destination = indices[k] + j * m;
                w.numerator[destination] += input[indices[k] + j * lda];
                w.denominator[destination] += 1;
            }
        }
        return;
    }
    // Every source row owns an independent q*n matrix. Transform all rows
    // with one dispatch per stage (each matrix keeps the single-matrix
    // operation sequence), then aggregate in the original row/j/k order so
    // every numerator sum is formed in the same order as before.
    const std::size_t size = std::size_t(q) * n;
    w.matrix.resize(size * m);
    if (o.wiener) w.reference.resize(size * m);
    for (int row = 0; row < m; ++row) {
        const int* indices = row_indices + row * q;
        float* matrix = w.matrix.data() + row * size;
        float* ref = o.wiener ? w.reference.data() + row * size : nullptr;
        for (int j = 0; j < n; ++j) for (int k = 0; k < q; ++k) {
            matrix[k + j * q] = input[indices[k] + j * lda];
            if (ref) ref[k + j * q] = reference[indices[k] + j * lda];
        }
    }
    nlh_haar2d_batch(w.matrix.data(), m, q, n, false);
    if (o.wiener) nlh_haar2d_batch(w.reference.data(), m, q, n, false);
    nlh_shrink_full_batch(w.matrix.data(), o.wiener ? w.reference.data() : nullptr, m, q, n, threshold, noise,
                          o.wiener_iterations, o.wiener);
    nlh_haar2d_batch(w.matrix.data(), m, q, n, true);
    // Each (row, k) adds 1 to every column j of its destination pixel, so the
    // denominator is a per-pixel count shared by all j (exact small integers).
    int counts[256] = {};
    for (int i = 0; i < m * q; ++i) ++counts[row_indices[i]];
    for (int j = 0; j < n; ++j) {
        double* den = w.denominator.data() + std::size_t(j) * m;
        for (int i = 0; i < m; ++i) den[i] = counts[i];
    }
    // Numerators keep the original per-element accumulation order (rows in
    // ascending order for every destination element).
    for (int row = 0; row < m; ++row) {
        const int* indices = row_indices + row * q;
        const float* matrix = w.matrix.data() + row * size;
        for (int j = 0; j < n; ++j) {
            double* num = w.numerator.data() + std::size_t(j) * m;
            const float* src = matrix + j * q;
            for (int k = 0; k < q; ++k) num[indices[k]] += src[k];
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
