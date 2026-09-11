#include "nss/cpu_twsc_full.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace nss {
void twsc_filter_full_batch(TwscFullBatchItem* items, int count) {
    if (count == 0) return;
    if (!items || count < 0 || count > 4096) throw std::invalid_argument("nss.TWSC: invalid batch");
    ResourceVector<int> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (items[a].m != items[b].m) return items[a].m < items[b].m;
        if (items[a].n != items[b].n) return items[a].n < items[b].n;
        return a < b;
    });
    for (int i = 0; i < count; ++i) if (!items[i].work || !items[i].options || !items[i].row_sigma || !items[i].column_sigma)
        throw std::invalid_argument("nss.TWSC: incomplete batch item");
    for (int begin = 0; begin < count;) {
        const int m = items[order[begin]].m, n = items[order[begin]].n;
        int end = begin + 1;
        while (end < count && items[order[end]].m == m && items[order[end]].n == n) ++end;
#if !NSS_ALIGNMENT_GENERIC
        bool uniform = n > 32 && m >= 1 && m <= kTwscMaxRows && n <= kTwscMaxColumns;
        for (int pos = begin; uniform && pos < end; ++pos) {
            const auto& item = items[order[pos]];
            for (int k = 1; k < m; ++k) uniform = uniform && item.row_sigma[k] == item.row_sigma[0];
        }
        if (uniform) {
            for (int start = begin; start < end; start += 16) {
                const int size = std::min(16, end - start);
                std::array<TwscWorkspace*, 16> work{};
                for (int i = 0; i < size; ++i) {
                    auto& item = items[order[start + i]]; work[i] = item.work;
                    twsc_prepare_group(item.group, m, n, item.lda, *item.work);
                }
                twsc_svd64_batch(work.data(), m, n, size);
                for (int i = 0; i < size; ++i) {
                    auto& item = items[order[start + i]]; auto& w = *item.work;
                    bool fallback = true;
                    if (!twsc_valid_svd(w.input.data(), m, n, m, w) && !twsc_svd(w.input.data(), m, n, m, w, fallback, true))
                        throw std::runtime_error("nss.TWSC: full batch SVD failed residual gate");
                    const auto result = twsc_finish_full(item.group, m, n, item.lda, item.row_sigma, item.column_sigma,
                                                        item.column_weight, *item.options, w, true);
                    if (item.stats) *item.stats = result;
                }
            }
        } else if (n == 8 && m >= 8 && m <= kSvdMaxM) {
            for (int start = begin; start < end; start += 16) {
                const int real_count = std::min(16, end - start);
                constexpr int lanes = 16; // fill all possible FP32 Highway lanes, including tails
                const int prefix = m * n + n + n * n;
                ResourceVector<float> dummy(std::size_t(lanes - real_count) * prefix);
                std::array<const float*, lanes> a{};
                std::array<float*, lanes> u{}, s{}, vt{};
                std::array<int, lanes> lda{}, ldu{}, ldvt{};
                for (int i = 0; i < real_count; ++i) {
                    auto& item = items[order[start + i]];
                    auto& w = *item.work;
                    twsc_prepare_group(item.group, m, n, item.lda, w);
                    w.small.resize(prefix); w.dictionary.resize(std::size_t(m) * n); w.singular.resize(n); w.vt.resize(n * n);
                    a[i] = w.input.data(); u[i] = w.small.data(); s[i] = u[i] + m * n; vt[i] = s[i] + n;
                }
                for (int i = real_count; i < lanes; ++i) {
                    a[i] = a[real_count - 1]; u[i] = dummy.data() + std::size_t(i - real_count) * prefix;
                    s[i] = u[i] + m * n; vt[i] = s[i] + n;
                }
                lda.fill(m); ldu.fill(m); ldvt.fill(n);
                const bool batch_ok = svd_economy_8_batch_hwy(m, a.data(), lda.data(), u.data(), ldu.data(), s.data(), vt.data(), ldvt.data(), lanes) == 0;
                for (int i = 0; i < real_count; ++i) {
                    auto& item = items[order[start + i]]; auto& w = *item.work;
                    bool fallback = false;
                    if (batch_ok) {
                        for (int k = 0; k < n; ++k) {
                            w.singular[k] = s[i][k];
                            for (int j = 0; j < m; ++j) w.dictionary[j + k * m] = u[i][j + k * m];
                            for (int j = 0; j < n; ++j) w.vt[k + j * n] = vt[i][k + j * n];
                        }
                    }
                    const bool clustered = batch_ok && twsc_svd_clustered(w);
                    if ((!batch_ok || clustered || !twsc_valid_svd(a[i], m, n, m, w)) && !twsc_svd(a[i], m, n, m, w, fallback, clustered))
                        throw std::runtime_error("nss.TWSC: batch SVD failed residual gate");
                    const auto result = twsc_finish_full(item.group, m, n, item.lda, item.row_sigma, item.column_sigma, item.column_weight, *item.options, w, fallback);
                    if (item.stats) *item.stats = result;
                }
            }
        } else
#endif
        {
            for (int pos = begin; pos < end; ++pos) {
                auto& item = items[order[pos]];
                const auto result = twsc_filter_full(item.group, item.m, item.n, item.lda, item.row_sigma, item.column_sigma,
                                                      item.column_weight, *item.options, *item.work);
                if (item.stats) *item.stats = result;
            }
        }
        begin = end;
    }
}
} // namespace nss
