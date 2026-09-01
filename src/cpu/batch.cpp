#include "nss/cpu_batch.hpp"

#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/cpu_twsc.hpp"
#include "cpu/bm/matcher.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace nss {
namespace {

template <typename Item>
inline void set_status(Item& item, int value) noexcept {
    if (item.status) {
        *item.status = value;
    }
}

template <typename Item, typename Less>
std::vector<int> bucket_order(const Item* items, int count, Less less) {
    std::vector<int> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (less(items[a], items[b])) {
            return true;
        }
        if (less(items[b], items[a])) {
            return false;
        }
        return a < b;
    });
    return order;
}

bool finite_singular_values(const float* values, int count) {
    if (!values || count < 1) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.f) {
            return false;
        }
    }
    return true;
}

void scale_twsc_rows(float* group, int m, int n, int lda, const float* row_weight, bool invert) {
    if (!row_weight) {
        return;
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float weight = row_weight[row];
            if (invert) {
                const float magnitude = std::max(std::fabs(weight), 1e-12f);
                weight = std::copysign(1.f / magnitude, weight);
            }
            group[row + col * lda] *= weight;
        }
    }
}

void finish_twsc_batch_item(TwscPcaBatchItem& item) {
    const int m = item.m;
    const int n = item.n;
    const int r = std::min(m, n);
    float* U = item.work;
    float* S = U + m * n;
    float* mean = S + n;
    float* B = mean + m;
    gemm_tn_hwy(m, n, r, U, m, item.group, item.lda, B, r);

    float sigmas[kSvdMaxN];
    float thresholds[kSvdMaxN];
    float row_values[kSvdMaxN];
    constexpr float kEps = 1e-6f;
    bool same = true;
    for (int col = 0; col < n; ++col) {
        float sigma = item.col_sigma ? item.col_sigma[col] : item.sigma;
        if (!std::isfinite(sigma) || sigma < 0.f) {
            sigma = 0.f;
        }
        sigmas[col] = sigma;
        if (col > 0 && sigma != sigmas[0]) {
            same = false;
        }
        if (item.col_weight) {
            item.col_weight[col] = 1.f / (sigma + kEps);
        }
    }
    const float sigma0 = sigmas[0];
    const float noise = static_cast<float>(n) * sigma0 * sigma0;
    for (int row = 0; row < r; ++row) {
        const float singular = S[row];
        S[row] = std::sqrt(std::max(singular * singular - noise, 0.f));
        const float denominator = S[row] + kEps;
        for (int col = 0; col < n; ++col) {
            row_values[col] = B[row + col * r];
        }
        if (same) {
            soft_threshold(row_values, n, sigma0 * sigma0 / denominator);
        } else {
            for (int col = 0; col < n; ++col) {
                thresholds[col] = sigmas[col] * sigmas[col] / denominator;
            }
            soft_threshold_var(row_values, thresholds, n);
        }
        for (int col = 0; col < n; ++col) {
            B[row + col * r] = row_values[col];
        }
    }
    gemm_nn_hwy(m, n, r, U, m, B, r, item.group, item.lda);
    group_center_add(item.group, m, n, item.lda, mean);
    scale_twsc_rows(item.group, m, n, item.lda, item.row_weight, true);
}

void finish_ncsr_batch_item(NcsrFilterBatchItem& item) {
    const int m = item.m;
    const int n = item.n;
    const int r = std::min(m, n);
    float* U = item.work;
    float* S = U + m * n;
    float* mean = S + n;
    float* B = mean + m;
    gemm_tn_hwy(m, n, r, U, m, item.group, item.lda, B, r);
    if (!(item.sigma > 0.f) || !std::isfinite(item.sigma)) {
        gemm_nn_hwy(m, n, r, U, m, B, r, item.group, item.lda);
        group_center_add(item.group, m, n, item.lda, mean);
        return;
    }

    float weights[kSvdMaxN];
    constexpr float kEps = 1e-12f;
    const float h = std::max(2.f * static_cast<float>(m) * item.sigma * item.sigma, kEps);
    float weight_sum = 0.f;
    for (int col = 0; col < n; ++col) {
        float distance = item.col_dist ? item.col_dist[col]
                                       : (col > 0 ? ssd_vec(item.group + col * item.lda, item.group, m) : 0.f);
        weights[col] = std::exp(-distance / h);
        weight_sum += weights[col];
    }
    if (!(weight_sum > 0.f)) {
        std::fill(weights, weights + n, 1.f);
        weight_sum = static_cast<float>(n);
    }
    const float inverse_weight = 1.f / weight_sum;
    constexpr float kMap = 2.8284271247461903f;
    const float sigma2 = item.sigma * item.sigma;
    float row_tau[kSvdMaxN];
    for (int row = 0; row < r; ++row) {
        float beta = 0.f;
        for (int col = 0; col < n; ++col) {
            beta += weights[col] * B[row + col * r];
        }
        beta *= inverse_weight;
        float variance = 0.f;
        for (int col = 0; col < n; ++col) {
            const float error = B[row + col * r] - beta;
            variance += weights[col] * error * error;
        }
        const float sigma_theta = std::sqrt(variance * inverse_weight);
        row_tau[row] = kMap * sigma2 / (sigma_theta + kEps);
    }
    ncsr_centralize_codes(B, r, n, r, std::fabs(item.sigma), weights, row_tau);
    gemm_nn_hwy(m, n, r, U, m, B, r, item.group, item.lda);
    group_center_add(item.group, m, n, item.lda, mean);
}

}  // namespace

int spatial_match_batch(const float* ref, int stride, int width, int height, const MatchBatchItem* items, int count,
                        Match* matches, int match_stride, int* counts) {
    if (count == 0) {
        return 0;
    }
    if (!ref || !items || !matches || !counts || count < 0 || match_stride < 1) {
        return -1;
    }
    const auto order = bucket_order(items, count, [](const MatchBatchItem& a, const MatchBatchItem& b) {
        if (a.block != b.block) {
            return a.block < b.block;
        }
        if (a.group != b.group) {
            return a.group < b.group;
        }
        if (a.bm_range != b.bm_range) {
            return a.bm_range < b.bm_range;
        }
        return false;
    });
    int first_error = 0;
    for (int index : order) {
        const auto& item = items[index];
        counts[index] = spatial_match(ref, stride, width, height, item.bx, item.by, item.block, item.bm_range,
                                      item.group, matches + static_cast<std::size_t>(index) * match_stride);
        if (counts[index] <= 0) {
            if (first_error == 0) {
                first_error = index + 1;
            }
        }
    }
    return first_error;
}

int spatial_match_nch_batch(const float* const* refs, const int* strides, int nch, int width, int height,
                            const MatchBatchItem* items, int count, Match* matches, int match_stride, int* counts) {
    if (count == 0) {
        return 0;
    }
    if (!refs || !strides || !items || !matches || !counts || count < 0 || match_stride < 1) {
        return -1;
    }
    const auto order = bucket_order(items, count, [](const MatchBatchItem& a, const MatchBatchItem& b) {
        if (a.block != b.block) {
            return a.block < b.block;
        }
        if (a.group != b.group) {
            return a.group < b.group;
        }
        return a.bm_range < b.bm_range;
    });
    int first_error = 0;
    for (int index : order) {
        const auto& item = items[index];
        counts[index] = spatial_match_nch(refs, strides, nch, width, height, item.bx, item.by, item.block,
                                          item.bm_range, item.group, matches + static_cast<std::size_t>(index) * match_stride);
        if (counts[index] <= 0) {
            if (first_error == 0) {
                first_error = index + 1;
            }
        }
    }
    return first_error;
}

int predictive_match_batch(const float* const* refs, const int* strides, int ntemp, int width, int height, int t0,
                           const SearchConfig& cfg, const MatchBatchItem* items, int count, Match* matches,
                           int match_stride, int* counts) {
    if (count == 0) {
        return 0;
    }
    if (!refs || !strides || !items || !matches || !counts || count < 0 || match_stride < 1) {
        return -1;
    }
    const auto order = bucket_order(items, count, [](const MatchBatchItem& a, const MatchBatchItem& b) {
        if (a.block != b.block) {
            return a.block < b.block;
        }
        if (a.group != b.group) {
            return a.group < b.group;
        }
        return a.bm_range < b.bm_range;
    });
    int first_error = 0;
    for (int index : order) {
        const auto& item = items[index];
        SearchConfig local = cfg;
        local.block = item.block;
        local.group = item.group;
        local.bm_range = item.bm_range;
        counts[index] = predictive_match(refs, strides, ntemp, width, height, item.bx, item.by, t0, local,
                                         matches + static_cast<std::size_t>(index) * match_stride);
        if (counts[index] <= 0) {
            if (first_error == 0) {
                first_error = index + 1;
            }
        }
    }
    return first_error;
}

int predictive_match_nch_batch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                               int t0, const SearchConfig& cfg, const MatchBatchItem* items, int count, Match* matches,
                               int match_stride, int* counts) {
    if (count == 0) {
        return 0;
    }
    if (!refs || !strides || !items || !matches || !counts || count < 0 || match_stride < 1) {
        return -1;
    }
    const auto order = bucket_order(items, count, [](const MatchBatchItem& a, const MatchBatchItem& b) {
        if (a.block != b.block) {
            return a.block < b.block;
        }
        if (a.group != b.group) {
            return a.group < b.group;
        }
        return a.bm_range < b.bm_range;
    });
    int first_error = 0;
    for (int index : order) {
        const auto& item = items[index];
        SearchConfig local = cfg;
        local.block = item.block;
        local.group = item.group;
        local.bm_range = item.bm_range;
        counts[index] = predictive_match_nch(refs, strides, nch, ntemp, width, height, item.bx, item.by, t0, local,
                                             matches + static_cast<std::size_t>(index) * match_stride);
        if (counts[index] <= 0) {
            if (first_error == 0) {
                first_error = index + 1;
            }
        }
    }
    return first_error;
}

int svd_economy_batch(SvdBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const SvdBatchItem& a, const SvdBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        return a.n < b.n;
    });
    int first_error = 0;
    for (int begin = 0; begin < count;) {
        int end = begin + 1;
        while (end < count && items[order[end]].m == items[order[begin]].m &&
               items[order[end]].n == items[order[begin]].n) {
            ++end;
        }

        const int m = items[order[begin]].m;
        const int n = items[order[begin]].n;
        std::vector<int> batch_indices;
        batch_indices.reserve(static_cast<std::size_t>(end - begin));
        for (int pos = begin; pos < end; ++pos) {
            const int index = order[pos];
            const auto& item = items[index];
            if (n == 8 && m >= 8 && m <= kSvdBatch8MaxM && item.A && item.U && item.S && item.Vt && item.lda >= m &&
                item.ldu >= m && item.ldvt >= 8) {
                batch_indices.push_back(index);
                continue;
            }
            if (svd_economy(item.m, item.n, item.A, item.lda, item.U, item.ldu, item.S, item.Vt, item.ldvt,
                            item.work, item.work_floats) != 0) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
            } else {
                set_status(items[index], 1);
            }
        }

        if (batch_indices.size() == 1) {
            const int index = batch_indices.front();
            const auto& item = items[index];
            if (svd_economy(item.m, item.n, item.A, item.lda, item.U, item.ldu, item.S, item.Vt, item.ldvt,
                            item.work, item.work_floats) != 0) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
            } else {
                set_status(items[index], 1);
            }
        } else if (!batch_indices.empty()) {
            const std::size_t size = batch_indices.size();
            std::vector<const float*> a(size);
            std::vector<int> lda(size);
            std::vector<float*> u(size);
            std::vector<int> ldu(size);
            std::vector<float*> s(size);
            std::vector<float*> vt(size);
            std::vector<int> ldvt(size);
            for (std::size_t i = 0; i < size; ++i) {
                const auto& item = items[batch_indices[i]];
                a[i] = item.A;
                lda[i] = item.lda;
                u[i] = item.U;
                ldu[i] = item.ldu;
                s[i] = item.S;
                vt[i] = item.Vt;
                ldvt[i] = item.ldvt;
            }
            if (svd_economy_8_batch_hwy(m, a.data(), lda.data(), u.data(), ldu.data(), s.data(), vt.data(),
                                        ldvt.data(), static_cast<int>(size)) != 0) {
                for (int index : batch_indices) {
                    if (first_error == 0) {
                        first_error = index + 1;
                    }
                }
            } else {
                for (int index : batch_indices) {
                    auto& item = items[index];
                    // AVX-512 -ffast-math can amplify a near-zero QR
                    // remainder for an exactly rank-deficient matrix. Keep
                    // the common 16-lane path, but fail closed per matrix.
                    if (!finite_singular_values(item.S, item.n) &&
                        svd_economy(item.m, item.n, item.A, item.lda, item.U, item.ldu, item.S, item.Vt, item.ldvt,
                                    item.work, item.work_floats) != 0) {
                        if (first_error == 0) {
                            first_error = index + 1;
                        }
                        continue;
                    }
                    set_status(item, 1);
                }
            }
        }
        begin = end;
    }
    return first_error;
}

void gemm_nn_batch(GemmNNBatchItem* items, int count) {
    if (!items || count <= 0) {
        return;
    }
    const auto order = bucket_order(items, count, [](const GemmNNBatchItem& a, const GemmNNBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        if (a.n != b.n) {
            return a.n < b.n;
        }
        return a.k < b.k;
    });
    for (int index : order) {
        const auto& item = items[index];
        gemm_nn_hwy(item.m, item.n, item.k, item.A, item.lda, item.B, item.ldb, item.C, item.ldc);
    }
}

int bm3d_filter_group_batch(Bm3dFilterBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const Bm3dFilterBatchItem& a, const Bm3dFilterBatchItem& b) {
        if (a.block != b.block) {
            return a.block < b.block;
        }
        if (a.group != b.group) {
            return a.group < b.group;
        }
        if (a.k != b.k) {
            return a.k < b.k;
        }
        return a.wiener < b.wiener;
    });
    int first_error = 0;
    for (int index : order) {
        auto& item = items[index];
        if (item.k == 0) {
            set_status(item, 1);
            continue;
        }
        if (!item.patches || !item.weight || !item.work || item.lda < item.block * item.block || item.group < 1 ||
            item.k < 1 || item.k > item.group || item.block < 1) {
            if (first_error == 0) {
                first_error = index + 1;
            }
            continue;
        }
        if (item.wiener && !item.ref_patches) {
            if (first_error == 0) {
                first_error = index + 1;
            }
            continue;
        }
        bm3d_filter_group(item.patches, item.lda, item.group, item.k, item.block, item.sigma, item.wiener,
                          item.wiener ? item.ref_patches : nullptr, item.weight, item.work);
        set_status(item, 1);
    }
    return first_error;
}

int wnnm_shrink_batch(WnnmShrinkBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const WnnmShrinkBatchItem& a, const WnnmShrinkBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        if (a.n != b.n) {
            return a.n < b.n;
        }
        if (a.residual != b.residual) {
            return a.residual < b.residual;
        }
        return a.adaptive < b.adaptive;
    });
    auto fail = [&](int index, int& first_error) {
        if (first_error == 0) {
            first_error = index + 1;
        }
    };
    auto finish = [](WnnmShrinkBatchItem& item) {
        const int m = item.m;
        const int n = item.n;
        float* U = item.work;
        float* S = U + m * n;
        float* Vt = S + n;
        float* mean = Vt + n * n;
        const float constant = 8.f * std::sqrt(2.0f * static_cast<float>(n)) * item.sigma * item.sigma;
        const int k = sv_shrink(S, std::min(m, n), constant, item.residual ? 0 : 1);
        if (item.adaptive_weight) {
            *item.adaptive_weight = item.adaptive && k > 0 ? 1.f / static_cast<float>(k) : 1.f;
        }
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < k; ++row) {
                Vt[row + col * n] *= S[row];
            }
        }
        if (item.lda == m) {
            gemm_nn_hwy(m, n, k, U, m, Vt, n, item.group, item.lda);
        } else {
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < m; ++row) {
                    float sum = 0.f;
                    for (int inner = 0; inner < k; ++inner) {
                        sum += U[row + inner * m] * Vt[inner + col * n];
                    }
                    item.group[row + col * item.lda] = sum;
                }
            }
        }
        if (item.residual) {
            group_center_add(item.group, m, n, item.lda, mean);
        }
    };

    int first_error = 0;
    for (int begin = 0; begin < count;) {
        int end = begin + 1;
        const auto& first = items[order[begin]];
        while (end < count) {
            const auto& next = items[order[end]];
            if (next.m != first.m || next.n != first.n || next.residual != first.residual ||
                next.adaptive != first.adaptive) {
                break;
            }
            ++end;
        }

        std::vector<int> batch_indices;
        batch_indices.reserve(static_cast<std::size_t>(end - begin));
        for (int pos = begin; pos < end; ++pos) {
            const int index = order[pos];
            auto& item = items[index];
            if (!item.group || item.m < 1 || item.n < 1 || item.lda < item.m || !item.work || item.work_floats < 1) {
                fail(index, first_error);
                continue;
            }
            if (item.n == 8 && item.m >= 8 && item.m <= kSvdBatch8MaxM &&
                item.work_floats >= wnnm_shrink_work_floats(item.m, item.n)) {
                batch_indices.push_back(index);
                continue;
            }
            if (wnnm_shrink(item.group, item.m, item.n, item.lda, item.sigma, item.residual, item.adaptive,
                            item.adaptive_weight, item.work, item.work_floats) != 0) {
                fail(index, first_error);
            } else {
                set_status(item, 1);
            }
        }

        if (batch_indices.size() == 1) {
            const int index = batch_indices.front();
            auto& item = items[index];
            if (wnnm_shrink(item.group, item.m, item.n, item.lda, item.sigma, item.residual, item.adaptive,
                            item.adaptive_weight, item.work, item.work_floats) != 0) {
                fail(index, first_error);
            } else {
                set_status(item, 1);
            }
        } else if (!batch_indices.empty()) {
            const std::size_t size = batch_indices.size();
            std::vector<const float*> a(size);
            std::vector<int> lda(size);
            std::vector<float*> u(size);
            std::vector<int> ldu(size);
            std::vector<float*> s(size);
            std::vector<float*> vt(size);
            std::vector<int> ldvt(size, 8);
            for (std::size_t pos = 0; pos < size; ++pos) {
                auto& item = items[batch_indices[pos]];
                float* U = item.work;
                float* S = U + item.m * item.n;
                float* Vt = S + item.n;
                float* mean = Vt + item.n * item.n;
                if (item.residual) {
                    group_center_sub(item.group, item.m, item.n, item.lda, mean);
                }
                a[pos] = item.group;
                lda[pos] = item.lda;
                u[pos] = U;
                ldu[pos] = item.m;
                s[pos] = S;
                vt[pos] = Vt;
            }
            if (svd_economy_8_batch_hwy(first.m, a.data(), lda.data(), u.data(), ldu.data(), s.data(), vt.data(),
                                        ldvt.data(), static_cast<int>(size)) != 0) {
                for (int index : batch_indices) {
                    auto& item = items[index];
                    if (item.residual) {
                        float* mean = item.work + item.m * item.n + item.n + item.n * item.n;
                        group_center_add(item.group, item.m, item.n, item.lda, mean);
                    }
                    fail(index, first_error);
                }
            } else {
                for (int index : batch_indices) {
                    auto& item = items[index];
                    float* U = item.work;
                    float* S = U + item.m * item.n;
                    float* Vt = S + item.n;
                    float* mean = Vt + item.n * item.n;
                    float* svd_work = mean + item.m;
                    const int svd_cap = item.work_floats - (item.m * item.n + item.n + item.n * item.n + item.m);
                    if (!finite_singular_values(S, item.n) &&
                        svd_economy(item.m, item.n, item.group, item.lda, U, item.m, S, Vt, item.n, svd_work,
                                    svd_cap) != 0) {
                        if (item.residual) {
                            group_center_add(item.group, item.m, item.n, item.lda, mean);
                        }
                        fail(index, first_error);
                        continue;
                    }
                    finish(item);
                    set_status(item, 1);
                }
            }
        }
        begin = end;
    }
    return first_error;
}

int mcwnnm_filter_group_batch(McwnnmFilterBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const McwnnmFilterBatchItem& a, const McwnnmFilterBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        if (a.n != b.n) {
            return a.n < b.n;
        }
        return a.residual < b.residual;
    });
    int first_error = 0;
    for (int index : order) {
        auto& item = items[index];
        if (item.m < 1 || item.n < 1 || !item.group || !item.sigma || !item.work || item.work_floats < 1) {
            if (first_error == 0) {
                first_error = index + 1;
            }
            continue;
        }
        if (mcwnnm_filter_group(item.group, item.m, item.n, item.lda, item.nch, item.sigma, item.admm_iter, item.rho,
                                item.mu, item.residual, item.adaptive, item.adaptive_weight, item.work,
                                item.work_floats) != 0) {
            if (first_error == 0) {
                first_error = index + 1;
            }
            continue;
        }
        set_status(item, 1);
    }
    return first_error;
}

int twsc_pca_soft_batch(TwscPcaBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const TwscPcaBatchItem& a, const TwscPcaBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        return a.n < b.n;
    });
    int first_error = 0;
    for (int begin = 0; begin < count;) {
        int end = begin + 1;
        while (end < count && items[order[end]].m == items[order[begin]].m &&
               items[order[end]].n == items[order[begin]].n) {
            ++end;
        }
        const int m = items[order[begin]].m;
        const int n = items[order[begin]].n;
        std::vector<int> batch_indices;
        batch_indices.reserve(static_cast<std::size_t>(end - begin));
        for (int pos = begin; pos < end; ++pos) {
            const int index = order[pos];
            auto& item = items[index];
            if (!item.group || !item.work || item.m < 1 || item.n < 1 || item.lda < item.m ||
                item.work_floats < 1) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
                continue;
            }
            if (n == 8 && m >= 8 && m <= kSvdBatch8MaxM &&
                item.work_floats >= twsc_pca_soft_work_floats(item.m, item.n)) {
                batch_indices.push_back(index);
                continue;
            }
            if (twsc_pca_soft(item.group, item.m, item.n, item.lda, item.sigma, item.work, item.work_floats,
                              item.col_sigma, item.col_weight, item.row_weight) != 0) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
            } else {
                set_status(item, 1);
            }
        }

        if (batch_indices.size() == 1) {
            const int index = batch_indices.front();
            auto& item = items[index];
            if (twsc_pca_soft(item.group, item.m, item.n, item.lda, item.sigma, item.work, item.work_floats,
                              item.col_sigma, item.col_weight, item.row_weight) != 0) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
            } else {
                set_status(item, 1);
            }
        } else if (!batch_indices.empty()) {
            const std::size_t size = batch_indices.size();
            std::vector<const float*> a(size);
            std::vector<int> lda(size);
            std::vector<float*> u(size);
            std::vector<int> ldu(size);
            std::vector<float*> s(size);
            for (std::size_t pos = 0; pos < size; ++pos) {
                auto& item = items[batch_indices[pos]];
                float* U = item.work;
                float* S = U + item.m * item.n;
                float* mean = S + item.n;
                float* B = mean + item.m;
                scale_twsc_rows(item.group, item.m, item.n, item.lda, item.row_weight, false);
                group_center_sub(item.group, item.m, item.n, item.lda, mean);
                a[pos] = item.group;
                lda[pos] = item.lda;
                u[pos] = U;
                ldu[pos] = item.m;
                s[pos] = S;
            }
            if (svd_economy_8_batch_u_hwy(m, a.data(), lda.data(), u.data(), ldu.data(), s.data(),
                                          static_cast<int>(size)) != 0) {
                for (int index : batch_indices) {
                    auto& item = items[index];
                    float* mean = item.work + item.m * item.n + item.n;
                    group_center_add(item.group, item.m, item.n, item.lda, mean);
                    scale_twsc_rows(item.group, item.m, item.n, item.lda, item.row_weight, true);
                    if (first_error == 0) {
                        first_error = index + 1;
                    }
                }
            } else {
                for (int index : batch_indices) {
                    auto& item = items[index];
                    float* U = item.work;
                    float* S = U + item.m * item.n;
                    float* mean = S + item.n;
                    float* B = mean + item.m;
                    float* Vt = B + item.m * item.n;
                    float* svd_work = Vt + item.n * item.n;
                    const int svd_cap = item.work_floats - static_cast<int>(svd_work - item.work);
                    if (!finite_singular_values(S, item.n) &&
                        svd_economy(item.m, item.n, item.group, item.lda, U, item.m, S, Vt, item.n, svd_work,
                                    svd_cap) != 0) {
                        group_center_add(item.group, item.m, item.n, item.lda, mean);
                        scale_twsc_rows(item.group, item.m, item.n, item.lda, item.row_weight, true);
                        if (first_error == 0) {
                            first_error = index + 1;
                        }
                        continue;
                    }
                    finish_twsc_batch_item(item);
                    set_status(item, 1);
                }
            }
        }
        begin = end;
    }
    return first_error;
}

int ncsr_filter_group_batch(NcsrFilterBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const NcsrFilterBatchItem& a, const NcsrFilterBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        return a.n < b.n;
    });
    int first_error = 0;
    for (int begin = 0; begin < count;) {
        int end = begin + 1;
        while (end < count && items[order[end]].m == items[order[begin]].m &&
               items[order[end]].n == items[order[begin]].n) {
            ++end;
        }
        const int m = items[order[begin]].m;
        const int n = items[order[begin]].n;
        std::vector<int> batch_indices;
        batch_indices.reserve(static_cast<std::size_t>(end - begin));
        for (int pos = begin; pos < end; ++pos) {
            const int index = order[pos];
            auto& item = items[index];
            if (!item.group || !item.work || item.m < 1 || item.n < 1 || item.lda < item.m ||
                item.work_floats < 1) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
                continue;
            }
            if (n == 8 && m >= 8 && m <= kSvdBatch8MaxM &&
                item.work_floats >= ncsr_filter_work_floats(item.m, item.n)) {
                batch_indices.push_back(index);
            } else if (ncsr_filter_group(item.group, item.m, item.n, item.lda, item.sigma, item.col_dist, item.work,
                                         item.work_floats) != 0) {
                if (first_error == 0) {
                    first_error = index + 1;
                }
            } else {
                set_status(item, 1);
            }
        }

        if (batch_indices.size() < 2) {
            for (int index : batch_indices) {
                auto& item = items[index];
                if (ncsr_filter_group(item.group, item.m, item.n, item.lda, item.sigma, item.col_dist, item.work,
                                      item.work_floats) != 0) {
                    if (first_error == 0) {
                        first_error = index + 1;
                    }
                } else {
                    set_status(item, 1);
                }
            }
            begin = end;
            continue;
        }

        const std::size_t size = batch_indices.size();
        std::vector<const float*> a(size);
        std::vector<int> lda(size);
        std::vector<float*> u(size);
        std::vector<int> ldu(size, m);
        std::vector<float*> s(size);
        std::vector<float*> vt(size);
        std::vector<int> ldvt(size, n);
        for (std::size_t pos = 0; pos < size; ++pos) {
            auto& item = items[batch_indices[pos]];
            float* U = item.work;
            float* S = U + m * n;
            float* mean = S + n;
            float* B = mean + m;
            float* Vt = B + m * n;
            group_center_sub(item.group, m, n, item.lda, mean);
            a[pos] = item.group;
            lda[pos] = item.lda;
            u[pos] = U;
            s[pos] = S;
            vt[pos] = Vt;
        }
        const bool batch_ok = svd_economy_8_batch_hwy(m, a.data(), lda.data(), u.data(), ldu.data(), s.data(),
                                                       vt.data(), ldvt.data(), static_cast<int>(size)) == 0;
        for (int index : batch_indices) {
            auto& item = items[index];
            float* U = item.work;
            float* S = U + m * n;
            float* mean = S + n;
            float* B = mean + m;
            float* Vt = B + m * n;
            float* svd_work = Vt + n * n;
            const int svd_cap = item.work_floats - static_cast<int>(svd_work - item.work);
            if ((!batch_ok || !finite_singular_values(S, n)) &&
                svd_economy(m, n, item.group, item.lda, U, m, S, Vt, n, svd_work, svd_cap) != 0) {
                group_center_add(item.group, m, n, item.lda, mean);
                if (first_error == 0) {
                    first_error = index + 1;
                }
                continue;
            }
            finish_ncsr_batch_item(item);
            set_status(item, 1);
        }
        begin = end;
    }
    return first_error;
}

int nlh_filter_group_batch(NlhFilterBatchItem* items, int count) {
    if (count == 0) {
        return 0;
    }
    if (!items || count < 0) {
        return -1;
    }
    for (int i = 0; i < count; ++i) {
        set_status(items[i], 0);
    }
    const auto order = bucket_order(items, count, [](const NlhFilterBatchItem& a, const NlhFilterBatchItem& b) {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        if (a.n != b.n) {
            return a.n < b.n;
        }
        return a.q < b.q;
    });
    int first_error = 0;
    for (int index : order) {
        auto& item = items[index];
        if (!item.patches || !item.weight || !item.work || item.m < 1 || item.n < 1 || item.lda < item.m ||
            item.q < 1 || item.work_floats < 1 || (item.wiener && !item.ref_patches)) {
            if (first_error == 0) {
                first_error = index + 1;
            }
            continue;
        }
        nlh_filter_group(item.patches, item.m, item.n, item.lda, item.q, item.sigma, item.wiener, item.ref_patches,
                         item.weight, item.work, item.work_floats);
        set_status(item, 1);
    }
    return first_error;
}

}  // namespace nss
