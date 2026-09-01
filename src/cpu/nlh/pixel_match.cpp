#include "nss/cpu_nlh.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <array>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/pixel_match.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void PixelMatch(const float* group, int m, int n, int lda, int q, int* idx) {
    if (!group || !idx || m < 1 || n < 1 || q < 1 || lda < m) {
        return;
    }
    const int qq = std::min(q, m);
    constexpr float kInf = 1.0e30f;
    constexpr int kFixedM = 256;
    constexpr int kFixedQ = 8;
    std::array<int, kFixedM * kFixedQ> fixed_idx;
    std::array<float, kFixedM * kFixedQ> fixed_dist;
    std::vector<int> dynamic_idx;
    std::vector<float> dynamic_dist;
    int* best_idx = nullptr;
    float* best_dist = nullptr;
    if (m <= kFixedM && qq <= kFixedQ) {
        best_idx = fixed_idx.data();
        best_dist = fixed_dist.data();
    } else {
        dynamic_idx.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(qq));
        dynamic_dist.resize(dynamic_idx.size());
        best_idx = dynamic_idx.data();
        best_dist = dynamic_dist.data();
    }
    for (int r = 0; r < m; ++r) {
        best_idx[r * qq] = r;
        best_dist[r * qq] = 0.f;
        for (int j = 1; j < qq; ++j) {
            best_idx[r * qq + j] = r;
            best_dist[r * qq + j] = kInf;
        }
    }

    std::array<float, kFixedM> fixed_pair;
    std::vector<float> dynamic_pair;
    float* pair_dist = nullptr;
    if (m <= kFixedM) {
        pair_dist = fixed_pair.data();
    } else {
        dynamic_pair.resize(static_cast<std::size_t>(m));
        pair_dist = dynamic_pair.data();
    }

    auto better = [](float d0, int i0, float d1, int i1) {
        return d0 < d1 || (d0 == d1 && i0 < i1);
    };
    auto insert = [&](int row, int candidate, float distance) {
        if (qq <= 1) {
            return;
        }
        if (qq == 4) {
            float* bd = best_dist + row * 4;
            int* bi = best_idx + row * 4;
            if (!better(distance, candidate, bd[3], bi[3])) {
                return;
            }
            if (better(distance, candidate, bd[1], bi[1])) {
                bd[3] = bd[2];
                bi[3] = bi[2];
                bd[2] = bd[1];
                bi[2] = bi[1];
                bd[1] = distance;
                bi[1] = candidate;
            } else if (better(distance, candidate, bd[2], bi[2])) {
                bd[3] = bd[2];
                bi[3] = bi[2];
                bd[2] = distance;
                bi[2] = candidate;
            } else {
                bd[3] = distance;
                bi[3] = candidate;
            }
            return;
        }
        int pos = qq - 1;
        if (!better(distance, candidate, best_dist[row * qq + pos], best_idx[row * qq + pos])) {
            return;
        }
        while (pos > 1 && better(distance, candidate, best_dist[row * qq + pos - 1], best_idx[row * qq + pos - 1])) {
            best_dist[row * qq + pos] = best_dist[row * qq + pos - 1];
            best_idx[row * qq + pos] = best_idx[row * qq + pos - 1];
            --pos;
        }
        best_dist[row * qq + pos] = distance;
        best_idx[row * qq + pos] = candidate;
    };

    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    using V = hn::Vec<hn::ScalableTag<float>>;
    // Compute only the upper triangle. Every distance is inserted into both
    // rows, so symmetric pairs cannot diverge through rounding or ties.
    for (int r = 0; r < m; ++r) {
        int s0 = r + 1;
        for (; s0 + 4 * N <= m; s0 += 4 * N) {
            V acc0 = hn::Zero(d);
            V acc1 = hn::Zero(d);
            V acc2 = hn::Zero(d);
            V acc3 = hn::Zero(d);
            for (int c = 0; c < n; ++c) {
                const auto vr = hn::Set(d, group[r + c * lda]);
                const float* col = group + c * lda + s0;
                const auto e0 = hn::Sub(hn::LoadU(d, col), vr);
                const auto e1 = hn::Sub(hn::LoadU(d, col + N), vr);
                const auto e2 = hn::Sub(hn::LoadU(d, col + 2 * N), vr);
                const auto e3 = hn::Sub(hn::LoadU(d, col + 3 * N), vr);
                acc0 = hn::MulAdd(e0, e0, acc0);
                acc1 = hn::MulAdd(e1, e1, acc1);
                acc2 = hn::MulAdd(e2, e2, acc2);
                acc3 = hn::MulAdd(e3, e3, acc3);
            }
            hn::StoreU(acc0, d, pair_dist + s0);
            hn::StoreU(acc1, d, pair_dist + s0 + N);
            hn::StoreU(acc2, d, pair_dist + s0 + 2 * N);
            hn::StoreU(acc3, d, pair_dist + s0 + 3 * N);
        }
        for (; s0 + N <= m; s0 += N) {
            V acc = hn::Zero(d);
            for (int c = 0; c < n; ++c) {
                const auto e = hn::Sub(hn::LoadU(d, group + c * lda + s0), hn::Set(d, group[r + c * lda]));
                acc = hn::MulAdd(e, e, acc);
            }
            hn::StoreU(acc, d, pair_dist + s0);
        }
        if (s0 < m) {
            const std::size_t remaining = static_cast<std::size_t>(m - s0);
            V acc = hn::Zero(d);
            for (int c = 0; c < n; ++c) {
                const auto e = hn::Sub(hn::LoadN(d, group + c * lda + s0, remaining),
                                       hn::Set(d, group[r + c * lda]));
                acc = hn::MulAdd(e, e, acc);
            }
            hn::StoreN(acc, d, pair_dist + s0, remaining);
        }
        for (int s = r + 1; s < m; ++s) {
            insert(r, s, pair_dist[s]);
            insert(s, r, pair_dist[s]);
        }
    }

    for (int r = 0; r < m; ++r) {
        int j = 0;
        for (; j < qq; ++j) {
            idx[r * q + j] = best_idx[r * qq + j];
        }
        for (; j < q; ++j) {
            idx[r * q + j] = r;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(PixelMatch);

void pixel_match(const float* group, int m, int n, int lda, int q, int* idx) {
    HWY_DYNAMIC_DISPATCH(PixelMatch)(group, m, n, lda, q, idx);
}

}  // namespace nss
#endif
