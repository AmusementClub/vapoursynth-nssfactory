// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "nss/cpu_nlh.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/nlh/pixel_match.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
#include "cpu/nlh/select_top3-inl.hpp"

// SSDs of row r against the K full vectors of candidates from s0 and then
// `tail` (< lanes) more, into out + s0: one pass over the columns with an
// accumulator per vector, each lane's sum in column order as before.
template <int K, class D>
HWY_INLINE void SsdRemainder(D d, const float* group, int n, int lda, int r, int s0, std::size_t tail, float* out) {
    const int N = static_cast<int>(hn::Lanes(d));
    auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), at = hn::Zero(d);
    for (int c = 0; c < n; ++c) {
        const auto vr = hn::Set(d, group[r + c * lda]);
        const float* col = group + c * lda + s0;
        if constexpr (K > 0) {
            const auto e = hn::Sub(hn::LoadU(d, col), vr);
            a0 = hn::MulAdd(e, e, a0);
        }
        if constexpr (K > 1) {
            const auto e = hn::Sub(hn::LoadU(d, col + N), vr);
            a1 = hn::MulAdd(e, e, a1);
        }
        if constexpr (K > 2) {
            const auto e = hn::Sub(hn::LoadU(d, col + 2 * N), vr);
            a2 = hn::MulAdd(e, e, a2);
        }
        if (tail) {
            const auto e = hn::Sub(hn::LoadN(d, col + K * N, tail), vr);
            at = hn::MulAdd(e, e, at);
        }
    }
    if constexpr (K > 0) hn::StoreU(a0, d, out + s0);
    if constexpr (K > 1) hn::StoreU(a1, d, out + s0 + N);
    if constexpr (K > 2) hn::StoreU(a2, d, out + s0 + 2 * N);
    if (tail) hn::StoreN(at, d, out + s0 + K * N, tail);
}

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
    nss::ResourceVector<int> dynamic_idx;
    nss::ResourceVector<float> dynamic_dist;
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
    nss::ResourceVector<float> dynamic_pair;
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
    constexpr int kSelM = 64;
    // Row r's SSDs against candidates s0.. into out[s0..m).
    auto ssd_from = [&](int r, int s0, float* out) {
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
            hn::StoreU(acc0, d, out + s0);
            hn::StoreU(acc1, d, out + s0 + N);
            hn::StoreU(acc2, d, out + s0 + 2 * N);
            hn::StoreU(acc3, d, out + s0 + 3 * N);
        }
        // The rest, under four vectors: one pass (a vector at a time it was
        // a chain of dependent FMAs per vector).
        if (s0 >= m) return;
        const std::size_t tail = static_cast<std::size_t>((m - s0) % N);
        switch ((m - s0) / N) {
            case 0: SsdRemainder<0>(d, group, n, lda, r, s0, tail, out); break;
            case 1: SsdRemainder<1>(d, group, n, lda, r, s0, tail, out); break;
            case 2: SsdRemainder<2>(d, group, n, lda, r, s0, tail, out); break;
            default: SsdRemainder<3>(d, group, n, lda, r, s0, tail, out); break;
        }
    };

    auto select_row = [&](float* distances, int r) {
        if ((HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX2) &&
            m == 64 && n == 16 && qq == 4 &&
            SelectTopThree64(d, distances, r, best_idx + r * 4 + 1, best_dist + r * 4 + 1)) return;
        const auto inf = hn::Set(d, kInf);
        distances[r] = kInf;
        for (int slot = 1; slot < qq; ++slot) {
            // Four running minima (one chain was latency-bound): the minimum
            // of the distances (never NaN for a finite group) in any order.
            auto v0 = inf, v1 = inf, v2 = inf, v3 = inf;
            int j = 0;
            for (; j + 4 * N <= m; j += 4 * N) {
                v0 = hn::Min(v0, hn::LoadU(d, distances + j));
                v1 = hn::Min(v1, hn::LoadU(d, distances + j + N));
                v2 = hn::Min(v2, hn::LoadU(d, distances + j + 2 * N));
                v3 = hn::Min(v3, hn::LoadU(d, distances + j + 3 * N));
            }
            for (; j + N <= m; j += N)
                v0 = hn::Min(v0, hn::LoadU(d, distances + j));
            const auto vmin = hn::Min(hn::Min(v0, v1), hn::Min(v2, v3));
            float best_d = hn::ReduceMin(d, vmin);
            for (; j < m; ++j) best_d = std::min(best_d, distances[j]);
            const auto vbest = hn::Set(d, best_d);
            int best_s = m;
            j = 0;
            for (; j + N <= m; j += N) {
                const auto eq = hn::Eq(hn::LoadU(d, distances + j), vbest);
                if (!hn::AllFalse(d, eq)) {
                    best_s = j + static_cast<int>(hn::FindFirstTrue(d, eq));
                    break;
                }
            }
            for (; j < m && best_s == m; ++j)
                if (distances[j] == best_d) best_s = j;
            if (!(best_d < kInf) || best_s < 0 || best_s >= m) break;
            best_idx[r * qq + slot] = best_s;
            best_dist[r * qq + slot] = best_d;
            distances[best_s] = kInf;
        }
    };

    // Repeating SSDs needs sufficient vector throughput. The measured policy
    // admits AVX3 and the tested group range; narrower targets retain the
    // upper-triangle method below.
    if (HWY_TARGET == HWY_AVX3 && m > kSelM && m <= kFixedM && n >= 8 && n <= 64 && qq <= 16 && qq > 1) {
        // A whole row permits SIMD selection without a quadratic scratch
        // matrix. SSD arithmetic is repeated for the symmetric row, keeping
        // the same column/FMA order; changing the difference sign is exact.
        // Every distance below kInf, by its bits (non-negative floats order
        // as their bits; overflowed and NaN distances fail), a vector at a
        // time: the scalar compares took a fifth of the stage.
        const hn::RebindToUnsigned<decltype(d)> du;
        const auto limit = hn::Set(du, std::bit_cast<std::uint32_t>(kInf));
        for (int r = 0; r < m; ++r) {
            ssd_from(r, 0, pair_dist);
            auto below = hn::FirstN(du, hn::Lanes(du));
            int s = 0;
            for (; s + N <= m; s += N) below = hn::And(below, hn::Lt(hn::BitCast(du, hn::LoadU(d, pair_dist + s)), limit));
            if (s < m) {
                const std::size_t rest = static_cast<std::size_t>(m - s);
                below = hn::And(below, hn::Lt(hn::BitCast(du, hn::LoadN(d, pair_dist + s, rest)), limit));
            }
            const bool bounded = hn::AllTrue(du, below);
            if (bounded) select_row(pair_dist, r);
            else {
                // The insertion path also defines finite sentinel ties and
                // exceptional-distance behavior; do not broaden that contract.
                for (int s = 0; s < m; ++s) if (s != r) insert(r, s, pair_dist[s]);
            }
        }
    } else if (m <= kSelM && qq > 1) {
        // Keep the column-major upper-triangle SSD; defer top-q so each row
        // selects from a dense distance row instead of updating two heaps
        // per pair. Ties keep the lowest index, matching better().
        std::array<float, kSelM * kSelM> dist;
        for (int r = 0; r < m; ++r) {
            // The row's upper part straight from the SSD; the lower part is
            // the transposed column of the rows before it.
            float* row = dist.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(m);
            ssd_from(r, r + 1, row);
            row[r] = 0.f;
            for (int s = r + 1; s < m; ++s) dist[static_cast<std::size_t>(s) * static_cast<std::size_t>(m) + static_cast<std::size_t>(r)] = row[s];
        }
        // Each row is consumed exactly once and select_row only mutates the
        // row it selects from, so select directly on the matrix row — no
        // per-row copy.
        for (int r = 0; r < m; ++r) {
            select_row(dist.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(m), r);
        }
    } else {
    // Compute only the upper triangle. Every distance is inserted into both
    // rows, so symmetric pairs cannot diverge through rounding or ties.
    for (int r = 0; r < m; ++r) {
        ssd_from(r, r + 1, pair_dist);
        for (int s = r + 1; s < m; ++s) {
            insert(r, s, pair_dist[s]);
            insert(s, r, pair_dist[s]);
        }
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
