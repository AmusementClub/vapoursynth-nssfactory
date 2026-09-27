// SPDX-License-Identifier: GPL-2.0-only
#include "cpu/finishers.hpp"
#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#include "nss/cpu_ncsr.hpp"

#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_twsc.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/ncsr/centralize.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/fast_math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static void GatherStrided(float* dst, const float* src, int stride, int n) {
    const hn::ScalableTag<float> d;
    const hn::Rebind<int32_t, hn::ScalableTag<float>> di;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(di, stride);
    int j = 0;
    for (; j + N <= n; j += N) {
        const auto idx = hn::Mul(hn::Iota(di, static_cast<int32_t>(j)), vs);
        hn::StoreU(hn::GatherIndex(d, src, idx), d, dst + j);
    }
    for (; j < n; ++j) {
        dst[j] = src[j * stride];
    }
}

static void ScatterStrided(float* dst, int stride, int n, const float* src) {
    const hn::ScalableTag<float> d;
    const hn::Rebind<int32_t, hn::ScalableTag<float>> di;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto vs = hn::Set(di, stride);
    int j = 0;
    for (; j + N <= n; j += N) {
        const auto idx = hn::Mul(hn::Iota(di, static_cast<int32_t>(j)), vs);
        hn::ScatterIndex(hn::LoadU(d, src + j), d, dst, idx);
    }
    for (; j < n; ++j) {
        dst[j * stride] = src[j];
    }
}

float NcsrGroupWeights(const float* col_dist, const float* group, int m, int n, int lda, float h, float* weights) {
    if (!weights || n < 1 || n > kSvdMaxN || !(h > 0.f) || !is_finite_bits(h) ||
        (!col_dist && (!group || m < 1 || lda < m))) {
        return 0.f;
    }
    float distance[kSvdMaxN];
    for (int col = 0; col < n; ++col) {
        distance[col] = col_dist ? col_dist[col]
                                 : (col > 0 ? ssd_vec(group + col * lda, group, m) : 0.f);
    }

    // One vector exp for complete and partial vectors. Weights are stored
    // first, then reduced in column order to keep the accumulation contract.
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto scale = hn::Set(d, -1.f / h);
    for (int col = 0; col < n; col += lanes) {
        const int count = n - col < lanes ? n - col : lanes;
        const auto exponent =
            hn::Mul(hn::LoadN(d, distance + col, static_cast<std::size_t>(count)), scale);
        hn::StoreN(hn::FastExp</*kHandleSubnormals=*/false>(d, exponent), d, weights + col,
                   static_cast<std::size_t>(count));
    }
    float sum = 0.f;
    for (int col = 0; col < n; ++col) {
        sum += weights[col];
    }
    return sum;
}

void NcsrCentralizeCodes(float* B, int r, int n, int ldb, float tau, const float* col_w, const float* row_tau) {
    if (!B || r < 1 || n < 1 || ldb < r || r > kSvdMaxN || n > kSvdMaxN) {
        return;
    }
    float w[kSvdMaxN];
    float wsum = 0.f;
    for (int j = 0; j < n; ++j) {
        const float wj = col_w ? std::max(col_w[j], 0.f) : 1.f;
        w[j] = wj;
        wsum += wj;
    }
    if (!(wsum > 0.f)) {
        for (int j = 0; j < n; ++j) {
            w[j] = 1.f;
        }
        wsum = static_cast<float>(n);
    }
    const float inv = 1.f / wsum;
    float row[kSvdMaxN];
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int i = 0; i < r; ++i) {
        GatherStrided(row, B + i, ldb, n);
        float s = 0.f;
        int j = 0;
        for (; j + N <= n; j += N) {
            s += hn::ReduceSum(d, hn::Mul(hn::LoadU(d, w + j), hn::LoadU(d, row + j)));
        }
        for (; j < n; ++j) {
            s += w[j] * row[j];
        }
        const float beta = s * inv;
        const auto vb = hn::Set(d, beta);
        j = 0;
        for (; j + N <= n; j += N) {
            hn::StoreU(hn::Sub(hn::LoadU(d, row + j), vb), d, row + j);
        }
        for (; j < n; ++j) {
            row[j] -= beta;
        }
        const float ti = row_tau ? row_tau[i] : tau;
        soft_threshold(row, n, ti);
        j = 0;
        for (; j + N <= n; j += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, row + j), vb), d, row + j);
        }
        for (; j < n; ++j) {
            row[j] += beta;
        }
        ScatterStrided(B + i, ldb, n, row);
    }
}

int NcsrFilterGroup(float* group, int m, int n, int lda, float sigma, const float* col_dist, float* work,
                    int work_floats) {
    if (!group || n <= 0 || m <= 0 || lda < m || m > kSvdMaxM || n > kSvdMaxN) {
        return -1;
    }
    const int need = ncsr_filter_work_floats(m, n);
    if (!work || work_floats < need) {
        return -1;
    }
    float* U = work;
    float* S = U + m * n;
    float* mean = S + n;
    float* B = mean + m;
    float* proj_work = B + m * n;
    const int proj_cap = work_floats - (m * n + n + m + m * n);
    const int r = pca_project(group, m, n, lda, U, S, B, mean, proj_work, proj_cap);
    if (r < 0) {
        return -1;
    }
    struct Rows {
        void gather(float* row, const float* codes, int i, int stride, int count) const {
            GatherStrided(row, codes + i, stride, count);
        }
    };
    detail::finish_ncsr_codes(group, m, n, lda, sigma, col_dist, B, Rows{}, NcsrGroupWeights, NcsrCentralizeCodes);
    detail::finish_pca_reconstruction(group, m, n, lda, U, B, mean);
    return 0;
}

// --- Row-major codes batch kernels (Track A) --------------------------------
// Brm[i*n + j] stores code row i of patch column j, so code rows are
// contiguous. Each kernel replicates the exact per-output accumulation order
// of the column-major composition it replaces (gemm_tn_hwy, finish_ncsr_codes
// + NcsrCentralizeCodes + soft_threshold, gemm_nn_hwy + GroupCenterAdd); only
// addresses and pass structure changed, never arithmetic order.

void NcsrProjectRM(const float* U, const float* group, int m, int n, int lda, float* Brm) {
    if (!U || !group || !Brm || m < 1 || n < 1 || lda < m || n > kSvdMaxN) {
        return;
    }
    const int r = std::min(m, n);
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        const float* bj = group + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        int t = 0;
        // Same accumulation shape as GemmTN (wnnm/jacobi8.cpp): 4-column
        // accumulator quads, N-lane chunks, ReduceSum, then the scalar tail.
        for (; t + 4 <= r; t += 4) {
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            const float* a0 = U + static_cast<std::size_t>(t) * static_cast<std::size_t>(m);
            const float* a1 = U + static_cast<std::size_t>(t + 1) * static_cast<std::size_t>(m);
            const float* a2 = U + static_cast<std::size_t>(t + 2) * static_cast<std::size_t>(m);
            const float* a3 = U + static_cast<std::size_t>(t + 3) * static_cast<std::size_t>(m);
            int i = 0;
            for (; i + N <= m; i += N) {
                const auto bv = hn::LoadU(d, bj + i);
                acc0 = hn::MulAdd(hn::LoadU(d, a0 + i), bv, acc0);
                acc1 = hn::MulAdd(hn::LoadU(d, a1 + i), bv, acc1);
                acc2 = hn::MulAdd(hn::LoadU(d, a2 + i), bv, acc2);
                acc3 = hn::MulAdd(hn::LoadU(d, a3 + i), bv, acc3);
            }
            float s0 = hn::ReduceSum(d, acc0);
            float s1 = hn::ReduceSum(d, acc1);
            float s2 = hn::ReduceSum(d, acc2);
            float s3 = hn::ReduceSum(d, acc3);
            for (; i < m; ++i) {
                const float bv = bj[i];
                s0 += a0[i] * bv;
                s1 += a1[i] * bv;
                s2 += a2[i] * bv;
                s3 += a3[i] * bv;
            }
            Brm[static_cast<std::size_t>(t) * n + j] = s0;
            Brm[static_cast<std::size_t>(t + 1) * n + j] = s1;
            Brm[static_cast<std::size_t>(t + 2) * n + j] = s2;
            Brm[static_cast<std::size_t>(t + 3) * n + j] = s3;
        }
        for (; t < r; ++t) {
            // Replicates DotN (wnnm/jacobi8.cpp) for r % 4 != 0 shapes.
            const float* a = U + static_cast<std::size_t>(t) * static_cast<std::size_t>(m);
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            int i = 0;
            for (; i + 4 * N <= m; i += 4 * N) {
                acc0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, bj + i), acc0);
                acc1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, bj + i + N), acc1);
                acc2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, bj + i + 2 * N), acc2);
                acc3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, bj + i + 3 * N), acc3);
            }
            auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
            for (; i + N <= m; i += N) {
                acc = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, bj + i), acc);
            }
            float s = hn::ReduceSum(d, acc);
            for (; i < m; ++i) {
                s += a[i] * bj[i];
            }
            Brm[static_cast<std::size_t>(t) * n + j] = s;
        }
    }
}

void NcsrFinishCodesRM(float* Brm, int r, int n, float sigma, const float* col_dist, const float* group, int m,
                       int lda) {
    if (!Brm || r < 1 || n < 1 || r > kSvdMaxN || n > kSvdMaxN) {
        return;
    }
    if (!(sigma > 0.f) || !is_finite_bits(sigma)) {
        return;
    }
    // Weights exactly as finish_ncsr_codes computes them.
    float weights[kSvdMaxN];
    constexpr float epsilon = 1e-12f;
    const float h = std::max(2.f * static_cast<float>(m) * sigma * sigma, epsilon);
    const float sum0 = NcsrGroupWeights(col_dist, group, m, n, lda, h, weights);
    float sum = sum0;
    if (!(sum > 0.f)) {
        std::fill_n(weights, n, 1.f);
        sum = static_cast<float>(n);
    }
    // 1/sum equals NcsrCentralizeCodes' 1/wsum: the clamped weights are
    // identical (FastExp outputs are non-negative) and the sequential column
    // order sum is the same value.
    const float inverse = 1.f / sum;
    constexpr float map_constant = 2.8284271247461903f;
    const float sigma2 = sigma * sigma;
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int i = 0; i < r; ++i) {
        float* row = Brm + static_cast<std::size_t>(i) * n;
        // finish_ncsr_codes pass: scalar column order on the original row.
        float mean = 0.f;
        for (int col = 0; col < n; ++col) mean += weights[col] * row[col];
        mean *= inverse;
        float variance = 0.f;
        for (int col = 0; col < n; ++col) {
            const float error = row[col] - mean;
            variance += weights[col] * error * error;
        }
        const float ti_raw = map_constant * sigma2 / (std::sqrt(variance * inverse) + epsilon);
        // NcsrCentralizeCodes pass: its own accumulation order, in place.
        float s = 0.f;
        int j = 0;
        for (; j + N <= n; j += N) {
            s += hn::ReduceSum(d, hn::Mul(hn::LoadU(d, weights + j), hn::LoadU(d, row + j)));
        }
        for (; j < n; ++j) {
            s += weights[j] * row[j];
        }
        const float beta = s * inverse;
        const auto vb = hn::Set(d, beta);
        j = 0;
        for (; j + N <= n; j += N) {
            hn::StoreU(hn::Sub(hn::LoadU(d, row + j), vb), d, row + j);
        }
        for (; j < n; ++j) {
            row[j] -= beta;
        }
        // Inlined soft_threshold (common/soft_threshold.cpp), same guarded tau
        // and the same vector/scalar split for this (N, n).
        const float t = is_finite_bits(ti_raw) && ti_raw > 0.f ? ti_raw : 0.f;
        const auto vt = hn::Set(d, t);
        const auto z = hn::Zero(d);
        j = 0;
        for (; j + N <= n; j += N) {
            const auto v = hn::LoadU(d, row + j);
            const auto a = hn::Abs(v);
            const auto shrunk = hn::CopySign(hn::Sub(a, vt), v);
            hn::StoreU(hn::IfThenElse(hn::Gt(a, vt), shrunk, z), d, row + j);
        }
        for (; j < n; ++j) {
            const float v = row[j];
            const float a = std::fabs(v);
            row[j] = (a > t) ? std::copysign(a - t, v) : 0.f;
        }
        j = 0;
        for (; j + N <= n; j += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, row + j), vb), d, row + j);
        }
        for (; j < n; ++j) {
            row[j] += beta;
        }
    }
}

void NcsrReconstructRM(float* group, int m, int n, int lda, const float* U, const float* Brm, const float* mean) {
    if (!group || !U || !Brm || !mean || m < 1 || n < 1 || lda < m || n > kSvdMaxN) {
        return;
    }
    const int r = std::min(m, n);
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    for (int j = 0; j < n; ++j) {
        float* cj = group + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        int i = 0;
        // Per-output t-ascending FMA chain identical to GemmNN, with the
        // GroupCenterAdd mean applied as the same single add before the store.
        for (; i + 4 * N <= m; i += 4 * N) {
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            for (int t = 0; t < r; ++t) {
                const float* at = U + static_cast<std::size_t>(t) * static_cast<std::size_t>(m) + i;
                const auto bt = hn::Set(d, Brm[static_cast<std::size_t>(t) * n + j]);
                acc0 = hn::MulAdd(hn::LoadU(d, at), bt, acc0);
                acc1 = hn::MulAdd(hn::LoadU(d, at + N), bt, acc1);
                acc2 = hn::MulAdd(hn::LoadU(d, at + 2 * N), bt, acc2);
                acc3 = hn::MulAdd(hn::LoadU(d, at + 3 * N), bt, acc3);
            }
            hn::StoreU(hn::Add(acc0, hn::LoadU(d, mean + i)), d, cj + i);
            hn::StoreU(hn::Add(acc1, hn::LoadU(d, mean + i + N)), d, cj + i + N);
            hn::StoreU(hn::Add(acc2, hn::LoadU(d, mean + i + 2 * N)), d, cj + i + 2 * N);
            hn::StoreU(hn::Add(acc3, hn::LoadU(d, mean + i + 3 * N)), d, cj + i + 3 * N);
        }
        for (; i + N <= m; i += N) {
            auto acc = hn::Zero(d);
            for (int t = 0; t < r; ++t) {
                acc = hn::MulAdd(hn::LoadU(d, U + static_cast<std::size_t>(t) * static_cast<std::size_t>(m) + i),
                                 hn::Set(d, Brm[static_cast<std::size_t>(t) * n + j]), acc);
            }
            hn::StoreU(hn::Add(acc, hn::LoadU(d, mean + i)), d, cj + i);
        }
        for (; i < m; ++i) {
            float sum = 0.f;
            for (int t = 0; t < r; ++t) {
                sum += U[i + static_cast<std::size_t>(t) * m] * Brm[static_cast<std::size_t>(t) * n + j];
            }
            cj[i] = sum + mean[i];
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(NcsrCentralizeCodes);
HWY_EXPORT(NcsrGroupWeights);
HWY_EXPORT(NcsrFilterGroup);
HWY_EXPORT(NcsrProjectRM);
HWY_EXPORT(NcsrFinishCodesRM);
HWY_EXPORT(NcsrReconstructRM);

void ncsr_centralize_codes(float* B, int r, int n, int ldb, float tau, const float* col_w, const float* row_tau) {
    HWY_DYNAMIC_DISPATCH(NcsrCentralizeCodes)(B, r, n, ldb, tau, col_w, row_tau);
}

float ncsr_group_weights(const float* col_dist, const float* group, int m, int n, int lda, float h, float* weights) {
    return HWY_DYNAMIC_DISPATCH(NcsrGroupWeights)(col_dist, group, m, n, lda, h, weights);
}

int ncsr_filter_group(float* group, int m, int n, int lda, float sigma, const float* col_dist, float* work,
                      int work_floats) {
    return HWY_DYNAMIC_DISPATCH(NcsrFilterGroup)(group, m, n, lda, sigma, col_dist, work, work_floats);
}

void ncsr_project_rm(const float* U, const float* group, int m, int n, int lda, float* Brm) {
    HWY_DYNAMIC_DISPATCH(NcsrProjectRM)(U, group, m, n, lda, Brm);
}

void ncsr_finish_codes_rm(float* Brm, int r, int n, float sigma, const float* col_dist, const float* group, int m,
                          int lda) {
    HWY_DYNAMIC_DISPATCH(NcsrFinishCodesRM)(Brm, r, n, sigma, col_dist, group, m, lda);
}

void ncsr_reconstruct_rm(float* group, int m, int n, int lda, const float* U, const float* Brm, const float* mean) {
    HWY_DYNAMIC_DISPATCH(NcsrReconstructRM)(group, m, n, lda, U, Brm, mean);
}

void ncsr_run_groups(const float* const* refs, const int* rstrides, const float* const* srcs, const int* sstrides,
                     int ntemp, int t0, int width, int height, const SearchConfig& cfg, float sigma, float* num,
                     float* den, float* patches, float* work) {
    if (!refs || !rstrides || !srcs || !sstrides || !num || !den || !patches || width < 1 || height < 1) {
        return;
    }
    const int block = cfg.block;
    const int step = cfg.step;
    const int group = cfg.group;
    if (block < 1 || step < 1 || group < 1) {
        return;
    }
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const int g = std::min(group, kWnnmMaxGroup);
    const int slices = ntemp < 1 ? 1 : ntemp;
    const int t_ref = std::clamp(t0, 0, slices - 1);
    const std::size_t plane_sz = static_cast<std::size_t>(width * height);
    const int shrink_n = ncsr_filter_work_floats(m, g);
    nss::ResourceVector<GroupJob> jobs;
    for (std::int64_t by0 = 0; by0 < static_cast<std::int64_t>(height) - block + step; by0 += step) {
        const int by = static_cast<int>(std::min<std::int64_t>(by0, std::max(0, height - block)));
        for (std::int64_t bx0 = 0; bx0 < static_cast<std::int64_t>(width) - block + step; bx0 += step) {
            const int bx = static_cast<int>(std::min<std::int64_t>(bx0, std::max(0, width - block)));
            jobs.push_back(GroupJob{static_cast<std::uint64_t>(jobs.size()), bx, by, t_ref,
                                    GroupKey{m, g, 1, GroupAlgorithm::NCSR, false, false}});
        }
    }
    // Chunk batch buffers are fully rewritten before every read (pack fills
    // each used patch column; group_center/SVD write the work regions first),
    // so allocate once per call instead of allocating and zeroing ~320 KB per
    // 32-group chunk.
    nss::ResourceVector<float> batch_patches(std::size_t{32} * static_cast<std::size_t>(g) *
                                             static_cast<std::size_t>(lda));
    nss::ResourceVector<float> batch_dist(std::size_t{32} * static_cast<std::size_t>(g));
    nss::ResourceVector<float> batch_work(std::size_t{32} * static_cast<std::size_t>(shrink_n));
    for (std::size_t begin = 0; begin < jobs.size(); begin += 32) {
        const std::size_t end = std::min(jobs.size(), begin + std::size_t{32});
        const int count = static_cast<int>(end - begin);
        std::array<MatchBatchItem, 32> match_items{};
        std::array<int, 32> counts{};
        std::array<Match, 32 * kWnnmMaxGroup> match_storage;
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] = MatchBatchItem{job.x, job.y, block, cfg.bm_range, g, nss::detail::avx2_policy(nss::detail::Avx2Algorithm::NCSR, block, g, cfg.radius, false, 0)};
        }
        const int match_rc = cfg.radius > 0
                                 ? predictive_match_batch(refs, rstrides, ntemp, width, height, t_ref, cfg,
                                                         match_items.data(), count, match_storage.data(),
                                                         kWnnmMaxGroup, counts.data())
                                 : spatial_match_batch(refs[t_ref], rstrides[t_ref], width, height, match_items.data(),
                                                       count, match_storage.data(), kWnnmMaxGroup, counts.data());
        if (match_rc != 0) {
            throw std::runtime_error("nss: matching failed for an active group");
        }
        std::array<int, 32> filter_status{};
        std::array<NcsrFilterBatchItem, 32> filter_items{};
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* p = batch_patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(g) * lda;
            float* dist = batch_dist.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(g);
            for (int j = 0; j < k; ++j) {
                const auto& mm = match_storage[static_cast<std::size_t>(i) * kWnnmMaxGroup + j];
                const int t = cfg.radius > 0 ? mm.t : t_ref;
                pack_patch(p + static_cast<std::size_t>(j) * lda, lda, srcs[t], sstrides[t], mm.x, mm.y, block, width,
                           height);
                dist[j] = mm.dist;
            }
            filter_items[static_cast<std::size_t>(i)] = NcsrFilterBatchItem{
                p, m, k, lda, sigma, dist,
                batch_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(shrink_n), shrink_n,
                &filter_status[static_cast<std::size_t>(i)]};
        }
        if (ncsr_filter_group_batch(filter_items.data(), count) != 0) throw std::runtime_error("nss: numerical group processing failed");
        struct Result {
            bool valid = false;
            int k = 0;
            Match matches[kWnnmMaxGroup]{};
            const float* patches = nullptr;
        };
        OrderedCommitQueue<Result> queue(jobs[begin].ordinal, 32);
        for (int i = 0; i < count; ++i) {
            Result result;
            result.valid = counts[static_cast<std::size_t>(i)] > 0 && filter_status[static_cast<std::size_t>(i)] != 0;
            result.k = counts[static_cast<std::size_t>(i)];
            result.patches = batch_patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(g) * lda;
            for (int j = 0; j < result.k; ++j) {
                result.matches[j] = match_storage[static_cast<std::size_t>(i) * kWnnmMaxGroup + j];
            }
            queue.push(jobs[begin + static_cast<std::size_t>(i)].ordinal, result);
            queue.drain([&](const Result& ready) {
                if (!ready.valid) {
                    return;
                }
                for (int j = 0; j < ready.k; ++j) {
                    int sl = 0;
                    if (cfg.radius > 0) {
                        sl = std::clamp(ready.matches[j].t - t_ref + cfg.radius, 0, slices - 1);
                    }
                    aggregate_add(num + static_cast<std::size_t>(sl) * plane_sz,
                                  den + static_cast<std::size_t>(sl) * plane_sz, width, ready.matches[j].x,
                                  ready.matches[j].y, ready.patches + static_cast<std::size_t>(j) * lda, block, width,
                                  height, 1.f);
                }
            });
        }
        queue.finish([&](const Result& ready) {
            if (!ready.valid) {
                return;
            }
            for (int j = 0; j < ready.k; ++j) {
                int sl = cfg.radius > 0 ? std::clamp(ready.matches[j].t - t_ref + cfg.radius, 0, slices - 1) : 0;
                aggregate_add(num + static_cast<std::size_t>(sl) * plane_sz,
                              den + static_cast<std::size_t>(sl) * plane_sz, width, ready.matches[j].x,
                              ready.matches[j].y, ready.patches + static_cast<std::size_t>(j) * lda, block, width,
                              height, 1.f);
            }
        });
    }
}

void ncsr_denoise_plane(const float* src, int width, int height, int sstride, float* dst, int dstride, int block,
                        int step, int group, int bm_range, float sigma, int iters, float delta, float* work,
                        int work_floats) {
    if (!src || !dst || !work || width < 1 || height < 1 || block < 1 || sstride < width || dstride < width) {
        return;
    }
    const int need = ncsr_denoise_work_floats(width, height, block, group);
    if (work_floats < need) {
        return;
    }
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const int g = group < 1 ? 1 : std::min(group, kWnnmMaxGroup);
    const int niter = iters < 1 ? 1 : iters;
    const std::size_t plane_sz = static_cast<std::size_t>(width * height);
    float* num = work;
    float* den = num + plane_sz;
    float* est = den + plane_sz;
    float* noisy = est + plane_sz;
    float* patches = noisy + plane_sz;
    float* shrink = patches + static_cast<std::size_t>(lda * g);

    for (int y = 0; y < height; ++y) {
        std::memcpy(est + static_cast<std::size_t>(y * width), src + y * sstride,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
    std::memcpy(noisy, est, plane_sz * sizeof(float));

    SearchConfig cfg;
    cfg.block = block;
    cfg.step = step < 1 ? 1 : step;
    cfg.group = g;
    cfg.bm_range = bm_range < 1 ? 1 : bm_range;
    cfg.radius = 0;

    const float* refs[1] = {est};
    const float* srcs[1] = {est};
    int strides[1] = {width};

    for (int iter = 0; iter < niter; ++iter) {
        if (iter > 0) {
            iter_regularize(est, noisy, width * height, delta);
        }
        std::memset(num, 0, plane_sz * sizeof(float));
        std::memset(den, 0, plane_sz * sizeof(float));
        ncsr_run_groups(refs, strides, srcs, strides, 1, 0, width, height, cfg, sigma, num, den, patches, shrink);
        aggregate_finish(est, num, den, est, width, height, width, width);
    }
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + y * dstride, est + static_cast<std::size_t>(y * width),
                    static_cast<std::size_t>(width) * sizeof(float));
    }
}

}  // namespace nss
#endif
