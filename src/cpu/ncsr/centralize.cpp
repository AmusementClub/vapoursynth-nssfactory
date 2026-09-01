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
    if (!(sigma > 0.f) || !std::isfinite(sigma)) {
        pca_reconstruct(group, m, n, lda, U, B, mean);
        return 0;
    }
    float w[kSvdMaxN];
    constexpr float kEps = 1e-12f;
    // Match.dist is total SSD; h = 2 m σ² so a noise-only pair has w ~ exp(-1/2).
    const float h = std::max(2.f * static_cast<float>(m) * sigma * sigma, kEps);
    float wsum = 0.f;
    for (int j = 0; j < n; ++j) {
        float dist = 0.f;
        if (col_dist) {
            dist = col_dist[j];
        } else if (j > 0) {
            dist = ssd_vec(group + j * lda, group, m);
        }
        w[j] = std::exp(-dist / h);
        wsum += w[j];
    }
    if (!(wsum > 0.f)) {
        for (int j = 0; j < n; ++j) {
            w[j] = 1.f;
        }
        wsum = static_cast<float>(n);
    }
    // TIP 2013 (17): τ_i = 2√2 σ² / σ_θ, σ_θ = weighted std of (α − β) on row i.
    constexpr float kMap = 2.8284271247461903f;
    float row_tau[kSvdMaxN];
    const float sig2 = sigma * sigma;
    const float inv_w = 1.f / wsum;
    float row[kSvdMaxN];
    for (int i = 0; i < r; ++i) {
        GatherStrided(row, B + i, r, n);
        float s = 0.f;
        for (int j = 0; j < n; ++j) {
            s += w[j] * row[j];
        }
        const float beta = s * inv_w;
        float var = 0.f;
        for (int j = 0; j < n; ++j) {
            const float e = row[j] - beta;
            var += w[j] * e * e;
        }
        const float sig_theta = std::sqrt(var * inv_w);
        row_tau[i] = kMap * sig2 / (sig_theta + kEps);
    }
    NcsrCentralizeCodes(B, r, n, r, std::fabs(sigma), w, row_tau);
    pca_reconstruct(group, m, n, lda, U, B, mean);
    return 0;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(NcsrCentralizeCodes);
HWY_EXPORT(NcsrFilterGroup);

void ncsr_centralize_codes(float* B, int r, int n, int ldb, float tau, const float* col_w, const float* row_tau) {
    HWY_DYNAMIC_DISPATCH(NcsrCentralizeCodes)(B, r, n, ldb, tau, col_w, row_tau);
}

int ncsr_filter_group(float* group, int m, int n, int lda, float sigma, const float* col_dist, float* work,
                      int work_floats) {
    return HWY_DYNAMIC_DISPATCH(NcsrFilterGroup)(group, m, n, lda, sigma, col_dist, work, work_floats);
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
    std::vector<GroupJob> jobs;
    for (int by0 = 0; by0 < height - block + step; by0 += step) {
        const int by = std::min(by0, std::max(0, height - block));
        for (int bx0 = 0; bx0 < width - block + step; bx0 += step) {
            const int bx = std::min(bx0, std::max(0, width - block));
            jobs.push_back(GroupJob{static_cast<std::uint64_t>(jobs.size()), bx, by, t_ref,
                                    GroupKey{m, g, 1, GroupAlgorithm::NCSR, false, false}});
        }
    }
    for (std::size_t begin = 0; begin < jobs.size(); begin += 32) {
        const std::size_t end = std::min(jobs.size(), begin + std::size_t{32});
        const int count = static_cast<int>(end - begin);
        std::array<MatchBatchItem, 32> match_items{};
        std::array<int, 32> counts{};
        std::array<Match, 32 * kWnnmMaxGroup> match_storage{};
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] = MatchBatchItem{job.x, job.y, block, cfg.bm_range, g};
        }
        const int match_rc = cfg.radius > 0
                                 ? predictive_match_batch(refs, rstrides, ntemp, width, height, t_ref, cfg,
                                                         match_items.data(), count, match_storage.data(),
                                                         kWnnmMaxGroup, counts.data())
                                 : spatial_match_batch(refs[t_ref], rstrides[t_ref], width, height, match_items.data(),
                                                       count, match_storage.data(), kWnnmMaxGroup, counts.data());
        if (match_rc < 0) {
            continue;
        }
        std::vector<float> batch_patches(static_cast<std::size_t>(count) * static_cast<std::size_t>(g) * lda, 0.f);
        std::vector<float> batch_dist(static_cast<std::size_t>(count) * static_cast<std::size_t>(g), 0.f);
        std::vector<float> batch_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(shrink_n), 0.f);
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
        (void)ncsr_filter_group_batch(filter_items.data(), count);
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
