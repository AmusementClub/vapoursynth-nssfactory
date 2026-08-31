#include "nss/cpu_nlh.hpp"
#include "cpu/hwy_config.hpp"

#include <algorithm>
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
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    float dist_s[256];
    std::vector<float> dist_v;
    float* dist = dist_s;
    if (m > 256) {
        dist_v.assign(static_cast<std::size_t>(m), 0.f);
        dist = dist_v.data();
    }
    constexpr float kInf = 1.0e30f;
    using V = hn::Vec<hn::ScalableTag<float>>;
    for (int r = 0; r < m; ++r) {
        int s0 = 0;
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
            hn::StoreU(acc0, d, dist + s0);
            hn::StoreU(acc1, d, dist + s0 + N);
            hn::StoreU(acc2, d, dist + s0 + 2 * N);
            hn::StoreU(acc3, d, dist + s0 + 3 * N);
        }
        for (; s0 + N <= m; s0 += N) {
            V acc = hn::Zero(d);
            for (int c = 0; c < n; ++c) {
                const auto e = hn::Sub(hn::LoadU(d, group + c * lda + s0), hn::Set(d, group[r + c * lda]));
                acc = hn::MulAdd(e, e, acc);
            }
            hn::StoreU(acc, d, dist + s0);
        }
        if (s0 < m) {
            for (int s = s0; s < m; ++s) {
                dist[s] = 0.f;
            }
            for (int c = 0; c < n; ++c) {
                const float vr = group[r + c * lda];
                const float* col = group + c * lda;
                for (int s = s0; s < m; ++s) {
                    const float e = col[s] - vr;
                    dist[s] += e * e;
                }
            }
        }
        dist[r] = kInf;
        idx[r * q] = r;
        int j = 1;
        const int need = qq - 1;
        for (; j <= need; ++j) {
            int bi = r;
            float bd = kInf;
            for (int s = 0; s < m; ++s) {
                const float ds = dist[s];
                if (ds < bd) {
                    bd = ds;
                    bi = s;
                }
            }
            if (bd >= kInf * 0.5f) {
                idx[r * q + j] = r;
            } else {
                idx[r * q + j] = bi;
                dist[bi] = kInf;
            }
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
