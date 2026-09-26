#include "cpu/lssc/cluster_ssd.hpp"
#include "cpu/hwy_config.hpp"

#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/lssc/cluster_ssd.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Lane-for-lane identical to SsdVec in src/cpu/common/vec.cpp: same ascending
// chunk order, same Sub + MulAdd sequence, same ReduceSum tree, same scalar
// tail. Keep the bodies textually aligned with that kernel.
float LsscSsdPair(const float* a, const float* b, int n) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    auto acc = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto e = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        acc = hn::MulAdd(e, e, acc);
    }
    float s = hn::ReduceSum(d, acc);
    for (; i < n; ++i) {
        const float e = a[i] - b[i];
        s += e * e;
    }
    return s;
}

// Four SSDs against consecutive centroids. Each output keeps the exact
// per-pair arithmetic of LsscSsdPair with an independent accumulator; only
// the patch load and loop scaffolding are shared across the four lanes of
// work. The scalar tail applies to each pair in ascending element order after
// its own ReduceSum, exactly like the single-pair kernel.
void LsscSsd4(const float* a, const float* cent, int n, float* out) {
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const float* c0 = cent;
    const float* c1 = cent + n;
    const float* c2 = c1 + n;
    const float* c3 = c2 + n;
    auto acc0 = hn::Zero(d);
    auto acc1 = hn::Zero(d);
    auto acc2 = hn::Zero(d);
    auto acc3 = hn::Zero(d);
    int i = 0;
    for (; i + N <= n; i += N) {
        const auto av = hn::LoadU(d, a + i);
        const auto e0 = hn::Sub(av, hn::LoadU(d, c0 + i));
        const auto e1 = hn::Sub(av, hn::LoadU(d, c1 + i));
        const auto e2 = hn::Sub(av, hn::LoadU(d, c2 + i));
        const auto e3 = hn::Sub(av, hn::LoadU(d, c3 + i));
        acc0 = hn::MulAdd(e0, e0, acc0);
        acc1 = hn::MulAdd(e1, e1, acc1);
        acc2 = hn::MulAdd(e2, e2, acc2);
        acc3 = hn::MulAdd(e3, e3, acc3);
    }
    float s0 = hn::ReduceSum(d, acc0);
    float s1 = hn::ReduceSum(d, acc1);
    float s2 = hn::ReduceSum(d, acc2);
    float s3 = hn::ReduceSum(d, acc3);
    for (; i < n; ++i) {
        const float av = a[i];
        const float e0 = av - c0[i];
        const float e1 = av - c1[i];
        const float e2 = av - c2[i];
        const float e3 = av - c3[i];
        s0 += e0 * e0;
        s1 += e1 * e1;
        s2 += e2 * e2;
        s3 += e3 * e3;
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

int LsscClusterAssignStep(const float* patches, int m, int n, int lda,
                          const float* cent, int k, int first, int* assign) {
    if (!patches || !cent || !assign || m < 1 || n < 1 || lda < m || k < 1) {
        return 0;
    }
    int changed = 0;
    float d4[4];
    for (int j = 0; j < n; ++j) {
        const float* p = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        int best = 0;
        float best_d = 0.f;
        int c = 0;
        for (; c + 4 <= k; c += 4) {
            LsscSsd4(p, cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m), m, d4);
            for (int r = 0; r < 4; ++r) {
                const int ci = c + r;
                if (ci == 0 || d4[r] < best_d) {
                    best_d = d4[r];
                    best = ci;
                }
            }
        }
        for (; c < k; ++c) {
            const float dv =
                LsscSsdPair(p, cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m), m);
            if (c == 0 || dv < best_d) {
                best_d = dv;
                best = c;
            }
        }
        // Reference short-circuit: at iteration 0 the flag is set without
        // reading the (possibly uninitialized) previous assignment.
        if (first) {
            changed = 1;
        } else if (assign[j] != best) {
            changed = 1;
        }
        assign[j] = best;
    }
    return changed;
}

void LsscClusterAccumStep(const float* patches, int m, int n, int lda,
                          const int* assign, float* cent, int* acc) {
    if (!patches || !assign || !cent || !acc || m < 1 || n < 1 || lda < m) {
        return;
    }
    const hn::ScalableTag<float> d;
    const int N = static_cast<int>(hn::Lanes(d));
    const auto one = hn::Set(d, 1.f);
    for (int j = 0; j < n; ++j) {
        const int c = assign[j];
        float* cj = cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m);
        const float* pj = patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        int i = 0;
        for (; i + N <= m; i += N) {
            hn::StoreU(hn::MulAdd(one, hn::LoadU(d, pj + i), hn::LoadU(d, cj + i)), d, cj + i);
        }
        for (; i < m; ++i) {
            cj[i] += 1.f * pj[i];
        }
        ++acc[c];
    }
}

int LsscClusterFarthest(const float* patches, int m, int n, int lda,
                        const int* assign, const int* acc, const float* cent) {
    if (!patches || !assign || !acc || !cent || m < 1 || n < 1 || lda < m) {
        return -1;
    }
    int steal = -1;
    float steal_d = -1.f;
    for (int j = 0; j < n; ++j) {
        const int oc = assign[j];
        if (acc[oc] <= 1) {
            continue;
        }
        const float dv =
            LsscSsdPair(patches + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda),
                        cent + static_cast<std::size_t>(oc) * static_cast<std::size_t>(m), m);
        if (dv > steal_d) {
            steal_d = dv;
            steal = j;
        }
    }
    return steal;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(LsscSsdPair);
HWY_EXPORT(LsscSsd4);
HWY_EXPORT(LsscClusterAssignStep);
HWY_EXPORT(LsscClusterAccumStep);
HWY_EXPORT(LsscClusterFarthest);

float lssc_ssd_pair(const float* a, const float* b, int n) {
    if (!a || !b || n < 1) {
        return 0.f;
    }
    return HWY_DYNAMIC_DISPATCH(LsscSsdPair)(a, b, n);
}

void lssc_ssd4(const float* a, const float* cent, int n, float* out) {
    if (!a || !cent || !out || n < 1) {
        return;
    }
    HWY_DYNAMIC_DISPATCH(LsscSsd4)(a, cent, n, out);
}

int lssc_cluster_assign_step(const float* patches, int m, int n, int lda,
                             const float* cent, int k, int first, int* assign) {
    return HWY_DYNAMIC_DISPATCH(LsscClusterAssignStep)(patches, m, n, lda, cent, k, first, assign);
}

void lssc_cluster_accum_step(const float* patches, int m, int n, int lda,
                             const int* assign, float* cent, int* acc) {
    HWY_DYNAMIC_DISPATCH(LsscClusterAccumStep)(patches, m, n, lda, assign, cent, acc);
}

int lssc_cluster_farthest(const float* patches, int m, int n, int lda,
                          const int* assign, const int* acc, const float* cent) {
    return HWY_DYNAMIC_DISPATCH(LsscClusterFarthest)(patches, m, n, lda, assign, acc, cent);
}

}  // namespace nss
#endif
