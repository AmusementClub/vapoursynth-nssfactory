// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_common.hpp"
#include "cpu/lssc/cluster_ssd.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace nss {

void lssc_cluster(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts) {
    if (!patches || !assign || m < 1 || n < 1 || lda < m) {
        return;
    }
    const int need = lssc_cluster_work_floats(m, n, nclusters);
    nss::ResourceVector<float> scratch(static_cast<std::size_t>(need), 0.f);
    nss::ResourceVector<int> count_fallback;
    int* count_out = counts;
    if (!count_out) {
        count_fallback.assign(static_cast<std::size_t>(std::max(1, nclusters)), 0);
        count_out = count_fallback.data();
    }
    lssc_cluster_workspace(patches, m, n, lda, nclusters, assign, count_out, scratch.data(), need);
}

void lssc_cluster_workspace(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts,
                            float* scratch, int scratch_floats) {
    if (!patches || !assign || !counts || !scratch || m < 1 || n < 1 || lda < m ||
        scratch_floats < lssc_cluster_work_floats(m, n, nclusters)) {
        return;
    }
    int k = nclusters;
    if (k < 1) {
        k = 1;
    }
    if (k > n) {
        k = n;
    }
    float* cent = scratch;
    // Keep the accumulator in the same caller-owned workspace. This avoids
    // the per-frame vector allocation that the legacy convenience wrapper
    // intentionally retains for standalone callers.
    int* acc = reinterpret_cast<int*>(scratch + static_cast<std::size_t>(k) * static_cast<std::size_t>(m));
    for (int c = 0; c < nclusters; ++c) {
        counts[c] = 0;
    }
    for (int c = 0; c < k; ++c) {
        acc[c] = 0;
    }

    if (k == 1) {
        for (int i = 0; i < n; ++i) {
            assign[i] = 0;
        }
        counts[0] = n;
        return;
    }

    std::fill(cent, cent + static_cast<std::size_t>(k) * static_cast<std::size_t>(m), 0.f);
    for (int c = 0; c < k; ++c) {
        const int src = (c * n) / k;
        std::memcpy(cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m),
                    patches + static_cast<std::size_t>(src) * static_cast<std::size_t>(lda),
                    static_cast<std::size_t>(m) * sizeof(float));
    }

    constexpr int kIters = 8;
    for (int iter = 0; iter < kIters; ++iter) {
        // Batched step kernels: each per-pair SSD keeps the exact SsdVec
        // reduction (see cluster_ssd.hpp), so assignments, the changed flag,
        // and steal decisions are bit-identical to the reference per-pair
        // loop; the Highway target is resolved once per step, not per pair.
        bool changed = lssc_cluster_assign_step(patches, m, n, lda, cent, k, iter == 0 ? 1 : 0, assign) != 0;

        std::fill(cent, cent + static_cast<std::size_t>(k) * static_cast<std::size_t>(m), 0.f);
        std::fill(acc, acc + k, 0);
        lssc_cluster_accum_step(patches, m, n, lda, assign, cent, acc);
        for (int c = 0; c < k; ++c) {
            if (acc[c] < 1) {
                continue;
            }
            const float inv = 1.f / static_cast<float>(acc[static_cast<std::size_t>(c)]);
            float* cj = cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m);
            scale_n(cj, inv, m);
        }
        for (int c = 0; c < k; ++c) {
            if (acc[static_cast<std::size_t>(c)] > 0) {
                continue;
            }
            int steal = lssc_cluster_farthest(patches, m, n, lda, assign, acc, cent);
            if (steal < 0) {
                steal = c % n;
            }
            const int oc = assign[steal];
            if (oc != c && acc[oc] > 0) {
                --acc[oc];
            }
            assign[steal] = c;
            acc[c] = 1;
            std::memcpy(cent + static_cast<std::size_t>(c) * static_cast<std::size_t>(m),
                        patches + static_cast<std::size_t>(steal) * static_cast<std::size_t>(lda),
                        static_cast<std::size_t>(m) * sizeof(float));
            changed = true;
        }
        if (!changed && iter > 0) {
            break;
        }
    }

    for (int c = 0; c < nclusters; ++c) {
        counts[c] = 0;
    }
    for (int j = 0; j < n; ++j) {
        const int c = assign[j];
        if (c >= 0 && c < nclusters) {
            ++counts[c];
        } else {
            assign[j] = 0;
            ++counts[0];
        }
    }
}

}  // namespace nss
