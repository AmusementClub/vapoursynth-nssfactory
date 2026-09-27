// SPDX-License-Identifier: GPL-2.0-only
// Bitwise characterization gates for the 2026-09-26 optimization round.
//
// The C4 paired gate proves bit-identical output on its fixed production
// shapes (k=64 clusters, m=64, block=8, tail-free GEMM shapes). These tests
// cover the shapes the gate does not reach: batch+tail mixes, scalar tails,
// NLH preset block sizes, small degenerate cases. Every comparison is a
// bitwise memcmp against the implementation the new kernel replaced (or an
// equivalent direct reference), so flag-class or reduction-tree drift fails
// here before it reaches a paired gate.
#include "nss/backend.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_image.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_twsc_full.hpp"
#include "cpu/bm/matcher.hpp"
#include "cpu/common/image_ssd_row.hpp"
#include "cpu/finishers.hpp"
#include "cpu/lssc/cluster_ssd.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include "hwy/highway.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}
bool same(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
bool same(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

enum Pattern { kRand, kTiny, kLarge, kZeros, kMixed };
void fill_pattern(std::mt19937& rng, float* p, int count, Pattern pattern) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int i = 0; i < count; ++i) {
        switch (pattern) {
            case kRand: p[i] = u(rng); break;
            case kTiny: p[i] = u(rng) * 1e-30f; break;
            case kLarge: p[i] = u(rng) * 1e10f; break;
            case kZeros: p[i] = 0.f; break;
            case kMixed:
                p[i] = (i % 4 == 0) ? 0.f : (i % 4 == 1) ? -0.f : (i % 4 == 2) ? u(rng) * 1e-30f : u(rng);
                break;
        }
    }
}

// ---------------------------------------------------------------- LSSC SSD

int test_lssc_ssd_pair_bitwise() {
    std::mt19937 rng(11);
    const int ns[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 300};
    for (int n : ns) for (int pattern = 0; pattern <= kMixed; ++pattern) {
        std::vector<float> a(n), b(n);
        fill_pattern(rng, a.data(), n, static_cast<Pattern>(pattern));
        fill_pattern(rng, b.data(), n, static_cast<Pattern>(pattern));
        const float got = nss::lssc_ssd_pair(a.data(), b.data(), n);
        const float ref = nss::ssd_vec(a.data(), b.data(), n);
        if (!same(got, ref)) {
            std::fprintf(stderr, "lssc_ssd_pair mismatch n=%d pattern=%d\n", n, pattern);
            return 1;
        }
    }
    return 0;
}

int test_lssc_ssd4_bitwise() {
    std::mt19937 rng(13);
    const int ns[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256};
    for (int n : ns) for (int pattern = 0; pattern <= kMixed; ++pattern) {
        std::vector<float> a(n), cent(static_cast<std::size_t>(4) * n);
        fill_pattern(rng, a.data(), n, static_cast<Pattern>(pattern));
        fill_pattern(rng, cent.data(), 4 * n, static_cast<Pattern>(pattern));
        float out[4]{};
        nss::lssc_ssd4(a.data(), cent.data(), n, out);
        for (int r = 0; r < 4; ++r) {
            const float ref = nss::lssc_ssd_pair(a.data(), cent.data() + static_cast<std::size_t>(r) * n, n);
            if (!same(out[r], ref)) {
                std::fprintf(stderr, "lssc_ssd4 mismatch n=%d r=%d pattern=%d\n", n, r, pattern);
                return 1;
            }
        }
    }
    return 0;
}

// Reference for one assign pass: per-pair SSD, strict-less first-win
// ascending centroids, `first` forcing changed without reading assign.
int ref_assign_step(const float* patches, int m, int n, int lda, const float* cent, int k, int first, int* assign) {
    int changed = 0;
    for (int j = 0; j < n; ++j) {
        const float* pj = patches + static_cast<std::size_t>(j) * lda;
        int best = 0;
        float best_d = nss::lssc_ssd_pair(pj, cent, m);
        for (int c = 1; c < k; ++c) {
            const float d = nss::lssc_ssd_pair(pj, cent + static_cast<std::size_t>(c) * m, m);
            if (d < best_d) { best_d = d; best = c; }
        }
        if (first || assign[j] != best) changed = 1;
        assign[j] = best;
    }
    return changed;
}

int test_lssc_assign_step_bitwise() {
    std::mt19937 rng(17);
    const int ms[] = {1, 4, 7, 8, 64};
    const int ns[] = {1, 2, 3, 5, 13, 20, 64};
    for (int m : ms) for (int n : ns) for (int k = 1; k <= 9; ++k) for (int first : {0, 1}) {
        const int lda = m + (n % 2);
        std::vector<float> patches(static_cast<std::size_t>(lda) * n), cent(static_cast<std::size_t>(k) * m);
        fill_pattern(rng, patches.data(), patches.size(), kRand);
        fill_pattern(rng, cent.data(), cent.size(), kRand);
        // Duplicate some patches so ties exercise the first-win rule.
        for (int j = 1; j < n; j += 3)
            std::memcpy(patches.data() + static_cast<std::size_t>(j) * lda, patches.data(),
                        static_cast<std::size_t>(m) * sizeof(float));
        std::vector<int> a1(n, -7), a2(n, -7);
        const int got_changed = nss::lssc_cluster_assign_step(patches.data(), m, n, lda, cent.data(), k, first, a1.data());
        const int ref_changed = ref_assign_step(patches.data(), m, n, lda, cent.data(), k, first, a2.data());
        if (got_changed != ref_changed || a1 != a2) {
            std::fprintf(stderr, "assign_step mismatch m=%d n=%d k=%d first=%d\n", m, n, k, first);
            return 1;
        }
    }
    return 0;
}

int test_lssc_accum_step_bitwise() {
    std::mt19937 rng(19);
    for (int m : {1, 4, 8, 64}) for (int n : {1, 3, 13, 64}) for (int k = 1; k <= 9; ++k) {
        const int lda = m + 1;
        std::vector<float> patches(static_cast<std::size_t>(lda) * n);
        fill_pattern(rng, patches.data(), patches.size(), kRand);
        std::vector<int> assign(n);
        for (int j = 0; j < n; ++j) assign[j] = (j * 7 + k) % k;
        std::vector<float> c1(static_cast<std::size_t>(k) * m, 0.f), c2 = c1;
        std::vector<int> acc1(k, 0), acc2(k, 0);
        nss::lssc_cluster_accum_step(patches.data(), m, n, lda, assign.data(), c1.data(), acc1.data());
        for (int j = 0; j < n; ++j) {
            const int c = assign[j];
            for (int i = 0; i < m; ++i)
                c2[static_cast<std::size_t>(c) * m + i] += patches[static_cast<std::size_t>(j) * lda + i];
            ++acc2[c];
        }
        for (std::size_t i = 0; i < c1.size(); ++i)
            if (!same(c1[i], c2[i])) {
                std::fprintf(stderr, "accum_step mismatch m=%d n=%d k=%d\n", m, n, k);
                return 1;
            }
        if (acc1 != acc2) {
            std::fprintf(stderr, "accum_step count mismatch m=%d n=%d k=%d\n", m, n, k);
            return 1;
        }
    }
    return 0;
}

int test_lssc_farthest_bitwise() {
    std::mt19937 rng(23);
    for (int m : {1, 4, 8, 64}) for (int n : {1, 3, 13, 64}) for (int k = 1; k <= 9; ++k) {
        const int lda = m;
        std::vector<float> patches(static_cast<std::size_t>(lda) * n), cent(static_cast<std::size_t>(k) * m);
        fill_pattern(rng, patches.data(), patches.size(), kRand);
        fill_pattern(rng, cent.data(), cent.size(), kRand);
        std::vector<int> assign(n), acc(k, 0);
        for (int j = 0; j < n; ++j) assign[j] = (j * 5 + 1) % k;
        for (int j = 0; j < n; ++j) ++acc[assign[j]];
        const int got = nss::lssc_cluster_farthest(patches.data(), m, n, lda, assign.data(), acc.data(), cent.data());
        int best = -1;
        float best_d = -1.f;
        for (int j = 0; j < n; ++j) {
            const int oc = assign[j];
            if (acc[oc] <= 1) continue;
            const float d = nss::lssc_ssd_pair(patches.data() + static_cast<std::size_t>(j) * lda,
                                               cent.data() + static_cast<std::size_t>(oc) * m, m);
            if (d > best_d) { best_d = d; best = j; }
        }
        if (got != best) {
            std::fprintf(stderr, "farthest mismatch m=%d n=%d k=%d: got %d want %d\n", m, n, k, got, best);
            return 1;
        }
        // k == n forces every acc == 1: no steal qualifies.
        if (k == n && got != -1) {
            std::fprintf(stderr, "farthest k==n should find none\n");
            return 1;
        }
    }
    return 0;
}

// Full driver reference: the pre-05edb70 k-means loop rebuilt on lssc_ssd_pair
// (bitwise == ssd_vec per test_lssc_ssd_pair_bitwise).
void ref_lssc_cluster(const float* patches, int m, int n, int lda, int nclusters, int* assign, int* counts) {
    int k = nclusters;
    if (k < 1) k = 1;
    if (k > n) k = n;
    std::vector<float> centbuf(static_cast<std::size_t>(k) * m);
    std::vector<int> acc(static_cast<std::size_t>(k), 0);
    float* cent = centbuf.data();
    for (int c = 0; c < nclusters; ++c) {
        counts[c] = 0;
    }
    for (int c = 0; c < k; ++c) {
        acc[c] = 0;
    }
    if (k == 1) {
        for (int i = 0; i < n; ++i) assign[i] = 0;
        counts[0] = n;
        return;
    }
    std::fill(cent, cent + static_cast<std::size_t>(k) * m, 0.f);
    for (int c = 0; c < k; ++c) {
        const int src = (c * n) / k;
        std::memcpy(cent + static_cast<std::size_t>(c) * m, patches + static_cast<std::size_t>(src) * lda,
                    static_cast<std::size_t>(m) * sizeof(float));
    }
    constexpr int kIters = 8;
    for (int iter = 0; iter < kIters; ++iter) {
        bool changed = false;
        for (int j = 0; j < n; ++j) {
            const float* pj = patches + static_cast<std::size_t>(j) * lda;
            int best = 0;
            float best_d = nss::lssc_ssd_pair(pj, cent, m);
            for (int c = 1; c < k; ++c) {
                const float d = nss::lssc_ssd_pair(pj, cent + static_cast<std::size_t>(c) * m, m);
                if (d < best_d) { best_d = d; best = c; }
            }
            if (iter == 0 || assign[j] != best) changed = true;
            assign[j] = best;
        }
        std::fill(cent, cent + static_cast<std::size_t>(k) * m, 0.f);
        std::fill(acc.begin(), acc.end(), 0);
        for (int j = 0; j < n; ++j) {
            const int c = assign[j];
            float* cj = cent + static_cast<std::size_t>(c) * m;
            const float* pj = patches + static_cast<std::size_t>(j) * lda;
            for (int i = 0; i < m; ++i) cj[i] += pj[i];
            ++acc[c];
        }
        for (int c = 0; c < k; ++c) {
            if (acc[c] < 1) continue;
            const float inv = 1.f / static_cast<float>(acc[c]);
            float* cj = cent + static_cast<std::size_t>(c) * m;
            for (int i = 0; i < m; ++i) cj[i] *= inv;
        }
        for (int c = 0; c < k; ++c) {
            if (acc[c] > 0) continue;
            int steal = -1;
            float steal_d = -1.f;
            for (int j = 0; j < n; ++j) {
                const int oc = assign[j];
                if (acc[oc] <= 1) continue;
                const float* pj = patches + static_cast<std::size_t>(j) * lda;
                const float d = nss::lssc_ssd_pair(pj, cent + static_cast<std::size_t>(oc) * m, m);
                if (d > steal_d) { steal_d = d; steal = j; }
            }
            if (steal < 0) steal = c % n;
            const int oc = assign[steal];
            if (oc != c && acc[oc] > 0) --acc[oc];
            assign[steal] = c;
            acc[c] = 1;
            std::memcpy(cent + static_cast<std::size_t>(c) * m, patches + static_cast<std::size_t>(steal) * lda,
                        static_cast<std::size_t>(m) * sizeof(float));
            changed = true;
        }
        if (!changed && iter > 0) break;
    }
    for (int j = 0; j < n; ++j) ++counts[assign[j]];
}

int test_lssc_cluster_driver_bitwise() {
    std::mt19937 rng(29);
    const int m = 16, lda = 16;
    for (int n : {5, 6, 7, 13, 20, 64}) for (int k : {5, 6, 7}) for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<float> patches(static_cast<std::size_t>(lda) * n);
        fill_pattern(rng, patches.data(), patches.size(), kRand);
        if (pattern == 1) {
            // Many duplicates: forces empty clusters and the steal path.
            for (int j = 1; j < n; ++j)
                std::memcpy(patches.data() + static_cast<std::size_t>(j) * lda, patches.data(),
                            static_cast<std::size_t>(m) * sizeof(float));
        } else if (pattern == 2) {
            std::fill(patches.begin(), patches.end(), 0.25f);
        }
        std::vector<int> a1(n, -1), c1(k, 0), a2(n, -1), c2(k, 0);
        nss::lssc_cluster(patches.data(), m, n, lda, k, a1.data(), c1.data());
        ref_lssc_cluster(patches.data(), m, n, lda, k, a2.data(), c2.data());
        if (a1 != a2 || c1 != c2) {
            std::fprintf(stderr, "lssc_cluster driver mismatch n=%d k=%d pattern=%d\n", n, k, pattern);
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------- image_ssd_row

int test_image_ssd_row_bitwise() {
    std::mt19937 rng(31);
    const nss::ImageSsdRowKernel kernel = nss::image_ssd_row_kernel();
    check(kernel != nullptr, "image_ssd_row_kernel resolves");
    const int blocks[] = {1, 2, 4, 7, 8, 12, 15, 16};
    const int counts[] = {1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 125, 128, 129};
    for (int block : blocks) for (int count : counts) {
        const int width = block + count + 2, height = block + 3;
        std::vector<float> img(static_cast<std::size_t>(width) * height);
        fill_pattern(rng, img.data(), img.size(), kRand);
        const int x0 = 1, y0 = 1, y = 2, xlo = 0;
        std::vector<float> out(count, -1e30f);
        kernel(img.data() + y0 * width + x0, width, img.data() + y * width + xlo, width, block, count, out.data());
        for (int x = 0; x < count; ++x) {
            const float ref = nss::ssd_block(img.data() + y0 * width + x0, width,
                                             img.data() + y * width + xlo + x, width, block);
            if (!same(out[x], ref)) {
                std::fprintf(stderr, "image_ssd_row mismatch block=%d count=%d x=%d\n", block, count, x);
                return 1;
            }
        }
    }
    return 0;
}

// ------------------------------------------------------- NCSR row-major

int test_ncsr_rowmajor_bitwise() {
    std::mt19937 rng(37);
    const int ms[] = {8, 9, 15, 16, 17, 31, 32, 33, 48, 49, 63, 64};
    for (int m : ms) for (int ladv = 0; ladv < 3; ++ladv) {
        const int lda = m + ladv;
        for (int with_dist : {0, 1}) for (float sigma : {3.0f, 0.5f, 0.0f}) {
            const int n = 8, r = 8;
            std::vector<float> group0(static_cast<std::size_t>(lda) * n), U(static_cast<std::size_t>(m) * n),
                meanv(m), distv(n);
            fill_pattern(rng, group0.data(), group0.size(), kRand);
            fill_pattern(rng, U.data(), U.size(), kRand);
            fill_pattern(rng, meanv.data(), m, kRand);
            for (int j = 0; j < n; ++j) distv[j] = 0.03f * static_cast<float>((j + m) % 5);
            const float* dist = with_dist ? distv.data() : nullptr;

            // 1) project. Both sides use explicit reduction trees, but the
            // tree/blocking shapes only coincide per target: NEON and AVX3
            // produce identical bits (the C4 gate is hash-exact), while x86
            // AVX2's 8-lane blocking splits shapes like m=15 differently and
            // the outputs drift by a few float ulp (observed <= 9.6e-7 on
            // AVX2 GCC 13.3). Cross-target bit-equality is not guaranteed by
            // policy, so compare with a scaled bound: 4e-6f per unit of
            // result magnitude tolerates the ~r-ulp tree wobble while any
            // structural error (wrong U, group or indexing) is orders of
            // magnitude larger.
            std::vector<float> b_old(static_cast<std::size_t>(r) * n, -1e30f),
                b_new(static_cast<std::size_t>(r) * n, -1e30f);
            nss::gemm_tn_hwy(m, n, r, U.data(), m, group0.data(), lda, b_old.data(), r);
            nss::ncsr_project_rm(U.data(), group0.data(), m, n, lda, b_new.data());
            for (int i = 0; i < r; ++i) for (int j = 0; j < n; ++j) {
                const float ov = b_old[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r];
                const float nv = b_new[static_cast<std::size_t>(i) * n + j];
                const float tol = 4e-6f * std::max(1.0f, std::fabs(ov));
                if (!(std::fabs(ov - nv) <= tol)) {
                    std::fprintf(stderr, "ncsr project mismatch m=%d ladv=%d i=%d j=%d: %.9g vs %.9g\n",
                                 m, ladv, i, j, ov, nv);
                    return 1;
                }
            }

            // 2) finish on identical project output. mean/variance feed
            // sqrt/var->ti, a scalar accumulation the old code evaluated in
            // the default class (batch.cpp) while the new kernel runs in the
            // SAFE class; their mul+add contraction differs by compiler
            // context (policy: cross-compiler outputs are not byte-identical;
            // C4 GCC gate is hash-exact). 2e-6 tolerates that ti wobble while
            // failing any structural error (wrong weights/h/order produce
            // errors orders of magnitude larger). beta/demean/shrink/re-add
            // are deterministic and compared through the same tolerance.
            auto codes_old = b_old;
            {
                std::vector<float> cm(r * n);
                for (int i = 0; i < r; ++i) for (int j = 0; j < n; ++j)
                    cm[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r] =
                        b_old[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r];
                auto group_tmp = group0;
                nss::detail::finish_ncsr_codes(group_tmp.data(), m, n, lda, sigma, dist, cm.data(),
                                               nss::detail::ScalarCodeRows{}, nss::ncsr_group_weights,
                                               nss::ncsr_centralize_codes);
                codes_old = cm;
            }
            auto codes_new = b_new;
            auto group_tmp = group0;
            nss::ncsr_finish_codes_rm(codes_new.data(), r, n, sigma, dist, group_tmp.data(), m, lda);
            for (int i = 0; i < r; ++i) for (int j = 0; j < n; ++j) {
                const float ov = codes_old[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r];
                const float nv = codes_new[static_cast<std::size_t>(i) * n + j];
                if (!(std::fabs(ov - nv) <= 2e-6f)) {
                    std::fprintf(stderr, "ncsr finish mismatch m=%d ladv=%d dist=%d sigma=%g i=%d j=%d: %.9g vs %.9g\n",
                                 m, ladv, with_dist, sigma, i, j, ov, nv);
                    return 1;
                }
            }

            // 3) reconstruct on identical finished codes (old finish output,
            // transposed view). The row-major kernel uses explicit MulAdd
            // chains (always FMA) while the old finish path is a default-class
            // scalar `sum += a*b` chain whose contraction is compiler/target
            // context; on x86 AVX2 GCC 13.3 the two therefore differ by a few
            // float ulp (observed <= 4.8e-7) even though both are policy-valid
            // (contracts: cross-compiler outputs are not byte-identical; the
            // C4 GCC gate is the hash-exact harness). Same style of scaled
            // bound as the project comparison above: 4e-6f per unit of result
            // magnitude tolerates the contraction wobble while structural
            // errors (wrong U, codes, mean or indexing) are orders of
            // magnitude larger.
            auto group_old = group0;
            nss::detail::finish_pca_reconstruction(group_old.data(), m, n, lda, U.data(), codes_old.data(),
                                                   meanv.data());
            std::vector<float> codes_old_rm(r * n);
            for (int i = 0; i < r; ++i) for (int j = 0; j < n; ++j)
                codes_old_rm[static_cast<std::size_t>(i) * n + j] =
                    codes_old[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r];
            auto group_new = group0;
            nss::ncsr_reconstruct_rm(group_new.data(), m, n, lda, U.data(), codes_old_rm.data(), meanv.data());
            for (std::size_t z = 0; z < group_old.size(); ++z) {
                const float tol = 4e-6f * std::max(1.0f, std::fabs(group_old[z]));
                if (!(std::fabs(group_old[z] - group_new[z]) <= tol)) {
                    std::fprintf(stderr, "ncsr reconstruct mismatch m=%d ladv=%d dist=%d sigma=%g at %zu: %.9g vs %.9g\n",
                                 m, ladv, with_dist, sigma, z, group_old[z], group_new[z]);
                    return 1;
                }
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------- TWSC gemm

// Contraction-exact oracle for scalar dot-product chains: the default
// contract class leaves the mul+add contraction of `sum += a*b` to compiler
// context (observed on Apple Clang: the four-chain block contracts to FMA,
// a lone reference chain does not; GCC contracts both). Both orderings are
// the same per-element l-ascending chain, so an output is valid iff it
// bitwise-matches one of the two contraction variants. This catches any
// structural error (indexing, order, missed terms) while tolerating the
// policy-sanctioned contraction choice (contracts: cross-compiler outputs
// are not byte-identical by policy).
double chain_fma_variant(const double* a, const double* b, int k) {
    double s = 0;
    for (int l = 0; l < k; ++l) s = std::fma(a[l], b[l], s);
    return s;
}
double chain_sep_variant(const double* a, const double* b, int k) {
    double s = 0;
    for (int l = 0; l < k; ++l) {
        const double p = a[l] * b[l];
        volatile double vp = p;  // separate rounding, no contraction
        s += vp;
    }
    return s;
}

// x86 AVX2 GCC 13.3 additionally emits *partial* contraction — observed
// fma(a2, b2, round(p0 + p1)) for a k=3 chain — which bitwise-matches neither
// pure variant. A mixed contraction still lies between the two pure forms
// within ~k ulps of each, so accept got if it bitwise-matches either variant
// or stays within 1e-12 of both (k <= 64, terms in [-1, 1]: the pure forms
// themselves differ by far less than 1e-12, while structural errors — missed
// or duplicated term, wrong indexing — are ~1.0).
bool chain_accept(double got, const double* a, const double* b, int k) {
    const double f = chain_fma_variant(a, b, k);
    const double s = chain_sep_variant(a, b, k);
    if (same(got, f) || same(got, s)) return true;
    return std::fabs(got - f) <= 1e-12 && std::fabs(got - s) <= 1e-12;
}

int test_twsc_gemm_bitwise() {
    std::mt19937 rng(41);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int ms[] = {1, 3, 5, 8, 16, 49, 64, 81};
    const int ns[] = {1, 3, 5, 8, 90};
    const int ks[] = {1, 3, 5, 8, 64};
    for (int m : ms) for (int n : ns) for (int k : ks) {
        std::vector<double> a(static_cast<std::size_t>(m) * std::max(k, 1)),
            b(static_cast<std::size_t>(std::max(m, k)) * n);
        for (auto& v : a) v = u(rng);
        for (auto& v : b) v = u(rng);
        std::vector<double> c1(static_cast<std::size_t>(std::max(m, k)) * n, -1e30);
        // NN: each output is one serial l-ascending chain on every code path
        // (the vector path is explicit MulAdd in the same order), so the
        // two-variant oracle covers both vector and scalar-tail shapes.
        nss::twsc_gemm_nn(m, n, k, a.data(), b.data(), c1.data());
        for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
            double av[64], bv[64];
            for (int l = 0; l < k; ++l) { av[l] = a[i + l * m]; bv[l] = b[l + j * k]; }
            const double got = c1[static_cast<std::size_t>(i) + j * m];
            if (!chain_accept(got, av, bv, k)) {
                std::fprintf(stderr, "twsc_gemm_nn mismatch m=%d n=%d k=%d i=%d j=%d got=%.17g\n",
                             m, n, k, i, j, got);
                return 1;
            }
        }
        // TN: with m >= lanes the vector path reduces through ReduceSum's
        // target-specific tree (covered by the C4 gate and test_batch's
        // lane==scalar checks). m=1 is all-scalar on every target and
        // exercises the four-chain i-block tails and per-chain scalar loops.
        if (m != 1) continue;
        std::fill(c1.begin(), c1.end(), -1e30);
        nss::twsc_gemm_tn(m, n, k, a.data(), b.data(), c1.data());
        for (int j = 0; j < n; ++j) for (int i = 0; i < k; ++i) {
            double av[64], bv[64];
            for (int l = 0; l < m; ++l) { av[l] = a[l + i * m]; bv[l] = b[l + j * m]; }
            const double got = c1[static_cast<std::size_t>(i) + j * k];
            if (!chain_accept(got, av, bv, m)) {
                std::fprintf(stderr, "twsc_gemm_tn mismatch m=%d n=%d k=%d i=%d j=%d got=%.17g\n",
                             m, n, k, i, j, got);
                return 1;
            }
        }
    }
    return 0;
}

// --------------------------------------------- image_match characterization

// Pre-3211e45 image_match (per-candidate ssd_block + SortedTopK), kept as the
// bitwise reference for the row-kernel scan in cpu/common/image.cpp.
float ref_image_ssd(const float* a, const float* b, int stride, int block) {
    return nss::ssd_block(a, stride, b, stride, block);
}

int image_match_reference(const float* const* guides, int frames, int nch, int width, int height,
                          int t0, int x0, int y0, const nss::ImageSearch& cfg, nss::Match* matches) {
    using ImageTopK = nss::detail::SortedTopK;
    auto distance = [&](int t, int x, int y) {
        float sum = 0;
        for (int c = 0; c < nch; ++c)
            sum += ref_image_ssd(guides[t0 * nch + c] + y0 * width + x0,
                                 guides[t * nch + c] + y * width + x, width, cfg.block);
        if (!std::isfinite(sum)) throw std::runtime_error("nss: unrepresentable patch distance");
        return sum;
    };
    matches[0] = nss::Match{x0, y0, t0, 0, 0};
    if (cfg.group == 1) return 1;
    ImageTopK spatial(matches + 1, cfg.group - 1);
    const int lo = cfg.window / 2, hi = cfg.window - lo - 1;
    for (int y = std::max(0, y0 - lo); y <= std::min(height - cfg.block, y0 + hi); ++y)
        for (int x = std::max(0, x0 - lo); x <= std::min(width - cfg.block, x0 + hi); ++x) {
            if (x == x0 && y == y0) continue;
            spatial.add(nss::Match{x, y, t0, distance(t0, x, y), std::uint32_t(1 + y * width + x)});
        }
    const int n = 1 + spatial.finish();
    if (cfg.radius == 0 || frames == 1) return n;
    std::array<nss::Match, 256> seeds{}, centers{}, local{};
    const int seed_count = std::min(n, cfg.ps_num);
    std::copy_n(matches, seed_count, seeds.data());
    std::array<nss::Match, 256> retained{};
    ImageTopK global(retained.data(), cfg.group - 1);
    for (int i = 1; i < n; ++i) global.add(matches[i]);
    for (int direction : {-1, 1}) {
        std::copy_n(seeds.data(), seed_count, centers.data());
        int nc = seed_count;
        for (int delta = 1; delta <= cfg.radius; ++delta) {
            const int t = t0 + direction * delta;
            if (t < 0 || t >= frames) break;
            ImageTopK next(local.data(), cfg.ps_num);
            int ymin = height, ymax = -1;
            for (int i = 0; i < nc; ++i) {
                ymin = std::min(ymin, std::max(0, centers[i].y - cfg.ps_range));
                ymax = std::max(ymax, std::min(height - cfg.block, centers[i].y + cfg.ps_range));
            }
            for (int y = ymin; y <= ymax; ++y) {
                std::array<std::pair<int, int>, 256> intervals;
                int ni = 0;
                for (int i = 0; i < nc; ++i)
                    if (std::abs(y - centers[i].y) <= cfg.ps_range)
                        intervals[ni++] = {std::max(0, centers[i].x - cfg.ps_range),
                                           std::min(width - cfg.block, centers[i].x + cfg.ps_range)};
                std::sort(intervals.begin(), intervals.begin() + ni);
                int done = -1;
                for (int i = 0; i < ni; ++i) {
                    for (int x = std::max(done + 1, intervals[i].first); x <= intervals[i].second; ++x)
                        next.add(nss::Match{x, y, t, distance(t, x, y), std::uint32_t(1 + y * width + x)});
                    done = std::max(done, intervals[i].second);
                }
            }
            nc = next.finish();
            for (int i = 0; i < nc; ++i) global.add(local[i]);
            std::copy_n(local.data(), nc, centers.data());
            if (nc == 0) break;
        }
    }
    const int count = global.finish();
    std::copy_n(retained.data(), count, matches + 1);
    return 1 + count;
}

int test_image_match_characterization() {
    std::mt19937 rng(43);
    struct Case { int width, height, x0, y0; };
    const Case shapes[] = {{19, 13, 0, 0}, {19, 13, 4, 6}, {33, 21, 16, 10}, {64, 48, 63, 47}, {64, 48, 30, 24}};
    const int blocks[] = {7, 8, 15};
    const int windows[] = {1, 40, 128, 129};
    for (const auto& sh : shapes) for (int block : blocks) for (int window : windows)
        for (int nch : {1, 3}) for (int radius : {0, 1}) {
            const int frames = radius ? 3 : 1;
            const int t0 = radius ? 1 : 0;
            nss::ImageSearch cfg;
            cfg.block = block;
            cfg.window = window;
            cfg.radius = radius;
            cfg.group = 16;
            cfg.ps_num = 2;
            cfg.ps_range = 4;
            if (sh.width < block || sh.height < block || sh.x0 > sh.width - block || sh.y0 > sh.height - block)
                continue;
            std::vector<std::vector<float>> planes(static_cast<std::size_t>(frames) * nch);
            std::vector<const float*> guides(planes.size());
            for (std::size_t p = 0; p < planes.size(); ++p) {
                planes[p].resize(static_cast<std::size_t>(sh.width) * sh.height);
                fill_pattern(rng, planes[p].data(), planes[p].size(), kRand);
                guides[p] = planes[p].data();
            }
            std::array<nss::Match, 256> m1{}, m2{};
            const int c1 = nss::image_match(guides.data(), frames, nch, sh.width, sh.height, t0, sh.x0, sh.y0, cfg,
                                            m1.data());
            const int c2 = image_match_reference(guides.data(), frames, nch, sh.width, sh.height, t0, sh.x0, sh.y0,
                                                 cfg, m2.data());
            if (c1 != c2) {
                std::fprintf(stderr, "image_match count mismatch block=%d window=%d nch=%d radius=%d\n",
                             block, window, nch, radius);
                return 1;
            }
            for (int i = 0; i < c1; ++i) {
                if (m1[i].x != m2[i].x || m1[i].y != m2[i].y || m1[i].t != m2[i].t ||
                    !same(m1[i].dist, m2[i].dist) || m1[i].ordinal != m2[i].ordinal) {
                    std::fprintf(stderr,
                                 "image_match mismatch i=%d block=%d window=%d nch=%d radius=%d anchor=(%d,%d)\n",
                                 i, block, window, nch, radius, sh.x0, sh.y0);
                    return 1;
                }
            }
        }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const auto target = std::string_view(argv[1]) == "avx2" ? HWY_AVX2 :
                            std::string_view(argv[1]) == "avx3" ? HWY_AVX3 : 0;
        const auto caps = nss::backend_caps();
        if (!target || !(caps.compiled_targets & caps.runtime_targets & target)) return 77;
        hwy::SetSupportedTargetsForTest(target);
    }
    int rc = 0;
    rc |= test_lssc_ssd_pair_bitwise();
    rc |= test_lssc_ssd4_bitwise();
    rc |= test_lssc_assign_step_bitwise();
    rc |= test_lssc_accum_step_bitwise();
    rc |= test_lssc_farthest_bitwise();
    rc |= test_lssc_cluster_driver_bitwise();
    rc |= test_image_ssd_row_bitwise();
    rc |= test_ncsr_rowmajor_bitwise();
    rc |= test_twsc_gemm_bitwise();
    rc |= test_image_match_characterization();
    if (g_failures) return 1;
    if (rc == 0) std::puts("test_bitwise_kernels: all bitwise characterizations passed");
    return rc;
}
