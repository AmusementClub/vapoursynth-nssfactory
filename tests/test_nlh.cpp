#include "nss/cpu_nlh.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

static int haar_roundtrip(int n) {
    std::mt19937 rng(static_cast<unsigned>(21 + n));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> orig(static_cast<std::size_t>(n));
    std::vector<float> buf(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        orig[static_cast<std::size_t>(i)] = dist(rng);
        buf[static_cast<std::size_t>(i)] = orig[static_cast<std::size_t>(i)];
    }
    nss::haar1d(buf.data(), buf.data(), n);
    nss::ihaar1d(buf.data(), buf.data(), n);
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(buf[static_cast<std::size_t>(i)] -
                                                                  orig[static_cast<std::size_t>(i)])));
    }
    std::printf("haar1d n=%d max_abs=%.6g\n", n, max_err);
    if (!(max_err < 1e-5)) {
        return fail("haar forward/inverse roundtrip");
    }
    return 0;
}

static int pixel_match_includes_self() {
    const int m = 6;
    const int n = 4;
    const int lda = 8;
    const int q = 4;
    std::vector<float> group(static_cast<std::size_t>(lda * n), 0.f);
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            group[static_cast<std::size_t>(r + c * lda)] = static_cast<float>(r) * 0.3f + static_cast<float>(c);
        }
    }
    std::vector<int> idx(static_cast<std::size_t>(m * q), -1);
    nss::pixel_match(group.data(), m, n, lda, q, idx.data());
    for (int r = 0; r < m; ++r) {
        bool has_self = false;
        for (int j = 0; j < q; ++j) {
            if (idx[static_cast<std::size_t>(r * q + j)] == r) {
                has_self = true;
                break;
            }
        }
        if (!has_self) {
            return fail("pixel_match must include self row");
        }
        if (idx[static_cast<std::size_t>(r * q)] != r) {
            return fail("pixel_match self must be first neighbor");
        }
    }
    std::printf("pixel_match includes self ok\n");
    return 0;
}

static int filter_group_snapshot_and_finite() {
    const int m = 8;
    const int n = 4;
    const int lda = 12;
    const int q = 4;
    std::vector<float> a(static_cast<std::size_t>(lda * n), 0.f);
    std::vector<float> b(static_cast<std::size_t>(lda * n), 0.f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = 0.2f * static_cast<float>(i) + 0.05f * static_cast<float>(j);
            a[static_cast<std::size_t>(i + j * lda)] = v;
            b[static_cast<std::size_t>(i + j * lda)] = v;
        }
    }
    const int wn = nss::nlh_filter_work_floats(m, n, q, lda);
    std::vector<float> work(static_cast<std::size_t>(wn));
    float wa = 0.f;
    float wb = 0.f;
    nss::nlh_filter_group(a.data(), m, n, lda, q, 0.02f, false, nullptr, &wa, work.data(), wn);
    nss::nlh_filter_group(b.data(), m, n, lda, q, 0.02f, false, nullptr, &wb, work.data(), wn);
    int nonfinite = 0;
    double diff = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float va = a[static_cast<std::size_t>(i + j * lda)];
            const float vb = b[static_cast<std::size_t>(i + j * lda)];
            if (!std::isfinite(va) || !std::isfinite(vb)) {
                ++nonfinite;
            }
            diff = std::max(diff, std::fabs(static_cast<double>(va - vb)));
        }
    }
    std::printf("nlh_filter_group finite=%d repeat_diff=%.6g wa=%.6g\n", nonfinite == 0, diff,
                static_cast<double>(wa));
    if (nonfinite != 0) {
        return fail("nlh_filter_group non-finite");
    }
    if (diff > 1e-5) {
        return fail("nlh_filter_group not deterministic");
    }
    if (!(wa > 0.f)) {
        return fail("nlh_filter_group weight");
    }
    std::vector<float> r = a;
    float ww = 0.f;
    nss::nlh_filter_group(a.data(), m, n, lda, q, 0.02f, true, r.data(), &ww, work.data(), wn);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(a[static_cast<std::size_t>(i + j * lda)])) {
                return fail("nlh wiener non-finite");
            }
        }
    }
    return 0;
}

static int match16_differential() {
    constexpr int width = 41;
    constexpr int height = 35;
    constexpr int stride = 48;
    std::vector<float> frame(static_cast<std::size_t>(stride * height));
    std::mt19937 rng(103);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& value : frame) {
        value = dist(rng);
    }

    auto compare = [&](int block, int range, int bx, int by) {
        nss::Match shared[16]{};
        nss::Match dedicated[16]{};
        const int shared_count =
            nss::spatial_match(frame.data(), stride, width, height, bx, by, block, range, 16, shared);
        const int dedicated_count =
            nss::nlh_spatial_match16(frame.data(), stride, width, height, bx, by, block, range, dedicated);
        if (shared_count != dedicated_count) {
            std::fprintf(stderr, "match16 count mismatch block=%d range=%d shared=%d dedicated=%d\n", block, range,
                         shared_count, dedicated_count);
            return 1;
        }
        for (int i = 0; i < shared_count; ++i) {
            const auto shared_bits = std::bit_cast<std::uint32_t>(shared[i].dist);
            const auto dedicated_bits = std::bit_cast<std::uint32_t>(dedicated[i].dist);
            const float distance_scale = std::max({1.f, std::fabs(shared[i].dist), std::fabs(dedicated[i].dist)});
            const bool same_distance =
                shared_bits == dedicated_bits ||
                (std::isfinite(shared[i].dist) && std::isfinite(dedicated[i].dist) &&
                 std::fabs(shared[i].dist - dedicated[i].dist) <= 8e-6f * distance_scale) ||
                (!std::isfinite(shared[i].dist) && !std::isfinite(dedicated[i].dist));
            if (shared[i].x != dedicated[i].x || shared[i].y != dedicated[i].y ||
                shared[i].t != dedicated[i].t || shared[i].ordinal != dedicated[i].ordinal ||
                !same_distance) {
                std::fprintf(stderr,
                             "match16 mismatch block=%d range=%d index=%d shared=(%d,%d,%u,%08x) "
                             "dedicated=(%d,%d,%u,%08x)\n",
                             block, range, i, shared[i].x, shared[i].y, shared[i].ordinal,
                             shared_bits, dedicated[i].x, dedicated[i].y, dedicated[i].ordinal, dedicated_bits);
                return 1;
            }
        }
        return 0;
    };

    int failed = 0;
    for (int block : {1, 2, 4, 8, 16}) {
        for (int range : {0, 1, 5, 20}) {
            failed |= compare(block, range, 0, 0);
            failed |= compare(block, range, width / 2, height / 2);
            failed |= compare(block, range, width - block, height - block);
        }
    }

    std::fill(frame.begin(), frame.end(), 0.25f);
    failed |= compare(8, 20, 17, 13);
    failed |= compare(16, 0, 25, 19);
    frame[13 * stride + 17] = std::numeric_limits<float>::quiet_NaN();
    nss::Match nonfinite_a[16]{};
    nss::Match nonfinite_b[16]{};
    const int nonfinite_count_a =
        nss::nlh_spatial_match16(frame.data(), stride, width, height, 17, 13, 8, 20, nonfinite_a);
    const int nonfinite_count_b =
        nss::nlh_spatial_match16(frame.data(), stride, width, height, 17, 13, 8, 20, nonfinite_b);
    bool seen_nonfinite = false;
    if (nonfinite_count_a != nonfinite_count_b || nonfinite_count_a != 16 || nonfinite_a[0].ordinal != 0) {
        failed |= fail("match16 non-finite count or self mismatch");
    }
    for (int i = 0; i < nonfinite_count_a; ++i) {
        if (nonfinite_a[i].x != nonfinite_b[i].x || nonfinite_a[i].y != nonfinite_b[i].y ||
            nonfinite_a[i].ordinal != nonfinite_b[i].ordinal ||
            std::bit_cast<std::uint32_t>(nonfinite_a[i].dist) !=
                std::bit_cast<std::uint32_t>(nonfinite_b[i].dist)) {
            failed |= fail("match16 non-finite result is not deterministic");
            break;
        }
        const bool finite = std::isfinite(nonfinite_a[i].dist);
        if (seen_nonfinite && finite) {
            failed |= fail("match16 ordered a finite result after a non-finite result");
            break;
        }
        seen_nonfinite = seen_nonfinite || !finite;
    }

    std::fill(frame.begin(), frame.end(), 0.5f);
    nss::MatchBatchItem items[2] = {{4, 3, 8, 20, 16}, {29, 23, 8, 20, 16}};
    nss::Match shared_batch[2 * nss::kBmMaxGroup]{};
    nss::Match dedicated_batch[2 * nss::kBmMaxGroup]{};
    int shared_counts[2]{};
    int dedicated_counts[2]{};
    const int shared_status = nss::spatial_match_batch(frame.data(), stride, width, height, items, 2, shared_batch,
                                                       nss::kBmMaxGroup, shared_counts);
    const int dedicated_status = nss::nlh_spatial_match_batch(frame.data(), stride, width, height, items, 2,
                                                              dedicated_batch, nss::kBmMaxGroup, dedicated_counts);
    if (shared_status != dedicated_status) {
        failed |= fail("match16 batch status mismatch");
    }
    for (int item = 0; item < 2; ++item) {
        if (shared_counts[item] != dedicated_counts[item]) {
            failed |= fail("match16 batch count mismatch");
            break;
        }
        for (int i = 0; i < shared_counts[item]; ++i) {
            const auto& shared = shared_batch[item * nss::kBmMaxGroup + i];
            const auto& dedicated = dedicated_batch[item * nss::kBmMaxGroup + i];
            if (shared.x != dedicated.x || shared.y != dedicated.y || shared.ordinal != dedicated.ordinal) {
                failed |= fail("match16 batch ordering mismatch");
                break;
            }
        }
    }
    if (nss::nlh_spatial_match16(nullptr, stride, width, height, 0, 0, 8, 20, nullptr) != 0) {
        failed |= fail("match16 invalid input must reject");
    }
    std::printf("nlh match16 differential checks complete\n");
    return failed;
}

int main() {
    int failed = 0;
    failed |= haar_roundtrip(16);
    failed |= haar_roundtrip(8);
    failed |= haar_roundtrip(4);
    failed |= haar_roundtrip(2);
    failed |= pixel_match_includes_self();
    failed |= filter_group_snapshot_and_finite();
    failed |= match16_differential();
    {
        std::vector<int> idx(6 * 4, -1);
        std::vector<float> group(8 * 4, 0.f);
        nss::pixel_match(group.data(), 6, 4, 5, 4, idx.data());
        if (idx[0] != -1) {
            return fail("pixel_match lda < m must reject");
        }
    }
    {
        // Query-only writeback: disjoint pairs must not overwrite each other.
        const int m = 4;
        const int n = 2;
        const int lda = 4;
        const int q = 2;
        std::vector<float> g(static_cast<std::size_t>(lda * n), 0.f);
        // Pair (0,1) around 1, pair (2,3) around 8 — distances don't cross.
        g[0] = 1.0f;
        g[1] = 1.1f;
        g[2] = 8.0f;
        g[3] = 8.2f;
        g[0 + lda] = 1.05f;
        g[1 + lda] = 1.15f;
        g[2 + lda] = 8.1f;
        g[3 + lda] = 8.3f;
        std::vector<float> pair(static_cast<std::size_t>(2 * lda), 0.f);
        pair[0] = g[0];
        pair[1] = g[1];
        pair[0 + lda] = g[0 + lda];
        pair[1 + lda] = g[1 + lda];
        const int wn = nss::nlh_filter_work_floats(m, n, q, lda);
        std::vector<float> work(static_cast<std::size_t>(wn));
        float w = 0.f;
        nss::nlh_filter_group(g.data(), m, n, lda, q, 0.4f, false, nullptr, &w, work.data(), wn);
        nss::nlh_filter_group(pair.data(), 2, n, lda, q, 0.4f, false, nullptr, &w, work.data(), wn);
        double diff = 0.0;
        for (int c = 0; c < n; ++c) {
            diff = std::max(diff, std::fabs(static_cast<double>(g[c * lda] - pair[c * lda])));
            diff = std::max(diff, std::fabs(static_cast<double>(g[1 + c * lda] - pair[1 + c * lda])));
        }
        std::printf("nlh query-only disjoint maxabs=%.6g\n", diff);
        if (diff > 1e-4) {
            return fail("nlh query-only writeback polluted by later rows");
        }
    }
    {
        // Extra match column is constant so pixel_match ranking matches n=4.
        // Haar uses n_use=4; column 4 must stay put.
        const int m = 8;
        const int lda = 8;
        const int q = 4;
        std::vector<float> g5(static_cast<std::size_t>(lda * 5), 0.f);
        std::vector<float> g4(static_cast<std::size_t>(lda * 4), 0.f);
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < m; ++i) {
                const float v = 0.15f * static_cast<float>(i) + 0.07f * static_cast<float>(j);
                g5[static_cast<std::size_t>(i + j * lda)] = v;
                g4[static_cast<std::size_t>(i + j * lda)] = v;
            }
        }
        for (int i = 0; i < m; ++i) {
            g5[static_cast<std::size_t>(i + 4 * lda)] = 0.5f;
        }
        const int wn5 = nss::nlh_filter_work_floats(m, 5, q, lda);
        const int wn4 = nss::nlh_filter_work_floats(m, 4, q, lda);
        std::vector<float> w5(static_cast<std::size_t>(wn5));
        std::vector<float> w4(static_cast<std::size_t>(wn4));
        float a = 0.f;
        float b = 0.f;
        nss::nlh_filter_group(g5.data(), m, 5, lda, q, 0.05f, false, nullptr, &a, w5.data(), wn5);
        nss::nlh_filter_group(g4.data(), m, 4, lda, q, 0.05f, false, nullptr, &b, w4.data(), wn4);
        double diff = 0.0;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < m; ++i) {
                diff = std::max(diff, std::fabs(static_cast<double>(g5[static_cast<std::size_t>(i + j * lda)] -
                                                                   g4[static_cast<std::size_t>(i + j * lda)])));
            }
        }
        for (int i = 0; i < m; ++i) {
            if (std::fabs(g5[static_cast<std::size_t>(i + 4 * lda)] - 0.5f) > 1e-6f) {
                return fail("nlh n=5 must not Haar the extra column");
            }
        }
        std::printf("nlh n=5 vs n=4 maxabs=%.6g\n", diff);
        if (diff > 1e-4) {
            return fail("nlh n=5 Haar prefix must match n=4");
        }
    }
    return failed ? 1 : 0;
}
