#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

struct RefMatch {
    int x;
    int y;
    float dist;
    std::uint32_t ordinal;
};

bool less_ref(const RefMatch& a, const RefMatch& b) {
    const bool a_finite = std::isfinite(a.dist);
    const bool b_finite = std::isfinite(b.dist);
    if (a_finite != b_finite) {
        return a_finite;
    }
    if (a_finite && a.dist != b.dist) {
        return a.dist < b.dist;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    return a.ordinal < b.ordinal;
}

float scalar_ssd(const float* a, int sa, const float* b, int sb, int block) {
    float sum = 0.f;
    for (int y = 0; y < block; ++y) {
        for (int x = 0; x < block; ++x) {
            const float d = a[y * sa + x] - b[y * sb + x];
            sum += d * d;
        }
    }
    return sum;
}

std::vector<RefMatch> scalar_spatial(const float* frame, int stride, int width, int height, int bx, int by, int block,
                                     int range, int group) {
    const int max_x = width - block;
    const int max_y = height - block;
    const int cx = std::clamp(bx, 0, max_x);
    const int cy = std::clamp(by, 0, max_y);
    const int left = std::max(0, cx - range);
    const int right = std::min(max_x, cx + range);
    const int top = std::max(0, cy - range);
    const int bottom = std::min(max_y, cy + range);
    std::vector<RefMatch> all;
    std::uint32_t ordinal = 1;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x, ++ordinal) {
            if (x == cx && y == cy) {
                continue;
            }
            all.push_back({x, y, scalar_ssd(frame + cy * stride + cx, stride, frame + y * stride + x, stride, block),
                           ordinal});
        }
    }
    std::stable_sort(all.begin(), all.end(), less_ref);
    if (static_cast<int>(all.size()) >= group) {
        all.resize(static_cast<std::size_t>(group - 1));
    }
    all.insert(all.begin(), RefMatch{cx, cy, 0.f, 0});
    return all;
}

bool close_distance(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) && std::isnan(b);
        }
        return a == b;
    }
    const float scale = std::max({1.f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= 8e-6f * scale;
}

bool ordered(const nss::Match& a, const nss::Match& b) {
    if (a.dist != b.dist) {
        return a.dist < b.dist;
    }
    const bool a_self = a.ordinal == 0;
    const bool b_self = b.ordinal == 0;
    if (a_self != b_self) {
        return a_self;
    }
    if (a.t != b.t) {
        return a.t < b.t;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    return a.ordinal < b.ordinal;
}

int compare_spatial(const std::vector<float>& frame, int stride, int width, int height, int block, int range, int group,
                    int bx, int by) {
    nss::Match got[nss::kBmMaxGroup]{};
    const int n = nss::spatial_match(frame.data(), stride, width, height, bx, by, block, range, group, got);
    const auto want = scalar_spatial(frame.data(), stride, width, height, bx, by, block, range, group);
    if (n != static_cast<int>(want.size())) {
        std::fprintf(stderr, "matcher count mismatch block=%d group=%d got=%d want=%zu\n", block, group, n, want.size());
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (got[i].x != want[static_cast<std::size_t>(i)].x || got[i].y != want[static_cast<std::size_t>(i)].y ||
            !close_distance(got[i].dist, want[static_cast<std::size_t>(i)].dist)) {
            std::fprintf(stderr, "matcher mismatch block=%d group=%d at=%d got=(%d,%d,%.8g) want=(%d,%d,%.8g)\n", block,
                         group, i, got[i].x, got[i].y, got[i].dist, want[static_cast<std::size_t>(i)].x,
                         want[static_cast<std::size_t>(i)].y, want[static_cast<std::size_t>(i)].dist);
            return 1;
        }
    }
    return 0;
}

int check_constant_ties() {
    constexpr int width = 24;
    constexpr int height = 20;
    constexpr int stride = 32;
    std::vector<float> frame(static_cast<std::size_t>(stride * height), 0.5f);
    nss::Match got[nss::kBmMaxGroup]{};
    const int n = nss::spatial_match(frame.data(), stride, width, height, 10, 9, 8, 7, 8, got);
    const auto want = scalar_spatial(frame.data(), stride, width, height, 10, 9, 8, 7, 8);
    if (n != 8) {
        std::fprintf(stderr, "constant tie count=%d\n", n);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (got[i].x != want[static_cast<std::size_t>(i)].x || got[i].y != want[static_cast<std::size_t>(i)].y) {
            std::fprintf(stderr, "constant tie order mismatch at %d got=(%d,%d) want=(%d,%d) n=%d\n", i, got[i].x,
                         got[i].y, want[static_cast<std::size_t>(i)].x, want[static_cast<std::size_t>(i)].y, n);
            for (int j = 0; j < n; ++j) {
                std::fprintf(stderr, "  %d: got=(%d,%d) want=(%d,%d)\n", j, got[j].x, got[j].y,
                             want[static_cast<std::size_t>(j)].x, want[static_cast<std::size_t>(j)].y);
            }
            return 1;
        }
    }
    return 0;
}

int check_short_window() {
    // A frame exactly one block wide/high has no non-self candidates.  The
    // AVX2 top-k path must return the same one-element set as the scalar path.
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int stride = 8;
    std::vector<float> frame(static_cast<std::size_t>(stride * height), 0.25f);
    nss::Match got[nss::kBmMaxGroup]{};
    const int n = nss::spatial_match(frame.data(), stride, width, height, 0, 0, 8, 64, 8, got);
    if (n != 1 || got[0].x != 0 || got[0].y != 0 || got[0].dist != 0.f) {
        std::fprintf(stderr, "short matcher window returned %d candidates\n", n);
        return 1;
    }
    return 0;
}

int check_multichannel() {
    constexpr int width = 23;
    constexpr int height = 19;
    constexpr int stride = 32;
    std::vector<float> p0(static_cast<std::size_t>(stride * height));
    std::vector<float> p1(p0.size());
    std::vector<float> p2(p0.size());
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (std::size_t i = 0; i < p0.size(); ++i) {
        p0[i] = dist(rng);
        p1[i] = dist(rng);
        p2[i] = dist(rng);
    }
    const float* refs[3] = {p0.data(), p1.data(), p2.data()};
    const int strides[3] = {stride, stride, stride};
    nss::Match got[8]{};
    const int n = nss::spatial_match_nch(refs, strides, 3, width, height, 8, 7, 4, 5, 8, got);
    std::vector<RefMatch> want;
    const int cx = 8;
    const int cy = 7;
    for (int y = std::max(0, cy - 5); y <= std::min(height - 4, cy + 5); ++y) {
        for (int x = std::max(0, cx - 5); x <= std::min(width - 4, cx + 5); ++x) {
            if (x == cx && y == cy) {
                continue;
            }
            float d = 0.f;
            for (int c = 0; c < 3; ++c) {
                d += scalar_ssd(refs[c] + cy * stride + cx, stride, refs[c] + y * stride + x, stride, 4);
            }
            want.push_back({x, y, d, static_cast<std::uint32_t>(want.size() + 1)});
        }
    }
    std::stable_sort(want.begin(), want.end(), less_ref);
    if (want.size() > 7) {
        want.resize(7);
    }
    want.insert(want.begin(), RefMatch{cx, cy, 0.f, 0});
    if (n != 8) {
        std::fprintf(stderr, "multichannel count=%d\n", n);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (got[i].x != want[static_cast<std::size_t>(i)].x || got[i].y != want[static_cast<std::size_t>(i)].y ||
            !close_distance(got[i].dist, want[static_cast<std::size_t>(i)].dist)) {
            std::fprintf(stderr, "multichannel mismatch at %d\n", i);
            return 1;
        }
    }

    constexpr int b8 = 8;
    nss::Match got8[8]{};
    const int n8 = nss::spatial_match_nch(refs, strides, 3, width, height, 8, 7, b8, 5, 8, got8);
    std::vector<RefMatch> want8;
    const int max_x8 = width - b8;
    const int max_y8 = height - b8;
    const int cx8 = std::clamp(8, 0, max_x8);
    const int cy8 = std::clamp(7, 0, max_y8);
    for (int y = std::max(0, cy8 - 5); y <= std::min(max_y8, cy8 + 5); ++y) {
        for (int x = std::max(0, cx8 - 5); x <= std::min(max_x8, cx8 + 5); ++x) {
            if (x == cx8 && y == cy8) {
                continue;
            }
            float d = 0.f;
            for (int c = 0; c < 3; ++c) {
                d += scalar_ssd(refs[c] + cy8 * stride + cx8, stride, refs[c] + y * stride + x, stride, b8);
            }
            want8.push_back({x, y, d, static_cast<std::uint32_t>(want8.size() + 1)});
        }
    }
    std::stable_sort(want8.begin(), want8.end(), less_ref);
    if (want8.size() > 7) {
        want8.resize(7);
    }
    want8.insert(want8.begin(), RefMatch{cx8, cy8, 0.f, 0});
    if (n8 != static_cast<int>(want8.size())) {
        std::fprintf(stderr, "multichannel8 count=%d want=%zu\n", n8, want8.size());
        return 1;
    }
    for (int i = 0; i < n8; ++i) {
        if (got8[i].x != want8[static_cast<std::size_t>(i)].x || got8[i].y != want8[static_cast<std::size_t>(i)].y ||
            !close_distance(got8[i].dist, want8[static_cast<std::size_t>(i)].dist)) {
            std::fprintf(stderr, "multichannel8 mismatch at %d\n", i);
            return 1;
        }
    }
    const float* pa8[3] = {p0.data() + 3 * stride + 2, p1.data() + 3 * stride + 2, p2.data() + 3 * stride + 2};
    const float* pb8[3] = {p0.data() + 4 * stride + 3, p1.data() + 4 * stride + 3, p2.data() + 4 * stride + 3};
    const int sa8[3] = {stride, stride, stride};
    float fused = nss::ssd_nch(pa8, sa8, pb8, sa8, 3, 8);
    float parts = 0.f;
    for (int c = 0; c < 3; ++c) {
        parts += nss::ssd_block(pa8[c], stride, pb8[c], stride, 8);
    }
    if (!close_distance(fused, parts)) {
        std::fprintf(stderr, "ssd_nch != sum ssd_block: %g vs %g\n", fused, parts);
        return 1;
    }
    return 0;
}

int check_predictive_stability() {
    constexpr int width = 20;
    constexpr int height = 18;
    constexpr int stride = 24;
    constexpr int frames = 3;
    std::vector<std::vector<float>> storage(static_cast<std::size_t>(frames),
                                            std::vector<float>(static_cast<std::size_t>(stride * height), 0.25f));
    const float* refs[frames];
    int strides[frames];
    for (int t = 0; t < frames; ++t) {
        refs[t] = storage[static_cast<std::size_t>(t)].data();
        strides[t] = stride;
    }
    nss::SearchConfig cfg;
    cfg.block = 4;
    cfg.group = 8;
    cfg.bm_range = 4;
    cfg.radius = 1;
    cfg.ps_num = 2;
    cfg.ps_range = 3;
    nss::Match a[8]{}, b[8]{};
    const int na = nss::predictive_match(refs, strides, frames, width, height, 8, 7, 1, cfg, a);
    const int nb = nss::predictive_match(refs, strides, frames, width, height, 8, 7, 1, cfg, b);
    if (na != nb || na < 1) {
        return 1;
    }
    for (int i = 0; i < na; ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].t != b[i].t || a[i].dist != b[i].dist ||
            (i > 0 && ordered(b[i], b[i - 1]))) {
            std::fprintf(stderr, "predictive tie ordering is unstable\n");
            return 1;
        }
    }
    return 0;
}

int check_nonfinite_fast_path() {
    struct Case {
        int width;
        int height;
        int stride;
        int bx;
        int by;
        int range;
        int bad_x;
        int bad_y;
        float bad_value;
        int bad2_x;
        int bad2_y;
        float bad2_value;
        const char* name;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float max_root = std::sqrt(std::numeric_limits<float>::max());
    const Case cases[] = {
        // One bad candidate is ignored once seven finite candidates have
        // filled the SIMD top-k set; no scalar fallback is needed.
        {24, 24, 32, 8, 8, 8, 0, 0, nan, -1, -1, 0.f, "nan-with-full-finite-topk"},
        // The reference block is non-finite, so every computed distance is
        // non-finite and the fast path must rebuild through StableTopK.
        {24, 24, 32, 8, 8, 8, 15, 15, inf, -1, -1, 0.f, "nonfinite-reference"},
        // Only six of the eight non-self candidates remain finite. This is
        // the finite-count fallback boundary for an eight-element group.
        {16, 8, 16, 0, 0, 64, 14, 0, nan, -1, -1, 0.f, "short-finite-set"},
        // The two squared terms sum to FLT_MAX. That distance compares equal
        // to the SIMD empty-lane sentinel, so the final count check must
        // retain it via the scalar fallback when all seven candidates are
        // required.
        {15, 8, 15, 0, 0, 64, 13, 0, max_root, 14, 0, std::ldexp(1.f, 52), "flt-max-distance"},
    };

    for (const Case& test : cases) {
        std::vector<float> frame(static_cast<std::size_t>(test.stride * test.height), 0.f);
        frame[static_cast<std::size_t>(test.bad_y * test.stride + test.bad_x)] = test.bad_value;
        if (test.bad2_x >= 0) {
            frame[static_cast<std::size_t>(test.bad2_y * test.stride + test.bad2_x)] = test.bad2_value;
        }
        nss::Match got[8]{};
        const int n = nss::spatial_match(frame.data(), test.stride, test.width, test.height, test.bx, test.by, 8,
                                         test.range, 8, got);
        const auto want = scalar_spatial(frame.data(), test.stride, test.width, test.height, test.bx, test.by, 8,
                                         test.range, 8);
        if (test.bad2_x >= 0 && (want.empty() || want.back().dist != std::numeric_limits<float>::max())) {
            std::fprintf(stderr, "%s did not construct an exact FLT_MAX distance\n", test.name);
            return 1;
        }
        if (n != static_cast<int>(want.size())) {
            std::fprintf(stderr, "%s count mismatch got=%d want=%zu\n", test.name, n, want.size());
            return 1;
        }
        for (int i = 0; i < n; ++i) {
            const auto& ref = want[static_cast<std::size_t>(i)];
            if (got[i].x != ref.x || got[i].y != ref.y || got[i].ordinal != ref.ordinal ||
                !close_distance(got[i].dist, ref.dist)) {
                std::fprintf(stderr,
                             "%s mismatch at %d got=(%d,%d,%.8g,%u) want=(%d,%d,%.8g,%u)\n", test.name, i,
                             got[i].x, got[i].y, got[i].dist, got[i].ordinal, ref.x, ref.y, ref.dist, ref.ordinal);
                return 1;
            }
        }
    }
    return 0;
}

int check_nonfinite_top1() {
    constexpr int width = 12;
    constexpr int height = 10;
    constexpr int stride = 16;
    constexpr int block = 4;
    constexpr int bx = 4;
    constexpr int by = 3;
    constexpr int range = 4;
    struct Case {
        int bad_x;
        int bad_y;
        float bad_value;
        const char* name;
    };
    const Case cases[] = {
        {0, 0, std::numeric_limits<float>::quiet_NaN(), "top1-nan-candidate"},
        {bx, by, std::numeric_limits<float>::infinity(), "top1-nonfinite-reference"},
    };
    for (const Case& test : cases) {
        std::vector<float> frame(static_cast<std::size_t>(stride * height), 0.25f);
        frame[static_cast<std::size_t>(test.bad_y * stride + test.bad_x)] = test.bad_value;
        nss::Match got[2]{};
        const int n = nss::spatial_match(frame.data(), stride, width, height, bx, by, block, range, 2, got);
        const auto want = scalar_spatial(frame.data(), stride, width, height, bx, by, block, range, 2);
        if (n != static_cast<int>(want.size())) {
            std::fprintf(stderr, "%s count mismatch got=%d want=%zu\n", test.name, n, want.size());
            return 1;
        }
        for (int i = 0; i < n; ++i) {
            const auto& ref = want[static_cast<std::size_t>(i)];
            if (got[i].x != ref.x || got[i].y != ref.y || got[i].ordinal != ref.ordinal ||
                !close_distance(got[i].dist, ref.dist)) {
                std::fprintf(stderr, "%s mismatch at %d\n", test.name, i);
                return 1;
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    constexpr int width = 23;
    constexpr int height = 19;
    constexpr int stride = 32;
    std::vector<float> frame(static_cast<std::size_t>(stride * height));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& v : frame) {
        v = dist(rng);
    }
    int failed = check_constant_ties() | check_short_window() | check_multichannel() | check_predictive_stability() |
                 check_nonfinite_fast_path() | check_nonfinite_top1();
    for (int block : {1, 2, 4, 8, 12, 16}) {
        if (block > width || block > height) {
            continue;
        }
        for (int group : {1, 2, 4, 8, 16, 32, 64}) {
            if (block != 8 && group > 16) {
                continue;
            }
            failed |= compare_spatial(frame, stride, width, height, block, 5, group, 0, 0);
            failed |= compare_spatial(frame, stride, width, height, block, 5, group, width - block, height - block);
        }
    }
    std::printf("matcher differential checks complete\n");
    return failed ? 1 : 0;
}
