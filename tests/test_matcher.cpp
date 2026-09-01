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
    if (a.dist != b.dist) {
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
    constexpr int width = 24;
    constexpr int height = 24;
    constexpr int stride = 32;
    std::vector<float> frame(static_cast<std::size_t>(stride * height), 0.25f);
    // These values force the block=8/group=8 Highway path to exercise its
    // deterministic scalar fallback rather than admitting a NaN distance.
    frame[1 * stride + 1] = std::numeric_limits<float>::quiet_NaN();
    frame[15 * stride + 15] = std::numeric_limits<float>::infinity();
    nss::Match a[8]{}, b[8]{};
    const int na = nss::spatial_match(frame.data(), stride, width, height, 8, 8, 8, 8, 8, a);
    const int nb = nss::spatial_match(frame.data(), stride, width, height, 8, 8, 8, 8, 8, b);
    if (na != nb || na < 1 || a[0].x != 8 || a[0].y != 8 || a[0].ordinal != 0) {
        std::fprintf(stderr, "non-finite matcher lost self result\n");
        return 1;
    }
    for (int i = 0; i < na; ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].t != b[i].t ||
            (std::isfinite(a[i].dist) != std::isfinite(b[i].dist))) {
            std::fprintf(stderr, "non-finite matcher is not deterministic\n");
            return 1;
        }
        if (i > 0 && !std::isfinite(a[i - 1].dist) && std::isfinite(a[i].dist)) {
            std::fprintf(stderr, "non-finite distance was ordered before finite distance\n");
            return 1;
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
                 check_nonfinite_fast_path();
    for (int block : {1, 2, 4, 8, 16}) {
        if (block > width || block > height) {
            continue;
        }
        for (int group : {1, 2, 4, 8, 16}) {
            failed |= compare_spatial(frame, stride, width, height, block, 5, group, 0, 0);
            failed |= compare_spatial(frame, stride, width, height, block, 5, group, width - block, height - block);
        }
    }
    std::printf("matcher differential checks complete\n");
    return failed ? 1 : 0;
}
