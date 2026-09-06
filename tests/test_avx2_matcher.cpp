#include "nss/cpu_api.hpp"
#include "hwy/targets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
// Independent exhaustive sorting oracle using the unchanged same-ISA SSD.
// This catches changed reduction order even when image-level tolerances pass.
bool less(const nss::Match& a, const nss::Match& b) {
    const bool af = std::isfinite(a.dist), bf = std::isfinite(b.dist);
    if (af != bf) return af;
    if (af && a.dist != b.dist) return a.dist < b.dist;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.ordinal < b.ordinal;
}

bool same_float(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
}

bool check_match(const std::vector<float>& image, int width, int height, int stride,
                 int block, int cx, int cy, int range, int group) {
    std::vector<nss::Match> expected;
    unsigned ordinal = 1;
    for (int y = std::max(0, cy - range); y <= std::min(height - block, cy + range); ++y) {
        for (int x = std::max(0, cx - range); x <= std::min(width - block, cx + range); ++x, ++ordinal) {
            if (x == cx && y == cy) continue;
            expected.push_back({x, y, 0, nss::ssd_block(image.data() + cy * stride + cx, stride,
                                                      image.data() + y * stride + x, stride, block), ordinal});
        }
    }
    std::stable_sort(expected.begin(), expected.end(), less);
    expected.resize(std::min(expected.size(), static_cast<std::size_t>(group - 1)));
    expected.insert(expected.begin(), {cx, cy, 0, 0.f, 0});
    nss::Match actual[nss::kBmMaxGroup];
    const int count = nss::spatial_match(image.data(), stride, width, height, cx, cy, block, range, group, actual, 15);
    if (count != static_cast<int>(expected.size())) return false;
    for (int i = 0; i < count; ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        if (a.x != e.x || a.y != e.y || a.t != e.t || a.ordinal != e.ordinal || !same_float(a.dist, e.dist)) {
            std::fprintf(stderr, "exact matcher mismatch b%d g%d at %d,%d range%d rank%d: %.9g != %.9g\n",
                         block, group, cx, cy, range, i, a.dist, e.dist);
            return false;
        }
    }
    return true;
}

bool check_unpack() {
    constexpr int width = 23, height = 21, stride = 29;
    std::mt19937 rng(91);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> patch(144), initial(stride * height);
    for (auto& v : patch) v = random(rng);
    for (auto& v : initial) v = random(rng);
    for (int x : {-3, 0, 1, 11, 20}) for (int y : {-2, 0, 1, 9, 20}) {
        auto n = initial, d = initial, refn = initial, refd = initial;
        nss::unpack_patch(n.data(), d.data(), stride, x, y, patch.data(), 12, width, height, 0.731f);
        nss::unpack_patch_fixed(refn.data(), refd.data(), stride, x, y, patch.data(), 12, width, height, 0.731f, 512);
        for (std::size_t i = 0; i < n.size(); ++i) {
            if (!same_float(n[i], refn[i]) || !same_float(d[i], refd[i])) {
                std::fprintf(stderr, "fixed unpack mismatch at %d,%d offset%zu\n", x, y, i);
                return false;
            }
        }
    }
    return true;
}
}  // namespace

int main() {
#if !(HWY_TARGETS & HWY_AVX2)
    std::puts("AVX2 matcher oracle skipped: AVX2 was not compiled");
    return 77;
#else
    if (!(hwy::SupportedTargets() & HWY_AVX2)) {
        std::puts("AVX2 matcher oracle skipped: hardware does not support AVX2");
        return 77;
    }
    // The exact oracle certifies the AVX2 recurrence, including in a
    // dynamic build on AVX-512 hardware. Do this before any dispatch call.
    hwy::SetSupportedTargetsForTest(HWY_AVX2);
    std::mt19937 rng(51);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    // Odd widths and padded rows exercise candidate tails and unaligned views.
    for (int block : {4, 8, 12, 16}) for (int extra : {0, 1, 9, 18}) {
        const int width = block + extra, height = block + extra / 2, stride = width + 7;
        std::vector<float> image(stride * height, std::numeric_limits<float>::quiet_NaN());
        for (int kind = 0; kind < 5; ++kind) {
            for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                image[y * stride + x] = kind == 0 ? 0.5f : kind == 1 ? float((x + y) % 3) * 0.125f : random(rng);
            }
            if (kind == 3) image[0] = std::numeric_limits<float>::quiet_NaN();
            if (kind == 4) image[(height - 1) * stride + width - 1] = std::numeric_limits<float>::infinity();
            for (int group : {1, 2, 8, 16, 32, 64}) for (int range : {0, 1, 7}) {
                // The pre-existing b8/g8 SIMD lane-sort path accumulates
                // four independent row streams; generic Ssd8 accumulates
                // all rows sequentially. It is unchanged by every port bit
                // and covered by test_matcher, so it is not a valid exact
                // generic-SSD oracle case. b8/g16+ (SortedTopK) remains exact.
                if (block == 8 && group == 8) continue;
                for (int corner = 0; corner < 3; ++corner) {
                    const int cx = corner == 0 ? 0 : corner == 1 ? extra / 2 : extra;
                    const int cy = corner == 0 ? 0 : corner == 1 ? (height - block) / 2 : height - block;
                    if (!check_match(image, width, height, stride, block, cx, cy, range, group)) return 1;
                }
            }
        }
    }
    if (!check_unpack()) return 1;
    std::puts("exact AVX2 matcher and fixed unpack checks passed");
    return 0;
#endif
}
