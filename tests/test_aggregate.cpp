#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

int check_radius(int radius) {
    constexpr int width = 13;
    constexpr int height = 7;
    constexpr int dstride = width + 3;
    constexpr int fstride = width + 5;
    constexpr int sstride = width + 7;
    const int slices = 2 * radius + 1;

    std::vector<float> dst(static_cast<std::size_t>(dstride) * height, -99.f);
    std::vector<float> src(static_cast<std::size_t>(sstride) * height, -77.f);
    std::vector<float> fat(static_cast<std::size_t>(fstride) * height * slices * 2, -55.f);
    std::vector<float> expected(static_cast<std::size_t>(width) * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            src[static_cast<std::size_t>(y) * sstride + x] = 1000.f + 10.f * y + x;
            float num = 0.f;
            float den = 0.f;
            for (int t = 0; t < slices; ++t) {
                const float n = static_cast<float>((t + 1) * (x + 2) + y);
                const float d = ((x + y) % 11 == 0) ? 0.f : static_cast<float>(t + 2);
                fat[(static_cast<std::size_t>(t * 2) * height + y) * fstride + x] = n;
                fat[(static_cast<std::size_t>(t * 2 + 1) * height + y) * fstride + x] = d;
                num += n;
                den += d;
            }
            expected[static_cast<std::size_t>(y) * width + x] =
                den > 1e-12f ? num / den : src[static_cast<std::size_t>(y) * sstride + x];
        }
    }

    nss::vaggregate_reduce(dst.data(), fat.data(), src.data(), width, height, dstride, fstride, sstride, radius);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float actual = dst[static_cast<std::size_t>(y) * dstride + x];
            const float want = expected[static_cast<std::size_t>(y) * width + x];
            if (std::fabs(actual - want) > 1e-5f) {
                std::fprintf(stderr, "VAggregate radius=%d mismatch at (%d,%d): %.9g != %.9g\n", radius, x, y,
                             actual, want);
                return 1;
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    for (int radius : {0, 1, 2, 3, 4, 7}) {
        if (check_radius(radius) != 0) {
            return 1;
        }
    }
    return 0;
}
