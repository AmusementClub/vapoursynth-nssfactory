// SPDX-License-Identifier: GPL-2.0-only
// The fused BM3D group filter against the generic path (pack, bm3d_filter_group,
// aggregate) for every shape it serves, spatial and temporal.
//
// Both compute the same transform with different arithmetic, so results agree
// to rounding (contracts/tolerances.json, fixed_group) except where a
// coefficient sits on the hard threshold. The hard-threshold stage is therefore
// checked at two sigmas that leave no coefficient near it: one that keeps
// every coefficient and one that kills all but the DC.
#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kWidth = 96, kHeight = 80, kFrames = 3;
constexpr double kMaxAbs = 2e-5, kRelative = 2e-4;  // fixed_group tolerance

struct Planes {
    std::vector<std::vector<float>> data;
    std::vector<const float*> ptr;
    std::vector<int> stride;
};

Planes make_planes(unsigned seed) {
    Planes p;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    for (int t = 0; t < kFrames; ++t) {
        std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                plane[static_cast<std::size_t>(y) * kWidth + x] =
                    0.5f + 0.3f * std::sin(0.11f * static_cast<float>(x + 2 * t)) * std::cos(0.07f * static_cast<float>(y)) +
                    noise(rng);
            }
        }
        p.data.push_back(std::move(plane));
    }
    for (const auto& plane : p.data) {
        p.ptr.push_back(plane.data());
        p.stride.push_back(kWidth);
    }
    return p;
}

int check(int block, int group, int radius, bool wiener, float sigma, int count, const char* label) {
    const Planes src = make_planes(11), ref = make_planes(23);
    const int slices = 2 * radius + 1, t0 = radius, area = block * block;
    const std::size_t plane = static_cast<std::size_t>(kWidth) * kHeight;
    std::vector<float> num_a(plane * slices, 0.f), den_a(plane * slices, 0.f), num_b = num_a, den_b = den_a;
    std::vector<float> cube(static_cast<std::size_t>(group) * area), refc(cube.size());
    std::vector<float> work(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)));
    std::mt19937 rng(static_cast<unsigned>(block * 131 + group * 17 + radius));
    for (int trial = 0; trial < 12; ++trial) {
        const int k = trial % 3 == 2 ? std::max(1, count / 2) : count;  // also groups shorter than `group`
        std::vector<nss::Match> matches(static_cast<std::size_t>(group));
        for (int j = 0; j < k; ++j) {
            matches[static_cast<std::size_t>(j)].x = static_cast<int>(rng() % static_cast<unsigned>(kWidth - block + 1));
            matches[static_cast<std::size_t>(j)].y = static_cast<int>(rng() % static_cast<unsigned>(kHeight - block + 1));
            matches[static_cast<std::size_t>(j)].t = radius > 0 ? static_cast<int>(rng() % static_cast<unsigned>(slices)) : 0;
        }
        // Generic path.
        std::fill(cube.begin(), cube.end(), 0.f);
        std::fill(refc.begin(), refc.end(), 0.f);
        for (int j = 0; j < k; ++j) {
            const nss::Match& m = matches[static_cast<std::size_t>(j)];
            const int t = radius > 0 ? m.t : t0;
            nss::pack_patch(cube.data() + static_cast<std::size_t>(j) * area, area, src.ptr[static_cast<std::size_t>(t)], kWidth,
                            m.x, m.y, block, kWidth, kHeight);
            if (wiener) {
                nss::pack_patch(refc.data() + static_cast<std::size_t>(j) * area, area, ref.ptr[static_cast<std::size_t>(t)],
                                kWidth, m.x, m.y, block, kWidth, kHeight);
            }
        }
        float weight = 1.f;
        nss::bm3d_filter_group(cube.data(), area, group, k, block, sigma, wiener, wiener ? refc.data() : nullptr, &weight,
                               work.data());
        for (int j = 0; j < k; ++j) {
            const nss::Match& m = matches[static_cast<std::size_t>(j)];
            const std::size_t slice = radius > 0 ? static_cast<std::size_t>(std::clamp(m.t - t0 + radius, 0, slices - 1)) : 0;
            nss::aggregate_add(num_a.data() + slice * plane, den_a.data() + slice * plane, kWidth, m.x, m.y,
                               cube.data() + static_cast<std::size_t>(j) * area, block, kWidth, kHeight, weight);
        }
        // Fused path.
        if (!nss::bm3d_filter_fused(block, group, src.ptr.data(), src.stride.data(), matches.data(), k, sigma,
                                    wiener ? ref.ptr.data() : nullptr, ref.stride.data(), num_b.data(), den_b.data(),
                                    kWidth, kWidth, kHeight, t0, radius, plane)) {
            std::printf("FAIL %s: no fused kernel\n", label);
            return 1;
        }
    }
    double worst = 0.0;
    int failures = 0;
    for (std::size_t i = 0; i < num_a.size(); ++i) {
        const double dn = std::fabs(static_cast<double>(num_a[i]) - num_b[i]);
        const double dd = std::fabs(static_cast<double>(den_a[i]) - den_b[i]);
        const double bound_n = kMaxAbs + kRelative * std::fabs(static_cast<double>(num_a[i]));
        const double bound_d = kMaxAbs + kRelative * std::fabs(static_cast<double>(den_a[i]));
        worst = std::max({worst, dn, dd});
        failures += dn > bound_n || dd > bound_d;
    }
    if (failures) std::printf("FAIL %s: %d sums out of tolerance, worst %.3g\n", label, failures, worst);
    return failures ? 1 : 0;
}

}  // namespace

int main() {
    if (!nss::bm3d_filter_fused(8, 16, nullptr, nullptr, nullptr, 0, 0.f, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
                                0)) {
        std::printf("test_bm3d_fused SKIP: no fused kernels on this target\n");
        return 77;
    }
    int failures = 0, cases = 0;
    for (const int block : {4, 8, 12, 16, 32}) {
        for (const int group : {4, 8, 16, 32, 64}) {
            const bool served = nss::bm3d_filter_fused(block, group, nullptr, nullptr, nullptr, 0, 0.f, nullptr, nullptr,
                                                       nullptr, nullptr, 0, 0, 0, 0, 0, 0);
            // 12 / 8 keeps its own path; 16 / 64, 32 / 32 and 32 / 64 keep the generic one.
            const bool expected = !(block == 12 && group == 8) && !(block == 16 && group == 64) &&
                                  !(block == 32 && group >= 32);
            if (served != expected) {
                std::printf("FAIL b%d g%d: served = %d\n", block, group, served ? 1 : 0);
                ++failures;
            }
            if (!served) continue;
            for (const int radius : {0, 1}) {
                char label[96];
                const struct {
                    const char* name;
                    bool wiener;
                    float sigma;
                } stages[] = {{"wiener", true, 0.02f}, {"hard keep-all", false, 1e-7f}, {"hard dc-only", false, 50.f}};
                for (const auto& stage : stages) {
                    std::snprintf(label, sizeof(label), "b%d g%d r%d %s", block, group, radius, stage.name);
                    failures += check(block, group, radius, stage.wiener, stage.sigma, group, label);
                    ++cases;
                }
            }
        }
    }
    std::printf("test_bm3d_fused: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
