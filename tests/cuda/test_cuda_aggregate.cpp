// SPDX-License-Identifier: GPL-2.0-only
// Patch aggregation: the ordered (deterministic) path must equal a host
// reference that accumulates in patch-id order with the same fmaf, bit for
// bit, and be bitwise repeatable; the atomic path must agree within float
// tolerance. --bench times both at 1080p BM3D-like loads (D17 study).
#include "cuda/common/aggregate.hpp"
#include "cuda/runtime/device.hpp"
#include "cuda/runtime/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Load {
    int width, height, block, slices;
    std::vector<nss_cuda::AggregatePatch> patches;
    std::vector<float> values;
};

// BM3D-like: raster references at `step`, `group` patches each within
// +-range of the reference, temporal slices assigned round-robin.
Load make_load(int width, int height, int block, int step, int group, int range, int slices, unsigned seed) {
    Load load{width, height, block, slices, {}, {}};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> offset(-range, range);
    std::uniform_real_distribution<float> value(-0.2f, 1.2f), weight(0.05f, 40.f);
    for (int y0 = 0; y0 < height - block + step; y0 += step) {
        const int y = std::min(y0, height - block);
        for (int x0 = 0; x0 < width - block + step; x0 += step) {
            const int x = std::min(x0, width - block);
            const float w = weight(rng);
            for (int j = 0; j < group; ++j) {
                const int px = j ? std::clamp(x + offset(rng), 0, width - block) : x;
                const int py = j ? std::clamp(y + offset(rng), 0, height - block) : y;
                load.patches.push_back({px, py, j % slices, w});
                for (int i = 0; i < block * block; ++i) load.values.push_back(value(rng));
            }
        }
    }
    return load;
}

struct Device {
    nss_cuda::DeviceBuffer values, patches, num, den;
    nss_cuda::AggregateTarget target{};
    explicit Device(const Load& load)
        : values(load.values.size() * sizeof(float)), patches(load.patches.size() * sizeof(nss_cuda::AggregatePatch)),
          num(static_cast<std::size_t>(load.width) * load.height * load.slices * sizeof(float)),
          den(num.bytes()) {
        NSS_CUDA_CHECK(cudaMemcpy(values.get(), load.values.data(), values.bytes(), cudaMemcpyHostToDevice));
        NSS_CUDA_CHECK(cudaMemcpy(patches.get(), load.patches.data(), patches.bytes(), cudaMemcpyHostToDevice));
        target = {num.as<float>(), den.as<float>(), load.width, load.height, load.width, load.slices,
                  static_cast<std::size_t>(load.width) * load.height};
    }
    std::vector<float> fetch(const nss_cuda::DeviceBuffer& buffer) const {
        std::vector<float> out(buffer.bytes() / sizeof(float));
        NSS_CUDA_CHECK(cudaMemcpy(out.data(), buffer.get(), buffer.bytes(), cudaMemcpyDeviceToHost));
        return out;
    }
};

int correctness(int block, int slices) {
    const Load load = make_load(181, 97, block, std::max(1, block / 2), 8, 9, slices, 11u + block);
    const std::size_t plane = static_cast<std::size_t>(load.width) * load.height;
    std::vector<float> num(plane * slices, 0.f), den(plane * slices, 0.f);
    for (std::size_t p = 0; p < load.patches.size(); ++p) {
        const auto& patch = load.patches[p];
        for (int r = 0; r < block; ++r) {
            for (int c = 0; c < block; ++c) {
                const std::size_t i = patch.slice * plane + static_cast<std::size_t>(patch.y + r) * load.width + patch.x + c;
                num[i] = std::fmaf(patch.weight, load.values[p * block * block + r * block + c], num[i]);
                den[i] += patch.weight;
            }
        }
    }
    Device dev(load);
    nss_cuda::OrderedAggregator ordered(load.width, load.height, slices, block, load.patches.size(), nullptr);
    ordered.run(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(), static_cast<int>(load.patches.size()),
                dev.target, nullptr);
    const auto num1 = dev.fetch(dev.num), den1 = dev.fetch(dev.den);
    ordered.run(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(), static_cast<int>(load.patches.size()),
                dev.target, nullptr);
    const auto num2 = dev.fetch(dev.num);
    int failures = 0;
    if (std::memcmp(num1.data(), num.data(), num.size() * sizeof(float)) ||
        std::memcmp(den1.data(), den.data(), den.size() * sizeof(float))) {
        std::printf("FAIL: ordered aggregation b%d s%d differs from the patch-order host reference\n", block, slices);
        ++failures;
    }
    if (std::memcmp(num1.data(), num2.data(), num.size() * sizeof(float))) {
        std::printf("FAIL: ordered aggregation b%d s%d is not repeatable\n", block, slices);
        ++failures;
    }
    NSS_CUDA_CHECK(cudaMemset(dev.num.get(), 0, dev.num.bytes()));
    NSS_CUDA_CHECK(cudaMemset(dev.den.get(), 0, dev.den.bytes()));
    nss_cuda::aggregate_atomic(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(),
                               static_cast<int>(load.patches.size()), block, dev.target, nullptr);
    const auto numa = dev.fetch(dev.num);
    double worst = 0.0;
    for (std::size_t i = 0; i < num.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(numa[i]) - num[i]) / std::max(1.0, std::fabs(1.0 * num[i])));
    }
    if (worst > 1e-5) {
        std::printf("FAIL: atomic aggregation b%d s%d relative error %.3g\n", block, slices, worst);
        ++failures;
    }
    return failures;
}

void bench() {
    for (const int slices : {1, 3}) {
        for (const int step : {8, 4}) {
            const Load load = make_load(1920, 1080, 8, step, 8, 7, slices, 5u);
            Device dev(load);
            const int n = static_cast<int>(load.patches.size());
            nss_cuda::OrderedAggregator ordered(load.width, load.height, slices, 8, load.patches.size(), nullptr);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            auto time = [&](auto&& body) {
                for (int i = 0; i < 3; ++i) body();
                cudaEventRecord(a);
                constexpr int kIters = 50;
                for (int i = 0; i < kIters; ++i) body();
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, a, b);
                return ms / kIters;
            };
            const float t_ordered = time([&] {
                ordered.run(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(), n, dev.target, nullptr);
            });
            const float t_atomic = time([&] {
                cudaMemsetAsync(dev.num.get(), 0, dev.num.bytes());
                cudaMemsetAsync(dev.den.get(), 0, dev.den.bytes());
                nss_cuda::aggregate_atomic(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(), n, 8,
                                           dev.target, nullptr);
            });
            // Repeatability of the atomic path over a few runs.
            std::vector<float> first;
            int differing = 0;
            for (int run = 0; run < 5; ++run) {
                cudaMemset(dev.num.get(), 0, dev.num.bytes());
                cudaMemset(dev.den.get(), 0, dev.den.bytes());
                nss_cuda::aggregate_atomic(dev.values.as<float>(), dev.patches.as<nss_cuda::AggregatePatch>(), n, 8,
                                           dev.target, nullptr);
                const auto out = dev.fetch(dev.num);
                if (run == 0) first = out;
                else differing += std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) != 0;
            }
            std::printf("bench 1920x1080 b8 g8 step%d slices%d: %d patches, ordered %.3f ms, atomic %.3f ms, "
                        "atomic runs differing from run 0: %d/4\n",
                        step, slices, n, t_ordered, t_atomic, differing);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (nss_cuda::runtime_info().device_count == 0) {
        std::printf("test_cuda_aggregate SKIP: no CUDA device\n");
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench();
        return 0;
    }
    int failures = 0;
    for (const int block : {1, 4, 8, 12, 16, 32}) {
        for (const int slices : {1, 3}) failures += correctness(block, slices);
    }
    std::printf("test_cuda_aggregate: %d failures\n", failures);
    return failures ? 1 : 0;
}
