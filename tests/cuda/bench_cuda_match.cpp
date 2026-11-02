// SPDX-License-Identifier: GPL-2.0-only
// Kernel-level timing of the exact spatial matchers on a 1080p plane (not a
// test): bench_cuda_match [reps]. Each line is one shape, every path that
// takes it timed with CUDA events over `reps` launches after a warm-up, the
// paths' outputs checked for the same bits.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"
#include "cuda/runtime/memory.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using Path = std::function<bool(const float*, const nss_cuda::MatchGeometry&, const nss_cuda::RasterGrid&, int, int,
                                nss_cuda::DeviceMatch*, int*, cudaStream_t)>;

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const int width = 1920, height = 1080;
    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.f, 0.04f);
    std::vector<float> plane(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            plane[static_cast<std::size_t>(y) * width + x] =
                0.5f + 0.25f * std::sin(x * 0.013f) * std::cos(y * 0.021f) + noise(rng);
        }
    }
    nss_cuda::DeviceBuffer d_plane(plane.size() * sizeof(float));
    NSS_CUDA_CHECK(cudaMemcpy(d_plane.get(), plane.data(), d_plane.bytes(), cudaMemcpyHostToDevice));
    const Path lanes = nss_cuda::spatial_match_lanes;
    const Path window = nss_cuda::spatial_match_window;
    const Path column = nss_cuda::spatial_match_column;
    struct Shape {
        int block, step, range, group;
    };
    std::vector<Shape> shapes;
    for (const int range : {4, 6, 9, 12, 16}) shapes.push_back({8, 8, range, 8});
    for (const int group : {2, 4, 16, 32}) shapes.push_back({8, 8, 9, group});
    for (const int step : {1, 2, 3, 4, 5, 6, 7}) shapes.push_back({8, step, 9, 8});
    for (const int step : {1, 3, 4, 6}) shapes.push_back({8, step, 9, 16});
    shapes.push_back({8, 4, 16, 16});
    shapes.push_back({8, 3, 7, 16});
    cudaEvent_t start, stop;
    NSS_CUDA_CHECK(cudaEventCreate(&start));
    NSS_CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("block step range group refs | lanes_us window_us column_us | window/lanes same\n");
    for (const Shape& s : shapes) {
        const auto grid = nss_cuda::make_raster_grid(width, height, s.block, s.step);
        const nss_cuda::MatchGeometry g{width, height, width, s.block, s.range, s.group};
        const int n = grid.count();
        nss_cuda::DeviceBuffer outs[3]{nss_cuda::DeviceBuffer(static_cast<std::size_t>(n) * s.group * sizeof(nss_cuda::DeviceMatch)),
                                       nss_cuda::DeviceBuffer(static_cast<std::size_t>(n) * s.group * sizeof(nss_cuda::DeviceMatch)),
                                       nss_cuda::DeviceBuffer(static_cast<std::size_t>(n) * s.group * sizeof(nss_cuda::DeviceMatch))};
        nss_cuda::DeviceBuffer counts[3]{nss_cuda::DeviceBuffer(n * sizeof(int)), nss_cuda::DeviceBuffer(n * sizeof(int)),
                                         nss_cuda::DeviceBuffer(n * sizeof(int))};
        const Path* paths[3]{&lanes, &window, &column};
        double us[3]{-1, -1, -1};
        for (int p = 0; p < 3; ++p) {
            if (p == 2 && !nss_cuda::spatial_match_column_takes(g, grid)) continue;
            if (p == 1 && !nss_cuda::spatial_match_window_takes(g, grid)) continue;
            NSS_CUDA_CHECK(cudaMemset(outs[p].get(), 0, outs[p].bytes()));
            const auto run = [&] {
                return (*paths[p])(d_plane.as<float>(), g, grid, 0, n, outs[p].as<nss_cuda::DeviceMatch>(),
                                   counts[p].as<int>(), nullptr);
            };
            if (!run()) continue;
            run();
            NSS_CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < reps; ++i) run();
            NSS_CUDA_CHECK(cudaEventRecord(stop));
            NSS_CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.f;
            NSS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            us[p] = 1000.0 * ms / reps;
        }
        bool same = true;
        if (us[0] >= 0 && us[1] >= 0) {
            std::vector<nss_cuda::DeviceMatch> a(static_cast<std::size_t>(n) * s.group), b(a.size());
            std::vector<int> ca(n), cb(n);
            NSS_CUDA_CHECK(cudaMemcpy(a.data(), outs[0].get(), outs[0].bytes(), cudaMemcpyDeviceToHost));
            NSS_CUDA_CHECK(cudaMemcpy(b.data(), outs[1].get(), outs[1].bytes(), cudaMemcpyDeviceToHost));
            NSS_CUDA_CHECK(cudaMemcpy(ca.data(), counts[0].get(), n * sizeof(int), cudaMemcpyDeviceToHost));
            NSS_CUDA_CHECK(cudaMemcpy(cb.data(), counts[1].get(), n * sizeof(int), cudaMemcpyDeviceToHost));
            for (int r = 0; r < n && same; ++r) {
                same = ca[r] == cb[r] && !std::memcmp(a.data() + static_cast<std::size_t>(r) * s.group,
                                                      b.data() + static_cast<std::size_t>(r) * s.group,
                                                      sizeof(nss_cuda::DeviceMatch) * ca[r]);
            }
        }
        std::printf("%5d %4d %5d %5d %6d | %8.1f %9.1f %9.1f | %6.3f %s\n", s.block, s.step, s.range, s.group, n, us[0],
                    us[1], us[2], us[0] > 0 && us[1] > 0 ? us[0] / us[1] : 0.0, same ? "yes" : "NO");
    }
    return 0;
}
