// SPDX-License-Identifier: GPL-2.0-only
// Kernel-level timing of the exact spatial matchers on a 1080p plane (not a
// test): bench_cuda_match [reps]. Each line is one shape, every path that
// takes it timed with CUDA events over `reps` launches after a warm-up, the
// paths' outputs checked for the same bits.
#include "cuda/common/match.hpp"
#include "cuda/runtime/error.hpp"
#include "cuda/runtime/memory.hpp"

#include <algorithm>
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
    const char* sweep = std::getenv("NSS_BENCH_SWEEP");
    if (sweep && *sweep == '1') {
        // Admission sweep: every block, sparse and dense steps, sides 5 to 33.
        for (int block = 2; block <= 8; ++block) {
            for (const int range : {2, 3, 4, 5, 6, 7, 9, 12, 16}) {
                shapes.push_back({block, block, range, 8});
                shapes.push_back({block, std::max(1, block / 2), range, 16});
            }
        }
    } else {
        for (const int range : {4, 6, 9, 12, 16}) shapes.push_back({8, 8, range, 8});
        for (const int group : {2, 4, 16, 32}) shapes.push_back({8, 8, 9, group});
        for (const int step : {1, 2, 3, 4, 5, 6, 7}) shapes.push_back({8, step, 9, 8});
        for (const int step : {1, 3, 4, 6}) shapes.push_back({8, step, 9, 16});
        shapes.push_back({8, 4, 16, 16});
        shapes.push_back({8, 3, 7, 16});
        for (const int block : {4, 5, 6, 7}) shapes.push_back({block, block, 9, 16});
    }
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
    // Temporal: the predictive search (radius 1, ps_num 2, ps_range 4) with
    // its centre by the lane kernels, and seeded by the window matcher.
    std::vector<nss_cuda::DeviceBuffer> frames;
    std::vector<const float*> pointers;
    for (int t = 0; t < 3; ++t) {
        frames.emplace_back(plane.size() * sizeof(float));
        NSS_CUDA_CHECK(cudaMemcpy(frames.back().get(), plane.data() + t * 3, (plane.size() - t * 3) * sizeof(float),
                                  cudaMemcpyHostToDevice));
        NSS_CUDA_CHECK(cudaMemcpy(frames.back().as<float>() + plane.size() - t * 3, plane.data(), t * 3 * sizeof(float),
                                  cudaMemcpyHostToDevice));
        pointers.push_back(frames.back().as<float>());
    }
    nss_cuda::DeviceBuffer table(pointers.size() * sizeof(const float*));
    NSS_CUDA_CHECK(cudaMemcpy(table.get(), pointers.data(), table.bytes(), cudaMemcpyHostToDevice));
    const nss_cuda::TemporalWindow tw{table.as<const float*>(), 3, 1, 1, 0, 3, 2, 4};
    std::printf("temporal block step range group refs | lanes_us window_seeded_us | speed same\n");
    for (const Shape& s : std::vector<Shape>{{8, 8, 9, 8}, {8, 8, 7, 8}, {8, 6, 9, 16}, {8, 8, 12, 16}, {6, 6, 9, 16}}) {
        const auto grid = nss_cuda::make_raster_grid(width, height, s.block, s.step);
        const nss_cuda::MatchGeometry g{width, height, width, s.block, s.range, s.group};
        const int n = grid.count();
        std::vector<nss_cuda::DeviceMatch> host[2];
        std::vector<int> host_counts[2];
        double us[2]{-1, -1};
        for (int p = 0; p < 2; ++p) {
            nss_cuda::DeviceBuffer o(static_cast<std::size_t>(n) * s.group * sizeof(nss_cuda::DeviceMatch));
            nss_cuda::DeviceBuffer c(n * sizeof(int));
            const auto run = [&] {
                if (p == 0) {
                    nss_cuda::predictive_match(g, tw, grid, 0, n, o.as<nss_cuda::DeviceMatch>(), c.as<int>(), nullptr);
                    return true;
                }
                return nss_cuda::predictive_match_window(g, tw, grid, 0, n, o.as<nss_cuda::DeviceMatch>(), c.as<int>(),
                                                         nullptr);
            };
            if (!run()) continue;
            NSS_CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < reps; ++i) run();
            NSS_CUDA_CHECK(cudaEventRecord(stop));
            NSS_CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.f;
            NSS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            us[p] = 1000.0 * ms / reps;
            host[p].resize(static_cast<std::size_t>(n) * s.group);
            host_counts[p].resize(n);
            NSS_CUDA_CHECK(cudaMemcpy(host[p].data(), o.get(), o.bytes(), cudaMemcpyDeviceToHost));
            NSS_CUDA_CHECK(cudaMemcpy(host_counts[p].data(), c.get(), c.bytes(), cudaMemcpyDeviceToHost));
        }
        bool same = us[1] >= 0;
        for (int r = 0; same && r < n; ++r) {
            same = host_counts[0][r] == host_counts[1][r] &&
                   !std::memcmp(host[0].data() + static_cast<std::size_t>(r) * s.group,
                                host[1].data() + static_cast<std::size_t>(r) * s.group,
                                sizeof(nss_cuda::DeviceMatch) * host_counts[0][r]);
        }
        std::printf("%8d %4d %5d %5d %6d | %8.1f %16.1f | %6.3f %s\n", s.block, s.step, s.range, s.group, n, us[0],
                    us[1], us[1] > 0 ? us[0] / us[1] : 0.0, same ? "yes" : "NO");
    }
    return 0;
}
