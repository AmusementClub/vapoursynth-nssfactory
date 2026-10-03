// SPDX-License-Identifier: GPL-2.0-only
// GPU block matching vs the CPU matcher contract (nss::spatial_match /
// nss::predictive_match). Positions must agree; a disagreement is accepted
// only as a near-tie (the two backends sum SSD in different orders), and on
// a piecewise-constant image with exact ties the order must match exactly.
#include "cuda/common/match.hpp"
#include "cuda/runtime/device.hpp"
#include "cuda/runtime/memory.hpp"
#include "nss/cpu_api.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            std::printf("FAIL: " __VA_ARGS__);   \
            std::printf("\n");                   \
            ++failures;                          \
        }                                        \
    } while (0)

std::vector<float> make_plane(int width, int height, unsigned seed, bool flat) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.05f);
    std::vector<float> plane(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float v = 0.5f + 0.3f * std::sin(x * 0.21f + seed) * std::cos(y * 0.17f);
            if (flat) v = static_cast<float>(((x / 8) + (y / 8) + seed) % 3) * 0.25f;  // exact ties
            else v += noise(rng);
            plane[static_cast<std::size_t>(y) * width + x] = v;
        }
    }
    return plane;
}

bool near_tie(float a, float b) {
    return std::fabs(a - b) <= 2e-5f * std::max(1.f, std::max(std::fabs(a), std::fabs(b)));
}

struct Stats {
    long refs = 0, entries = 0, flips = 0;
};

void compare_group(const nss::Match* cpu, int cpu_n, const nss_cuda::DeviceMatch* gpu, int gpu_n, bool exact,
                   Stats& stats, const char* label, int ref) {
    stats.refs++;
    CHECK(cpu_n == gpu_n, "%s ref %d: count cpu %d gpu %d", label, ref, cpu_n, gpu_n);
    const int n = std::min(cpu_n, gpu_n);
    for (int i = 0; i < n; ++i) {
        stats.entries++;
        const bool same = cpu[i].x == gpu[i].x && cpu[i].y == gpu[i].y && cpu[i].t == gpu[i].t;
        if (!same) {
            stats.flips++;
            CHECK(!exact, "%s ref %d entry %d: exact-tie order differs cpu (%d,%d,%d) gpu (%d,%d,%d)", label, ref, i,
                  cpu[i].x, cpu[i].y, cpu[i].t, gpu[i].x, gpu[i].y, gpu[i].t);
            CHECK(near_tie(cpu[i].dist, gpu[i].dist), "%s ref %d entry %d: not a near tie (cpu %.9g gpu %.9g)", label,
                  ref, i, cpu[i].dist, gpu[i].dist);
        } else {
            CHECK(near_tie(cpu[i].dist, gpu[i].dist), "%s ref %d entry %d: distance cpu %.9g gpu %.9g", label, ref, i,
                  cpu[i].dist, gpu[i].dist);
        }
    }
}

void spatial_case(int width, int height, int block, int group, int step, int range, bool flat, Stats& stats) {
    const auto plane = make_plane(width, height, 7u + block + group, flat);
    nss_cuda::DeviceBuffer d_plane(plane.size() * sizeof(float));
    NSS_CUDA_CHECK(cudaMemcpy(d_plane.get(), plane.data(), plane.size() * sizeof(float), cudaMemcpyHostToDevice));
    const auto grid = nss_cuda::make_raster_grid(width, height, block, step);
    nss_cuda::DeviceBuffer d_out(static_cast<std::size_t>(grid.count()) * group * sizeof(nss_cuda::DeviceMatch));
    nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(grid.count()) * sizeof(int));
    const nss_cuda::MatchGeometry g{width, height, width, block, range, group};
    nss_cuda::spatial_match(d_plane.as<float>(), g, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(), d_counts.as<int>(), nullptr);
    std::vector<nss_cuda::DeviceMatch> out(static_cast<std::size_t>(grid.count()) * group);
    std::vector<int> counts(grid.count());
    NSS_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    NSS_CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    char label[96];
    std::snprintf(label, sizeof label, "spatial %dx%d b%d g%d s%d r%d%s", width, height, block, group, step, range,
                  flat ? " flat" : "");
    for (int ref = 0; ref < grid.count(); ++ref) {
        nss::Match cpu[nss_cuda::kMaxGroup];
        const int n = nss::spatial_match(plane.data(), width, width, height, grid.x(ref), grid.y(ref), block, range,
                                         group, cpu);
        compare_group(cpu, n, out.data() + static_cast<std::size_t>(ref) * group, counts[ref], flat, stats, label, ref);
    }
}

void temporal_case(int width, int height, int block, int group, int step, int range, int radius, int center,
                   int frame_count, bool flat, Stats& stats) {
    const int ntemp = 2 * radius + 1;
    std::vector<std::vector<float>> frames;
    std::vector<nss_cuda::DeviceBuffer> d_frames;
    std::vector<const float*> host_ptrs, dev_ptrs;
    std::vector<int> strides(ntemp, width);
    for (int t = 0; t < ntemp; ++t) {
        frames.push_back(make_plane(width, height, 31u * t + 3u, flat));
        d_frames.emplace_back(frames.back().size() * sizeof(float));
        NSS_CUDA_CHECK(cudaMemcpy(d_frames.back().get(), frames.back().data(), frames.back().size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        host_ptrs.push_back(frames.back().data());
        dev_ptrs.push_back(d_frames.back().as<float>());
    }
    nss_cuda::DeviceBuffer d_ptrs(dev_ptrs.size() * sizeof(float*));
    NSS_CUDA_CHECK(cudaMemcpy(d_ptrs.get(), dev_ptrs.data(), d_ptrs.bytes(), cudaMemcpyHostToDevice));

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = range;
    cfg.radius = radius;
    cfg.ps_num = std::min(2, group);
    cfg.ps_range = 4;
    cfg.valid_t_begin = std::max(0, radius - center);
    cfg.valid_t_end = radius + std::min(radius + 1, frame_count - center);

    const auto grid = nss_cuda::make_raster_grid(width, height, block, step);
    nss_cuda::DeviceBuffer d_out(static_cast<std::size_t>(grid.count()) * group * sizeof(nss_cuda::DeviceMatch));
    nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(grid.count()) * sizeof(int));
    const nss_cuda::MatchGeometry g{width, height, width, block, range, group};
    const nss_cuda::TemporalWindow w{d_ptrs.as<const float*>(), ntemp, radius, radius, cfg.valid_t_begin,
                                     cfg.valid_t_end, cfg.ps_num, cfg.ps_range};
    nss_cuda::predictive_match(g, w, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(), d_counts.as<int>(), nullptr);
    std::vector<nss_cuda::DeviceMatch> out(static_cast<std::size_t>(grid.count()) * group);
    std::vector<int> counts(grid.count());
    NSS_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    NSS_CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    char label[128];
    std::snprintf(label, sizeof label, "temporal %dx%d b%d g%d s%d r%d R%d c%d/%d%s", width, height, block, group, step,
                  range, radius, center, frame_count, flat ? " flat" : "");
    for (int ref = 0; ref < grid.count(); ++ref) {
        nss::Match cpu[nss_cuda::kMaxGroup];
        const int n = nss::predictive_match(host_ptrs.data(), strides.data(), ntemp, width, height, grid.x(ref),
                                            grid.y(ref), radius, cfg, cpu);
        compare_group(cpu, n, out.data() + static_cast<std::size_t>(ref) * group, counts[ref], flat, stats, label, ref);
    }
}

}  // namespace

int main() {
    if (nss_cuda::runtime_info().device_count == 0) {
        std::printf("test_cuda_match SKIP: no CUDA device\n");
        return 77;
    }
    Stats stats;
    for (const bool flat : {false, true}) {
        for (const int block : {4, 8, 12, 16}) {
            for (const int group : {1, 8, 16, 32}) {
                spatial_case(97, 83, block, group, 3, 7, flat, stats);
                spatial_case(64, 61, block, group, 8, 12, flat, stats);
            }
        }
        spatial_case(80, 72, 8, 64, 4, 40, flat, stats);  // > one top-K chunk of candidates
        for (const int radius : {1, 2}) {
            temporal_case(71, 66, 8, 8, 4, 7, radius, radius, 2 * radius + 1, flat, stats);
            temporal_case(71, 66, 8, 16, 5, 7, radius, 0, 9, flat, stats);  // clip start: earlier frames invalid
            temporal_case(71, 66, 4, 32, 6, 9, radius, 8, 9, flat, stats);  // clip end
        }
    }
    std::printf("test_cuda_match: %ld refs, %ld entries, %ld near-tie flips, %d failures\n", stats.refs, stats.entries,
                stats.flips, failures);
    if (stats.flips * 100 > stats.entries) {
        std::printf("FAIL: near-tie flips exceed 1%% of entries\n");
        ++failures;
    }
    return failures ? 1 : 0;
}
