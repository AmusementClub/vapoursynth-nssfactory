// SPDX-License-Identifier: GPL-2.0-only
// GPU block matching vs the CPU matcher contract (nss::spatial_match /
// nss::predictive_match, and nss::image_match for joint channels, exact-size
// windows and groups up to 256). Positions must agree; a disagreement is accepted
// only as a near-tie (the two backends sum SSD in different orders), and on
// a piecewise-constant image with exact ties the order must match exactly.
#include "cuda/common/match.hpp"
#include "cuda/runtime/device.hpp"
#include "cuda/runtime/memory.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_image.hpp"

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

// nss::image_match: `nch` channels summed, the exact-size window
// [c - window / 2, c + window - window / 2 - 1], groups up to 256, and the
// predictive search over `frames` frames from reference frame t0.
void image_case(int width, int height, int block, int group, int step, int window, int nch, int frames, int t0,
                int radius, int ps_num, bool flat, Stats& stats) {
    const std::size_t plane_floats = static_cast<std::size_t>(width) * height;
    std::vector<std::vector<float>> host(frames);  // per frame: nch stacked planes
    std::vector<nss_cuda::DeviceBuffer> device;
    std::vector<const float*> guides, dev_ptrs;
    for (int t = 0; t < frames; ++t) {
        for (int c = 0; c < nch; ++c) {
            const auto plane = make_plane(width, height, 17u * t + 5u * c + 1u, flat);
            host[t].insert(host[t].end(), plane.begin(), plane.end());
        }
        device.emplace_back(host[t].size() * sizeof(float));
        NSS_CUDA_CHECK(cudaMemcpy(device.back().get(), host[t].data(), host[t].size() * sizeof(float), cudaMemcpyHostToDevice));
        dev_ptrs.push_back(device.back().as<float>());
    }
    for (int t = 0; t < frames; ++t) {
        for (int c = 0; c < nch; ++c) guides.push_back(host[t].data() + c * plane_floats);
    }
    nss_cuda::DeviceBuffer d_ptrs(dev_ptrs.size() * sizeof(float*));
    NSS_CUDA_CHECK(cudaMemcpy(d_ptrs.get(), dev_ptrs.data(), d_ptrs.bytes(), cudaMemcpyHostToDevice));

    const auto grid = nss_cuda::make_raster_grid(width, height, block, step);
    nss_cuda::DeviceBuffer d_out(static_cast<std::size_t>(grid.count()) * group * sizeof(nss_cuda::DeviceMatch));
    nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(grid.count()) * sizeof(int));
    nss_cuda::MatchGeometry g{width, height, width, block, window / 2, group};
    g.range_hi = window - window / 2 - 1;
    g.channels = nch;
    g.channel_step = static_cast<long long>(plane_floats);
    if (radius > 0 && frames > 1) {
        const nss_cuda::TemporalWindow w{d_ptrs.as<const float*>(), frames, t0, radius, 0, frames, ps_num, 4};
        nss_cuda::predictive_match(g, w, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(), d_counts.as<int>(), nullptr);
    } else {
        nss_cuda::spatial_match(dev_ptrs[t0], g, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(),
                                d_counts.as<int>(), nullptr);
    }
    std::vector<nss_cuda::DeviceMatch> out(static_cast<std::size_t>(grid.count()) * group);
    std::vector<int> counts(grid.count());
    NSS_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    NSS_CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    char label[128];
    std::snprintf(label, sizeof label, "image %dx%d b%d g%d s%d w%d ch%d f%d t%d R%d%s", width, height, block, group,
                  step, window, nch, frames, t0, radius, flat ? " flat" : "");
    const nss::ImageSearch search{block, step, group, window, radius, ps_num, 4};
    std::vector<nss::Match> cpu(256);
    for (int ref = 0; ref < grid.count(); ++ref) {
        const int n = nss::image_match(guides.data(), frames, nch, width, height, t0, grid.x(ref), grid.y(ref), search,
                                       cpu.data());
        nss_cuda::DeviceMatch* gpu = out.data() + static_cast<std::size_t>(ref) * group;
        // The spatial kernels report slot 0; the CPU reports the sequence index.
        if (!(radius > 0 && frames > 1)) {
            for (int i = 0; i < counts[ref]; ++i) gpu[i].t = t0;
        }
        compare_group(cpu.data(), n, gpu, counts[ref], flat, stats, label, ref);
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
    for (const bool flat : {false, true}) {
        for (const int nch : {1, 2, 3}) {
            image_case(61, 57, 8, 8, 5, 15, nch, 1, 0, 0, 2, flat, stats);    // MCWNNM-like joint groups
            image_case(72, 64, 8, 16, 6, 40, nch, 1, 0, 0, 2, flat, stats);   // NLH window (even: asymmetric)
        }
        image_case(72, 64, 8, 90, 7, 60, 1, 1, 0, 0, 2, flat, stats);         // TWSC default group
        image_case(72, 64, 4, 256, 9, 33, 2, 1, 0, 0, 2, flat, stats);        // group limit
        image_case(40, 36, 16, 64, 8, 129, 1, 1, 0, 0, 2, flat, stats);       // window larger than the plane
    }
    // Predictive search (ties across frames are ordered differently, so only
    // the noisy planes): warp path, block top-K, and the 256-entry kernel.
    image_case(64, 56, 8, 16, 6, 40, 1, 3, 1, 1, 2, false, stats);
    image_case(64, 56, 8, 16, 6, 21, 3, 5, 0, 2, 3, false, stats);
    image_case(64, 56, 8, 16, 6, 21, 1, 5, 4, 2, 3, false, stats);
    image_case(64, 56, 8, 64, 8, 21, 2, 3, 1, 1, 20, false, stats);
    image_case(64, 56, 4, 200, 8, 25, 1, 3, 2, 1, 100, false, stats);
    // Lane kernels: several overlapping windows per layer, ps_num equal to
    // the group, two columns per lane (block 16, 8 lanes), parts (block 4
    // and 8 with 16 and 32 lanes) and the largest group they take.
    image_case(64, 56, 8, 8, 6, 15, 1, 5, 2, 2, 8, false, stats);
    image_case(64, 56, 4, 4, 5, 15, 1, 3, 1, 1, 4, false, stats);
    image_case(64, 56, 16, 8, 7, 15, 1, 3, 1, 1, 4, false, stats);
    image_case(64, 56, 16, 16, 7, 15, 1, 5, 2, 2, 6, false, stats);
    image_case(64, 56, 4, 16, 5, 21, 1, 3, 1, 1, 12, false, stats);
    image_case(64, 56, 8, 33, 6, 21, 1, 3, 1, 1, 20, false, stats);
    image_case(64, 56, 16, 33, 9, 21, 1, 3, 0, 1, 5, false, stats);
    // Blocks 1, 2, 12 (lanes without a column) and 32 (one reference per warp).
    for (const bool flat : {false, true}) {
        for (const int block : {1, 2, 32}) {
            for (const int group : {2, 8, 16, 32}) spatial_case(97, 83, block, group, std::min(block, 3), 7, flat, stats);
        }
        spatial_case(97, 83, 12, 33, 5, 9, flat, stats);
        temporal_case(71, 66, 12, 8, 5, 7, 1, 1, 3, flat, stats);
        temporal_case(71, 66, 12, 16, 5, 7, 2, 0, 9, flat, stats);
        temporal_case(71, 66, 2, 8, 2, 7, 1, 1, 3, flat, stats);
        temporal_case(71, 66, 1, 16, 1, 5, 1, 1, 3, flat, stats);
        temporal_case(71, 66, 32, 8, 9, 7, 1, 1, 3, flat, stats);
    }
    for (const bool flat : {false, true}) {
        temporal_case(71, 66, 16, 8, 6, 7, 1, 1, 3, flat, stats);
        temporal_case(71, 66, 16, 16, 6, 7, 2, 2, 5, flat, stats);
        temporal_case(71, 66, 4, 8, 5, 7, 2, 0, 9, flat, stats);
    }
    std::printf("test_cuda_match: %ld refs, %ld entries, %ld near-tie flips, %d failures\n", stats.refs, stats.entries,
                stats.flips, failures);
    if (stats.flips * 100 > stats.entries) {
        std::printf("FAIL: near-tie flips exceed 1%% of entries\n");
        ++failures;
    }
    return failures ? 1 : 0;
}
