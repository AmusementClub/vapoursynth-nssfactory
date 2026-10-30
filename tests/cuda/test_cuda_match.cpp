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
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <utility>
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

// The exact column matcher against the existing lane path and the CPU, in
// uneven batches that do not start on grid rows.
void column_case(int block, int width, int height, int group, int step, int range, int batches, bool flat, Stats& stats) {
    const auto plane = make_plane(width, height, 19u + block + group + step + range, flat);
    nss_cuda::DeviceBuffer d_plane(plane.size() * sizeof(float));
    NSS_CUDA_CHECK(cudaMemcpy(d_plane.get(), plane.data(), plane.size() * sizeof(float), cudaMemcpyHostToDevice));
    const auto grid = nss_cuda::make_raster_grid(width, height, block, step);
    const int n = grid.count();
    const nss_cuda::MatchGeometry g{width, height, width, block, range, group};
    CHECK(nss_cuda::spatial_match_column_takes(g, grid), "column %dx%d b%d g%d s%d not supported", width, height, block,
          group, step);
    std::vector<nss_cuda::DeviceMatch> out[2];
    std::vector<int> counts[2];
    for (int path = 0; path < 2; ++path) {
        nss_cuda::DeviceBuffer d_out(static_cast<std::size_t>(n) * group * sizeof(nss_cuda::DeviceMatch));
        nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(n) * sizeof(int));
        NSS_CUDA_CHECK(cudaMemset(d_out.get(), 0, d_out.bytes()));
        NSS_CUDA_CHECK(cudaMemset(d_counts.get(), 0, d_counts.bytes()));
        for (int b = 0; b < batches; ++b) {
            const int begin = static_cast<int>(static_cast<long long>(n) * b * b / (batches * batches));
            const int end = static_cast<int>(static_cast<long long>(n) * (b + 1) * (b + 1) / (batches * batches));
            if (path) {
                CHECK(nss_cuda::spatial_match_column(d_plane.as<float>(), g, grid, begin, end - begin,
                                                    d_out.as<nss_cuda::DeviceMatch>() +
                                                        static_cast<std::size_t>(begin) * group,
                                                    d_counts.as<int>() + begin, nullptr),
                      "column %dx%d b%d g%d s%d batch declined", width, height, block, group, step);
            } else {
                nss_cuda::spatial_match(d_plane.as<float>(), g, grid, begin, end - begin,
                                        d_out.as<nss_cuda::DeviceMatch>() + static_cast<std::size_t>(begin) * group,
                                        d_counts.as<int>() + begin, nullptr);
            }
        }
        out[path].resize(static_cast<std::size_t>(n) * group);
        counts[path].resize(n);
        NSS_CUDA_CHECK(cudaMemcpy(out[path].data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
        NSS_CUDA_CHECK(cudaMemcpy(counts[path].data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    }
    char label[112];
    std::snprintf(label, sizeof label, "column %dx%d b%d g%d s%d r%d batches %d%s", width, height, block, group, step,
                  range, batches, flat ? " flat" : "");
    for (int ref = 0; ref < n; ++ref) {
        CHECK(counts[0][ref] == counts[1][ref], "%s ref %d: lane count %d column count %d", label, ref, counts[0][ref],
              counts[1][ref]);
        const int count = std::min(counts[0][ref], counts[1][ref]);
        CHECK(std::memcmp(out[0].data() + static_cast<std::size_t>(ref) * group,
                          out[1].data() + static_cast<std::size_t>(ref) * group,
                          sizeof(nss_cuda::DeviceMatch) * count) == 0,
              "%s ref %d: lane and column results differ", label, ref);
        nss::Match cpu[nss_cuda::kMaxGroup];
        const int m = nss::spatial_match(plane.data(), width, width, height, grid.x(ref), grid.y(ref), block, range, group,
                                         cpu);
        compare_group(cpu, m, out[1].data() + static_cast<std::size_t>(ref) * group, counts[1][ref], flat, stats, label,
                      ref);
    }
}

// The square matcher (block = reuse * step) against the CPU, in `batches`
// uneven batches that do not start on grid rows.
void square_case(int q, int width, int height, int group, int range, int batches, bool flat, Stats& stats,
                 int reuse = 2) {
    const auto plane = make_plane(width, height, 13u + q + group + range, flat);
    nss_cuda::DeviceBuffer d_plane(plane.size() * sizeof(float));
    NSS_CUDA_CHECK(cudaMemcpy(d_plane.get(), plane.data(), plane.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int block = reuse * q;
    const auto grid = nss_cuda::make_raster_grid(width, height, block, q);
    const int n = grid.count();
    nss_cuda::DeviceBuffer d_out(static_cast<std::size_t>(n) * group * sizeof(nss_cuda::DeviceMatch));
    nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(n) * sizeof(int));
    NSS_CUDA_CHECK(cudaMemset(d_counts.get(), 0, d_counts.bytes()));
    const nss_cuda::MatchGeometry g{width, height, width, block, range, group};
    CHECK(nss_cuda::spatial_match_square_shape(g, grid), "square b%d s%d %dx%d g%d not supported", block, q, width,
          height, group);
    for (int b = 0; b < batches; ++b) {
        const int begin = static_cast<int>(static_cast<long long>(n) * b * b / (batches * batches));
        const int end = static_cast<int>(static_cast<long long>(n) * (b + 1) * (b + 1) / (batches * batches));
        nss_cuda::spatial_match_square(d_plane.as<float>(), g, grid, begin, end - begin,
                                       d_out.as<nss_cuda::DeviceMatch>() + static_cast<std::size_t>(begin) * group,
                                       d_counts.as<int>() + begin, nullptr, true);
    }
    std::vector<nss_cuda::DeviceMatch> out(static_cast<std::size_t>(n) * group);
    std::vector<int> counts(n);
    NSS_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    NSS_CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    char label[96];
    std::snprintf(label, sizeof label, "square b%d s%d %dx%d g%d r%d batches %d%s", block, q, width, height, group,
                  range, batches, flat ? " flat" : "");
    for (int ref = 0; ref < n; ++ref) {
        CHECK(counts[ref] >= 1, "%s ref %d: no result", label, ref);
        nss::Match cpu[nss_cuda::kMaxGroup];
        const int m = nss::spatial_match(plane.data(), width, width, height, grid.x(ref), grid.y(ref), block, range,
                                         group, cpu);
        compare_group(cpu, m, out.data() + static_cast<std::size_t>(ref) * group, counts[ref], flat, stats, label,
                      ref);
    }
}

void temporal_case(int width, int height, int block, int group, int step, int range, int radius, int center,
                   int frame_count, bool flat, Stats& stats, bool square = false) {
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
    if (square) {
        CHECK(nss_cuda::predictive_match_square(g, w, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(),
                                                d_counts.as<int>(), nullptr, true),
              "square temporal %dx%d b%d s%d not taken", width, height, block, step);
    } else {
        nss_cuda::predictive_match(g, w, grid, 0, grid.count(), d_out.as<nss_cuda::DeviceMatch>(), d_counts.as<int>(),
                                   nullptr);
    }
    std::vector<nss_cuda::DeviceMatch> out(static_cast<std::size_t>(grid.count()) * group);
    std::vector<int> counts(grid.count());
    NSS_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    NSS_CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
    char label[128];
    std::snprintf(label, sizeof label, "temporal%s %dx%d b%d g%d s%d r%d R%d c%d/%d%s", square ? " square" : "", width,
                  height, block, group, step, range, radius, center, frame_count, flat ? " flat" : "");
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

// The tensor-core matcher (spatial_match with a prepared workspace) against
// the lane kernels on the same plane: the same match lists, byte for byte, for
// whole grids and for batches that do not start on a multiple of 8. `kind`: 0
// noisy, 1 exact ties, 2 a NaN, 3 a value FP16 cannot hold (2 and 3 take the
// exact scan of every candidate).
int tensor_runs = 0;
void tensor_case(int width, int height, int group, int step, int lo, int hi, int kind) {
    auto plane = make_plane(width, height, 11u + group + step, kind == 1);
    if (kind == 2) plane[static_cast<std::size_t>(height / 2) * width + width / 3] = std::nanf("");
    if (kind == 3) plane[static_cast<std::size_t>(height / 3) * width + width / 2] = 1.0e6f;
    nss_cuda::MatchGeometry g{width, height, width, 8, lo, group};
    if (hi >= 0) g.range_hi = hi;
    const auto grid = nss_cuda::make_raster_grid(width, height, 8, step);
    if (!nss_cuda::spatial_match_tensor(g, grid)) {
        // Refused only below sm_80, or where the device runs the compute_75 PTX (JIT forced).
        CHECK(nss_cuda::device_info(0, nss_cuda::runtime_info()).cc_major < 8 || std::getenv("CUDA_FORCE_PTX_JIT"),
              "tensor %dx%d g%d s%d [-%d,+%d] not taken", width, height, group, step, lo, g.hi());
        return;
    }
    ++tensor_runs;
    nss_cuda::DeviceBuffer d_plane(plane.size() * sizeof(float));
    NSS_CUDA_CHECK(cudaMemcpy(d_plane.get(), plane.data(), plane.size() * sizeof(float), cudaMemcpyHostToDevice));
    nss_cuda::DeviceBuffer scratch(nss_cuda::spatial_match_scratch_bytes(width, height));
    nss_cuda::spatial_match_prepare(d_plane.as<float>(), g, scratch.get(), nullptr);
    const int n = grid.count();
    const std::size_t matches = static_cast<std::size_t>(n) * group;
    for (const int batch : {n, 37}) {
        const int start = batch == n ? 0 : 5;
        std::vector<nss_cuda::DeviceMatch> out[2];
        std::vector<int> counts[2];
        for (int path = 0; path < 2; ++path) {
            nss_cuda::DeviceBuffer d_out(matches * sizeof(nss_cuda::DeviceMatch));
            nss_cuda::DeviceBuffer d_counts(static_cast<std::size_t>(n) * sizeof(int));
            NSS_CUDA_CHECK(cudaMemset(d_out.get(), 0, d_out.bytes()));
            NSS_CUDA_CHECK(cudaMemset(d_counts.get(), 0, d_counts.bytes()));
            for (int begin = start; begin < n; begin += batch) {
                const int count = std::min(batch, n - begin);
                nss_cuda::spatial_match(d_plane.as<float>(), g, grid, begin, count,
                                        d_out.as<nss_cuda::DeviceMatch>() + static_cast<std::size_t>(begin) * group,
                                        d_counts.as<int>() + begin, nullptr, path ? scratch.get() : nullptr);
            }
            out[path].resize(matches);
            counts[path].resize(n);
            NSS_CUDA_CHECK(cudaMemcpy(out[path].data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
            NSS_CUDA_CHECK(cudaMemcpy(counts[path].data(), d_counts.get(), d_counts.bytes(), cudaMemcpyDeviceToHost));
        }
        int bad = 0;
        for (int ref = start; ref < n; ++ref) {
            const bool same = counts[0][ref] == counts[1][ref] &&
                              std::memcmp(out[0].data() + static_cast<std::size_t>(ref) * group,
                                          out[1].data() + static_cast<std::size_t>(ref) * group,
                                          sizeof(nss_cuda::DeviceMatch) * std::max(0, counts[0][ref])) == 0;
            bad += !same;
        }
        CHECK(bad == 0, "tensor %dx%d g%d s%d [-%d,+%d] kind %d batch %d: %d of %d references differ", width, height,
              group, step, lo, g.hi(), kind, batch, bad, n - start);
    }
}

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
        // Joint lane kernels: blocks below 8 and every lane count.
        image_case(61, 57, 4, 8, 5, 15, 3, 1, 0, 0, 2, flat, stats);
        image_case(61, 57, 2, 16, 3, 11, 3, 1, 0, 0, 2, flat, stats);
        image_case(61, 57, 1, 32, 3, 9, 3, 1, 0, 0, 2, flat, stats);
        image_case(61, 57, 8, 32, 6, 21, 3, 1, 0, 0, 2, flat, stats);
        image_case(72, 64, 8, 90, 7, 60, 1, 1, 0, 0, 2, flat, stats);         // TWSC default group
        image_case(72, 64, 4, 256, 9, 33, 2, 1, 0, 0, 2, flat, stats);        // group limit
        image_case(40, 36, 16, 64, 8, 129, 1, 1, 0, 0, 2, flat, stats);       // window larger than the plane
    }
    // Predictive search (ties across frames are ordered differently, so only
    // the noisy planes): lane kernels (one channel, and three jointly), warp
    // path (two channels), block top-K, and the 256-entry kernel.
    image_case(64, 56, 8, 16, 6, 40, 1, 3, 1, 1, 2, false, stats);
    image_case(64, 56, 8, 16, 6, 21, 3, 5, 0, 2, 3, false, stats);
    image_case(64, 56, 4, 8, 6, 15, 3, 5, 2, 2, 4, false, stats);
    image_case(64, 56, 8, 32, 6, 15, 3, 3, 1, 1, 6, false, stats);
    image_case(64, 56, 8, 16, 6, 21, 2, 5, 0, 2, 3, false, stats);
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
    // Blocks 1, 2, 7, 12, 15 (lanes without a column) and 32 (one reference per warp).
    for (const bool flat : {false, true}) {
        for (const int block : {7, 15}) {
            for (const int group : {2, 8, 16, 32}) spatial_case(97, 83, block, group, 3, 7, flat, stats);
            image_case(72, 64, block, 16, block - 3, 40, 1, 1, 0, 0, 2, flat, stats);  // NLH presets
            temporal_case(71, 66, block, 16, 5, 7, 1, 1, 3, flat, stats);
        }
        for (const int block : {1, 2, 32}) {
            for (const int group : {2, 8, 16, 32}) spatial_case(97, 83, block, group, std::min(block, 3), 7, flat, stats);
        }
        // Blocks that had no lane instance before 2026-10-19 (the warp kernel served them).
        for (const int block : {3, 5, 6, 9, 10, 11, 13, 14}) {
            for (const int group : {2, 8, 16, 32}) spatial_case(97, 83, block, group, 3, 7, flat, stats);
            temporal_case(71, 66, block, 16, 5, 7, 1, 1, 3, flat, stats);
        }
        for (const int block : {3, 5, 6, 7}) image_case(61, 57, block, 8, 3, 15, 3, 1, 0, 0, 2, flat, stats);
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
    // Exact column reuse at steps 1 to 3, every supported block/group,
    // clamped edges, flat images, one-reference grids and oversized windows.
    for (const bool flat : {false, true}) {
        for (const int group : {8, 16, 32}) {
            for (const int step : {1, 2, 3}) {
                for (const int block : {4, 8, 12, 16}) {
                    const auto grid = nss_cuda::make_raster_grid(97, 83, block, step);
                    const nss_cuda::MatchGeometry g{97, 83, 97, block, 7, group};
                    if ((block == 4 && step == 2) || (block == 16 && group == 32) ||
                        (block == 16 && group == 8 && step == 3)) {
                        CHECK(!nss_cuda::spatial_match_column_takes(g, grid),
                              "rejected shape taken by the column matcher: b%d g%d s%d", block, group, step);
                        continue;
                    }
                    column_case(block, 97, 83, group, step, 7, 3, flat, stats);
                    column_case(block, 101, 76, group, step, 10, 2, flat, stats);
                }
            }
        }
        for (const int block : {4, 8, 12, 16}) {
            column_case(block, 40, 36, 16, 3, 40, 2, flat, stats);
            column_case(block, block, block, 16, 1, 40, 1, flat, stats);
        }
    }
    {
        const auto grid = nss_cuda::make_raster_grid(97, 83, 8, 4);
        const nss_cuda::MatchGeometry g{97, 83, 97, 8, 7, 8};
        CHECK(!nss_cuda::spatial_match_column_takes(g, grid), "step 4 taken by the column matcher");
        const auto unsupported_grid = nss_cuda::make_raster_grid(97, 83, 32, 1);
        const nss_cuda::MatchGeometry unsupported{97, 83, 97, 32, 7, 32};
        CHECK(!nss_cuda::spatial_match_column_takes(unsupported, unsupported_grid),
              "block 32 taken by the column matcher");
    }
    // Square matcher, every quarter side: regular grids (max_x, max_y
    // multiples of q), a clamped last column, row or both, a single
    // reference, windows wider than the plane, every group it takes, uneven
    // batches. The sizes are scaled with q so that every case clamps as named.
    for (const bool flat : {false, true}) {
        for (const int q : {1, 2, 4, 6, 8}) {
            const int u = 25 * q;  // a regular width / height is 2q + k q
            for (const int group : {2, 3, 4, 8, 16, 32}) {
                square_case(q, u, u - q, group, 7, 3, flat, stats);                 // regular
                if (q > 1) {
                    square_case(q, u - 1, u + 1, group, 9, 4, flat, stats);         // both clamped
                    square_case(q, u + 1, u - q, group, 10, 2, flat, stats);        // column clamped
                    square_case(q, u, u + q - 1, group, 10, 2, flat, stats);        // row clamped
                }
            }
            square_case(q, 2 * q, 2 * q, 8, 9, 1, flat, stats);
            square_case(q, 2 * q + 2, 2 * q + 1, 8, 9, 1, flat, stats);
            square_case(q, 10 * q, 9 * q, 16, 40, 2, flat, stats);
            square_case(q, 67 * q, 35 * q, 32, 19, 5, flat, stats);  // several blocks each way
            square_case(q, 331, 203, 16, 12, 7, flat, stats);
        }
    }
    for (const bool flat : {false, true}) {
        for (const auto& shape : {std::pair<int, int>{3, 4}, {4, 3}, {4, 4}, {2, 4}}) {
            const int q = shape.first, reuse = shape.second, u = 25 * q;
            for (const int group : {2, 3, 4, 8, 16, 32}) {
                square_case(q, u, u - q, group, 7, 3, flat, stats, reuse);
                square_case(q, u - 1, u + 1, group, 9, 4, flat, stats, reuse);
                square_case(q, u + 1, u - q, group, 10, 2, flat, stats, reuse);
                square_case(q, u, u + q - 1, group, 10, 2, flat, stats, reuse);
            }
            square_case(q, reuse * q, reuse * q, 8, 9, 1, flat, stats, reuse);
            square_case(q, reuse * q + 2, reuse * q + 1, 8, 9, 1, flat, stats, reuse);
            square_case(q, 10 * q, 9 * q, 16, 40, 2, flat, stats, reuse);
            square_case(q, 67 * q, 35 * q, 32, 19, 5, flat, stats, reuse);
            square_case(q, 331, 203, 16, 12, 7, flat, stats, reuse);
        }
    }
    // Predictive search with the centre frame by the square matcher, every
    // shape, clamped edges, a clipped window at either end of the clip.
    for (const bool flat : {false, true}) {
        temporal_case(161, 123, 8, 16, 4, 9, 1, 3, 9, flat, stats, true);
        temporal_case(161, 123, 8, 8, 4, 9, 2, 0, 9, flat, stats, true);  // clip start
        temporal_case(203, 151, 4, 16, 2, 7, 1, 4, 9, flat, stats, true);
        temporal_case(363, 271, 16, 16, 8, 9, 1, 8, 9, flat, stats, true);  // clip end
        temporal_case(245, 183, 12, 32, 6, 5, 1, 2, 9, flat, stats, true);
        temporal_case(67, 53, 2, 8, 1, 4, 1, 1, 3, flat, stats, true);
        temporal_case(160, 120, 8, 2, 4, 12, 3, 3, 7, flat, stats, true);  // regular grid
        temporal_case(161, 123, 12, 16, 3, 9, 1, 3, 9, flat, stats, true);
        temporal_case(203, 151, 12, 16, 4, 7, 1, 4, 9, flat, stats, true);
        temporal_case(161, 123, 16, 16, 4, 9, 1, 3, 9, flat, stats, true);
        temporal_case(203, 151, 8, 16, 2, 7, 1, 4, 9, flat, stats, true);
    }
    // Tensor-core matcher: windows of side 33 to 129 (symmetric and exact-size), every step, groups 2 to 16.
    for (const int kind : {0, 1}) {
        for (const int group : {2, 8, 16}) {
            tensor_case(200, 150, group, 8, 30, 29, kind);  // TWSC default window
            tensor_case(200, 150, group, 4, 20, 19, kind);  // blind estimate window, two-row warps
            tensor_case(160, 120, group, 8, 20, -1, kind);  // a side of 41
        }
        for (const int step : {1, 3, 5, 6, 7}) tensor_case(180, 96, 9, step, 24, 23, kind);
        tensor_case(220, 140, 12, 8, 64, -1, kind);   // largest side, wider than the plane
        tensor_case(220, 140, 4, 2, 40, 40, kind);
    }
    // Below a side of 40 the lane kernels serve (they are faster there).
    {
        nss_cuda::MatchGeometry g{160, 120, 160, 8, 19, 8};
        CHECK(!nss_cuda::spatial_match_tensor(g, nss_cuda::make_raster_grid(160, 120, 8, 8)), "side 39 taken by the tensor path");
    }
    tensor_case(200, 150, 8, 8, 30, 29, 2);
    tensor_case(200, 150, 16, 4, 20, 19, 3);
    std::printf("test_cuda_match: %d tensor-core cases against the lane kernels\n", tensor_runs);
    std::printf("test_cuda_match: %ld refs, %ld entries, %ld near-tie flips, %d failures\n", stats.refs, stats.entries,
                stats.flips, failures);
    if (stats.flips * 100 > stats.entries) {
        std::printf("FAIL: near-tie flips exceed 1%% of entries\n");
        ++failures;
    }
    return failures ? 1 : 0;
}
