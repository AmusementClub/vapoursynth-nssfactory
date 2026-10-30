// SPDX-License-Identifier: GPL-2.0-only
// Vulkan block matching (vulkan/common/match.hpp: the lane kernels and the
// block fallback) against the CPU matcher contract (nss::spatial_match,
// nss::predictive_match, and nss::image_match for joint channels and
// exact-size windows), as
// test_cuda_match: positions must agree, a disagreement is accepted only as a
// near tie (the backends sum SSDs in different orders), and on a
// piecewise-constant image with exact ties the order must match exactly. The
// native and the emulated warp layers must give the same bits. Exits 77
// without a supported Vulkan device; an explicit device id may name an
// unsupported one (lavapipe).
//
// usage: test_vk_match [device_id]
#include "nss/cpu_api.hpp"
#include "nss/cpu_image.hpp"
#include "vulkan/common/match.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace nss_vk;

int failures = 0;

#define CHECK(cond, ...)                       \
    do {                                       \
        if (!(cond)) {                         \
            if (failures < 40) {               \
                std::printf("FAIL: " __VA_ARGS__); \
                std::printf("\n");             \
            }                                  \
            ++failures;                        \
        }                                      \
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

void compare_group(const nss::Match* cpu, int cpu_n, const DeviceMatch* gpu, int gpu_n, bool exact, Stats& stats,
                   const char* label, int ref) {
    stats.refs++;
    CHECK(cpu_n == gpu_n, "%s ref %d: count cpu %d vk %d", label, ref, cpu_n, gpu_n);
    const int n = std::min(cpu_n, gpu_n);
    for (int i = 0; i < n; ++i) {
        stats.entries++;
        const bool same = cpu[i].x == gpu[i].x && cpu[i].y == gpu[i].y && cpu[i].t == gpu[i].t;
        if (!same) {
            stats.flips++;
            CHECK(!exact, "%s ref %d entry %d: exact-tie order differs cpu (%d,%d,%d) vk (%d,%d,%d)", label, ref, i,
                  cpu[i].x, cpu[i].y, cpu[i].t, gpu[i].x, gpu[i].y, gpu[i].t);
            CHECK(near_tie(cpu[i].dist, gpu[i].dist), "%s ref %d entry %d: not a near tie (cpu %.9g vk %.9g)", label,
                  ref, i, cpu[i].dist, gpu[i].dist);
        } else {
            CHECK(near_tie(cpu[i].dist, gpu[i].dist), "%s ref %d entry %d: distance cpu %.9g vk %.9g", label, ref, i,
                  cpu[i].dist, gpu[i].dist);
        }
    }
}

struct Run {
    std::vector<DeviceMatch> matches;
    std::vector<int> counts;
};

// Runs one search on both warp layers (native where the device has the
// lanes), and once more in rows of three workgroups (the dispatch split of
// searches wider than 65535 workgroups), and checks that they agree bit for
// bit; returns the native result.
template <class Record>
Run run_both(const std::shared_ptr<Device>& device, Matcher& matcher, const MatchGeometry& g, int ps_num, int refs,
             Record&& record, const char* label) {
    std::vector<Run> runs;
    for (int pass = 0; pass < 3; ++pass) {
        const bool emulated = pass == 1;
        matcher.force_emulation(emulated);
        matcher.limit_groups_x(pass == 2 ? 3 : 65535);
        if (!emulated && matcher.emulated(g, ps_num)) continue;  // no native layer for these lanes here
        DownloadPlane out(device, static_cast<std::size_t>(refs) * g.group * sizeof(DeviceMatch), 1);
        DownloadPlane counts(device, static_cast<std::size_t>(refs) * sizeof(int), 1);
        Slot slot(device);
        record(slot, out.address(), counts.address());
        out.record_download(slot.commands(Slot::Download));
        counts.record_download(slot.commands(Slot::Download));
        slot.submit();
        slot.wait();
        Run run{std::vector<DeviceMatch>(static_cast<std::size_t>(refs) * g.group), std::vector<int>(refs)};
        out.read(run.matches.data(), 0, run.matches.size() * sizeof(DeviceMatch));
        counts.read(run.counts.data(), 0, run.counts.size() * sizeof(int));
        runs.push_back(std::move(run));
    }
    matcher.force_emulation(false);
    matcher.limit_groups_x(65535);
    for (std::size_t i = 1; i < runs.size(); ++i) {
        CHECK(std::memcmp(runs[0].matches.data(), runs[i].matches.data(), runs[0].matches.size() * sizeof(DeviceMatch)) ==
                      0 &&
                  runs[0].counts == runs[i].counts,
              "%s: %s differs", label, i + 1 == runs.size() ? "the dispatch in rows" : "the emulated layer");
    }
    return std::move(runs[0]);
}

// pad: extra floats a device row (pitch > width); split: the references in
// two dispatches (a nonzero ref_begin).
void spatial_case(const std::shared_ptr<Device>& device, Matcher& matcher, int width, int height, int block, int group,
                  int step, int range, bool flat, Stats& stats, int pad = 0, bool split = false) {
    const auto plane = make_plane(width, height, 7u + block + group, flat);
    const int pitch = width + pad;
    UploadPlane d_plane(device, static_cast<std::size_t>(pitch) * sizeof(float), height);
    d_plane.write(plane.data(), static_cast<std::ptrdiff_t>(width * sizeof(float)), width * sizeof(float));
    const RasterGrid grid = make_raster_grid(width, height, block, step);
    const MatchGeometry g{width, height, pitch, block, range, group};
    char label[128];
    std::snprintf(label, sizeof label, "spatial %dx%d b%d g%d s%d r%d%s%s%s", width, height, block, group, step, range,
                  flat ? " flat" : "", pad ? " padded" : "", split ? " split" : "");
    if (!matcher.supports(g, 0)) {
        CHECK(false, "%s: not supported", label);
        return;
    }
    const int before = failures;
    const Run run = run_both(device, matcher, g, 0, grid.count(), [&](Slot& slot, VkDeviceAddress out, VkDeviceAddress counts) {
        if (d_plane.needs_copy()) d_plane.record_upload(slot.commands(Slot::Upload));
        const int first = split ? grid.count() / 3 : grid.count();
        matcher.spatial(slot.commands(Slot::Compute), d_plane.address(), g, grid, 0, first, out, counts);
        matcher.spatial(slot.commands(Slot::Compute), d_plane.address(), g, grid, first, grid.count() - first,
                        out + static_cast<VkDeviceAddress>(first) * group * sizeof(DeviceMatch),
                        counts + static_cast<VkDeviceAddress>(first) * sizeof(int));
    }, label);
    for (int ref = 0; ref < grid.count(); ++ref) {
        nss::Match cpu[256];
        const int n = nss::spatial_match(plane.data(), width, width, height, grid.x(ref), grid.y(ref), block, range,
                                         group, cpu);
        compare_group(cpu, n, run.matches.data() + static_cast<std::size_t>(ref) * group, run.counts[ref], flat, stats,
                      label, ref);
    }
    if (failures != before) std::printf("%s: %d failures\n", label, failures - before);
}


void temporal_case(const std::shared_ptr<Device>& device, Matcher& matcher, int width, int height, int block, int group,
                   int step, int range, int radius, int center, int frame_count, bool flat, Stats& stats, int ps_num = 2,
                   int ps_range = 4, SquareMatcher* square = nullptr) {
    const int ntemp = 2 * radius + 1;
    std::vector<std::vector<float>> frames;
    std::vector<std::unique_ptr<UploadPlane>> d_frames;
    std::vector<const float*> host_ptrs;
    std::vector<VkDeviceAddress> addresses;
    std::vector<int> strides(ntemp, width);
    for (int t = 0; t < ntemp; ++t) {
        frames.push_back(make_plane(width, height, 31u * t + 3u, flat));
        d_frames.push_back(std::make_unique<UploadPlane>(device, static_cast<std::size_t>(width) * sizeof(float), height));
        d_frames.back()->write(frames.back().data(), static_cast<std::ptrdiff_t>(width * sizeof(float)), width * sizeof(float));
        host_ptrs.push_back(frames.back().data());
        addresses.push_back(d_frames.back()->address());
    }
    UploadPlane table(device, addresses.size() * sizeof(VkDeviceAddress), 1);
    table.write(addresses.data(), 0, addresses.size() * sizeof(VkDeviceAddress));

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = range;
    cfg.radius = radius;
    cfg.ps_num = std::min(ps_num, group);
    cfg.ps_range = ps_range;
    cfg.valid_t_begin = std::max(0, radius - center);
    cfg.valid_t_end = radius + std::min(radius + 1, frame_count - center);

    const RasterGrid grid = make_raster_grid(width, height, block, step);
    const MatchGeometry g{width, height, width, block, range, group};
    const TemporalWindow w{table.address(), ntemp, radius, radius, cfg.valid_t_begin, cfg.valid_t_end, cfg.ps_num,
                           cfg.ps_range};
    char label[128];
    std::snprintf(label, sizeof label, "temporal %dx%d b%d g%d s%d r%d R%d c%d/%d p%d/%d%s", width, height, block,
                  group, step, range, radius, center, frame_count, cfg.ps_num, cfg.ps_range, flat ? " flat" : "");
    if (!matcher.supports(g, cfg.ps_num)) {
        CHECK(false, "%s: not supported", label);
        return;
    }
    const int before = failures;
    const Run run = run_both(device, matcher, g, cfg.ps_num, grid.count(),
                             [&](Slot& slot, VkDeviceAddress out, VkDeviceAddress counts) {
                                 for (auto& f : d_frames) {
                                     if (f->needs_copy()) f->record_upload(slot.commands(Slot::Upload));
                                 }
                                 if (table.needs_copy()) table.record_upload(slot.commands(Slot::Upload));
                                 matcher.predictive(slot.commands(Slot::Compute), g, w, grid, 0, grid.count(), out, counts);
                             },
                             label);
    for (int ref = 0; ref < grid.count(); ++ref) {
        nss::Match cpu[256];
        const int n = nss::predictive_match(host_ptrs.data(), strides.data(), ntemp, width, height, grid.x(ref),
                                            grid.y(ref), radius, cfg, cpu);
        compare_group(cpu, n, run.matches.data() + static_cast<std::size_t>(ref) * group, run.counts[ref], flat, stats,
                      label, ref);
    }
    if (square) {
        const Run seeded = run_both(device, matcher, g, cfg.ps_num, grid.count(),
                                    [&](Slot& slot, VkDeviceAddress out, VkDeviceAddress counts) {
                                        for (auto& f : d_frames) {
                                            if (f->needs_copy()) f->record_upload(slot.commands(Slot::Upload));
                                        }
                                        if (table.needs_copy()) table.record_upload(slot.commands(Slot::Upload));
                                        square->temporal(slot.commands(Slot::Compute), table.address(), radius, g, grid, 0,
                                                         grid.count(), out, counts);
                                        compute_barrier(*device, slot.commands(Slot::Compute));
                                        matcher.predictive(slot.commands(Slot::Compute), g, w, grid, 0, grid.count(), out,
                                                           counts, true);
                                    },
                                    "seeded square");
        for (int ref = 0; ref < grid.count(); ++ref) {
            CHECK(seeded.counts[ref] == run.counts[ref], "seeded square ref %d: count %d != %d", ref,
                  seeded.counts[ref], run.counts[ref]);
            for (int i = 0; i < std::min(seeded.counts[ref], run.counts[ref]); ++i) {
                const DeviceMatch& a = seeded.matches[static_cast<std::size_t>(ref) * group + i];
                const DeviceMatch& b = run.matches[static_cast<std::size_t>(ref) * group + i];
                const bool same = a.x == b.x && a.y == b.y && a.t == b.t;
                CHECK((flat && same) || near_tie(a.dist, b.dist), "seeded square ref %d entry %d differs", ref, i);
            }
        }
    }
    if (failures != before) std::printf("%s: %d failures\n", label, failures - before);
}

void square_case(const std::shared_ptr<Device>& device, SquareMatcher& square, int width, int height, int block,
                 int group, int step, int range, bool flat, Stats& stats, int pad = 0, bool split = false) {
    const auto plane = make_plane(width, height, 7u + block + group, flat);
    const int pitch = width + pad;
    UploadPlane d_plane(device, static_cast<std::size_t>(pitch) * sizeof(float), height);
    d_plane.write(plane.data(), static_cast<std::ptrdiff_t>(width * sizeof(float)), width * sizeof(float));
    const RasterGrid grid = make_raster_grid(width, height, block, step);
    const MatchGeometry g{width, height, pitch, block, range, group};
    CHECK(spatial_match_square_takes(g, grid), "square b%d g%d s%d: shape rejected", block, group, step);
    DownloadPlane out(device, static_cast<std::size_t>(grid.count()) * group * sizeof(DeviceMatch), 1);
    DownloadPlane counts(device, static_cast<std::size_t>(grid.count()) * sizeof(int), 1);
    Slot slot(device);
    if (d_plane.needs_copy()) d_plane.record_upload(slot.commands(Slot::Upload));
    const int first = split ? grid.count() / 3 : grid.count();
    square.spatial(slot.commands(Slot::Compute), d_plane.address(), g, grid, 0, first, out.address(), counts.address());
    square.spatial(slot.commands(Slot::Compute), d_plane.address(), g, grid, first, grid.count() - first,
                   out.address() + static_cast<VkDeviceAddress>(first) * group * sizeof(DeviceMatch),
                   counts.address() + static_cast<VkDeviceAddress>(first) * sizeof(int));
    out.record_download(slot.commands(Slot::Download));
    counts.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    Run run{std::vector<DeviceMatch>(static_cast<std::size_t>(grid.count()) * group), std::vector<int>(grid.count())};
    out.read(run.matches.data(), 0, run.matches.size() * sizeof(DeviceMatch));
    counts.read(run.counts.data(), 0, run.counts.size() * sizeof(int));
    for (int ref = 0; ref < grid.count(); ++ref) {
        nss::Match cpu[256];
        const int n = nss::spatial_match(plane.data(), width, width, height, grid.x(ref), grid.y(ref), block, range,
                                         group, cpu);
        compare_group(cpu, n, run.matches.data() + static_cast<std::size_t>(ref) * group, run.counts[ref], flat, stats,
                      "square spatial", ref);
    }
}

// nss::image_match: `nch` channels summed, the exact-size window
// [c - window / 2, c + window - window / 2 - 1], groups up to 256, and the
// predictive search over `frames` frames from reference frame t0 (as
// test_cuda_match's image_case).
void image_case(const std::shared_ptr<Device>& device, Matcher& matcher, int width, int height, int block, int group,
                int step, int window, int nch, int frames, int t0, int radius, int ps_num, bool flat, Stats& stats) {
    const std::size_t plane_floats = static_cast<std::size_t>(width) * height;
    std::vector<std::vector<float>> host(frames);  // per frame: nch stacked planes
    std::vector<std::unique_ptr<UploadPlane>> d_frames;
    std::vector<const float*> guides;
    std::vector<VkDeviceAddress> addresses;
    for (int t = 0; t < frames; ++t) {
        for (int c = 0; c < nch; ++c) {
            const auto plane = make_plane(width, height, 17u * t + 5u * c + 1u, flat);
            host[t].insert(host[t].end(), plane.begin(), plane.end());
        }
        d_frames.push_back(std::make_unique<UploadPlane>(device, host[t].size() * sizeof(float), 1));
        d_frames.back()->write(host[t].data(), 0, host[t].size() * sizeof(float));
        addresses.push_back(d_frames.back()->address());
    }
    for (int t = 0; t < frames; ++t) {
        for (int c = 0; c < nch; ++c) guides.push_back(host[t].data() + c * plane_floats);
    }
    UploadPlane table(device, addresses.size() * sizeof(VkDeviceAddress), 1);
    table.write(addresses.data(), 0, addresses.size() * sizeof(VkDeviceAddress));

    const RasterGrid grid = make_raster_grid(width, height, block, step);
    MatchGeometry g{width, height, width, block, window / 2, group};
    g.range_hi = window - window / 2 - 1;
    g.channels = nch;
    g.channel_step = static_cast<long long>(plane_floats);
    const bool temporal = radius > 0 && frames > 1;
    const int search_ps = temporal ? ps_num : 0;
    char label[128];
    std::snprintf(label, sizeof label, "image %dx%d b%d g%d s%d w%d ch%d f%d t%d R%d p%d%s %s", width, height, block,
                  group, step, window, nch, frames, t0, radius, ps_num, flat ? " flat" : "",
                  matcher.lanes(g, search_ps) ? "lanes" : "block");
    if (!matcher.supports(g, search_ps)) {
        CHECK(false, "%s: not supported", label);
        return;
    }
    const int before = failures;
    Run run = run_both(device, matcher, g, search_ps, grid.count(),
                       [&](Slot& slot, VkDeviceAddress out, VkDeviceAddress counts) {
                           for (auto& f : d_frames) {
                               if (f->needs_copy()) f->record_upload(slot.commands(Slot::Upload));
                           }
                           if (table.needs_copy()) table.record_upload(slot.commands(Slot::Upload));
                           if (temporal) {
                               const TemporalWindow w{table.address(), frames, t0, radius, 0, frames, ps_num, 4};
                               matcher.predictive(slot.commands(Slot::Compute), g, w, grid, 0, grid.count(), out, counts);
                           } else {
                               matcher.spatial(slot.commands(Slot::Compute), addresses[static_cast<std::size_t>(t0)], g,
                                               grid, 0, grid.count(), out, counts);
                           }
                       },
                       label);
    const nss::ImageSearch search{block, step, group, window, radius, ps_num, 4};
    std::vector<nss::Match> cpu(256);
    for (int ref = 0; ref < grid.count(); ++ref) {
        const int n = nss::image_match(guides.data(), frames, nch, width, height, t0, grid.x(ref), grid.y(ref), search,
                                       cpu.data());
        DeviceMatch* gpu = run.matches.data() + static_cast<std::size_t>(ref) * group;
        // The spatial kernels report slot 0; the CPU reports the sequence index.
        if (!temporal) {
            for (int i = 0; i < run.counts[ref]; ++i) {
                CHECK(gpu[i].t == 0, "%s ref %d entry %d: spatial slot %d", label, ref, i, gpu[i].t);
                gpu[i].t = t0;
            }
        }
        compare_group(cpu.data(), n, gpu, run.counts[ref], flat, stats, label, ref);
    }
    if (failures != before) std::printf("%s: %d failures\n", label, failures - before);
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk match SKIP: no Vulkan device (%s)\n", vk.error.c_str());
        return 77;
    }
    int index = -1;
    if (argc > 1) {
        index = std::atoi(argv[1]);
    } else {
        for (int i = 0; i < static_cast<int>(vk.devices.size()) && index < 0; ++i) {
            if (classify(query_caps(vk.devices[static_cast<std::size_t>(i)])).support != Support::unsupported) index = i;
        }
    }
    if (index < 0) {
        std::printf("vk match SKIP: no supported Vulkan device\n");
        return 77;
    }
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    Matcher matcher(device);
    SquareMatcher square(device);
    Stats stats;
    for (const bool flat : {false, true}) {
        square_case(device, square, 71, 66, 2, 8, 1, 7, flat, stats, 3, true);
        square_case(device, square, 71, 66, 4, 16, 2, 7, flat, stats);
        square_case(device, square, 71, 66, 8, 8, 4, 7, flat, stats, 0, true);
        square_case(device, square, 71, 66, 12, 32, 6, 7, flat, stats);
        square_case(device, square, 71, 66, 16, 16, 8, 7, flat, stats, 5, true);
        for (const int q : {1, 2, 4, 6, 8}) {
            temporal_case(device, matcher, 71, 66, 2 * q, 8, q, 7, 1, 1, 3, flat, stats, 2, 4, &square);
        }
        temporal_case(device, matcher, 71, 66, 8, 32, 4, 7, 2, 2, 5, flat, stats, 2, 4, &square);
        temporal_case(device, matcher, 71, 66, 8, 2, 4, 7, 1, 1, 3, flat, stats, 2, 4, &square);
        temporal_case(device, matcher, 71, 66, 8, 8, 4, 7, 1, 1, 3, flat, stats, 4, 4, &square);
        temporal_case(device, matcher, 71, 66, 8, 8, 4, 7, 1, 1, 3, flat, stats, 1, 4, &square);
        temporal_case(device, matcher, 71, 66, 16, 16, 8, 7, 1, 1, 3, flat, stats, 2, 4, &square);
    }
    for (const bool flat : {false, true}) {
        for (const int block : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
            for (const int group : {2, 8, 16, 32}) {
                spatial_case(device, matcher, 97, 83, block, group, std::min(block, 3), 7, flat, stats);
            }
        }
        // The block fallback: groups beyond the lanes (BM3D's 64), one top-K
        // chunk exceeded (window 81 x 81), and block 32 with group 64.
        spatial_case(device, matcher, 97, 83, 8, 64, 3, 7, flat, stats);
        spatial_case(device, matcher, 97, 83, 4, 48, 5, 9, flat, stats);
        spatial_case(device, matcher, 80, 72, 8, 64, 4, 40, flat, stats);
        spatial_case(device, matcher, 97, 83, 32, 64, 9, 7, flat, stats);
        spatial_case(device, matcher, 97, 83, 8, 1, 4, 7, flat, stats);  // group 1: the reference alone
        spatial_case(device, matcher, 64, 61, 8, 16, 8, 12, flat, stats);
        spatial_case(device, matcher, 97, 83, 12, 33, 5, 9, flat, stats);
        spatial_case(device, matcher, 97, 83, 32, 8, 9, 7, flat, stats);
        for (const int radius : {1, 2}) {
            temporal_case(device, matcher, 71, 66, 8, 8, 4, 7, radius, radius, 2 * radius + 1, flat, stats);
            temporal_case(device, matcher, 71, 66, 8, 16, 5, 7, radius, 0, 9, flat, stats);   // clip start
            temporal_case(device, matcher, 71, 66, 4, 32, 6, 9, radius, 8, 9, flat, stats);   // clip end
        }
        temporal_case(device, matcher, 71, 66, 12, 8, 5, 7, 1, 1, 3, flat, stats);
        temporal_case(device, matcher, 71, 66, 2, 8, 2, 7, 1, 1, 3, flat, stats);
        temporal_case(device, matcher, 71, 66, 1, 16, 1, 5, 1, 1, 3, flat, stats);
        temporal_case(device, matcher, 71, 66, 32, 8, 9, 7, 1, 1, 3, flat, stats);
        temporal_case(device, matcher, 71, 66, 16, 16, 6, 7, 2, 2, 5, flat, stats);
        for (const int ps_range : {0, 1, 7}) {
            temporal_case(device, matcher, 71, 66, 16, 16, 6, 7, 1, 1, 3, flat, stats, 2, ps_range);
        }
        temporal_case(device, matcher, 71, 66, 8, 8, 4, 7, 1, 1, 3, flat, stats, 1);
        temporal_case(device, matcher, 71, 66, 8, 16, 5, 7, 2, 0, 9, flat, stats, 1);
        // ps_num up to the lanes of a reference (the shuffle sources reach L - 1).
        temporal_case(device, matcher, 71, 66, 8, 8, 4, 5, 1, 1, 3, flat, stats, 8);
        temporal_case(device, matcher, 71, 66, 8, 16, 5, 5, 1, 1, 3, flat, stats, 16);
        temporal_case(device, matcher, 71, 66, 4, 32, 6, 5, 1, 1, 3, flat, stats, 32);
        spatial_case(device, matcher, 97, 83, 8, 16, 3, 7, flat, stats, 13, true);
        spatial_case(device, matcher, 97, 83, 16, 32, 3, 7, flat, stats, 5, true);
        spatial_case(device, matcher, 97, 83, 8, 64, 3, 7, flat, stats, 13, true);  // block fallback, padded, split
        // Block fallback, predictive: group 64, and ps_num beyond the lanes.
        temporal_case(device, matcher, 71, 66, 8, 64, 4, 7, 1, 1, 3, flat, stats, 4);
        temporal_case(device, matcher, 71, 66, 8, 64, 5, 7, 2, 0, 9, flat, stats, 8);
        temporal_case(device, matcher, 71, 66, 4, 64, 6, 5, 1, 1, 3, flat, stats, 40);
    }
    for (const bool flat : {false, true}) {
        for (const int nch : {1, 2, 3}) {
            image_case(device, matcher, 61, 57, 8, 8, 5, 15, nch, 1, 0, 0, 2, flat, stats);   // MCWNNM-like joint groups
            image_case(device, matcher, 72, 64, 8, 16, 6, 40, nch, 1, 0, 0, 2, flat, stats);  // NLH window (asymmetric)
        }
        // Joint lane kernels: blocks below 8 and every lane count; joint
        // blocks beyond 8 and large groups on the block fallback.
        image_case(device, matcher, 61, 57, 4, 8, 5, 15, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 61, 57, 2, 16, 3, 11, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 61, 57, 1, 32, 3, 9, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 61, 57, 7, 32, 6, 21, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 61, 57, 12, 16, 6, 15, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 61, 57, 16, 8, 6, 15, 3, 1, 0, 0, 2, flat, stats);
        image_case(device, matcher, 72, 64, 8, 90, 7, 60, 1, 1, 0, 0, 2, flat, stats);    // TWSC default group
        image_case(device, matcher, 72, 64, 4, 256, 9, 33, 2, 1, 0, 0, 2, flat, stats);   // group limit
        image_case(device, matcher, 40, 36, 16, 64, 8, 129, 1, 1, 0, 0, 2, flat, stats);  // window larger than the plane
    }
    // Predictive (ties across frames are ordered differently, so only the
    // noisy planes): lanes for one and three channels, the block fallback for
    // two channels, groups beyond 33 and the 256-entry kernel.
    image_case(device, matcher, 64, 56, 8, 16, 6, 40, 1, 3, 1, 1, 2, false, stats);
    image_case(device, matcher, 64, 56, 8, 16, 6, 21, 3, 5, 0, 2, 3, false, stats);
    image_case(device, matcher, 64, 56, 4, 8, 6, 15, 3, 5, 2, 2, 4, false, stats);
    image_case(device, matcher, 64, 56, 8, 32, 6, 15, 3, 3, 1, 1, 6, false, stats);
    image_case(device, matcher, 64, 56, 8, 16, 6, 21, 2, 5, 0, 2, 3, false, stats);
    image_case(device, matcher, 64, 56, 8, 16, 6, 21, 1, 5, 4, 2, 3, false, stats);
    image_case(device, matcher, 64, 56, 8, 64, 8, 21, 2, 3, 1, 1, 20, false, stats);
    image_case(device, matcher, 64, 56, 4, 200, 8, 25, 1, 3, 2, 1, 100, false, stats);
    image_case(device, matcher, 64, 56, 12, 16, 7, 15, 3, 3, 1, 1, 4, false, stats);
    std::printf("vk match: %ld references, %ld entries, %ld near-tie flips, %d failures\n", stats.refs, stats.entries,
                stats.flips, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
