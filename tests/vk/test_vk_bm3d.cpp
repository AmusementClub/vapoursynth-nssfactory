// SPDX-License-Identifier: GPL-2.0-only
// Vulkan BM3D group kernels (vulkan/bm3d/kernels.hpp) against the CPU's
// generic group filter (nss::bm3d_filter_group) and nss::unpack_patch on the
// same matches, for every block and group of BM3D and both stages, on the
// native and the emulated warp layer: the finished planes (num / den) must
// agree within a PSNR bound (the device sums the patches in fixed point and
// reduces in other orders; a coefficient on the threshold may flip), and the
// two layers bit for bit. Exits 77 without a supported Vulkan device; an
// explicit device id may name an unsupported one (lavapipe).
//
// usage: test_vk_bm3d [device_id [label substring]]
#include "nss/cpu_api.hpp"
#include "vulkan/bm3d/kernels.hpp"
#include "vulkan/common/aggregate.hpp"
#include "vulkan/common/match.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace nss_vk;

int failures = 0;
const char* only = nullptr;  // run the cases whose label contains it

#define CHECK(cond, ...)                           \
    do {                                           \
        if (!(cond)) {                             \
            if (failures < 40) {                   \
                std::printf("FAIL: " __VA_ARGS__); \
                std::printf("\n");                 \
            }                                      \
            ++failures;                            \
        }                                          \
    } while (0)

std::vector<float> make_plane(int width, int height, unsigned seed, float noise_sigma) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, noise_sigma);
    std::vector<float> plane(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = 0.5f + 0.3f * std::sin(x * 0.19f) * std::cos(y * 0.13f) + ((x / 9 + y / 7) % 2) * 0.1f;
            plane[static_cast<std::size_t>(y) * width + x] = v + noise(rng);
        }
    }
    return plane;
}

double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) se += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    const double mse = se / static_cast<double>(a.size());
    return mse == 0.0 ? 999.0 : 10.0 * std::log10(1.0 / mse);
}

struct Case {
    int width, height, block, group, step, range;
    bool wiener;
};

// The CPU's result: every reference's group filtered and unpacked in raster order.
std::vector<float> cpu_result(const Case& c, const std::vector<float>& src, const std::vector<float>& ref,
                              const std::vector<std::vector<nss::Match>>& matches, float sigma) {
    const int area = c.block * c.block;
    std::vector<float> num(src.size(), 0.f), den(src.size(), 0.f);
    std::vector<float> cube(static_cast<std::size_t>(c.group) * area), refc(cube.size());
    std::vector<float> work(static_cast<std::size_t>(nss::bm3d_filter_work_floats(c.group, c.block)));
    for (const auto& group : matches) {
        const int k = static_cast<int>(group.size());
        std::fill(cube.begin(), cube.end(), 0.f);
        std::fill(refc.begin(), refc.end(), 0.f);
        for (int i = 0; i < k; ++i) {
            nss::pack_patch(cube.data() + static_cast<std::size_t>(i) * area, area, src.data(), c.width, group[i].x,
                            group[i].y, c.block, c.width, c.height);
            if (c.wiener) {
                nss::pack_patch(refc.data() + static_cast<std::size_t>(i) * area, area, ref.data(), c.width,
                                group[i].x, group[i].y, c.block, c.width, c.height);
            }
        }
        float weight = 1.f;
        nss::bm3d_filter_group(cube.data(), area, c.group, k, c.block, sigma, c.wiener,
                               c.wiener ? refc.data() : nullptr, &weight, work.data());
        for (int i = 0; i < k; ++i) {
            nss::unpack_patch(num.data(), den.data(), c.width, group[i].x, group[i].y,
                              cube.data() + static_cast<std::size_t>(i) * area, c.block, c.width, c.height, weight);
        }
    }
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = den[i] > 1e-12f ? num[i] / den[i] : src[i];
    return out;
}

std::vector<float> gpu_result(const std::shared_ptr<Device>& device, Bm3dKernels& kernels, const Aggregator& aggregator,
                              const Case& c, const std::vector<float>& src, const std::vector<float>& ref,
                              const std::vector<DeviceMatch>& matches, const std::vector<int>& counts, int batch,
                              float sigma) {
    const std::size_t floats = src.size();
    const int refs = static_cast<int>(counts.size());
    UploadPlane d_src(device, floats * sizeof(float), 1), d_ref(device, floats * sizeof(float), 1);
    d_src.write(src.data(), 0, floats * sizeof(float));
    d_ref.write(ref.data(), 0, floats * sizeof(float));
    const VkDeviceAddress tables[2] = {d_src.address(), d_ref.address()};
    UploadPlane d_tables(device, sizeof tables, 1);
    d_tables.write(tables, 0, sizeof tables);
    UploadPlane d_matches(device, matches.size() * sizeof(DeviceMatch), 1);
    d_matches.write(matches.data(), 0, matches.size() * sizeof(DeviceMatch));
    UploadPlane d_counts(device, counts.size() * sizeof(int), 1);
    d_counts.write(counts.data(), 0, counts.size() * sizeof(int));
    const std::size_t scratch_floats = bm3d_scratch_floats(c.block, c.group, c.wiener);
    Buffer scratch(device, std::max<std::size_t>(4, scratch_floats * batch * sizeof(float)), Memory::Device);
    Buffer num(device, floats * 8, Memory::Device), den(device, floats * 8, Memory::Device);
    Buffer rows(device, floats * 8, Memory::Device);
    DownloadPlane out(device, floats * sizeof(float), 1);

    Slot slot(device);
    VkCommandBuffer upload = slot.commands(Slot::Upload);
    for (const UploadPlane* p : {&d_src, &d_ref, &d_tables, &d_matches, &d_counts}) {
        if (p->needs_copy()) p->record_upload(upload);
    }
    VkCommandBuffer compute = slot.commands(Slot::Compute);
    fixed_clear(*device, compute, num.get(), 0, floats * 8);
    fixed_clear(*device, compute, den.get(), 0, floats * 8);
    Bm3dGroupArgs args{};
    args.src = d_tables.address();
    args.ref = c.wiener ? d_tables.address() + sizeof(VkDeviceAddress) : 0;
    args.scratch = scratch.address();
    args.pitch = c.width;
    args.block = c.block;
    args.group = c.group;
    args.sigma = sigma;
    args.fused.num = num.address();
    args.fused.den = den.address();
    args.fused.pitch = c.width;
    args.fused.slice_step = static_cast<std::int64_t>(floats);
    for (int begin = 0; begin < refs; begin += batch) {
        args.batch = std::min(batch, refs - begin);
        args.matches = d_matches.address() + static_cast<VkDeviceAddress>(begin) * c.group * sizeof(DeviceMatch);
        args.counts = d_counts.address() + static_cast<VkDeviceAddress>(begin) * sizeof(int);
        kernels.filter(compute, args);
        compute_barrier(*device, compute);  // the scratch is reused
    }
    aggregator.fixed_finish(compute, num.address(), den.address(), c.width, c.block, d_src.address(), c.width, c.height,
                            rows.address(), out.address());
    out.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    std::vector<float> result(floats);
    out.read(result.data(), 0, floats * sizeof(float));
    return result;
}

void run_case(const std::shared_ptr<Device>& device, Bm3dKernels& kernels, const Aggregator& aggregator, const Case& c) {
    char label[96];
    std::snprintf(label, sizeof label, "bm3d %dx%d b%d g%d s%d %s", c.width, c.height, c.block, c.group, c.step,
                  c.wiener ? "wiener" : "hard");
    if (only && !std::strstr(label, only)) return;
    const float sigma = 0.1f;
    const auto src = make_plane(c.width, c.height, 5u + c.block * 7u + c.group, 0.08f);
    const auto ref = make_plane(c.width, c.height, 5u + c.block * 7u + c.group, 0.f);  // the clean plane as a basic estimate
    const RasterGrid grid = make_raster_grid(c.width, c.height, c.block, c.step);
    std::vector<std::vector<nss::Match>> cpu_matches(static_cast<std::size_t>(grid.count()));
    std::vector<DeviceMatch> matches(static_cast<std::size_t>(grid.count()) * c.group);
    std::vector<int> counts(static_cast<std::size_t>(grid.count()));
    for (int r = 0; r < grid.count(); ++r) {
        nss::Match m[64];
        const int n = nss::spatial_match((c.wiener ? ref : src).data(), c.width, c.width, c.height, grid.x(r), grid.y(r),
                                         c.block, c.range, c.group, m);
        cpu_matches[static_cast<std::size_t>(r)].assign(m, m + n);
        counts[static_cast<std::size_t>(r)] = n;
        for (int i = 0; i < n; ++i) matches[static_cast<std::size_t>(r) * c.group + i] = DeviceMatch{m[i].x, m[i].y, 0, m[i].dist};
    }
    const std::vector<float> want = cpu_result(c, src, ref, cpu_matches, sigma);
    std::vector<float> got[2];
    for (int emulated = 0; emulated < 2; ++emulated) {
        kernels.force_emulation(emulated != 0);
        // Batches that do not divide the grid (the warp kernel's idle lanes).
        const char* batch = std::getenv("NSS_VK_TEST_BATCH");
        got[emulated] = gpu_result(device, kernels, aggregator, c, src, ref, matches, counts, batch ? std::atoi(batch) : 37,
                                   sigma);
    }
    kernels.force_emulation(false);
    CHECK(got[0] == got[1], "%s: native and emulated layers differ", label);
    const double db = psnr(want, got[0]);
    double worst = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(want[i]) - got[0][i]));
    CHECK(db >= 75.0, "%s: PSNR against the CPU %.1f dB (max abs %.3g)", label, db, worst);
    std::printf("%-34s PSNR %6.1f dB, max abs %.3g\n", label, db, worst);
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk bm3d SKIP: no Vulkan device (%s)\n", vk.error.c_str());
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
        std::printf("vk bm3d SKIP: no supported Vulkan device\n");
        return 77;
    }
    if (argc > 2) only = argv[2];
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    Bm3dKernels kernels(device);
    const Aggregator aggregator(device);
    for (const bool wiener : {false, true}) {
        for (const int block : {1, 2, 4, 8, 12, 16, 32}) {
            for (const int group : {1, 2, 4, 8, 16, 32, 64}) {
                // Big blocks with big groups: a plane wide enough for whole groups.
                const int side = block >= 16 ? 96 : 61;
                run_case(device, kernels, aggregator, Case{side + 3, side, block, group, std::max(1, block / 2), 7, wiener});
            }
        }
    }
    std::printf("vk bm3d: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
