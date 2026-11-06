// SPDX-License-Identifier: GPL-2.0-only
// Vulkan WNNM group kernels (vulkan/wnnm/kernels.hpp) against the CPU's group
// shrinkage (nss::wnnm_shrink) and nss::unpack_patch on the same matches, for
// block sizes up to 16 and every eigen path (groups up to 8: the three Gram
// kernels; 9 to 16: the parallel Jacobi; 17 to 32: divide and conquer), with
// and without residual and adaptive aggregation: the finished planes must
// agree within a PSNR bound (FP32 Gram path, fixed-point sums), and the native
// layer, the emulated layer and the dispatch in rows of three workgroups bit
// for bit. Exits 77 without a supported Vulkan device.
//
// NSS_VK_TEST_TIME=1: also the kernels' time on a 1920 x 1080 plane (step 4,
// range 15), native, per group size, from the compute queue's timestamps.
//
// usage: test_vk_wnnm [device_id [label substring]]
#include "nss/cpu_api.hpp"
#include "vulkan/common/aggregate.hpp"
#include "vulkan/common/match.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/slot.hpp"
#include "vulkan/wnnm/kernels.hpp"

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
    bool residual, adaptive;
};

struct Matched {
    std::vector<std::vector<nss::Match>> cpu;
    std::vector<DeviceMatch> device;
    std::vector<int> counts;
};

Matched match_plane(const Case& c, const std::vector<float>& src) {
    const RasterGrid grid = make_raster_grid(c.width, c.height, c.block, c.step);
    Matched m;
    m.cpu.resize(static_cast<std::size_t>(grid.count()));
    m.device.resize(static_cast<std::size_t>(grid.count()) * c.group);
    m.counts.resize(static_cast<std::size_t>(grid.count()));
    for (int r = 0; r < grid.count(); ++r) {
        nss::Match found[64];
        const int n = nss::spatial_match(src.data(), c.width, c.width, c.height, grid.x(r), grid.y(r), c.block, c.range,
                                         c.group, found);
        m.cpu[static_cast<std::size_t>(r)].assign(found, found + n);
        m.counts[static_cast<std::size_t>(r)] = n;
        for (int i = 0; i < n; ++i) m.device[static_cast<std::size_t>(r) * c.group + i] = DeviceMatch{found[i].x, found[i].y, 0, found[i].dist};
    }
    return m;
}

// The CPU's result: every reference's group shrunk and unpacked in raster order.
std::vector<float> cpu_result(const Case& c, const std::vector<float>& src, const Matched& m, float sigma) {
    const int area = c.block * c.block;
    std::vector<float> num(src.size(), 0.f), den(src.size(), 0.f);
    std::vector<float> group(static_cast<std::size_t>(c.group) * area);
    std::vector<float> work(static_cast<std::size_t>(nss::wnnm_shrink_work_floats(area, c.group)));
    for (const auto& g : m.cpu) {
        const int n = static_cast<int>(g.size());
        for (int i = 0; i < n; ++i) {
            nss::pack_patch(group.data() + static_cast<std::size_t>(i) * area, area, src.data(), c.width, g[i].x, g[i].y,
                            c.block, c.width, c.height);
        }
        float weight = 1.f;
        nss::wnnm_shrink(group.data(), area, n, area, sigma, c.residual, c.adaptive, &weight, work.data(),
                         static_cast<int>(work.size()));
        for (int i = 0; i < n; ++i) {
            nss::unpack_patch(num.data(), den.data(), c.width, g[i].x, g[i].y,
                              group.data() + static_cast<std::size_t>(i) * area, c.block, c.width, c.height,
                              c.adaptive ? weight : 1.f);
        }
    }
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = den[i] > 1e-12f ? num[i] / den[i] : src[i];
    return out;
}

// The device's result; with ms, also the kernels' time (ms, compute queue).
std::vector<float> gpu_result(const std::shared_ptr<Device>& device, WnnmKernels& kernels, const Aggregator& aggregator,
                              const Case& c, const std::vector<float>& src, const Matched& m, int batch, float sigma,
                              double* ms = nullptr) {
    const std::size_t floats = src.size();
    const int refs = static_cast<int>(m.counts.size());
    UploadPlane d_src(device, floats * sizeof(float), 1);
    d_src.write(src.data(), 0, floats * sizeof(float));
    const VkDeviceAddress table = d_src.address();
    UploadPlane d_table(device, sizeof table, 1);
    d_table.write(&table, 0, sizeof table);
    UploadPlane d_matches(device, m.device.size() * sizeof(DeviceMatch), 1);
    d_matches.write(m.device.data(), 0, m.device.size() * sizeof(DeviceMatch));
    UploadPlane d_counts(device, m.counts.size() * sizeof(int), 1);
    d_counts.write(m.counts.data(), 0, m.counts.size() * sizeof(int));
    batch = std::min(batch, refs);
    const std::size_t scratch_floats = wnnm_scratch_floats(c.group);
    Buffer scratch(device, std::max<std::size_t>(4, scratch_floats * batch * sizeof(float)), Memory::Device);
    Buffer num(device, floats * 8, Memory::Device), den(device, floats * 8, Memory::Device);
    Buffer rows(device, floats * 8, Memory::Device);
    DownloadPlane out(device, floats * sizeof(float), 1);

    Slot slot(device, ms ? 2u : 0u);
    VkCommandBuffer upload = slot.commands(Slot::Upload);
    for (const UploadPlane* p : {&d_src, &d_table, &d_matches, &d_counts}) {
        if (p->needs_copy()) p->record_upload(upload);
    }
    VkCommandBuffer compute = slot.commands(Slot::Compute);
    fixed_clear(*device, compute, num.get(), 0, floats * 8);
    fixed_clear(*device, compute, den.get(), 0, floats * 8);
    if (ms) slot.mark(Slot::Compute, 0);
    WnnmGroupArgs args{};
    args.src = d_table.address();
    args.scratch = scratch.address();
    args.pitch = c.width;
    args.block = c.block;
    args.group = c.group;
    args.sigma = sigma;
    args.residual = c.residual;
    args.adaptive = c.adaptive;
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
    if (ms) slot.mark(Slot::Compute, 1);
    aggregator.fixed_finish(compute, num.address(), den.address(), c.width, c.block, d_src.address(), c.width, c.height,
                            rows.address(), out.address());
    out.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    if (ms) *ms = (slot.timestamps_ns()[1] - slot.timestamps_ns()[0]) * 1e-6;
    std::vector<float> result(floats);
    out.read(result.data(), 0, floats * sizeof(float));
    return result;
}

void run_case(const std::shared_ptr<Device>& device, WnnmKernels& kernels, const Aggregator& aggregator, const Case& c) {
    char label[96];
    std::snprintf(label, sizeof label, "wnnm %dx%d b%d g%d s%d r%d a%d", c.width, c.height, c.block, c.group, c.step,
                  c.residual ? 1 : 0, c.adaptive ? 1 : 0);
    if (only && !std::strstr(label, only)) return;
    const float sigma = 0.08f;
    const auto src = make_plane(c.width, c.height, 5u + c.block * 7u + c.group, 0.08f);
    const Matched m = match_plane(c, src);
    const std::vector<float> want = cpu_result(c, src, m, sigma);
    // Native, emulated, and native in rows of three workgroups; batches that
    // do not divide the grid (idle lanes and groups of a workgroup).
    std::vector<float> got[3];
    for (int pass = 0; pass < 3; ++pass) {
        kernels.force_emulation(pass == 1);
        kernels.limit_groups_x(pass == 2 ? 3 : 65535);
        const char* batch = std::getenv("NSS_VK_TEST_BATCH");
        got[pass] = gpu_result(device, kernels, aggregator, c, src, m, batch ? std::atoi(batch) : 37, sigma);
    }
    kernels.force_emulation(false);
    kernels.limit_groups_x(65535);
    CHECK(got[0] == got[1], "%s: native and emulated layers differ", label);
    CHECK(got[0] == got[2], "%s: the dispatch in rows differs", label);
    const double db = psnr(want, got[0]);
    double worst = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(want[i]) - got[0][i]));
    CHECK(db >= 70.0, "%s: PSNR against the CPU %.1f dB (max abs %.3g)", label, db, worst);
    std::printf("%-34s PSNR %6.1f dB, max abs %.3g\n", label, db, worst);
}

void time_kernels(const std::shared_ptr<Device>& device, WnnmKernels& kernels, const Aggregator& aggregator) {
    for (const int group : {8, 16, 32}) {
        const Case c{1920, 1080, 8, group, 4, 15, false, true};
        const auto src = make_plane(c.width, c.height, 11u, 0.08f);
        const Matched m = match_plane(c, src);
        double best = 1e30;
        for (int run = 0; run < 5; ++run) {
            double ms = 0.0;
            gpu_result(device, kernels, aggregator, c, src, m, 1 << 30, 0.08f, &ms);
            best = std::min(best, ms);
        }
        std::printf("time wnnm 1920x1080 b8 g%d s4 range15: %zu groups, %.3f ms\n", group, m.counts.size(), best);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk wnnm SKIP: no Vulkan device (%s)\n", vk.error.c_str());
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
        std::printf("vk wnnm SKIP: no supported Vulkan device\n");
        return 77;
    }
    if (argc > 2) only = argv[2];
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    WnnmKernels kernels(device);
    const Aggregator aggregator(device);
    for (const int block : {1, 3, 4, 8, 11, 16}) {
        for (const int group : {1, 2, 5, 8, 9, 13, 16, 17, 24, 32}) {
            for (const int flags : {0, 1, 2, 3}) {
                const bool residual = (flags & 1) != 0, adaptive = (flags & 2) != 0;
                if (block != 8 && flags != 2) continue;  // every option at block 8, the default elsewhere
                const int side = block >= 11 ? 88 : 61;
                run_case(device, kernels, aggregator,
                         Case{side + 3, side, block, group, std::max(1, block / 2), 7, residual, adaptive});
            }
        }
    }
    const char* timing = std::getenv("NSS_VK_TEST_TIME");
    if (timing && timing[0] == '1') time_kernels(device, kernels, aggregator);
    std::printf("vk wnnm: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
