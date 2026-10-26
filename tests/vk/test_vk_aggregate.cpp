// SPDX-License-Identifier: GPL-2.0-only
// Vulkan V3 primitives of the patch-group filters:
// - DCT lines (shaders/common/dct.glsl) of every length against the CPU's
//   nss::dct_lines (orthonormal; tolerance) and against a host model of the
//   same arithmetic (the genfft codelets with std::fma, and the 2 / 4-point
//   products): bit for bit on NVIDIA, which fuses fma; Mesa splits it (D26).
//   Forward then inverse returns the input.
// - Fixed-point aggregation (common/fixed.glsl, common/aggregate.hpp): the
//   cells equal the host's exact integer sums, also into a ring of slices
//   that drops the slots outside it, two runs give the same bits, and
//   fixed_finish matches the host quotient within Vulkan's division
//   precision (2.5 ULP).
// Exits 77 without a supported Vulkan device; an explicit device id may name
// an unsupported one (lavapipe).
//
// usage: test_vk_aggregate [device_id]
#include "gpuhost/fixed.hpp"
#include "nss/cpu_api.hpp"
#include "vulkan/common/aggregate.hpp"
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

namespace host_codelet {
// The CPU's genfft files with scalar fma, as nss_cuda's dct_codelets.cuh.
#define R float
#define V float
#define HWY_INLINE inline
#define DVK(name, value) const float name = static_cast<float>(value)
#define LDK(name) (name)
#define LD(ptr, vstride, alignment) (*(ptr))
#define ST(ptr, value, vstride, alignment) (*(ptr) = (value))
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMUL(a, b) ((a) * (b))
#define VFMA(a, b, c) std::fma((a), (b), (c))
#define VFMS(a, b, c) std::fma((a), (b), -(c))
#define VFNMS(a, b, c) std::fma(-(a), (b), (c))
#define VNEG(a) (-(a))
#define VLEAVE() ((void)0)
#include "cpu/bm/dct_codelet_fwd_n12_is8.hpp"
#include "cpu/bm/dct_codelet_fwd_n16_is8.hpp"
#include "cpu/bm/dct_codelet_fwd_n32_is8.hpp"
#include "cpu/bm/dct_codelet_fwd_n64_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n12_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n16_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n32_is8.hpp"
#include "cpu/bm/dct_codelet_inv_n64_is8.hpp"
#undef R
#undef V
#undef HWY_INLINE
#undef DVK
#undef LDK
#undef LD
#undef ST
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VFMS
#undef VFNMS
#undef VNEG
#undef VLEAVE
}  // namespace host_codelet

namespace {

using namespace nss_vk;

int failures = 0;

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

// The host model of dct_line<n> where it is the CPU's arithmetic (n = 2, 4
// and the codelets); false for the others.
bool host_line(int n, const float* in, float* out, bool inverse) {
    if (n == 2 || n == 4) {
        const float r = 0.707106781186547524400844362104849039284835938f;
        const float a = 0.653281482438188263928322f, b = 0.270598050073098492199862f;
        const float t2[2][2] = {{r, r}, {r, -r}};
        const float t4[4][4] = {{0.5f, 0.5f, 0.5f, 0.5f}, {a, b, -b, -a}, {0.5f, -0.5f, -0.5f, 0.5f}, {b, -a, a, -b}};
        for (int k = 0; k < n; ++k) {
            float acc = 0.f;
            for (int j = 0; j < n; ++j) {
                const float term = n == 2 ? (inverse ? t2[j][k] : t2[k][j]) : (inverse ? t4[j][k] : t4[k][j]);
                acc = std::fma(term, in[j], acc);
            }
            out[k] = acc;
        }
        return true;
    }
    float i8[8 * 64]{}, o8[8 * 64]{};
    for (int k = 0; k < n; ++k) i8[8 * k] = in[k];
    using namespace host_codelet;
    switch (n * 2 + (inverse ? 1 : 0)) {
        case 24: nss_dct12_fwd_is8(0, i8, o8); break;
        case 25: nss_dct12_inv_is8(0, i8, o8); break;
        case 32: nss_dct16_fwd_is8(0, i8, o8); break;
        case 33: nss_dct16_inv_is8(0, i8, o8); break;
        case 64: nss_dct32_fwd_is8(0, i8, o8); break;
        case 65: nss_dct32_inv_is8(0, i8, o8); break;
        case 128: nss_dct64_fwd_is8(0, i8, o8); break;
        case 129: nss_dct64_inv_is8(0, i8, o8); break;
        default: return false;
    }
    for (int k = 0; k < n; ++k) out[k] = o8[8 * k];
    return true;
}

std::vector<float> run_lines(const std::shared_ptr<Device>& device, int n, const std::vector<float>& in, bool inverse) {
    struct Push {
        VkDeviceAddress src, dst;
        std::int32_t lines, inverse;
    };
    const int lines = static_cast<int>(in.size()) / n;
    UploadPlane src(device, in.size() * sizeof(float), 1);
    src.write(in.data(), 0, in.size() * sizeof(float));
    DownloadPlane dst(device, in.size() * sizeof(float), 1);
    Pipeline pipeline(device, {"test_dct_n" + std::to_string(n), sizeof(Push), 0, {}});
    Slot slot(device);
    if (src.needs_copy()) src.record_upload(slot.commands(Slot::Upload));
    const Push push{src.address(), dst.address(), lines, inverse ? 1 : 0};
    pipeline.dispatch(slot.commands(Slot::Compute), &push, static_cast<std::uint32_t>((lines + 63) / 64));
    dst.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    std::vector<float> out(in.size());
    dst.read(out.data(), 0, out.size() * sizeof(float));
    return out;
}

void dct_case(const std::shared_ptr<Device>& device, int n, bool exact_device) {
    constexpr int kLines = 300;
    std::mt19937 rng(17u + static_cast<unsigned>(n));
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    std::vector<float> in(static_cast<std::size_t>(kLines) * n);
    for (float& v : in) v = value(rng);
    const float scale = n == 8 ? 4.f : 1.f;  // dct_line_scale
    for (const bool inverse : {false, true}) {
        const std::vector<float> gpu = run_lines(device, n, in, inverse);
        // The CPU's orthonormal transform of the same lines.
        std::vector<float> cpu = in;
        if (n > 1) nss::dct_lines(cpu.data(), n, n, 1, kLines, inverse);
        double worst = 0.0;
        long same = 0, modelled = 0;
        for (int l = 0; l < kLines; ++l) {
            const float* g = gpu.data() + static_cast<std::size_t>(l) * n;
            for (int k = 0; k < n; ++k) {
                worst = std::max(worst, std::fabs(static_cast<double>(g[k]) / scale - cpu[static_cast<std::size_t>(l) * n + k]));
            }
            float model[64];
            if (host_line(n, in.data() + static_cast<std::size_t>(l) * n, model, inverse)) {
                ++modelled;
                same += std::memcmp(model, g, sizeof(float) * n) == 0;
            }
        }
        CHECK(worst < 2e-5 * std::sqrt(static_cast<double>(n)), "dct n%d %s: max |vk / %g - cpu| %.3g", n,
              inverse ? "inverse" : "forward", scale, worst);
        if (exact_device) {
            CHECK(same == modelled, "dct n%d %s: %ld of %ld lines differ from the host model of the same arithmetic", n,
                  inverse ? "inverse" : "forward", modelled - same, modelled);
        }
        std::printf("dct n%-2d %-7s max |vk - cpu| %.3g, %ld/%ld lines equal to the host model\n", n,
                    inverse ? "inverse" : "forward", worst, same, modelled);
    }
    // Round trip.
    const std::vector<float> back = run_lines(device, n, run_lines(device, n, in, false), true);
    double worst = 0.0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(back[i]) / (scale * scale) - in[i]));
    }
    CHECK(worst < 4e-6 * std::sqrt(static_cast<double>(n)), "dct n%d round trip: max error %.3g", n, worst);
}

struct Patch {
    std::int32_t x, y, t;
    float weight;
};

std::uint64_t host_units(float v) {
    const float scaled = v * nss::gpuhost::kFixedScale;
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(std::nearbyint(scaled)));
}

// npatch weighted block x block patches over `frames` window slots into a
// width x height target of `slices` slices, kept as a ring of `ring` cells
// from `first` (slot t -> logical slice t + base).
void fixed_case(const std::shared_ptr<Device>& device, const Aggregator& aggregator, int width, int height, int block,
                int npatch, int frames, int slices, int base, int ring, int first) {
    std::mt19937 rng(3u + static_cast<unsigned>(width * 7 + block + npatch));
    std::uniform_int_distribution<int> px(0, width - block), py(0, height - block), pt(0, frames - 1);
    std::uniform_real_distribution<float> weight(0.01f, 1.f), value(-0.25f, 1.25f);
    std::vector<Patch> patches(static_cast<std::size_t>(npatch));
    for (Patch& p : patches) p = Patch{px(rng), py(rng), pt(rng), weight(rng)};
    patches[0].weight = 0.f;  // a zero weight adds zeros
    std::vector<float> values(static_cast<std::size_t>(npatch) * block * block);
    for (float& v : values) v = value(rng);
    values[1] = std::nanf("");  // a NaN sample adds nothing
    const int pitch = width + 3;
    const std::size_t slice_cells = static_cast<std::size_t>(pitch) * height;
    const std::size_t cells = slice_cells * ring;

    // Host: exact integer sums.
    std::vector<std::uint64_t> num(cells, 0), den(cells, 0);
    const int area = block * block;
    for (int i = 0; i < npatch; ++i) {
        const Patch& p = patches[static_cast<std::size_t>(i)];
        const int logical = p.t + base;
        if (logical < 0 || logical >= slices) continue;
        const int cell = (logical + first) % ring;
        const std::size_t corner = cell * slice_cells + static_cast<std::size_t>(p.y) * pitch + p.x;
        for (int k = 0; k < area; ++k) {
            const float v = p.weight * values[static_cast<std::size_t>(i) * area + k];
            num[corner + static_cast<std::size_t>(k / block) * pitch + k % block] += std::isnan(v) ? 0 : host_units(v);
        }
        den[corner] += host_units(p.weight);
    }

    struct Push {
        FixedTarget target;
        VkDeviceAddress patches, values;
        std::int32_t npatch, block;
    };
    static_assert(sizeof(Push) == 88, "matches the push block of shaders/test/fixed.comp");
    Buffer d_num(device, cells * 8, Memory::Device), d_den(device, cells * 8, Memory::Device);
    Buffer rows(device, slice_cells * 8, Memory::Device);
    UploadPlane d_patches(device, patches.size() * sizeof(Patch), 1);
    d_patches.write(patches.data(), 0, patches.size() * sizeof(Patch));
    UploadPlane d_values(device, values.size() * sizeof(float), 1);
    d_values.write(values.data(), 0, values.size() * sizeof(float));
    std::vector<float> src(slice_cells, 0.5f);
    UploadPlane d_src(device, src.size() * sizeof(float), 1);
    d_src.write(src.data(), 0, src.size() * sizeof(float));
    DownloadPlane out_num(device, cells * 8, 1), out_den(device, cells * 8, 1);
    DownloadPlane out(device, slice_cells * sizeof(float), static_cast<int>(ring));
    Pipeline add(device, {"test_fixed_add", sizeof(Push), 0, {}});

    std::vector<std::uint64_t> runs[2][2];
    std::vector<float> finished(slice_cells * ring);
    for (int run = 0; run < 2; ++run) {
        Slot slot(device);
        VkCommandBuffer upload = slot.commands(Slot::Upload);
        for (const UploadPlane* p : {&d_patches, &d_values, &d_src}) {
            if (p->needs_copy()) p->record_upload(upload);
        }
        VkCommandBuffer compute = slot.commands(Slot::Compute);
        fixed_clear(*device, compute, d_num.get(), 0, cells * 8);
        fixed_clear(*device, compute, d_den.get(), 0, cells * 8);
        Push push{};
        push.target.num = d_num.address();
        push.target.den = d_den.address();
        push.target.slice_step = static_cast<std::int64_t>(slice_cells);
        push.target.pitch = pitch;
        push.target.slice_base = base;
        push.target.slices = slices;
        push.target.slice_first = first;
        push.target.slice_ring = ring;
        push.patches = d_patches.address();
        push.values = d_values.address();
        push.npatch = npatch;
        push.block = block;
        add.dispatch(compute, &push, static_cast<std::uint32_t>((npatch * area + 63) / 64));
        compute_barrier(*device, compute);
        for (int s = 0; s < ring; ++s) {
            const VkDeviceAddress offset = static_cast<VkDeviceAddress>(s) * slice_cells * 8;
            aggregator.fixed_finish(compute, d_num.address() + offset, d_den.address() + offset, pitch, block,
                                    d_src.address(), width, height, rows.address(),
                                    out.address() + static_cast<VkDeviceAddress>(s) * slice_cells * sizeof(float));
            compute_barrier(*device, compute);  // rows is reused
        }
        // The cells themselves, through the download planes.
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        device->vk().vkCmdPipelineBarrier2(compute, &dependency);
        VkBufferCopy copy{0, 0, cells * 8};
        device->vk().vkCmdCopyBuffer(compute, d_num.get(), out_num.buffer(), 1, &copy);
        device->vk().vkCmdCopyBuffer(compute, d_den.get(), out_den.buffer(), 1, &copy);
        VkCommandBuffer download = slot.commands(Slot::Download);
        out_num.record_download(download);
        out_den.record_download(download);
        out.record_download(download);
        slot.submit();
        slot.wait();
        runs[run][0].resize(cells);
        runs[run][1].resize(cells);
        out_num.read(runs[run][0].data(), 0, cells * 8);
        out_den.read(runs[run][1].data(), 0, cells * 8);
        out.read(finished.data(), static_cast<std::ptrdiff_t>(slice_cells * sizeof(float)), slice_cells * sizeof(float));
    }
    char label[96];
    std::snprintf(label, sizeof label, "fixed %dx%d b%d n%d f%d slices %d base %d ring %d first %d", width, height, block,
                  npatch, frames, slices, base, ring, first);
    CHECK(runs[0][0] == num, "%s: num cells differ from the host's sums", label);
    CHECK(runs[0][1] == den, "%s: den cells differ from the host's sums", label);
    CHECK(runs[0][0] == runs[1][0] && runs[0][1] == runs[1][1], "%s: two runs differ", label);
    // Finish: the den of a pixel is the sum of the corners that cover it.
    double worst = 0.0;
    for (int s = 0; s < ring; ++s) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t at = s * slice_cells + static_cast<std::size_t>(y) * pitch + x;
                std::uint64_t w = 0;
                for (int k = 0; k < block && k <= y; ++k) {
                    for (int j = 0; j < block && j <= x; ++j) w += den[at - static_cast<std::size_t>(k) * pitch - j];
                }
                const float unit = 1.f / 4294967296.f;
                const float n_ = static_cast<float>(static_cast<std::int64_t>(num[at])) * unit;
                const float d_ = static_cast<float>(static_cast<std::int64_t>(w)) * unit;
                const float want = d_ > 1e-12f ? n_ / d_ : 0.5f;
                const float got = finished[s * slice_cells + static_cast<std::size_t>(y) * pitch + x];
                worst = std::max(worst, static_cast<double>(std::fabs(got - want)) / std::max(1e-3f, std::fabs(want)));
            }
        }
    }
    CHECK(worst < 4e-7, "%s: finish relative error %.3g", label, worst);
    std::printf("%s: cells exact, finish max rel %.3g\n", label, worst);
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk aggregate SKIP: no Vulkan device (%s)\n", vk.error.c_str());
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
        std::printf("vk aggregate SKIP: no supported Vulkan device\n");
        return 77;
    }
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    // NVIDIA fuses fma(): the host model gives its bits. Mesa splits it (D26).
    const bool exact = device->caps().vendor_id == 0x10de;
    for (const int n : {1, 2, 4, 8, 12, 16, 32, 64}) dct_case(device, n, exact);
    const Aggregator aggregator(device);
    fixed_case(device, aggregator, 61, 47, 8, 900, 1, 1, 0, 1, 0);    // spatial
    fixed_case(device, aggregator, 64, 40, 4, 1500, 3, 3, 0, 3, 0);   // legacy: one slice per slot
    fixed_case(device, aggregator, 50, 33, 16, 300, 3, 2, -1, 4, 3);  // rolling ring: slot 0 dropped, wraps
    fixed_case(device, aggregator, 37, 29, 1, 2000, 5, 5, 0, 5, 2);   // block 1, wrapping ring
    std::printf("vk aggregate: %d failures\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
