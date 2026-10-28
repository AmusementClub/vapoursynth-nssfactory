// SPDX-License-Identifier: GPL-2.0-only
// The arithmetic of shaders/pnlm/common.glsl against the model
// (include/nss/params/pnlm.hpp) on the host: pnlm_exp, correctly rounded
// quotients (div_rn), weights and tents, bit for bit. Reports mismatches;
// fails on NVIDIA (fused fma, IEEE operations), only reports elsewhere (D26).
// Exits 77 without a supported Vulkan device.
//
// usage: test_vk_pnlm_math [device_id]
#include "nss/params/pnlm.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace nss_vk;

namespace {

struct Push {
    VkDeviceAddress in4, out4, flat_in, flat_out;
    std::int32_t count;
    float inv_area, offset, inv_h2, inv_s2;
};

std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    int index = -1;
    if (argc > 1) {
        index = std::atoi(argv[1]);
    } else {
        for (int i = 0; i < static_cast<int>(vk.devices.size()) && index < 0; ++i) {
            if (classify(query_caps(vk.devices[static_cast<std::size_t>(i)])).support != Support::unsupported) index = i;
        }
    }
    if (index < 0) {
        std::printf("vk pnlm math SKIP: no supported Vulkan device\n");
        return 77;
    }
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    const bool exact = device->caps().vendor_id == 0x10de;
    const nss::PnlmScale scale = nss::pnlm_scale(6.f, 6);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> ux(-95.f, 0.f), ua(0.f, 60.f), ub(0.5f, 1200.f), ud(0.f, 0.08f);
    const int count = 1 << 18;
    std::vector<float> in(4 * static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        in[4 * i] = i < 64 ? -static_cast<float>(i) * 1.5f : ux(rng);
        in[4 * i + 1] = ua(rng);
        in[4 * i + 2] = i % 7 == 0 ? std::max(1.f, std::floor(ub(rng))) : ub(rng);
        in[4 * i + 3] = ud(rng);
    }
    // Flat rule inputs: the sums of k p^2 values near a level, as in a patch group.
    const float inv_s2 = scale.inv_s2;
    std::vector<float> flat_in(4 * static_cast<std::size_t>(count));
    std::uniform_real_distribution<float> level(0.f, 1.f), noise(-0.06f, 0.06f), raw(0.f, 1.f);
    std::uniform_int_distribution<int> ks(1, 32);
    for (int i = 0; i < count; ++i) {
        const int n = ks(rng) * 36;
        const float m = level(rng);
        float s1 = 0.f, s2 = 0.f;
        for (int j = 0; j < n; ++j) {
            const float v = m + noise(rng) * static_cast<float>(i % 5) * 0.5f;
            s1 = s1 + v;
            s2 = std::fma(v, v, s2);
        }
        flat_in[4 * i] = s1;
        flat_in[4 * i + 1] = s2;
        flat_in[4 * i + 2] = static_cast<float>(n);
        flat_in[4 * i + 3] = raw(rng);
    }
    const std::size_t bytes = in.size() * sizeof(float);
    UploadPlane d_in(device, bytes, 1), d_flat(device, bytes, 1);
    d_in.write(in.data(), 0, bytes);
    d_flat.write(flat_in.data(), 0, bytes);
    DownloadPlane d_out(device, bytes, 1), d_flat_out(device, bytes / 4, 1);
    Pipeline kernel(device, {"test_pnlm_math", sizeof(Push), 0, {}});
    Slot slot(device);
    if (d_in.needs_copy()) d_in.record_upload(slot.commands(Slot::Upload));
    if (d_flat.needs_copy()) d_flat.record_upload(slot.commands(Slot::Upload));
    const Push push{d_in.address(), d_out.address(), d_flat.address(), d_flat_out.address(), count,
                    scale.inv_area, scale.offset, scale.inv_h2, inv_s2};
    kernel.dispatch(slot.commands(Slot::Compute), &push, static_cast<std::uint32_t>((count + 63) / 64));
    d_out.record_download(slot.commands(Slot::Download));
    d_flat_out.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    std::vector<float> out(in.size());
    d_out.read(out.data(), static_cast<std::ptrdiff_t>(bytes), bytes);
    std::vector<float> flat_out(static_cast<std::size_t>(count));
    d_flat_out.read(flat_out.data(), static_cast<std::ptrdiff_t>(bytes / 4), bytes / 4);
    int flat_bad = 0;
    nss::PnlmScale flat_scale = scale;
    flat_scale.inv_s2 = inv_s2;
    for (int i = 0; i < count; ++i) {
        const nss::PnlmFlat f = nss::pnlm_flat(flat_in[4 * i], flat_in[4 * i + 1], flat_in[4 * i + 2], flat_scale);
        const float want = nss::pnlm_mix(flat_in[4 * i + 3], f);
        if (bits(want) != bits(flat_out[static_cast<std::size_t>(i)]) && flat_bad++ < 4) {
            std::printf("  flat(%g, %g, %g, %g): device %.9g host %.9g\n", flat_in[4 * i], flat_in[4 * i + 1],
                        flat_in[4 * i + 2], flat_in[4 * i + 3], flat_out[static_cast<std::size_t>(i)], want);
        }
    }
    int bad[4] = {};
    for (int i = 0; i < count; ++i) {
        const float want[4] = {nss::pnlm_exp(in[4 * i]), in[4 * i + 1] / in[4 * i + 2],
                               nss::pnlm_weight(in[4 * i + 3], scale), nss::pnlm_tent(i % 6, 6)};
        for (int k = 0; k < 4; ++k) {
            if (bits(want[k]) == bits(out[4 * i + k])) continue;
            if (bad[k]++ < 4) {
                std::printf("  %s(%g%s%g): device %.9g host %.9g\n", k == 0 ? "exp" : k == 1 ? "div" : k == 2 ? "weight" : "tent",
                            k == 1 ? in[4 * i + 1] : k == 0 ? in[4 * i] : in[4 * i + 3], k == 1 ? ", " : "",
                            k == 1 ? in[4 * i + 2] : 0.0, out[4 * i + k], want[k]);
            }
        }
    }
    std::printf("vk pnlm math: %d inputs; mismatches exp %d, div %d, weight %d, tent %d, flat %d%s\n", count, bad[0],
                bad[1], bad[2], bad[3], flat_bad, exact ? "" : " (reported only: not NVIDIA)");
    return exact && (bad[0] || bad[1] || bad[2] || bad[3] || flat_bad) ? EXIT_FAILURE : EXIT_SUCCESS;
}
