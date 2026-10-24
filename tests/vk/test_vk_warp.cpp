// SPDX-License-Identifier: GPL-2.0-only
// D23 warp layer (shaders/common/warp.glsl): the native layer (subgroup
// operations at a required subgroup size) and the emulated one (shared
// memory and barriers) on the same inputs, lane by lane against a host model
// of nss_cuda's warp operations, for segments of 4 to 32 lanes. Exits 77
// without a Vulkan device; an explicit device id may name an unsupported one
// (lavapipe).
//
// usage: test_vk_warp [device_id]
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace nss_vk;

constexpr int kOut = 10;
constexpr int kGroups = 8, kThreads = 128;
constexpr int kLanes = kGroups * kThreads;

struct Push {
    VkDeviceAddress input, result;
};

float bits_float(std::uint32_t u) {
    return std::bit_cast<float>(((u >> 9) & 0x7fffffu) | ((127u + (u % 8u)) << 23));
}

// The expected record of every lane; o[9] (w_any) is a lower bound.
std::vector<std::uint32_t> model(const std::vector<std::uint32_t>& in, int L) {
    std::vector<std::uint32_t> out(static_cast<std::size_t>(kLanes) * kOut);
    for (int seg = 0; seg < kLanes / L; ++seg) {
        const std::uint32_t* u = in.data() + seg * L;
        std::vector<std::uint64_t> wide(L);
        std::vector<float> sum(L);
        std::uint32_t ballot = 0, count = 0;
        bool any = false;
        for (int i = 0; i < L; ++i) {
            wide[i] = (std::uint64_t{u[i]} << 32) | std::uint64_t{static_cast<std::uint32_t>(u[i] * 2654435761u)};
            sum[i] = bits_float(u[i]);
            ballot |= (u[i] & 1u) << i;
            count += u[i] % 3u == 0u;
            any = any || u[i] == static_cast<std::uint32_t>(seg);
        }
        for (int m = L / 2; m >= 1; m /= 2) {  // the butterfly of w_sum_f
            std::vector<float> next(L);
            for (int i = 0; i < L; ++i) next[i] = sum[i] + sum[i ^ m];
            sum = next;
        }
        for (int i = 0; i < L; ++i) {
            std::uint32_t* o = out.data() + (static_cast<std::size_t>(seg) * L + i) * kOut;
            const int src = (i * 3 + 1) % L;
            const std::uint64_t up = i == 0 ? wide[0] : wide[i - 1];
            o[0] = u[i ^ (L > 1 ? L / 2 : 0)];
            o[1] = u[src];
            o[2] = static_cast<std::uint32_t>(wide[src] >> 32);
            o[3] = static_cast<std::uint32_t>(wide[src]);
            o[4] = static_cast<std::uint32_t>(up >> 32);
            o[5] = static_cast<std::uint32_t>(up);
            o[6] = ballot;
            o[7] = count;
            o[8] = std::bit_cast<std::uint32_t>(sum[i]);
            o[9] = any ? 1u : 0u;
        }
    }
    return out;
}

std::vector<std::uint32_t> run(const std::shared_ptr<Device>& device, const std::string& id, std::uint32_t subgroup,
                               const std::vector<std::uint32_t>& in) {
    Pipeline pipeline(device, {id, sizeof(Push), subgroup, {}});
    UploadPlane input(device, in.size() * sizeof(std::uint32_t), 1);
    DownloadPlane result(device, static_cast<std::size_t>(kLanes) * kOut * sizeof(std::uint32_t), 1);
    Slot slot(device);
    input.write(in.data(), 0, in.size() * sizeof(std::uint32_t));
    if (input.needs_copy()) input.record_upload(slot.commands(Slot::Upload));
    const Push push{input.address(), result.address()};
    pipeline.dispatch(slot.commands(Slot::Compute), &push, kGroups);
    result.record_download(slot.commands(Slot::Download));
    slot.submit();
    slot.wait();
    std::vector<std::uint32_t> out(static_cast<std::size_t>(kLanes) * kOut);
    result.read(out.data(), 0, out.size() * sizeof(std::uint32_t));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        std::printf("vk warp SKIP: no Vulkan device (%s)\n", vk.error.c_str());
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
        std::printf("vk warp SKIP: no supported Vulkan device\n");
        return 77;
    }
    std::shared_ptr<Device> device = Device::open(index, argc > 1);
    const DeviceCaps& caps = device->caps();
    std::mt19937 rng(23);
    std::vector<std::uint32_t> in(kLanes);
    for (std::uint32_t& u : in) u = rng();
    for (int i = 0; i < kLanes; i += 37) in[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i / 4);  // w_any hits
    int failures = 0, checked = 0;
    for (int L : {4, 8, 16, 32}) {
        const std::vector<std::uint32_t> expected = model(in, L);
        // Native at the widest selectable subgroup up to 32 that L divides.
        const std::uint32_t width = std::min<std::uint32_t>(caps.max_subgroup_size, 32);
        const bool native = width >= static_cast<std::uint32_t>(L) && width >= caps.min_subgroup_size;
        for (int emulated = 0; emulated < 2; ++emulated) {
            if (!emulated && !native) {
                std::printf("L = %d: native layer skipped (subgroups of at most %u lanes)\n", L, caps.max_subgroup_size);
                continue;
            }
            const std::string id = "test_warp_l" + std::to_string(L) + (emulated ? "_e" : "_n");
            const std::vector<std::uint32_t> got = run(device, id, emulated ? 0 : width, in);
            int bad = 0;
            for (int lane = 0; lane < kLanes; ++lane) {
                for (int k = 0; k < kOut; ++k) {
                    const std::size_t at = static_cast<std::size_t>(lane) * kOut + k;
                    // w_any: true whenever the segment's predicate is (more lanes may count).
                    const bool ok = k == kOut - 1 ? got[at] >= expected[at] : got[at] == expected[at];
                    if (!ok && bad++ < 5) {
                        std::fprintf(stderr, "FAIL: %s lane %d op %d: %08x != %08x\n", id.c_str(), lane, k, got[at],
                                     expected[at]);
                    }
                }
            }
            failures += bad;
            ++checked;
            std::printf("L = %d %s: %s\n", L, emulated ? "emulated" : "native", bad ? "FAIL" : "ok");
        }
    }
    if (failures) return EXIT_FAILURE;
    std::printf("vk warp OK (%d layers)\n", checked);
    return EXIT_SUCCESS;
}
