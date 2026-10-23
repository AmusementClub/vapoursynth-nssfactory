// SPDX-License-Identifier: GPL-2.0-only
// D22/D23 support classification (as amended 2026-10-24) on synthetic device
// capabilities; needs no Vulkan device or loader.
#include "vulkan/runtime/device.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool lists(const nss_vk::Classification& c, const std::string& prefix) {
    for (const std::string& item : c.missing) {
        if (item.compare(0, prefix.size(), prefix) == 0) return true;
    }
    return false;
}

// Every D22 requirement met, subgroup sizes [min, max], shared memory in KiB.
nss_vk::DeviceCaps device(std::uint32_t min_size, std::uint32_t max_size, std::uint32_t shared_kib) {
    nss_vk::DeviceCaps caps;
    caps.api_version = VK_API_VERSION_1_3;
    caps.has_compute_queue = true;
    caps.buffer_device_address = true;
    caps.timeline_semaphore = true;
    caps.shader_int64 = true;
    caps.buffer_int64_atomics = true;
    caps.subgroup_size_control = true;
    caps.compute_full_subgroups = true;
    caps.required_size_in_compute = true;
    caps.subgroup_in_compute = true;
    caps.subgroup_ops = nss_vk::kRequiredSubgroupOps | VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
    caps.subgroup_size = max_size;
    caps.min_subgroup_size = min_size;
    caps.max_subgroup_size = max_size;
    caps.shared_memory = shared_kib * 1024;
    return caps;
}

}  // namespace

int main() {
    using nss_vk::Support;
    // RTX 5080 (NVIDIA 595): subgroup 32 only, 48 KiB.
    const auto nvidia = nss_vk::classify(device(32, 32, 48));
    expect(nvidia.support == Support::native && nvidia.native_lanes == 32 && nvidia.missing.empty(), "nvidia native");
    // AMD RDNA: wave32 and wave64, 64 KiB.
    const auto rdna = nss_vk::classify(device(32, 64, 64));
    expect(rdna.support == Support::native && rdna.native_lanes == 64, "rdna native");
    // GCN, wave64 only: 32-lane kernels run in 32-lane segments of the wave (L divides 64).
    const auto gcn = nss_vk::classify(device(64, 64, 64));
    expect(gcn.support == Support::native && gcn.native_lanes == 64, "wave64-only native");
    // Intel Xe: 8 to 32.
    const auto xe = nss_vk::classify(device(8, 32, 64));
    expect(xe.support == Support::native && xe.native_lanes == 32, "xe native");
    // A device whose widest subgroup is 16: lane kernels up to L = 16 native, wider ones emulated.
    const auto narrow = nss_vk::classify(device(8, 16, 48));
    expect(narrow.support == Support::emulated && narrow.native_lanes == 16, "narrow emulated");
    // lavapipe (Mesa 26): subgroup 8 and 32 KiB, below D22's 48 KiB (no CI tier, 2026-10-24).
    const auto lavapipe = nss_vk::classify(device(8, 8, 32));
    expect(lavapipe.support == Support::unsupported, "lavapipe unsupported");
    expect(lavapipe.missing.size() == 1 && lists(lavapipe, "maxComputeSharedMemorySize 32 KiB"), "lavapipe reason");
    expect(lavapipe.native_lanes == 0, "unsupported has no native lanes");

    auto old = device(32, 32, 48);
    old.api_version = VK_API_VERSION_1_2;
    expect(lists(nss_vk::classify(old), "Vulkan 1.3"), "1.2 device");
    auto no_atomics = device(32, 32, 48);
    no_atomics.buffer_int64_atomics = false;
    expect(lists(nss_vk::classify(no_atomics), "shaderBufferInt64Atomics"), "int64 atomics");
    auto no_shuffle = device(32, 32, 48);
    no_shuffle.subgroup_ops &= ~VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    expect(lists(nss_vk::classify(no_shuffle), "subgroup shuffle"), "shuffle");
    auto no_control = device(32, 32, 48);
    no_control.required_size_in_compute = false;
    no_control.compute_full_subgroups = false;
    const auto control = nss_vk::classify(no_control);
    expect(lists(control, "computeFullSubgroups") && lists(control, "required subgroup size"), "size control");
    auto bad_range = device(64, 32, 48);
    expect(lists(nss_vk::classify(bad_range), "subgroup size range"), "inverted range");
    // FP64, cooperative matrix and host import stay optional (D22).
    auto optional = device(32, 32, 48);
    optional.shader_float64 = true;
    optional.cooperative_matrix = true;
    optional.transfer_queues = 2;
    expect(nss_vk::join(nss_vk::optional_features(optional)) == "shaderFloat64,cooperative_matrix,transfer_queue",
           "optional list");
    expect(nss_vk::join({}).empty(), "empty join");

    if (failures) return EXIT_FAILURE;
    std::printf("vk support OK\n");
    return EXIT_SUCCESS;
}
