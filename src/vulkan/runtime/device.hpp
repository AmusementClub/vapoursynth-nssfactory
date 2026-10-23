// SPDX-License-Identifier: GPL-2.0-only
// Device capabilities and the D22/D23 support classification (docs/vulkan-plan.md,
// as amended 2026-10-24). DeviceCaps is plain data so classify() is tested
// without a device (tests/vk/test_vk_support.cpp).
#pragma once

#include "vulkan/runtime/vk.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nss_vk {

// D22: shared memory every V1/V2 kernel may assume.
inline constexpr std::uint32_t kRequiredSharedMemory = 48u * 1024u;
// D22: subgroup operations the warp layer uses in compute shaders.
inline constexpr VkSubgroupFeatureFlags kRequiredSubgroupOps =
    VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
// D23: the CUDA kernels' warp width.
inline constexpr std::uint32_t kWarpLanes = 32;

struct DeviceCaps {
    std::string name;
    std::uint32_t vendor_id = 0;
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    std::uint32_t api_version = 0;
    std::string driver_name;  // VkPhysicalDeviceDriverProperties
    std::string driver_info;

    // Required (D22).
    bool buffer_device_address = false;
    bool timeline_semaphore = false;
    bool shader_int64 = false;
    bool buffer_int64_atomics = false;
    bool subgroup_size_control = false;
    bool compute_full_subgroups = false;
    bool required_size_in_compute = false;  // requiredSubgroupSizeStages has compute
    bool subgroup_in_compute = false;       // subgroup supportedStages has compute
    VkSubgroupFeatureFlags subgroup_ops = 0;
    std::uint32_t subgroup_size = 0;  // default width
    std::uint32_t min_subgroup_size = 0;
    std::uint32_t max_subgroup_size = 0;
    std::uint32_t shared_memory = 0;  // maxComputeSharedMemorySize, bytes
    bool has_compute_queue = false;

    // Optional, reported (D22, D28).
    bool shader_float64 = false;
    bool host_query_reset = false;      // timestamps reset from the host (Slot)
    bool cooperative_matrix = false;    // VK_KHR_cooperative_matrix
    bool external_memory_host = false;  // VK_EXT_external_memory_host (not used: D28)
    std::uint32_t transfer_queues = 0;  // queues in copy-only (transfer, no compute/graphics/video) families
    std::uint64_t device_local_bytes = 0;
    std::uint64_t host_visible_device_bytes = 0;  // largest device-local + host-visible heap (ReBAR or the 256 MiB BAR)
};

// native: every kernel runs on the subgroup warp layer. emulated: the device
// meets D22, but its widest subgroup is below 32, so kernels with more lanes
// per reference than `native_lanes` run on the emulated layer (bit-identical,
// slower). unsupported: a D22 requirement is missing (`missing` lists them).
enum class Support { native, emulated, unsupported };

struct Classification {
    Support support = Support::unsupported;
    // Widest selectable subgroup: a kernel with L lanes per reference runs
    // native when L <= native_lanes (L and subgroup sizes are powers of two,
    // so L then divides the width; D23 as amended 2026-10-24).
    std::uint32_t native_lanes = 0;
    std::vector<std::string> missing;
};

const char* support_name(Support support);
const char* device_type_name(VkPhysicalDeviceType type);
Classification classify(const DeviceCaps& caps);

// Queries a physical device of the shared instance.
DeviceCaps query_caps(VkPhysicalDevice device);

// "a,b,c"; "" for none.
std::string join(const std::vector<std::string>& items);
// Present optional features, for Backend().
std::vector<std::string> optional_features(const DeviceCaps& caps);

}  // namespace nss_vk
