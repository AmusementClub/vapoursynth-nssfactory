// SPDX-License-Identifier: GPL-2.0-only
// The process-wide Vulkan instance of libnss_vk. Created on first use and kept
// for the life of the process (no teardown at unload: driver atexit order is
// not ours to control). Without a loader, or with a loader older than 1.3,
// `handle` stays null and `error` says why; nothing throws.
#pragma once

#include "vulkan/runtime/vk.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nss_vk {

// The API version every nss_vk device and the instance must reach (D22).
inline constexpr std::uint32_t kRequiredApiVersion = VK_API_VERSION_1_3;

struct Instance {
    VkInstance handle = VK_NULL_HANDLE;
    std::uint32_t loader_version = 0;  // vkEnumerateInstanceVersion; 0 without a loader
    std::vector<VkPhysicalDevice> devices;  // loader order; device_id indexes this
    std::string error;                 // non-empty when the loader or instance is unusable
};

const Instance& instance();

// "1.4.341"; "none" for 0.
std::string version_text(std::uint32_t version);

}  // namespace nss_vk
