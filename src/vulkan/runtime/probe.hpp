// SPDX-License-Identifier: GPL-2.0-only
// Load probe for Backend(): creates a device, runs shaders/runtime/probe.comp
// once at a required subgroup size and checks the result. Every wait is bounded.
#pragma once

#include "vulkan/runtime/device.hpp"

#include <cstdint>
#include <string>

namespace nss_vk {

struct ProbeResult {
    bool ok = false;
    double milliseconds = 0.0;      // device creation through the checked result
    std::uint32_t subgroup_size = 0;  // the required size the probe ran at
    std::string error;
};

ProbeResult probe(VkPhysicalDevice device, const DeviceCaps& caps);

}  // namespace nss_vk
