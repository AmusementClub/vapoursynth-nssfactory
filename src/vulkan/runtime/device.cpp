// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/device.hpp"

#include "vulkan/runtime/instance.hpp"

#include <algorithm>
#include <cstring>

namespace nss_vk {

const char* support_name(Support support) {
    switch (support) {
        case Support::native: return "native";
        case Support::emulated: return "emulated";
        case Support::unsupported: break;
    }
    return "unsupported";
}

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: break;
    }
    return "other";
}

Classification classify(const DeviceCaps& caps) {
    Classification result;
    std::vector<std::string>& missing = result.missing;
    if (caps.api_version < kRequiredApiVersion) {
        missing.push_back("Vulkan 1.3 (device has " + version_text(caps.api_version) + ")");
    }
    if (!caps.has_compute_queue) missing.push_back("compute queue");
    if (!caps.buffer_device_address) missing.push_back("bufferDeviceAddress");
    if (!caps.timeline_semaphore) missing.push_back("timelineSemaphore");
    if (!caps.shader_int64) missing.push_back("shaderInt64");
    if (!caps.buffer_int64_atomics) missing.push_back("shaderBufferInt64Atomics");
    if (!caps.subgroup_in_compute) missing.push_back("subgroup operations in compute");
    const struct {
        VkSubgroupFeatureFlags bit;
        const char* name;
    } ops[] = {{VK_SUBGROUP_FEATURE_BASIC_BIT, "subgroup basic"},
               {VK_SUBGROUP_FEATURE_VOTE_BIT, "subgroup vote"},
               {VK_SUBGROUP_FEATURE_BALLOT_BIT, "subgroup ballot"},
               {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "subgroup shuffle"},
               {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "subgroup arithmetic"}};
    for (const auto& op : ops) {
        if (!(caps.subgroup_ops & op.bit)) missing.push_back(op.name);
    }
    if (!caps.subgroup_size_control) missing.push_back("subgroupSizeControl");
    if (!caps.compute_full_subgroups) missing.push_back("computeFullSubgroups");
    if (!caps.required_size_in_compute) missing.push_back("required subgroup size in compute");
    if (caps.shared_memory < kRequiredSharedMemory) {
        missing.push_back("maxComputeSharedMemorySize " + std::to_string(caps.shared_memory / 1024) + " KiB < " +
                          std::to_string(kRequiredSharedMemory / 1024) + " KiB");
    }
    const std::uint32_t widest = caps.max_subgroup_size;
    if (widest == 0 || (widest & (widest - 1)) != 0 || caps.min_subgroup_size == 0 ||
        caps.min_subgroup_size > widest) {
        missing.push_back("subgroup size range [" + std::to_string(caps.min_subgroup_size) + ", " +
                          std::to_string(widest) + "]");
    }
    if (!missing.empty()) {
        return result;
    }
    result.native_lanes = widest;
    result.support = widest >= kWarpLanes ? Support::native : Support::emulated;
    return result;
}

std::string join(const std::vector<std::string>& items) {
    std::string text;
    for (const std::string& item : items) {
        if (!text.empty()) text += ",";
        text += item;
    }
    return text;
}

std::vector<std::string> optional_features(const DeviceCaps& caps) {
    std::vector<std::string> present;
    if (caps.shader_float64) present.push_back("shaderFloat64");
    if (caps.cooperative_matrix) present.push_back("cooperative_matrix");
    if (caps.external_memory_host) present.push_back("external_memory_host");
    if (caps.transfer_queues > 0) present.push_back("transfer_queue");
    if (caps.host_visible_device_bytes > 0) present.push_back("host_visible_device_memory");
    return present;
}

DeviceCaps query_caps(VkPhysicalDevice device) {
    DeviceCaps caps;

    std::uint32_t extension_count = 0;
    std::vector<VkExtensionProperties> extensions;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr) == VK_SUCCESS) {
        extensions.resize(extension_count);
        if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, extensions.data()) < VK_SUCCESS) {
            extension_count = 0;
        }
        extensions.resize(extension_count);
    }
    auto has_extension = [&](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
    };
    caps.cooperative_matrix = has_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    caps.external_memory_host = has_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);

    VkPhysicalDeviceProperties basic{};
    vkGetPhysicalDeviceProperties(device, &basic);
    caps.api_version = basic.apiVersion;
    // The 1.1–1.3 property and feature structs need a 1.3 device; older ones
    // are classified unsupported by their version alone.
    const bool v13 = basic.apiVersion >= kRequiredApiVersion;

    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &driver};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, &p11};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
    if (v13) {
        vkGetPhysicalDeviceProperties2(device, &properties);
    } else {
        properties.properties = basic;
    }
    const VkPhysicalDeviceProperties& p = properties.properties;
    caps.name = p.deviceName;
    caps.vendor_id = p.vendorID;
    caps.type = p.deviceType;
    caps.shared_memory = p.limits.maxComputeSharedMemorySize;
    if (v13) {
        caps.driver_name = driver.driverName;
        caps.driver_info = driver.driverInfo;
        caps.subgroup_size = p11.subgroupSize;
        caps.subgroup_in_compute = (p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        caps.subgroup_ops = p11.subgroupSupportedOperations;
        caps.min_subgroup_size = p13.minSubgroupSize;
        caps.max_subgroup_size = p13.maxSubgroupSize;
        caps.required_size_in_compute = (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;

        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
        vkGetPhysicalDeviceFeatures2(device, &features);
        caps.shader_int64 = features.features.shaderInt64;
        caps.shader_float64 = features.features.shaderFloat64;
        caps.buffer_device_address = f12.bufferDeviceAddress;
        caps.timeline_semaphore = f12.timelineSemaphore;
        caps.buffer_int64_atomics = f12.shaderBufferInt64Atomics;
        caps.subgroup_size_control = f13.subgroupSizeControl;
        caps.compute_full_subgroups = f13.computeFullSubgroups;
    }

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
    families.resize(family_count);
    for (const VkQueueFamilyProperties& family : families) {
        const VkQueueFlags flags = family.queueFlags;
        if (flags & VK_QUEUE_COMPUTE_BIT) {
            caps.has_compute_queue = true;
        } else if ((flags & ~(VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_PROTECTED_BIT)) == VK_QUEUE_TRANSFER_BIT) {
            // Copy-only families (D28); video and optical-flow families also
            // carry the transfer bit but are not counted.
            caps.transfer_queues += family.queueCount;
        }
    }

    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    for (std::uint32_t h = 0; h < memory.memoryHeapCount; ++h) {
        if (memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            caps.device_local_bytes += memory.memoryHeaps[h].size;
        }
    }
    constexpr VkMemoryPropertyFlags bar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    for (std::uint32_t t = 0; t < memory.memoryTypeCount; ++t) {
        if ((memory.memoryTypes[t].propertyFlags & bar) == bar) {
            caps.host_visible_device_bytes = std::max<std::uint64_t>(caps.host_visible_device_bytes,
                                                       memory.memoryHeaps[memory.memoryTypes[t].heapIndex].size);
        }
    }
    return caps;
}

}  // namespace nss_vk
