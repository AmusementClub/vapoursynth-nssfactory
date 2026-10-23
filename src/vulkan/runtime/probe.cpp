// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/probe.hpp"

#include "vulkan/runtime/error.hpp"
#include "vulkan/runtime/shaders.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <vector>

namespace nss_vk {
namespace {

constexpr std::uint32_t kGroupSize = 128;  // local_size_x of probe.comp
constexpr std::uint32_t kGroups = 4;
constexpr std::uint32_t kInvocations = kGroupSize * kGroups;
constexpr std::uint64_t kWaitNs = 5'000'000'000ull;  // the GPU may drive a desktop: never wait unbounded

// Owns every object the probe creates; destroys them in reverse order.
struct ProbeObjects {
    VkDevice device = VK_NULL_HANDLE;
    VolkDeviceTable vk{};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    ~ProbeObjects() {
        if (device == VK_NULL_HANDLE) return;
        if (fence) vk.vkDestroyFence(device, fence, nullptr);
        if (pool) vk.vkDestroyCommandPool(device, pool, nullptr);
        if (pipeline) vk.vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vk.vkDestroyPipelineLayout(device, layout, nullptr);
        if (module) vk.vkDestroyShaderModule(device, module, nullptr);
        if (buffer) vk.vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vk.vkFreeMemory(device, memory, nullptr);
        vk.vkDestroyDevice(device, nullptr);
    }
};

struct PushConstants {
    VkDeviceAddress result;
};

void run_probe(VkPhysicalDevice physical, const DeviceCaps& caps, ProbeResult& result) {
    // The probe's own needs; the full D22 list is Backend()'s classification.
    std::vector<std::string> missing;
    if (caps.api_version < VK_API_VERSION_1_3) missing.push_back("Vulkan 1.3");
    if (!caps.has_compute_queue) missing.push_back("compute queue");
    if (!caps.buffer_device_address) missing.push_back("bufferDeviceAddress");
    if (!caps.shader_int64) missing.push_back("shaderInt64");
    if (!caps.buffer_int64_atomics) missing.push_back("shaderBufferInt64Atomics");
    if (!caps.subgroup_size_control || !caps.compute_full_subgroups || !caps.required_size_in_compute) {
        missing.push_back("subgroup size control in compute");
    }
    if ((caps.subgroup_ops & kRequiredSubgroupOps) != kRequiredSubgroupOps || !caps.subgroup_in_compute) {
        missing.push_back("subgroup operations");
    }
    if (!missing.empty()) {
        result.error = "probe needs " + join(missing);
        return;
    }
    // The warp width when selectable, else the nearest size the device allows.
    const std::uint32_t width = std::clamp(kWarpLanes, caps.min_subgroup_size, caps.max_subgroup_size);
    result.subgroup_size = width;
    if (width < 2 || kGroupSize % width != 0) {
        result.error = "probe cannot run at subgroup size " + std::to_string(width);
        return;
    }

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    std::uint32_t family = 0;
    while (family < family_count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;

    ProbeObjects o;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderBufferInt64Atomics = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    features.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    NSS_VK_CHECK(vkCreateDevice(physical, &device_info, nullptr, &o.device));
    volkLoadDeviceTable(&o.vk, o.device);
    const VolkDeviceTable& vk = o.vk;
    VkQueue queue = VK_NULL_HANDLE;
    vk.vkGetDeviceQueue(o.device, family, 0, &queue);

    // Result buffer: host-visible, coherent, addressable from the shader.
    const VkDeviceSize bytes = sizeof(std::uint64_t) + kInvocations * sizeof(std::uint32_t);
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    NSS_VK_CHECK(vk.vkCreateBuffer(o.device, &buffer_info, nullptr, &o.buffer));
    VkMemoryRequirements requirements{};
    vk.vkGetBufferMemoryRequirements(o.device, o.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t type = 0;
    while (type < memory.memoryTypeCount &&
           !((requirements.memoryTypeBits >> type & 1u) && (memory.memoryTypes[type].propertyFlags & wanted) == wanted)) {
        ++type;
    }
    if (type == memory.memoryTypeCount) {
        result.error = "probe: no host-visible coherent memory type for a storage buffer";
        return;
    }
    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags_info};
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = type;
    NSS_VK_CHECK(vk.vkAllocateMemory(o.device, &allocate_info, nullptr, &o.memory));
    NSS_VK_CHECK(vk.vkBindBufferMemory(o.device, o.buffer, o.memory, 0));
    void* mapped = nullptr;
    NSS_VK_CHECK(vk.vkMapMemory(o.device, o.memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    std::memset(mapped, 0, static_cast<std::size_t>(bytes));
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = o.buffer;
    const PushConstants push{vk.vkGetBufferDeviceAddress(o.device, &address_info)};

    const Spirv code = shader("runtime_probe");
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.count * sizeof(std::uint32_t);
    module_info.pCode = code.words;
    NSS_VK_CHECK(vk.vkCreateShaderModule(o.device, &module_info, nullptr, &o.module));
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    NSS_VK_CHECK(vk.vkCreatePipelineLayout(o.device, &layout_info, nullptr, &o.layout));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size_info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    size_info.requiredSubgroupSize = width;
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &size_info,
                           VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
                           VK_SHADER_STAGE_COMPUTE_BIT, o.module, "main", nullptr};
    pipeline_info.layout = o.layout;
    NSS_VK_CHECK(vk.vkCreateComputePipelines(o.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &o.pipeline));

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    NSS_VK_CHECK(vk.vkCreateCommandPool(o.device, &pool_info, nullptr, &o.pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = o.pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    NSS_VK_CHECK(vk.vkAllocateCommandBuffers(o.device, &command_info, &commands));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    NSS_VK_CHECK(vk.vkBeginCommandBuffer(commands, &begin));
    vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, o.pipeline);
    vk.vkCmdPushConstants(commands, o.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vk.vkCmdDispatch(commands, kGroups, 1, 1);
    // Make the shader writes visible to the host read after the fence.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                            &barrier, 0, nullptr, 0, nullptr);
    NSS_VK_CHECK(vk.vkEndCommandBuffer(commands));

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    NSS_VK_CHECK(vk.vkCreateFence(o.device, &fence_info, nullptr, &o.fence));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    NSS_VK_CHECK(vk.vkQueueSubmit(queue, 1, &submit, o.fence));
    const VkResult waited = vk.vkWaitForFences(o.device, 1, &o.fence, VK_TRUE, kWaitNs);
    NSS_VK_CHECK(waited);
    if (waited == VK_TIMEOUT) {
        // Destroying objects still in use is undefined; wait for the queue
        // before ProbeObjects tears them down.
        vk.vkQueueWaitIdle(queue);
        result.error = "probe: dispatch did not finish within 5 s";
        return;
    }

    std::uint64_t sum = 0;
    std::memcpy(&sum, mapped, sizeof(sum));
    std::vector<std::uint32_t> lanes(kInvocations);
    std::memcpy(lanes.data(), static_cast<const char*>(mapped) + sizeof(sum), lanes.size() * sizeof(std::uint32_t));
    const std::uint64_t expected = std::uint64_t{kInvocations} * (kInvocations + 1) / 2;
    const auto bad = std::find_if(lanes.begin(), lanes.end(), [&](std::uint32_t v) { return v != width; });
    if (bad != lanes.end()) {
        result.error = "probe: invocation " + std::to_string(bad - lanes.begin()) + " reported " +
                       std::to_string(*bad) + " (expected subgroup width " + std::to_string(width) + ")";
    } else if (sum != expected) {
        result.error = "probe: 64-bit atomic sum " + std::to_string(sum) + " != " + std::to_string(expected);
    } else {
        result.ok = true;
    }
}

}  // namespace

ProbeResult probe(VkPhysicalDevice device, const DeviceCaps& caps) {
    ProbeResult result;
    const auto start = std::chrono::steady_clock::now();
    try {
        run_probe(device, caps, result);
    } catch (const std::exception& error) {
        result.ok = false;
        result.error = std::string("probe: ") + error.what();
    }
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

}  // namespace nss_vk
