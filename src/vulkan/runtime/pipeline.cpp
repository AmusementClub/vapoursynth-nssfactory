// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/pipeline.hpp"

#include "vulkan/runtime/error.hpp"
#include "vulkan/runtime/shaders.hpp"

#include <stdexcept>
#include <utility>

namespace nss_vk {

Pipeline::Pipeline(std::shared_ptr<Device> device, const PipelineDesc& desc)
    : device_(std::move(device)), push_bytes_(desc.push_bytes) {
    const VolkDeviceTable& vk = device_->vk();
    const VkDevice handle = device_->handle();
    if (desc.push_bytes > 128 || desc.push_bytes % 4 != 0) {
        throw std::logic_error("nss_vk: push constants must be a multiple of 4 bytes, at most 128");
    }
    if (desc.subgroup_size != 0) {
        const DeviceCaps& caps = device_->caps();
        if (!caps.subgroup_size_control || !caps.compute_full_subgroups || !caps.required_size_in_compute) {
            throw std::logic_error("nss_vk: required subgroup size without subgroup size control");
        }
        if ((desc.subgroup_size & (desc.subgroup_size - 1)) != 0 || desc.subgroup_size < caps.min_subgroup_size ||
            desc.subgroup_size > caps.max_subgroup_size) {
            throw std::logic_error("nss_vk: subgroup size " + std::to_string(desc.subgroup_size) +
                                   " is not a power of two in the device's range");
        }
    }
    const Spirv code = shader(desc.shader);
    VkShaderModule module = VK_NULL_HANDLE;
    try {
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, desc.push_bytes};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = desc.push_bytes ? 1u : 0u;
        layout.pPushConstantRanges = &range;
        NSS_VK_CHECK(vk.vkCreatePipelineLayout(handle, &layout, nullptr, &layout_));

        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = code.count * sizeof(std::uint32_t);
        module_info.pCode = code.words;
        NSS_VK_CHECK(vk.vkCreateShaderModule(handle, &module_info, nullptr, &module));

        std::vector<VkSpecializationMapEntry> entries(desc.constants.size());
        for (std::size_t i = 0; i < entries.size(); ++i) {
            entries[i] = {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(i * sizeof(std::uint32_t)),
                          sizeof(std::uint32_t)};
        }
        VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(),
                                            desc.constants.size() * sizeof(std::uint32_t), desc.constants.data()};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        size.requiredSubgroupSize = desc.subgroup_size;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      desc.subgroup_size ? &size : nullptr,
                      desc.subgroup_size ? VkPipelineShaderStageCreateFlags(
                                               VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
                                         : 0u,
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      module,
                      "main",
                      entries.empty() ? nullptr : &specialization};
        info.layout = layout_;
        NSS_VK_CHECK(vk.vkCreateComputePipelines(handle, device_->pipeline_cache(), 1, &info, nullptr, &pipeline_));
    } catch (...) {
        if (module) vk.vkDestroyShaderModule(handle, module, nullptr);
        reset();
        throw;
    }
    vk.vkDestroyShaderModule(handle, module, nullptr);
}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = std::move(other.device_);
        layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
        pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
        push_bytes_ = std::exchange(other.push_bytes_, 0);
    }
    return *this;
}

void Pipeline::reset() noexcept {
    if (device_ && !device_->poisoned()) {
        if (pipeline_) device_->vk().vkDestroyPipeline(device_->handle(), pipeline_, nullptr);
        if (layout_) device_->vk().vkDestroyPipelineLayout(device_->handle(), layout_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    device_.reset();
}

void Pipeline::dispatch(VkCommandBuffer commands, const void* push, std::uint32_t x, std::uint32_t y,
                        std::uint32_t z) const {
    const VolkDeviceTable& vk = device_->vk();
    vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    if (push_bytes_) vk.vkCmdPushConstants(commands, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
    vk.vkCmdDispatch(commands, x, y, z);
}

void compute_barrier(const Device& device, VkCommandBuffer commands) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    device.vk().vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace nss_vk
