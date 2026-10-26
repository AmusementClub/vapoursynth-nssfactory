// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/common/aggregate.hpp"

#include <stdexcept>

namespace nss_vk {
namespace {

struct FinishPush {
    VkDeviceAddress num, den, rows, src, dst;
    std::int32_t block, width, height, pitch;
};
static_assert(sizeof(FinishPush) == 56, "matches the push block of shaders/common/aggregate.comp");

}  // namespace

void fixed_clear(const Device& device, VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset,
                 VkDeviceSize bytes) {
    if (bytes == 0) return;
    // Earlier compute work in the command buffer (a finish reading the
    // cells, the kernels adding to them) completes before the fill...
    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    before.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    before.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    // ...and the compute work after it sees the zeros.
    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &before;
    device.vk().vkCmdPipelineBarrier2(commands, &dependency);
    device.vk().vkCmdFillBuffer(commands, buffer, offset, bytes, 0u);
    dependency.pMemoryBarriers = &after;
    device.vk().vkCmdPipelineBarrier2(commands, &dependency);
}

Aggregator::Aggregator(std::shared_ptr<Device> device)
    : device_(std::move(device)),
      rows_(device_, {"aggregate_fixed_rows", sizeof(FinishPush), 0, {}}),
      finish_(device_, {"aggregate_fixed_finish", sizeof(FinishPush), 0, {}}) {}

void Aggregator::fixed_finish(VkCommandBuffer commands, VkDeviceAddress num, VkDeviceAddress den, int pitch, int block,
                              VkDeviceAddress src, int width, int height, VkDeviceAddress rows,
                              VkDeviceAddress out) const {
    if (width <= 0 || height <= 0) return;
    if (static_cast<long long>(pitch) * height >= (1LL << 31)) {
        throw std::logic_error("nss_vk: fixed_finish plane beyond 32-bit indices");
    }
    const FinishPush push{num, den, rows, src, out, block, width, height, pitch};
    const std::uint32_t gx = static_cast<std::uint32_t>((width + 31) / 32);
    const std::uint32_t gy = static_cast<std::uint32_t>((height + 7) / 8);
    rows_.dispatch(commands, &push, gx, gy);
    compute_barrier(*device_, commands);
    finish_.dispatch(commands, &push, gx, gy);
}

}  // namespace nss_vk
