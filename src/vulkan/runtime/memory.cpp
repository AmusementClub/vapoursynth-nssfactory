// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/memory.hpp"

#include "vulkan/runtime/error.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace nss_vk {

void copy_rows(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
               int rows) {
    const auto* in = static_cast<const std::uint8_t*>(src);
    auto* out = static_cast<std::uint8_t*>(dst);
    if (src_stride == dst_stride && static_cast<std::size_t>(src_stride) == row_bytes) {
        std::memcpy(out, in, row_bytes * static_cast<std::size_t>(rows));
        return;
    }
    for (int row = 0; row < rows; ++row) {
        std::memcpy(out + row * dst_stride, in + row * src_stride, row_bytes);
    }
}

Buffer::Buffer(std::shared_ptr<Device> device, std::size_t bytes, Memory memory,
               const std::shared_ptr<nss::ResourceBudget>& budget)
    : device_(std::move(device)), bytes_(bytes), memory_(memory) {
    if (bytes == 0) throw std::logic_error("nss_vk: zero-sized buffer");
    const bool host = memory == Memory::Staging || memory == Memory::Readback;
    // Charge first, so an exhausted budget is reported as such and not as a
    // device allocation failure.
    charge_ = BudgetCharge(budget, host ? nss::ResourceKind::Pinned : nss::ResourceKind::Workspace, bytes);

    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    // The copy queues may be other families than compute: shared buffers
    // avoid queue-family ownership transfers between the stages of a frame.
    const std::vector<std::uint32_t>& families = device_->families();
    if (families.size() > 1) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    } else {
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    VmaAllocationCreateInfo allocation{};
    switch (memory) {
        case Memory::Device:
            allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        case Memory::Rebar:
            allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case Memory::Staging:
            allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case Memory::Readback:
            allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }
    // Plane-sized buffers get memory of their own: freed at once, never part
    // of a VMA block that outlives them uncharged.
    if (bytes >= kDedicatedBytes) allocation.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VmaAllocationInfo result{};
    NSS_VK_CHECK(vmaCreateBuffer(device_->allocator(), &info, &allocation, &buffer_, &allocation_, &result));
    mapped_ = result.pMappedData;
    // Charge what the device really reserved (alignment, dedicated rounding).
    if (result.size > bytes) {
        try {
            padding_ = BudgetCharge(budget, host ? nss::ResourceKind::Pinned : nss::ResourceKind::Workspace,
                                    static_cast<std::size_t>(result.size - bytes));
        } catch (...) {
            vmaDestroyBuffer(device_->allocator(), buffer_, allocation_);
            buffer_ = VK_NULL_HANDLE;
            allocation_ = VK_NULL_HANDLE;
            throw;
        }
    }
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = buffer_;
    address_ = device_->vk().vkGetBufferDeviceAddress(device_->handle(), &address);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = std::move(other.device_);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        address_ = std::exchange(other.address_, 0);
        bytes_ = std::exchange(other.bytes_, 0);
        memory_ = other.memory_;
        mapped_ = std::exchange(other.mapped_, nullptr);
        charge_ = std::move(other.charge_);
        padding_ = std::move(other.padding_);
    }
    return *this;
}

void Buffer::reset() noexcept {
    // A poisoned device may still use the buffer: leak it.
    if (buffer_ != VK_NULL_HANDLE && !device_->poisoned()) vmaDestroyBuffer(device_->allocator(), buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    address_ = 0;
    bytes_ = 0;
    mapped_ = nullptr;
    charge_ = BudgetCharge();
    padding_ = BudgetCharge();
    device_.reset();
}

void Buffer::flush(std::size_t offset, std::size_t bytes) const {
    NSS_VK_CHECK(vmaFlushAllocation(device_->allocator(), allocation_, offset, bytes));
}

void Buffer::invalidate(std::size_t offset, std::size_t bytes) const {
    NSS_VK_CHECK(vmaInvalidateAllocation(device_->allocator(), allocation_, offset, bytes));
}

UploadPlane::UploadPlane(std::shared_ptr<Device> device, std::size_t pitch, int rows,
                         const std::shared_ptr<nss::ResourceBudget>& budget, Mode mode)
    : device_(std::move(device)), pitch_(pitch), rows_(rows) {
    const std::size_t bytes = pitch * static_cast<std::size_t>(rows);
    if (mode == Mode::automatic && device_->rebar()) {
        try {
            device_buffer_ = Buffer(device_, bytes, Memory::Rebar, budget);
            return;
        } catch (const VkError&) {
            // BAR memory exhausted (other users of the device): fall back to staging.
        }
    }
    device_buffer_ = Buffer(device_, bytes, Memory::Device, budget);
    staging_ = Buffer(device_, bytes, Memory::Staging, budget);
}

void UploadPlane::write(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes) {
    const Buffer& target = staging_ ? staging_ : device_buffer_;
    copy_rows(host, host_stride, target.mapped(), static_cast<std::ptrdiff_t>(pitch_), row_bytes, rows_);
    target.flush(0, pitch_ * static_cast<std::size_t>(rows_));
}

void UploadPlane::record_upload(VkCommandBuffer commands) const {
    if (!staging_) return;
    const VkBufferCopy region{0, 0, pitch_ * static_cast<std::size_t>(rows_)};
    device_->vk().vkCmdCopyBuffer(commands, staging_.get(), device_buffer_.get(), 1, &region);
}

DownloadPlane::DownloadPlane(std::shared_ptr<Device> device, std::size_t pitch, int rows,
                             const std::shared_ptr<nss::ResourceBudget>& budget)
    : device_(std::move(device)), pitch_(pitch), rows_(rows) {
    const std::size_t bytes = pitch * static_cast<std::size_t>(rows);
    device_buffer_ = Buffer(device_, bytes, Memory::Device, budget);
    readback_ = Buffer(device_, bytes, Memory::Readback, budget);
}

void DownloadPlane::record_download(VkCommandBuffer commands) const {
    const VkBufferCopy region{0, 0, pitch_ * static_cast<std::size_t>(rows_)};
    device_->vk().vkCmdCopyBuffer(commands, device_buffer_.get(), readback_.get(), 1, &region);
    record_host_barrier(commands);
}

void DownloadPlane::record_download(VkCommandBuffer commands, int first, int rows, VkBuffer target,
                                    VkDeviceSize target_offset) const {
    const VkDeviceSize offset = pitch_ * static_cast<std::size_t>(first);
    const VkBufferCopy region{offset, target ? target_offset : offset, pitch_ * static_cast<std::size_t>(rows)};
    device_->vk().vkCmdCopyBuffer(commands, device_buffer_.get(), target ? target : readback_.get(), 1, &region);
}

void DownloadPlane::record_host_barrier(VkCommandBuffer commands) const {
    // The host reads after the slot's wait: the copy's writes must reach the
    // host domain (the semaphore signal alone only makes them available to
    // the device).
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    device_->vk().vkCmdPipelineBarrier2(commands, &dependency);
}

void DownloadPlane::read(void* host, std::ptrdiff_t host_stride, std::size_t row_bytes) const {
    read(host, host_stride, row_bytes, 0, rows_);
}

void DownloadPlane::read(void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int first, int rows) const {
    const std::size_t offset = pitch_ * static_cast<std::size_t>(first);
    readback_.invalidate(offset, pitch_ * static_cast<std::size_t>(rows));
    copy_rows(readback_.as<std::uint8_t>() + offset, static_cast<std::ptrdiff_t>(pitch_), host, host_stride, row_bytes,
              rows);
}

PlaneStore::PlaneStore(std::shared_ptr<Device> device, std::size_t region_bytes, int regions,
                       const std::shared_ptr<nss::ResourceBudget>& budget, UploadPlane::Mode mode)
    : device_(std::move(device)), region_bytes_(region_bytes), regions_(regions) {
    const std::size_t bytes = region_bytes * static_cast<std::size_t>(regions);
    if (mode == UploadPlane::Mode::automatic && device_->rebar()) {
        try {
            device_buffer_ = Buffer(device_, bytes, Memory::Rebar, budget);
            return;
        } catch (const VkError&) {
            // BAR memory exhausted: fall back to staging.
        }
    }
    device_buffer_ = Buffer(device_, bytes, Memory::Device, budget);
    staging_ = Buffer(device_, bytes, Memory::Staging, budget);
}

void PlaneStore::write(int region, std::size_t offset, const void* host, std::ptrdiff_t host_stride,
                       std::size_t row_bytes, int rows, std::size_t pitch) {
    const Buffer& target = staging_ ? staging_ : device_buffer_;
    const std::size_t at = region_bytes_ * static_cast<std::size_t>(region) + offset;
    copy_rows(host, host_stride, target.as<std::uint8_t>() + at, static_cast<std::ptrdiff_t>(pitch), row_bytes, rows);
    target.flush(at, pitch * static_cast<std::size_t>(rows));
}

void PlaneStore::record_upload(VkCommandBuffer commands, int region) const {
    if (!staging_) return;
    const VkDeviceSize at = region_bytes_ * static_cast<std::size_t>(region);
    const VkBufferCopy copy{at, at, region_bytes_};
    device_->vk().vkCmdCopyBuffer(commands, staging_.get(), device_buffer_.get(), 1, &copy);
}

void PlaneStore::record_upload(VkCommandBuffer commands, int region, std::size_t offset, std::size_t bytes) const {
    if (!staging_) return;
    const VkDeviceSize at = region_bytes_ * static_cast<std::size_t>(region) + offset;
    const VkBufferCopy copy{at, at, bytes};
    device_->vk().vkCmdCopyBuffer(commands, staging_.get(), device_buffer_.get(), 1, &copy);
}

void PlaneStore::record_upload(VkCommandBuffer commands, int region, std::size_t offset, std::size_t bytes,
                               VkBuffer source, VkDeviceSize source_offset) const {
    const VkDeviceSize at = region_bytes_ * static_cast<std::size_t>(region) + offset;
    const VkBufferCopy copy{source_offset, at, bytes};
    device_->vk().vkCmdCopyBuffer(commands, source, device_buffer_.get(), 1, &copy);
}

}  // namespace nss_vk
