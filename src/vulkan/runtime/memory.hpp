// SPDX-License-Identifier: GPL-2.0-only
// Owning VMA buffers charged to the filter's resource budget (memory_limit_mb,
// D25), and the plane transfer buffers built on them (D28 as amended
// 2026-10-24: uploads into resizable BAR when the device has it, else through
// host staging; downloads through host-cached staging).
#pragma once

#include "nss/resources.hpp"
#include "vulkan/runtime/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace nss_vk {

// Charges `bytes` to `budget` under `kind` for the lifetime of the object
// (as nss_cuda's BudgetCharge).
class BudgetCharge {
public:
    BudgetCharge() = default;
    BudgetCharge(const std::shared_ptr<nss::ResourceBudget>& budget, nss::ResourceKind kind, std::size_t bytes)
        : account_(budget ? std::make_shared<nss::ResourceAccount>(budget, kind) : nullptr), bytes_(bytes) {
        if (account_) account_->acquire(bytes_);
    }
    BudgetCharge(const BudgetCharge&) = delete;
    BudgetCharge& operator=(const BudgetCharge&) = delete;
    BudgetCharge(BudgetCharge&& other) noexcept
        : account_(std::move(other.account_)), bytes_(std::exchange(other.bytes_, 0)) {}
    BudgetCharge& operator=(BudgetCharge&& other) noexcept {
        if (this != &other) {
            release();
            account_ = std::move(other.account_);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }
    ~BudgetCharge() { release(); }

private:
    void release() noexcept {
        if (account_) account_->release(bytes_);
        account_.reset();
        bytes_ = 0;
    }
    std::shared_ptr<nss::ResourceAccount> account_;
    std::size_t bytes_ = 0;
};

enum class Memory {
    Device,    // device-local; kernels only
    Rebar,     // device-local and host-visible (resizable BAR): the host writes, kernels read
    Staging,   // host memory the host writes and the copy queues read
    Readback,  // host-cached memory the copy queues write and the host reads
};

// Buffers this size or larger get a dedicated allocation.
inline constexpr std::size_t kDedicatedBytes = 4u << 20;

// One VkBuffer with its own VMA allocation, addressable from shaders (buffer
// device address) and usable by every queue family of the device. Host
// memory kinds are persistently mapped. Device and Rebar memory is charged as
// Workspace, Staging and Readback as Pinned.
class Buffer {
public:
    Buffer() = default;
    // Throws std::runtime_error("nss: memory_limit_mb exceeded") when the
    // budget is exhausted, VkError when the allocation fails.
    Buffer(std::shared_ptr<Device> device, std::size_t bytes, Memory memory,
           const std::shared_ptr<nss::ResourceBudget>& budget = nullptr);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept { *this = std::move(other); }
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer() { reset(); }

    void reset() noexcept;
    explicit operator bool() const noexcept { return buffer_ != VK_NULL_HANDLE; }
    VkBuffer get() const noexcept { return buffer_; }
    VkDeviceAddress address() const noexcept { return address_; }
    std::size_t bytes() const noexcept { return bytes_; }
    Memory memory() const noexcept { return memory_; }
    void* mapped() const noexcept { return mapped_; }
    template <class T> T* as() const noexcept { return static_cast<T*>(mapped_); }
    // Host writes reach the device / device writes reach the host; no-ops on
    // coherent memory.
    void flush(std::size_t offset = 0, std::size_t bytes = VK_WHOLE_SIZE) const;
    void invalidate(std::size_t offset = 0, std::size_t bytes = VK_WHOLE_SIZE) const;

private:
    std::shared_ptr<Device> device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    std::size_t bytes_ = 0;
    Memory memory_ = Memory::Device;
    void* mapped_ = nullptr;
    BudgetCharge charge_;
    BudgetCharge padding_;  // allocation size beyond `bytes`
};

// A pitched plane (`pitch` bytes per row) the host fills and kernels read.
// With resizable BAR the host writes the device buffer itself; otherwise it
// writes a staging copy that record_upload() copies on the upload queue.
class UploadPlane {
public:
    enum class Mode { automatic, staging };  // staging: never use BAR (tests)

    UploadPlane() = default;
    UploadPlane(std::shared_ptr<Device> device, std::size_t pitch, int rows,
                const std::shared_ptr<nss::ResourceBudget>& budget = nullptr, Mode mode = Mode::automatic);

    // Host rows (any stride) into the plane; flushes when needed. Must happen
    // before the Slot::submit() of the frame that reads it (host writes are
    // made visible by the submission).
    void write(const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes);
    // Copy staging -> device; nothing to record with BAR.
    bool needs_copy() const noexcept { return static_cast<bool>(staging_); }
    void record_upload(VkCommandBuffer commands) const;
    VkDeviceAddress address() const noexcept { return device_buffer_.address(); }
    VkBuffer buffer() const noexcept { return device_buffer_.get(); }
    std::size_t pitch() const noexcept { return pitch_; }
    int rows() const noexcept { return rows_; }

private:
    std::shared_ptr<Device> device_;
    Buffer device_buffer_;
    Buffer staging_;
    std::size_t pitch_ = 0;
    int rows_ = 0;
};

// A pitched plane kernels write and the host reads back: record_download()
// copies it into host-cached memory on the download queue; after the slot's
// wait, read() copies the rows out.
class DownloadPlane {
public:
    DownloadPlane() = default;
    DownloadPlane(std::shared_ptr<Device> device, std::size_t pitch, int rows,
                  const std::shared_ptr<nss::ResourceBudget>& budget = nullptr);

    void record_download(VkCommandBuffer commands) const;
    // Rows [first, first + rows) into `target` at `target_offset` (a frame
    // plane imported by FrameMemory), or into the readback buffer when
    // `target` is null; then record_host_barrier() once for the stage.
    void record_download(VkCommandBuffer commands, int first, int rows, VkBuffer target,
                         VkDeviceSize target_offset) const;
    void record_host_barrier(VkCommandBuffer commands) const;
    void read(void* host, std::ptrdiff_t host_stride, std::size_t row_bytes) const;
    // Rows [first, first + rows) only (several planes stacked in one buffer).
    void read(void* host, std::ptrdiff_t host_stride, std::size_t row_bytes, int first, int rows) const;
    VkDeviceAddress address() const noexcept { return device_buffer_.address(); }
    VkBuffer buffer() const noexcept { return device_buffer_.get(); }
    std::size_t pitch() const noexcept { return pitch_; }
    int rows() const noexcept { return rows_; }

private:
    std::shared_ptr<Device> device_;
    Buffer device_buffer_;
    Buffer readback_;
    std::size_t pitch_ = 0;
    int rows_ = 0;
};

// `regions` equal regions of device memory kernels read (frames of a cache):
// the host writes a region into BAR memory directly when the device has it,
// else into a staging mirror that record_upload() copies on the upload queue.
// A region may be rewritten only once no submitted work reads it.
class PlaneStore {
public:
    PlaneStore() = default;
    PlaneStore(std::shared_ptr<Device> device, std::size_t region_bytes, int regions,
               const std::shared_ptr<nss::ResourceBudget>& budget = nullptr,
               UploadPlane::Mode mode = UploadPlane::Mode::automatic);

    // Host rows (any stride) into region `region` at byte `offset`, `pitch`
    // bytes per row in the store; flushes when needed. Before Slot::submit().
    void write(int region, std::size_t offset, const void* host, std::ptrdiff_t host_stride, std::size_t row_bytes,
               int rows, std::size_t pitch);
    bool needs_copy() const noexcept { return static_cast<bool>(staging_); }
    void record_upload(VkCommandBuffer commands, int region) const;
    // `bytes` at `offset` of region `region` only: from the staging mirror
    // (needs_copy()), or from `source` at `source_offset` (a frame plane
    // imported by FrameMemory, which the host did not write; any store).
    void record_upload(VkCommandBuffer commands, int region, std::size_t offset, std::size_t bytes) const;
    void record_upload(VkCommandBuffer commands, int region, std::size_t offset, std::size_t bytes, VkBuffer source,
                       VkDeviceSize source_offset) const;
    VkDeviceAddress address() const noexcept { return device_buffer_.address(); }
    VkBuffer buffer() const noexcept { return device_buffer_.get(); }
    std::size_t region_bytes() const noexcept { return region_bytes_; }
    int regions() const noexcept { return regions_; }

private:
    std::shared_ptr<Device> device_;
    Buffer device_buffer_;
    Buffer staging_;
    std::size_t region_bytes_ = 0;
    int regions_ = 0;
};

// Rows of `row_bytes` between two strided layouts (one memcpy when both are tight).
void copy_rows(const void* src, std::ptrdiff_t src_stride, void* dst, std::ptrdiff_t dst_stride, std::size_t row_bytes,
               int rows);

}  // namespace nss_vk
