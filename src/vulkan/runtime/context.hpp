// SPDX-License-Identifier: GPL-2.0-only
// The logical device every nss_vk filter on one physical device shares, and
// the filter-creation helpers (backend argument tail, device acquisition with
// the D22/D23 support policy). The counterpart of nss_cuda's context.hpp plus
// its primary context: one VkDevice per physical device and process, created
// on first use and kept for the life of the process.
#pragma once

#include "vulkan/runtime/device.hpp"
#include "vulkan/runtime/vk.hpp"

#include <VapourSynth4.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nss_vk {

// Appended to every shared k<Filter>Signature (D21).
inline constexpr const char* kBackendSignature = "device_id:int:opt;num_streams:int:opt;";
inline constexpr int kDefaultStreams = 1;
inline constexpr int kMaxStreams = 16;
// Host waits on the device are bounded (the GPU may drive a desktop). A frame
// that has not finished after this long is an error; NSS_VK_WAIT_TIMEOUT_MS
// overrides it.
inline constexpr std::uint64_t kDefaultWaitTimeoutMs = 60'000;
std::uint64_t wait_timeout_ns();

std::string signature(const char* shared);

struct BackendArgs {
    int device_id = 0;
    int num_streams = kDefaultStreams;
    bool streams_explicit = false;
};
// Throws std::invalid_argument("nss_vk.<filter>: ...") on invalid values.
BackendArgs parse_backend_args(const VSAPI* vsapi, const VSMap* in, const char* filter);

// One queue and the mutex its submissions take (vkQueueSubmit2 needs external
// synchronization). Several roles may name the same queue.
struct Queue {
    VkQueue handle = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    std::uint32_t index = 0;
    std::uint32_t timestamp_bits = 0;  // 0: no timestamps on this family
    std::mutex* mutex = nullptr;
};

class Device {
public:
    // The shared device for `index` (vkEnumeratePhysicalDevices order). Throws
    // std::runtime_error when the instance has no such device, the device is
    // unsupported (D22) and `allow_unsupported` is false, or creation fails.
    // Only tests pass allow_unsupported (lavapipe primitive tests).
    static std::shared_ptr<Device> open(int index, bool allow_unsupported = false);

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    int index() const noexcept { return index_; }
    VkPhysicalDevice physical() const noexcept { return physical_; }
    VkDevice handle() const noexcept { return device_; }
    const VolkDeviceTable& vk() const noexcept { return table_; }
    VmaAllocator allocator() const noexcept { return allocator_; }
    const DeviceCaps& caps() const noexcept { return caps_; }
    const Classification& support() const noexcept { return support_; }
    std::string label() const;

    // D28 as amended 2026-10-24: kernels run on `compute`; uploads and
    // downloads take one copy-only queue each when the device has two, share
    // one when it has one, and use the compute queue otherwise.
    const Queue& compute() const noexcept { return compute_; }
    const Queue& upload() const noexcept { return upload_; }
    const Queue& download() const noexcept { return download_; }
    // Distinct queue families of the three roles, for concurrent buffers.
    const std::vector<std::uint32_t>& families() const noexcept { return families_; }
    void submit(const Queue& queue, const VkSubmitInfo2& info) const;

    // Uploads write device-local, host-visible memory when it is resizable BAR
    // (more than the legacy 256 MiB window) and NSS_VK_REBAR is not 0.
    bool rebar() const noexcept { return rebar_; }
    float timestamp_period_ns() const noexcept { return timestamp_period_; }
    bool debug_labels() const noexcept { return debug_labels_; }

    VkPipelineCache pipeline_cache() const noexcept { return pipeline_cache_; }
    // Writes the pipeline cache to disk when it changed since the last write
    // (no-op when the cache is disabled). Errors are not fatal. Filters call
    // it once after creating their pipelines.
    void save_pipeline_cache();

    // After a failed wait the device may still run work on objects their
    // owners are about to free: from then on Buffers, Pipelines, Slots and the
    // device itself leak instead of destroying anything.
    void poison() noexcept { poisoned_.store(true, std::memory_order_relaxed); }
    bool poisoned() const noexcept { return poisoned_.load(std::memory_order_relaxed); }
    const std::filesystem::path& pipeline_cache_file() const noexcept { return cache_file_; }

private:
    Device() = default;
    void create(int index);
    void load_pipeline_cache();

    int index_ = 0;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VolkDeviceTable table_{};
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    DeviceCaps caps_;
    Classification support_;
    Queue compute_, upload_, download_;
    std::vector<std::uint32_t> families_;
    std::unique_ptr<std::mutex[]> queue_mutexes_;
    bool rebar_ = false;
    float timestamp_period_ = 1.0f;
    bool debug_labels_ = false;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    std::filesystem::path cache_file_;
    std::mutex cache_mutex_;
    std::uint64_t cache_saved_hash_ = 0;
    std::atomic<bool> poisoned_{false};
};

// Resolves device_id for a filter: unsupported devices are a creation error,
// emulated ones log one warning per device and process. Throws
// std::invalid_argument with the "nss_vk.<filter>: " prefix.
std::shared_ptr<Device> acquire_device(int device_id, const char* filter, VSCore* core, const VSAPI* vsapi);

// The pipeline cache directory: NSS_VK_PIPELINE_CACHE when set ("0" or "off"
// disables the cache), else the user cache directory
// ($XDG_CACHE_HOME or ~/.cache, %LOCALAPPDATA% on Windows) + nssfactory/vk.
// Empty when disabled or no directory is known.
std::filesystem::path pipeline_cache_directory();

}  // namespace nss_vk
