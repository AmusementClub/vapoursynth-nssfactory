// SPDX-License-Identifier: GPL-2.0-only
// The logical device every nss_vk filter on one physical device shares, and
// the filter-creation helpers (backend argument tail, device acquisition with
// the D22/D23 support policy). The counterpart of nss_cuda's context.hpp plus
// its primary context: one VkDevice per physical device and process, created
// on first use and kept for the life of the process.
#pragma once

#include "nss/params/temporal.hpp"
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

struct VSVULKANAPI;
struct VSGPUMemoryReservation;

namespace nss_vk {

// Appended to every shared k<Filter>Signature (D21).
inline constexpr const char* kBackendSignature = "device_id:int:opt;num_streams:int:opt;";
inline constexpr int kDefaultStreams = 1;
// temporal_mode and the rolling cache when not given, as nss_cuda (D21: one
// interface golden for both GPU namespaces).
inline constexpr nss::TemporalMode kDefaultTemporalMode = nss::TemporalMode::Rolling;
inline constexpr nss::RollingCache kRollingCache = nss::RollingCache::Adaptive;
// memory_limit_mb when the argument is not given: 0, no limit (as nss_cuda).
inline constexpr int kDefaultMemoryLimitMb = 0;
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
    bool device_explicit = false;
    bool streams_explicit = false;
};
// Throws std::invalid_argument("nss_vk.<filter>: ...") on invalid values.
BackendArgs parse_backend_args(const VSAPI* vsapi, const VSMap* in, const char* filter);

// One queue and the mutex its submissions take (vkQueueSubmit2 needs external
// synchronization). Several roles may name the same queue. On VapourSynth's
// device the core's queue lock is taken instead (vs_queue: VSVulkanQueueType).
struct Queue {
    VkQueue handle = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    std::uint32_t index = 0;
    std::uint32_t timestamp_bits = 0;  // 0: no timestamps on this family
    std::mutex* mutex = nullptr;
    int vs_queue = -1;
};

class Device {
public:
    // The shared device for `index` (vkEnumeratePhysicalDevices order). Throws
    // std::runtime_error when the instance has no such device, the device is
    // unsupported (D22) and `allow_unsupported` is false, or creation fails.
    // Only tests pass allow_unsupported (lavapipe primitive tests).
    static std::shared_ptr<Device> open(int index, bool allow_unsupported = false);
    // VapourSynth's own Vulkan device (R80+ GPU frames, D29), adopted from
    // the core: GPU frames must be processed on it, and plugins may not make
    // a device of their own for them. One per core, kept while filters on it
    // live; it never destroys the VkDevice. Kernels, uploads and downloads
    // all run on the core's compute queue under its queue lock, buffers come
    // from VMA as on our own devices (memory_limit_mb) and are reported to
    // the core's VRAM budget (reserveGPUMemory), and slots signal timelines
    // from createGPUTimeline, so that their values can be published on the
    // planes they write. Throws std::runtime_error.
    static std::shared_ptr<Device> open_core(VSCore* core, const VSVULKANAPI* api);

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    // vkEnumeratePhysicalDevices index in nss_vk's instance (device_id); -1
    // for VapourSynth's device when nss_vk's instance does not list it.
    int index() const noexcept { return index_; }
    VkPhysicalDevice physical() const noexcept { return physical_; }
    VkDevice handle() const noexcept { return device_; }
    const VolkDeviceTable& vk() const noexcept { return table_; }
    VmaAllocator allocator() const noexcept { return allocator_; }
    const DeviceCaps& caps() const noexcept { return caps_; }
    const Classification& support() const noexcept { return support_; }
    std::string label() const;

    // D28 as amended 2026-10-24: kernels run on `compute`; uploads and
    // downloads take one copy-only queue each when the device has two. With
    // one, both use it and downloads are submitted once their kernel is done
    // (defer_downloads, so uploads run ahead); with none, everything is on the
    // compute queue.
    const Queue& compute() const noexcept { return compute_; }
    const Queue& upload() const noexcept { return upload_; }
    const Queue& download() const noexcept { return download_; }
    // Distinct queue families of the three roles, for concurrent buffers.
    const std::vector<std::uint32_t>& families() const noexcept { return families_; }
    bool defer_downloads() const noexcept { return defer_downloads_; }
    void submit(const Queue& queue, const VkSubmitInfo2& info) const;

    // VapourSynth's device (open_core): the core and its GPU API, else null.
    VSCore* core() const noexcept { return core_; }
    const VSVULKANAPI* core_api() const noexcept { return core_api_; }
    bool core_device() const noexcept { return core_ != nullptr; }
    // Device memory nss_vk holds on VapourSynth's device, declared to the
    // core before it is allocated (reserveGPUMemory: its cache makes room)
    // and taken back once it is freed. No-op on nss_vk's own devices.
    void reserve(std::int64_t bytes) noexcept;

    // Uploads write device-local, host-visible memory when it is resizable BAR
    // (more than the legacy 256 MiB window) and NSS_VK_REBAR is not 0.
    bool rebar() const noexcept { return rebar_; }
    // VK_EXT_external_memory_host is enabled: VapourSynth frame memory can be
    // imported (frame_memory.hpp; vapoursynth-classic only).
    bool host_import() const noexcept { return host_import_; }
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
    void create_core(VSCore* core, const VSVULKANAPI* api);
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
    bool host_import_ = false;
    bool defer_downloads_ = false;
    float timestamp_period_ = 1.0f;
    bool debug_labels_ = false;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    std::filesystem::path cache_file_;
    std::mutex cache_mutex_;
    std::uint64_t cache_saved_hash_ = 0;
    std::atomic<bool> poisoned_{false};
    VSCore* core_ = nullptr;
    const VSVULKANAPI* core_api_ = nullptr;
    VSGPUMemoryReservation* reservation_ = nullptr;
    std::mutex reserve_mutex_;
    std::int64_t reserved_ = 0;
};

// Resolves device_id for a filter: unsupported devices are a creation error,
// emulated ones log one warning per device and process. Throws
// std::invalid_argument with the "nss_vk.<filter>: " prefix.
std::shared_ptr<Device> acquire_device(int device_id, const char* filter, VSCore* core, const VSAPI* vsapi);
// The same for GPU frames: VapourSynth's device. An explicit device_id must
// name that device (as nss_vk's instance lists it); `api` is the core's GPU
// API (gpu_frames.hpp take_clips).
std::shared_ptr<Device> acquire_core_device(const BackendArgs& backend, const char* filter, VSCore* core,
                                            const VSAPI* vsapi, const VSVULKANAPI* api);

// The pipeline cache directory: NSS_VK_PIPELINE_CACHE when set ("0" or "off"
// disables the cache), else the user cache directory
// ($XDG_CACHE_HOME or ~/.cache, %LOCALAPPDATA% on Windows) + nssfactory/vk.
// Empty when disabled or no directory is known.
std::filesystem::path pipeline_cache_directory();

}  // namespace nss_vk
