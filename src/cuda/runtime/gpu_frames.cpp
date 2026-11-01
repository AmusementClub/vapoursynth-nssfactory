// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/gpu_frames.hpp"

#include "cuda/runtime/error.hpp"
#include "cuda/runtime/gpu_fill.hpp"
#include "cuda/runtime/vs_api.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#if NSS_CUDA_GPU_FRAMES
#include <VSVulkan4.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#endif

namespace nss_cuda {
namespace {

std::string prefix(const char* filter) { return std::string("nss_cuda.") + filter + ": "; }

bool requested() {
    const char* value = std::getenv("NSS_CUDA_GPU_FRAMES");
    return value && (!std::strcmp(value, "1") || !std::strcmp(value, "on") || !std::strcmp(value, "force"));
}

int core_release(VSCore* core, const VSAPI* vsapi) {
    VSCoreInfo info{};
    vsapi->getCoreInfo(core, &info);
    return info.core;
}

#if NSS_CUDA_GPU_FRAMES
constexpr int kTestedCore = 81;
constexpr std::size_t kMaxMemoryImports = 16;
constexpr std::size_t kMaxSemaphoreImports = 16;

nss::NodeRef download(const nss::NodeRef& node, const char* filter, VSCore* core, const VSAPI* vsapi) {
    VSPlugin* std_plugin = vsapi->getPluginByID("com.vapoursynth.std", core);
    VSMap* args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", node, maReplace);
    VSMap* result = std_plugin ? vsapi->invoke(std_plugin, "GPUDownload", args) : nullptr;
    vsapi->freeMap(args);
    if (!result) throw std::runtime_error(prefix(filter) + "std.GPUDownload is not available");
    if (const char* error = vsapi->mapGetError(result)) {
        const std::string message = prefix(filter) + "std.GPUDownload failed: " + error;
        vsapi->freeMap(result);
        throw std::runtime_error(message);
    }
    nss::NodeRef downloaded(vsapi->mapGetNode(result, "clip", 0, nullptr), vsapi);
    vsapi->freeMap(result);
    return downloaded;
}

void close_handle(std::intptr_t handle) {
#ifdef _WIN32
    if (handle) CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
    if (handle >= 0) close(static_cast<int>(handle));
#endif
}

void check_memory_handle(int actual) {
#ifdef _WIN32
    constexpr int expected = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr int expected = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    if (actual != expected) throw std::runtime_error("nss_cuda: unsupported VapourSynth external-memory handle type");
}

bool semaphore_handle_supported(int actual) {
#ifdef _WIN32
    constexpr int expected = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    constexpr int expected = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    return actual == expected;
}
#endif

}  // namespace

bool gpu_frames_registration(const VSPLUGINAPI* vspapi) {
#if NSS_CUDA_GPU_FRAMES
    return requested() && vs_api::gpu_frames(vspapi);
#else
    (void)vspapi;
    return false;
#endif
}

std::string gpu_frames_status(VSCore* core, const VSAPI* vsapi) {
#if !NSS_CUDA_GPU_FRAMES
    (void)core, (void)vsapi;
    return "no (built without CUDA GPU frames)";
#else
    if (!vs_api::gpu_frames(vsapi)) return "no (VapourSynth API before 4.3)";
    if (!requested()) return "off (NSS_CUDA_GPU_FRAMES=1 enables experimental GPU frames)";
    const int release = core_release(core, vsapi);
    if (release == kTestedCore) return "yes (R81, experimental, all filters)";
    return "downloaded (R" + std::to_string(release) + " untested; experimental GPU frames support R81)";
#endif
}

std::string residency_args(const std::string& signature, bool gpu_frames) {
    if (!gpu_frames) return signature;
    std::string result;
    std::size_t at = 0;
    while (true) {
        const std::size_t found = signature.find(":vnode", at);
        if (found == std::string::npos) break;
        std::size_t end = found + 6;
        if (signature.compare(end, 2, "[]") == 0) end += 2;
        result.append(signature, at, end - at);
        result += ":all";
        at = end;
    }
    result.append(signature, at, std::string::npos);
    return result;
}

const char* clip_return(bool gpu_frames) { return gpu_frames ? "clip:vnode:all;" : "clip:vnode;"; }

const VSVULKANAPI* take_clips(const char* filter, std::initializer_list<ClipArg> clips, VSCore* core,
                              const VSAPI* vsapi) {
#if !NSS_CUDA_GPU_FRAMES
    (void)filter, (void)clips, (void)core, (void)vsapi;
    return nullptr;
#else
    const ClipArg* first = nullptr;
    const ClipArg* other = nullptr;
    bool gpu = false;
    for (const ClipArg& clip : clips) {
        if (!clip.node || !*clip.node) continue;
        const bool resident = vs_api::gpu_node(vsapi, *clip.node);
        if (!first) {
            first = &clip;
            gpu = resident;
        } else if (resident != gpu && !other) {
            other = &clip;
        }
    }
    if (other) {
        throw std::invalid_argument(prefix(filter) + first->name + " and " + other->name +
                                    " must both be CPU frames or both GPU frames "
                                    "(pass one through std.GPUUpload or std.GPUDownload)");
    }
    if (!gpu) return nullptr;
    const int release = core_release(core, vsapi);
    if (requested() && release == kTestedCore) return vs_api::vulkan(vsapi);
    {
        static std::mutex mutex;
        static std::set<VSCore*> warned;
        std::lock_guard lock(mutex);
        if (warned.insert(core).second) {
            const std::string message = !requested()
                ? "nss_cuda: NSS_CUDA_GPU_FRAMES is off: GPU clips are taken as CPU frames (std.GPUDownload)"
                : "nss_cuda: VapourSynth R" + std::to_string(release) +
                  " is not the tested R81 core: its GPU clips are taken as CPU frames (std.GPUDownload)";
            vsapi->logMessage(mtWarning, message.c_str(), core);
        }
    }
    for (const ClipArg& clip : clips) {
        if (clip.node && *clip.node) *clip.node = download(*clip.node, filter, core, vsapi);
    }
    return nullptr;
#endif
}

DeviceInfo acquire_core_device(const BackendArgs& backend, const char* filter, VSCore* core, const VSAPI* vsapi,
                               const VSVULKANAPI* api) {
#if !NSS_CUDA_GPU_FRAMES
    (void)backend, (void)filter, (void)core, (void)vsapi, (void)api;
    throw std::logic_error("nss_cuda: GPU frames were not built");
#else
    VSVulkanCoreInfo info{};
    char error[512] = {};
    if (api->getVulkanCoreInfo(core, &info, error, sizeof(error)) != 0) {
        throw std::invalid_argument(prefix(filter) + "VapourSynth's Vulkan device is not available: " + error);
    }
    if (!info.exportHandleType) {
        throw std::invalid_argument(prefix(filter) + "VapourSynth's device cannot export GPU planes to CUDA");
    }
    const RuntimeInfo runtime = runtime_info();
    int matched = -1;
    for (int device = 0; device < runtime.device_count; ++device) {
        cudaDeviceProp prop{};
        NSS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (!std::memcmp(prop.uuid.bytes, info.deviceUUID, VK_UUID_SIZE)) {
            matched = device;
            break;
        }
    }
    if (matched < 0) {
        throw std::invalid_argument(prefix(filter) +
                                    "VapourSynth's Vulkan device has no matching CUDA device (deviceUUID)");
    }
    if (backend.device_explicit && backend.device_id != matched) {
        throw std::invalid_argument(prefix(filter) + "GPU frames are processed on VapourSynth's device, CUDA device " +
                                    std::to_string(matched) + ", which is not device_id " +
                                    std::to_string(backend.device_id) + ": omit device_id or select that GPU for the core");
    }
    return acquire_device(matched, filter, core, vsapi);
#endif
}

GpuFrames::GpuFrames(VSCore* core, const VSVULKANAPI* api, const VSAPI* vsapi)
    : core_(core), api_(api), vsapi_(vsapi) {
#if NSS_CUDA_GPU_FRAMES
    char error[256] = {};
    reservation_ = api_->reserveGPUMemory(core_, 0, error, sizeof(error));
#endif
}

GpuFrames::~GpuFrames() {
#if NSS_CUDA_GPU_FRAMES
    if (reservation_) api_->releaseGPUMemoryReservation(reservation_);
#endif
}

void GpuFrames::reserve(std::int64_t bytes) noexcept {
#if NSS_CUDA_GPU_FRAMES
    if (!reservation_) return;
    std::lock_guard lock(reserve_mutex_);
    reserved_ = std::max<std::int64_t>(0, bytes);
    api_->updateGPUMemoryReservation(reservation_, reserved_);
#else
    (void)bytes;
#endif
}

const VSFrame* GpuFrames::get_frame(int n, VSNode* node, VSFrameContext* context, nss::FrameScope& frames) const {
#if NSS_CUDA_GPU_FRAMES
    char error[512] = {};
    const VSFrame* frame = api_->getExportableFrameFilter(n, node, context, error, sizeof(error));
    if (!frame) throw std::runtime_error(std::string("nss_cuda: cannot take an exportable GPU frame: ") + error);
    return frames.adoptInput(frame);
#else
    (void)n, (void)node, (void)context, (void)frames;
    throw std::logic_error("nss_cuda: GPU frames were not built");
#endif
}

GpuFrameCache::~GpuFrameCache() {
    for (const auto& [key, frame] : frames_) api_->freeFrame(frame);
}

const VSFrame* GpuFrameCache::get(const GpuFrames& gpu, int n, VSNode* node, VSFrameContext* context,
                                  nss::FrameScope& frames) {
    const auto key = std::make_pair(node, n);
    if (const auto found = frames_.find(key); found != frames_.end()) {
        return frames.adoptInput(api_->addFrameRef(found->second));
    }
    const VSFrame* frame = gpu.get_frame(n, node, context, frames);
    const VSFrame* cached = api_->addFrameRef(frame);
    try {
        frames_.emplace(key, cached);
    } catch (...) {
        api_->freeFrame(cached);
        throw;
    }
    return frame;
}

VSFrame* GpuFrames::new_frame(nss::FrameScope& frames, const VSVideoFormat* format, int width, int height,
                              const VSFrame* props) const {
#if NSS_CUDA_GPU_FRAMES
    return frames.adoptOutput(api_->newGPUVideoFrame(format, width, height, props, core_));
#else
    (void)frames, (void)format, (void)width, (void)height, (void)props;
    throw std::logic_error("nss_cuda: GPU frames were not built");
#endif
}

#if NSS_CUDA_GPU_FRAMES
struct GpuStream::Impl {
    struct MemoryImport {
        std::uint64_t id = 0;
        cudaExternalMemory_t memory = nullptr;
        void* base = nullptr;
        std::uint64_t used = 0;
    };
    struct SemaphoreImport {
        VkSemaphore source = VK_NULL_HANDLE;
        cudaExternalSemaphore_t semaphore = nullptr;
    };

    Impl(std::shared_ptr<GpuFrames> frames_, cudaStream_t stream_) : frames(std::move(frames_)), stream(stream_) {
        imports.reserve(kMaxMemoryImports);
        semaphores.reserve(kMaxSemaphoreImports);
        char error[256] = {};
        timeline = frames->api()->createGPUTimeline(frames->core(), error, sizeof(error));
        if (!timeline) throw std::runtime_error(std::string("nss_cuda: createGPUTimeline failed: ") + error);
        try {
            timeline_sem = frames->api()->getGPUTimelineSemaphore(timeline);
            cuda_timeline = import_semaphore(timeline_sem);
        } catch (...) {
            frames->api()->freeGPUTimeline(timeline);
            timeline = nullptr;
            throw;
        }
    }
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        for (MemoryImport& import : imports) {
            if (import.base) cudaFree(import.base);
            if (import.memory) cudaDestroyExternalMemory(import.memory);
        }
        for (const SemaphoreImport& import : semaphores) {
            if (import.semaphore) cudaDestroyExternalSemaphore(import.semaphore);
        }
        if (cuda_timeline) cudaDestroyExternalSemaphore(cuda_timeline);
        if (timeline) frames->api()->freeGPUTimeline(timeline);
    }

    cudaExternalSemaphore_t import_semaphore(VkSemaphore source) {
        VSVulkanExportedSemaphore exported{};
        char error[256] = {};
        if (frames->api()->exportGPUSemaphore(frames->core(), source, &exported, error, sizeof(error)) != 0) return nullptr;
        if (!semaphore_handle_supported(exported.handleType)) {
            close_handle(exported.handle);
            return nullptr;
        }
        cudaExternalSemaphoreHandleDesc desc{};
#ifdef _WIN32
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        desc.handle.win32.handle = reinterpret_cast<void*>(exported.handle);
#else
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        desc.handle.fd = static_cast<int>(exported.handle);
#endif
        cudaExternalSemaphore_t semaphore = nullptr;
        const cudaError_t result = cudaImportExternalSemaphore(&semaphore, &desc);
#ifndef _WIN32
        if (result == cudaSuccess) exported.handle = -1;
#endif
        close_handle(exported.handle);
        if (result != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        return semaphore;
    }

    cudaExternalSemaphore_t map_semaphore(VkSemaphore source) {
        for (const SemaphoreImport& import : semaphores) {
            if (import.source == source) return import.semaphore;
        }
        if (semaphores.size() >= kMaxSemaphoreImports) return nullptr;
        cudaExternalSemaphore_t semaphore = import_semaphore(source);
        if (semaphore) semaphores.push_back({source, semaphore});
        return semaphore;
    }

    void drop_import(std::size_t index) noexcept {
        if (imports[index].base) cudaFree(imports[index].base);
        if (imports[index].memory) cudaDestroyExternalMemory(imports[index].memory);
        imports[index] = std::move(imports.back());
        imports.pop_back();
    }

    void* map_plane(const VSFrame* frame, int plane) {
        VSVulkanExportedMemory exported{};
        char error[512] = {};
        if (frames->api()->exportGPUPlane(frame, plane, &exported, error, sizeof(error)) != 0) {
            throw std::runtime_error(std::string("nss_cuda: exportGPUPlane failed: ") + error);
        }
        try {
            check_memory_handle(exported.handleType);
        } catch (...) {
            close_handle(exported.handle);
            throw;
        }
        const std::uint64_t calls = ++map_calls;
        for (MemoryImport& import : imports) {
            if (import.id == exported.memoryId) {
                import.used = calls;
                close_handle(exported.handle);
                return static_cast<std::uint8_t*>(import.base) + exported.offset;
            }
        }
        if (imports.size() >= kMaxMemoryImports) {
            const auto victim = std::min_element(imports.begin(), imports.end(), [](const MemoryImport& a, const MemoryImport& b) {
                return a.used < b.used;
            });
            NSS_CUDA_CHECK(cudaStreamSynchronize(stream));
            drop_import(static_cast<std::size_t>(victim - imports.begin()));
        }
        cudaExternalMemoryHandleDesc desc{};
#ifdef _WIN32
        desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = reinterpret_cast<void*>(exported.handle);
#else
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = static_cast<int>(exported.handle);
#endif
        desc.size = exported.memorySize;
        cudaExternalMemory_t memory = nullptr;
        cudaError_t result = cudaImportExternalMemory(&memory, &desc);
#ifndef _WIN32
        if (result == cudaSuccess) exported.handle = -1;
#endif
        if (result != cudaSuccess) {
            close_handle(exported.handle);
            NSS_CUDA_CHECK(result);
        }
        cudaExternalMemoryBufferDesc buffer{};
        buffer.size = exported.memorySize;
        void* base = nullptr;
        result = cudaExternalMemoryGetMappedBuffer(&base, memory, &buffer);
        if (result != cudaSuccess) {
            cudaDestroyExternalMemory(memory);
            close_handle(exported.handle);
            NSS_CUDA_CHECK(result);
        }
        close_handle(exported.handle);
        imports.push_back({exported.memoryId, memory, base, calls});
        return static_cast<std::uint8_t*>(base) + exported.offset;
    }

    void wait_input(const VSFrame* frame, int plane) {
        VSVulkanPlaneInfo info{};
        if (frames->api()->getGPUPlane(frame, plane, &info) != 0) {
            throw std::runtime_error("nss_cuda: input plane is not GPU resident");
        }
        if (!host_sync && info.readySemaphore) {
            cudaExternalSemaphore_t semaphore = map_semaphore(info.readySemaphore);
            if (semaphore) {
                cudaExternalSemaphoreWaitParams params{};
                params.params.fence.value = info.readyValue;
                NSS_CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &params, 1, stream));
            } else {
                host_sync = true;
            }
        }
        if (host_sync && std::find(host_waited.begin(), host_waited.end(), frame) == host_waited.end()) {
            char error[512] = {};
            if (frames->api()->waitGPUFrame(frame, error, sizeof(error)) != 0) {
                throw std::runtime_error(std::string("nss_cuda: waitGPUFrame failed: ") + error);
            }
            host_waited.push_back(frame);
        }
    }

    std::shared_ptr<GpuFrames> frames;
    cudaStream_t stream = nullptr;
    VSGPUTimeline* timeline = nullptr;
    VkSemaphore timeline_sem = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cuda_timeline = nullptr;
    std::uint64_t next_value = 0;
    std::vector<MemoryImport> imports;
    std::vector<SemaphoreImport> semaphores;
    std::uint64_t map_calls = 0;
    bool host_sync = true;
    std::vector<const VSFrame*> host_waited;
};
#else
struct GpuStream::Impl {};
#endif

GpuStream::GpuStream(std::shared_ptr<GpuFrames> frames, cudaStream_t stream) {
#if NSS_CUDA_GPU_FRAMES
    impl_ = std::make_unique<Impl>(std::move(frames), stream);
#else
    (void)frames, (void)stream;
    throw std::logic_error("nss_cuda: GPU frames were not built");
#endif
}

GpuStream::~GpuStream() = default;

void GpuStream::begin() {
#if NSS_CUDA_GPU_FRAMES
    impl_->host_sync = impl_->cuda_timeline == nullptr;
    impl_->host_waited.clear();
#endif
}

void GpuStream::read(const VSFrame* frame, int plane, void* device, std::size_t device_pitch, std::size_t row_bytes,
                     int rows) {
#if NSS_CUDA_GPU_FRAMES
    impl_->wait_input(frame, plane);
    const void* source = impl_->map_plane(frame, plane);
    const std::ptrdiff_t pitch = impl_->frames->vsapi()->getStride(frame, plane);
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(device, device_pitch, source, static_cast<std::size_t>(pitch), row_bytes,
                                     static_cast<std::size_t>(rows), cudaMemcpyDeviceToDevice, impl_->stream));
#else
    (void)frame, (void)plane, (void)device, (void)device_pitch, (void)row_bytes, (void)rows;
#endif
}

void GpuStream::write(const void* device, std::size_t device_pitch, VSFrame* frame, int plane, std::size_t row_bytes,
                      int rows) {
    write_rows(device, device_pitch, frame, plane, row_bytes, rows, 0);
}

void GpuStream::write_rows(const void* device, std::size_t device_pitch, VSFrame* frame, int plane,
                           std::size_t row_bytes, int rows, int first_row) {
#if NSS_CUDA_GPU_FRAMES
    auto* target = static_cast<std::uint8_t*>(impl_->map_plane(frame, plane));
    const std::ptrdiff_t pitch = impl_->frames->vsapi()->getStride(frame, plane);
    target += pitch * static_cast<std::ptrdiff_t>(first_row);
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(target, static_cast<std::size_t>(pitch), device, device_pitch, row_bytes,
                                     static_cast<std::size_t>(rows), cudaMemcpyDeviceToDevice, impl_->stream));
#else
    (void)device, (void)device_pitch, (void)frame, (void)plane, (void)row_bytes, (void)rows, (void)first_row;
#endif
}

void GpuStream::copy(const VSFrame* source, int source_plane, VSFrame* target, int target_plane, std::size_t row_bytes,
                     int rows) {
    copy_rows(source, source_plane, 0, target, target_plane, 0, row_bytes, rows);
}

void GpuStream::copy_rows(const VSFrame* source, int source_plane, int source_first_row, VSFrame* target,
                          int target_plane, int target_first_row, std::size_t row_bytes, int rows) {
#if NSS_CUDA_GPU_FRAMES
    impl_->wait_input(source, source_plane);
    const auto* from = static_cast<const std::uint8_t*>(impl_->map_plane(source, source_plane));
    auto* to = static_cast<std::uint8_t*>(impl_->map_plane(target, target_plane));
    const std::ptrdiff_t source_pitch = impl_->frames->vsapi()->getStride(source, source_plane);
    const std::ptrdiff_t target_pitch = impl_->frames->vsapi()->getStride(target, target_plane);
    from += source_pitch * static_cast<std::ptrdiff_t>(source_first_row);
    to += target_pitch * static_cast<std::ptrdiff_t>(target_first_row);
    NSS_CUDA_CHECK(cudaMemcpy2DAsync(to, static_cast<std::size_t>(target_pitch), from,
                                     static_cast<std::size_t>(source_pitch), row_bytes, static_cast<std::size_t>(rows),
                                     cudaMemcpyDeviceToDevice, impl_->stream));
#else
    (void)source, (void)source_plane, (void)source_first_row, (void)target, (void)target_plane, (void)target_first_row,
    (void)row_bytes, (void)rows;
#endif
}

void GpuStream::fill(VSFrame* frame, int plane, int first_row, int rows, std::uint32_t word) {
#if NSS_CUDA_GPU_FRAMES
    auto* target = static_cast<std::uint8_t*>(impl_->map_plane(frame, plane));
    const std::ptrdiff_t pitch = impl_->frames->vsapi()->getStride(frame, plane);
    target += pitch * static_cast<std::ptrdiff_t>(first_row);
    gpu_fill(target, static_cast<std::size_t>(pitch), rows, word, impl_->stream);
#else
    (void)frame, (void)plane, (void)first_row, (void)rows, (void)word;
#endif
}

void GpuStream::publish(VSFrame* frame, const int* planes, std::size_t count) {
#if NSS_CUDA_GPU_FRAMES
    if (impl_->host_sync || !count) return;
    const std::uint64_t value = ++impl_->next_value;
    cudaExternalSemaphoreSignalParams params{};
    params.params.fence.value = value;
    NSS_CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&impl_->cuda_timeline, &params, 1, impl_->stream));
    for (std::size_t i = 0; i < count; ++i) {
        impl_->frames->api()->setGPUPlaneProducer(frame, planes[i], impl_->timeline, value);
    }
#else
    (void)frame, (void)planes, (void)count;
#endif
}

}  // namespace nss_cuda
