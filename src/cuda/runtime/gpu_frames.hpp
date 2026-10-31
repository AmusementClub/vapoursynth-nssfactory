// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include "cuda/runtime/context.hpp"
#include "frontend/ownership.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>

struct VSGPUMemoryReservation;
struct VSGPUTimeline;
struct VSVULKANAPI;

namespace nss_cuda {

bool gpu_frames_registration(const VSPLUGINAPI* vspapi);
std::string gpu_frames_status(VSCore* core, const VSAPI* vsapi);
std::string residency_args(const std::string& signature, bool gpu_frames);
const char* clip_return(bool gpu_frames);

struct ClipArg {
    const char* name;
    nss::NodeRef* node;
};
const VSVULKANAPI* take_clips(const char* filter, std::initializer_list<ClipArg> clips, VSCore* core,
                              const VSAPI* vsapi);
DeviceInfo acquire_core_device(const BackendArgs& backend, const char* filter, VSCore* core, const VSAPI* vsapi,
                               const VSVULKANAPI* api);

class GpuFrames {
public:
    GpuFrames(VSCore* core, const VSVULKANAPI* api, const VSAPI* vsapi);
    ~GpuFrames();
    GpuFrames(const GpuFrames&) = delete;
    GpuFrames& operator=(const GpuFrames&) = delete;

    VSCore* core() const noexcept { return core_; }
    const VSVULKANAPI* api() const noexcept { return api_; }
    const VSAPI* vsapi() const noexcept { return vsapi_; }
    void reserve(std::int64_t bytes) noexcept;
    const VSFrame* get_frame(int n, VSNode* node, VSFrameContext* context, nss::FrameScope& frames) const;
    VSFrame* new_frame(nss::FrameScope& frames, const VSVideoFormat* format, int width, int height,
                       const VSFrame* props) const;

private:
    VSCore* core_;
    const VSVULKANAPI* api_;
    const VSAPI* vsapi_;
    VSGPUMemoryReservation* reservation_ = nullptr;
    std::mutex reserve_mutex_;
    std::int64_t reserved_ = 0;
};

class GpuStream {
public:
    GpuStream(std::shared_ptr<GpuFrames> frames, cudaStream_t stream);
    ~GpuStream();
    GpuStream(const GpuStream&) = delete;
    GpuStream& operator=(const GpuStream&) = delete;

    void begin();
    void read(const VSFrame* frame, int plane, void* device, std::size_t device_pitch, std::size_t row_bytes,
              int rows);
    void write(const void* device, std::size_t device_pitch, VSFrame* frame, int plane, std::size_t row_bytes,
               int rows);
    void copy(const VSFrame* source, int source_plane, VSFrame* target, int target_plane, std::size_t row_bytes,
              int rows);
    void publish(VSFrame* frame, const int* planes, std::size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nss_cuda
