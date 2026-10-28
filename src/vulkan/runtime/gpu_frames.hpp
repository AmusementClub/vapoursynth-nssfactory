// SPDX-License-Identifier: GPL-2.0-only
// VapourSynth R80+ GPU frames for the nss_vk filters (D29; docs/vulkan-plan.md
// VG1, VG2). On a core with API 4.3 every clip argument is registered
// `vnode:all`: a CPU clip runs as before (nss_vk's own device, uploads and
// downloads), a GPU-resident one on VapourSynth's device (Device::open_core)
// with its frames read and written there, and the output is as resident as
// the input. The device is the core's; the kernels, the slots and the memory
// plan are the CPU-frame path's.
//
// A plane of a GPU frame is a pitched buffer with the CPU frame's stride. It
// is copied into the buffers the kernels read, and the results into the
// output frame's planes, on the device (vkCmdCopyBuffer, or plane_copy.comp
// where the rows are padded), in the stages where the CPU-frame path uploads
// and downloads; the copies wait for each input plane's producer pair, keep
// the input frames until the work is done, and the output planes are
// published with the slot's timeline (setGPUPlaneProducer).
//
// One binary for every core (D29): GPU frames are taken only on the cores
// this build was tested with (kTestedCores). On any other core with API 4.3,
// a GPU clip is downloaded (std.GPUDownload) and the filter runs on CPU frames,
// with one warning per core; NSS_VK_GPU_FRAMES=force takes GPU frames on any
// core with API 4.3, NSS_VK_GPU_FRAMES=0 never does.
#pragma once

#include "frontend/ownership.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <VapourSynth4.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>

namespace nss_vk {

// VapourSynth releases whose VSVULKANAPI layout this build was tested with
// (the table is not versioned and changes in lockstep with the core; R81
// pins the headers, R80's table is its prefix).
inline constexpr int kTestedCores[] = {80, 81};

// The registered arguments and return type: every `vnode` becomes
// `vnode:all` when the core offers GPU frames (API 4.3), else they are
// unchanged.
std::string residency_args(const std::string& signature, bool gpu_frames);
const char* clip_return(bool gpu_frames);

// Backend()'s gpu_frames: "yes (R81)", "yes (R82, NSS_VK_GPU_FRAMES=force)",
// "downloaded (R82 untested; NSS_VK_GPU_FRAMES=force uses them)", "off
// (NSS_VK_GPU_FRAMES=0)", "no (VapourSynth API R4.2: R80 or later)" or "no
// (built without VSVulkan4.h)".
std::string gpu_frames_status(VSCore* core, const VSAPI* vsapi);

// A filter's clip arguments (null nodes are skipped).
struct ClipArg {
    const char* name;
    nss::NodeRef* node;
};
// Decides how the filter takes its clips: CPU frames (null), or GPU frames on
// a tested core (the core's GPU API). The clips must share a residency
// (std::invalid_argument otherwise). GPU clips on other cores, or with
// NSS_VK_GPU_FRAMES=0, are replaced by their std.GPUDownload.
const VSVULKANAPI* take_clips(const char* filter, std::initializer_list<ClipArg> clips, VSCore* core,
                              const VSAPI* vsapi);

// Plane I/O of GPU frames on VapourSynth's device, for one filter instance.
class GpuFrames {
public:
    GpuFrames(std::shared_ptr<Device> device, const VSAPI* vsapi);

    // A new GPU frame (newGPUVideoFrame), held by `frames` as an output.
    VSFrame* new_frame(nss::FrameScope& frames, const VSVideoFormat* format, int width, int height,
                       const VSFrame* props, VSCore* core) const;

    // `rows` rows of `row_bytes` of plane `plane` of the GPU frame f into
    // `to` at `to_offset` (tight rows), recorded into `stage` of the slot:
    // the stage waits for the plane's producer, and the slot keeps f until
    // its work is done.
    void read(Slot& slot, Slot::Stage stage, const VSFrame* f, int plane, VkBuffer to, VkDeviceAddress to_address,
              VkDeviceSize to_offset, std::size_t row_bytes, int rows) const;
    // `rows` tight rows from `from` at `from_offset` into plane `plane` of
    // the GPU frame f (one this filter made), from row `first_row` on. After
    // the slot's submit(), publish() the plane.
    void write(Slot& slot, Slot::Stage stage, VkBuffer from, VkDeviceAddress from_address, VkDeviceSize from_offset,
               VSFrame* f, int plane, std::size_t row_bytes, int rows, int first_row = 0) const;
    // Plane `plane` of the GPU frame `src` into plane `plane` of f from row
    // `first_row` on (as read() for the source, write() for the target).
    void copy(Slot& slot, Slot::Stage stage, const VSFrame* src, int plane, VSFrame* f, int first_row,
              std::size_t row_bytes, int rows) const;
    // Rows [first_row, first_row + rows) of plane `plane` of f, padding
    // included, set to the 32-bit word `word`.
    void fill(Slot& slot, Slot::Stage stage, VSFrame* f, int plane, int first_row, int rows, std::uint32_t word) const;
    // After the slot's submit(): the plane's producer is the slot's last
    // stage of this frame.
    void publish(const Slot& slot, const VSFrame* f, int plane) const;

private:
    struct Plane {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        std::size_t pitch = 0;
        VkSemaphore ready = VK_NULL_HANDLE;
        std::uint64_t ready_value = 0;
    };
    Plane plane(const VSFrame* f, int plane) const;
    // Rows between two pitched layouts: one copy region where the pitches
    // agree, else plane_copy.comp.
    void record(VkCommandBuffer commands, VkBuffer from, VkDeviceAddress from_address, VkDeviceSize from_offset,
                std::size_t from_pitch, VkBuffer to, VkDeviceAddress to_address, VkDeviceSize to_offset,
                std::size_t to_pitch, std::size_t row_bytes, int rows) const;

    std::shared_ptr<Device> device_;
    const VSAPI* vsapi_;
    Pipeline repack_;
};

}  // namespace nss_vk
