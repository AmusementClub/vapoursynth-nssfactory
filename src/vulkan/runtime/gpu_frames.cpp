// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/gpu_frames.hpp"

#include "vulkan/runtime/vs_api.hpp"

#include <stdexcept>
#include <utility>
#if NSS_VK_GPU_FRAMES
#include <VSVulkan4.h>
#endif

namespace nss_vk {
namespace {

std::string prefix(const char* filter) {
    return std::string("nss_vk.") + filter + ": ";
}

struct RepackPush {
    VkDeviceAddress src, dst;
    std::uint32_t width, height, src_pitch, dst_pitch;  // words
};

int core_release(VSCore* core, const VSAPI* vsapi) {
    VSCoreInfo info{};
    vsapi->getCoreInfo(core, &info);
    return info.core;
}

}  // namespace

std::string residency_args(const std::string& signature, bool gpu_frames) {
    if (!gpu_frames) return signature;
    // "name:vnode[]:flags;" -> "name:vnode[]:all:flags;" (R81 vscore.cpp: the
    // residency is one of the flags after the type).
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

const char* clip_return(bool gpu_frames) {
    return gpu_frames ? "clip:vnode:all;" : "clip:vnode;";
}

std::string gpu_frames_status(VSCore* core, const VSAPI* vsapi) {
#if !NSS_VK_GPU_FRAMES
    (void)core, (void)vsapi;
    return "no (built without VSVulkan4.h)";
#else
    if (!vs_api::gpu_frames(vsapi)) {
        const int api = vsapi->getAPIVersion();
        return "no (VapourSynth API R" + std::to_string(api >> 16) + "." + std::to_string(api & 0xffff) +
               ": R80 or later)";
    }
    return "yes (R" + std::to_string(core_release(core, vsapi)) + ")";
#endif
}

const VSVULKANAPI* take_clips(const char* filter, std::initializer_list<ClipArg> clips, VSCore*,
                              const VSAPI* vsapi) {
    // The first clip given and the first whose residency differs from it.
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
    return gpu ? vs_api::vulkan(vsapi) : nullptr;
}

GpuFrames::GpuFrames(std::shared_ptr<Device> device, const VSAPI* vsapi)
    : device_(std::move(device)), vsapi_(vsapi), repack_(device_, {"runtime_plane_copy", sizeof(RepackPush)}) {}

VSFrame* GpuFrames::new_frame(nss::FrameScope& frames, const VSVideoFormat* format, int width, int height,
                              const VSFrame* props, VSCore* core) const {
#if NSS_VK_GPU_FRAMES
    return frames.adoptOutput(device_->core_api()->newGPUVideoFrame(format, width, height, props, core));
#else
    (void)frames, (void)format, (void)width, (void)height, (void)props, (void)core;
    throw std::logic_error("nss_vk: GPU frames without VSVulkan4.h");
#endif
}

GpuFrames::Plane GpuFrames::plane(const VSFrame* f, int index) const {
    Plane result;
#if NSS_VK_GPU_FRAMES
    VSVulkanPlaneInfo info{};
    if (device_->core_api()->getGPUPlane(f, index, &info) != 0) {
        throw std::runtime_error("nss_vk: plane " + std::to_string(index) + " of a frame is not GPU resident");
    }
    result.buffer = info.buffer;
    result.pitch = static_cast<std::size_t>(vsapi_->getStride(f, index));
    result.ready = info.readySemaphore;
    result.ready_value = info.readyValue;
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = info.buffer;
    result.address = device_->vk().vkGetBufferDeviceAddress(device_->handle(), &address);
#else
    (void)f, (void)index;
#endif
    return result;
}

void GpuFrames::record(VkCommandBuffer commands, VkBuffer from, VkDeviceAddress from_address, VkDeviceSize from_offset,
                       std::size_t from_pitch, VkBuffer to, VkDeviceAddress to_address, VkDeviceSize to_offset,
                       std::size_t to_pitch, std::size_t row_bytes, int rows) const {
    if (rows <= 0) return;
    if (from_pitch == to_pitch) {
        // The padding of the rows between comes along: one region.
        const VkBufferCopy region{from_offset, to_offset, from_pitch * static_cast<std::size_t>(rows - 1) + row_bytes};
        device_->vk().vkCmdCopyBuffer(commands, from, to, 1, &region);
        return;
    }
    const RepackPush push{from_address + from_offset,
                          to_address + to_offset,
                          static_cast<std::uint32_t>(row_bytes / 4),
                          static_cast<std::uint32_t>(rows),
                          static_cast<std::uint32_t>(from_pitch / 4),
                          static_cast<std::uint32_t>(to_pitch / 4)};
    repack_.dispatch(commands, &push, static_cast<std::uint32_t>((push.width + 63) / 64),
                     static_cast<std::uint32_t>((push.height + 3) / 4));
}

void GpuFrames::read(Slot& slot, Slot::Stage stage, const VSFrame* f, int index, VkBuffer to,
                     VkDeviceAddress to_address, VkDeviceSize to_offset, std::size_t row_bytes, int rows) const {
    const Plane p = plane(f, index);
    if (p.ready) slot.wait_for(stage, p.ready, p.ready_value);
    slot.retain(f, vsapi_);
    record(slot.commands(stage), p.buffer, p.address, 0, p.pitch, to, to_address, to_offset, row_bytes, row_bytes,
           rows);
}

void GpuFrames::write(Slot& slot, Slot::Stage stage, VkBuffer from, VkDeviceAddress from_address,
                      VkDeviceSize from_offset, VSFrame* f, int index, std::size_t row_bytes, int rows,
                      int first_row) const {
    const Plane p = plane(f, index);
    slot.retain(f, vsapi_);
    record(slot.commands(stage), from, from_address, from_offset, row_bytes, p.buffer, p.address,
           p.pitch * static_cast<std::size_t>(first_row), p.pitch, row_bytes, rows);
}

void GpuFrames::copy(Slot& slot, Slot::Stage stage, const VSFrame* src, int index, VSFrame* f, int first_row,
                     std::size_t row_bytes, int rows) const {
    const Plane from = plane(src, index), to = plane(f, index);
    if (from.ready) slot.wait_for(stage, from.ready, from.ready_value);
    slot.retain(src, vsapi_);
    slot.retain(f, vsapi_);
    record(slot.commands(stage), from.buffer, from.address, 0, from.pitch, to.buffer, to.address,
           to.pitch * static_cast<std::size_t>(first_row), to.pitch, row_bytes, rows);
}

void GpuFrames::fill(Slot& slot, Slot::Stage stage, VSFrame* f, int index, int first_row, int rows,
                     std::uint32_t word) const {
    if (rows <= 0) return;
    const Plane p = plane(f, index);
    slot.retain(f, vsapi_);
    device_->vk().vkCmdFillBuffer(slot.commands(stage), p.buffer, p.pitch * static_cast<std::size_t>(first_row),
                                  p.pitch * static_cast<std::size_t>(rows), word);
}

void GpuFrames::publish(const Slot& slot, const VSFrame* f, int index) const {
#if NSS_VK_GPU_FRAMES
    device_->core_api()->setGPUPlaneProducer(f, index, slot.gpu_timeline(), slot.last_value());
#else
    (void)slot, (void)f, (void)index;
#endif
}

}  // namespace nss_vk
