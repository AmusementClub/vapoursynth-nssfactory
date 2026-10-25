// SPDX-License-Identifier: GPL-2.0-only
// Direct I/O between the device and VapourSynth frames, where the host offers
// vapoursynth-classic's VSFrameMemory.h extension: the nss_vk backend of
// frontend/frame_memory.hpp (D28 as amended 2026-10-25). A pool block is
// imported into the device as host memory (VK_EXT_external_memory_host) on
// its second transfer, with a transfer buffer over the whole block, and freed
// by the pool's release hook. The copy queues then read input planes from and
// write output planes into the frames themselves, instead of the host copying
// them through BAR memory, staging or readback buffers. At most half of the
// frame cache is imported (NSS_VK_DIRECT_IO_MAX_MB overrides).
//
// Without the extension (upstream VapourSynth, a device without
// VK_EXT_external_memory_host, or NSS_VK_DIRECT_IO=0 in the environment)
// pinned() is always empty and the filters write and read planes on the host
// as before. The bytes moved are the same either way, so outputs are
// identical. NSS_VK_DIRECT_IO_STATS prints counts when a core's last
// instance goes.
#pragma once

#include "frontend/frame_memory.hpp"
#include "vulkan/runtime/context.hpp"

#include <VapourSynth4.h>

#include <cstddef>
#include <memory>
#include <optional>

namespace nss_vk {

// A plane in an imported block: a transfer source or destination.
struct HostPlane {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    explicit operator bool() const noexcept { return buffer != VK_NULL_HANDLE; }
};

// One per filter instance, declared after what keeps its device alive: the
// imports are freed when the device's last instance on a core goes.
class FrameMemory {
public:
    FrameMemory(std::shared_ptr<Device> device, VSCore* core, const VSAPI* vsapi);

    // The imported block of plane `plane` of f, holding `rows` rows of
    // `row_bytes` at `data` (getReadPtr, or getWritePtr for a frame being
    // written), so that a copy may read or write the plane while f is held.
    // Empty when the rows are not tight (stride != row_bytes: one copy region
    // per plane), and as frontend::FrameMemory::pinned (first use, cap,
    // failure, no extension).
    HostPlane pinned(const VSFrame* f, int plane, const void* data, std::ptrdiff_t stride, std::size_t row_bytes);
    bool available() const noexcept { return memory_ && memory_->available(); }

private:
    std::shared_ptr<Device> device_;
    std::optional<nss::frontend::FrameMemory> memory_;  // only on a device that imports
};

}  // namespace nss_vk
