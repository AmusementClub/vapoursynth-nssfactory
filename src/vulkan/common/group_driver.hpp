// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for the patch-group filters (BM3D, WNNM, MCWNNM, NCSR;
// docs/vulkan-plan.md V5), as nss_cuda's group_driver.hpp: a filter parses
// its arguments, fills a GroupFilterConfig and supplies `launch`, which turns
// one batch of matched groups into weighted patches added to fixed-point sums
// (nss_vk aggregates every shape so, owner 2026-10-26); the driver owns
//   - the memory plan against memory_limit_mb (slots, batches),
//   - frame I/O: BAR or staged uploads and readback, or direct copies to and
//     from VapourSynth frames under vapoursynth-classic (FrameMemory), or
//     GPU frames on VapourSynth's device (GpuFrames, D29),
//   - matching (common/match.hpp, spatial or predictive, on the guide when
//     there is one),
//   - the three output modes: spatial, the legacy fat intermediate (float
//     num / den slices from the fixed-point sums) and rolling (V6),
// with two stages (BM3D final) and units of three planes matched on the
// first (CBM3D chroma). Joint units and outer rounds (V7, V8) come later.
//
// A spatial or legacy frame is one submission of its slot: every plane's
// uploads, then the units one after the other on the compute queue, then
// every plane's downloads, and one wait. A rolling chunk is one submission
// of the slot its turn gives it (the window tables, then per plane its
// centers with the ring uploads between them, then the chunk's frames into
// the host chunk store), as nss_cuda's chunks (the same carry, chunk cache
// and plan order, src/gpuhost).
#pragma once

#include "frontend/ownership.hpp"
#include "nss/contracts.hpp"
#include "nss/params/temporal.hpp"
#include "vulkan/common/aggregate.hpp"
#include "vulkan/common/match.hpp"
#include "vulkan/runtime/context.hpp"

#include <VapourSynth4.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace nss_vk {

enum class GroupMode { Spatial, Legacy, Rolling };

struct GroupPlane {
    // Filled by the filter:
    bool active = false;  // false copies the plane
    int block = 0, group = 0, step = 0, range = 0, ps_num = 0, ps_range = 0;
    float sigma = 0.f;  // filter-defined scale, passed through to launch
    // Filled by the driver:
    bool wiener = false;  // its launches get a guide (the Wiener reference)
    int width = 0, height = 0;
    std::size_t floats = 0;
    RasterGrid grid{};
    int batch = 0;
};

// One batch of matched groups, recorded into `commands` (compute).
struct GroupLaunch {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkDeviceAddress src = 0;    // table: window slot t -> source unit (2R+1 entries)
    VkDeviceAddress guide = 0;  // same for the guide (Wiener reference), 0 without one
    int pitch = 0;              // floats (and cells of `fused` per row)
    VkDeviceAddress matches = 0;  // batch * plane->group (group-strided)
    VkDeviceAddress counts = 0;   // batch
    int batch = 0;
    const GroupPlane* plane = nullptr;  // plane[c]: channel c's parameters (sigma) in a unit
    int channels = 1;
    long long channel_step = 0;  // floats from channel c to c + 1 of a unit
    VkDeviceAddress scratch = 0;  // batch * scratch_floats(plane) floats, or 0
    const bool* channel_active = nullptr;  // shared_match: channel c is filtered
    // The unit's sums: channel c's cells start c * fused.channel_step cells
    // further; a match of window slot t goes to the slice fused gives it.
    FixedTarget fused{};
};

struct GroupFilterConfig {
    std::string name;  // "BM3D": node name and error prefix nss_vk.<name>
    nss::Model model = nss::Model::BM3D;
    nss::NodeRef node, guide;  // guide: matching clip (ref / rclip), may be null
    VSVideoInfo vi{};
    int radius = 0;
    GroupMode mode = GroupMode::Legacy;  // for radius > 0; radius 0 is always spatial
    nss::RollingParams rolling{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    std::shared_ptr<Device> device;
    // GPU frames (gpu_frames.hpp take_clips): the clips are GPU resident and
    // `device` is VapourSynth's; null for CPU frames.
    const VSVULKANAPI* gpu_api = nullptr;
    GroupPlane planes[3];
    // Shared groups (CBM3D): with channels == 3 the groups are matched on
    // channel 0 and every channel is filtered with them (planes[c].sigma,
    // planes[0]'s geometry); planes[c].active says whether channel c is
    // filtered.
    int channels = 1;
    bool shared_match = false;
    // Two stages in one node (BM3D final): the source is first filtered with
    // `basic` and no guide, and that estimate stays on the device as the
    // guide of the second stage. The node takes no guide clip then; spatial
    // and rolling output only.
    bool two_stage = false;
    GroupPlane basic[3];
    // Extra device floats per reference group (second argument: guide present).
    std::function<std::size_t(const GroupPlane&, bool)> scratch_floats;
    // Creates the kernels a plane's stage will use (at filter creation).
    std::function<void(const GroupPlane&, bool wiener)> prepare;
    std::function<void(const GroupLaunch&)> launch;
};

// Plans memory, builds the slots and creates the VapourSynth node.
void group_filter_install(GroupFilterConfig&& config, VSMap* out, VSCore* core, const VSAPI* vsapi);

}  // namespace nss_vk
