// SPDX-License-Identifier: GPL-2.0-only
// BM3D collaborative filtering on the device (docs/vulkan-plan.md V5): the
// group kernels of shaders/bm3d (the warp kernel for the shapes whose cube
// fits registers, else one workgroup per group), as nss_cuda's
// bm3d_filter_groups. Every shape adds its weighted patches to fixed-point
// sums itself (common/aggregate.hpp; owner 2026-10-26).
#pragma once

#include "vulkan/common/aggregate.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace nss_vk {

struct Bm3dGroupArgs {
    // Device tables of plane addresses indexed by DeviceMatch::t (one entry
    // when spatial). ref == 0: the hard threshold, else the Wiener stage.
    VkDeviceAddress src = 0;
    VkDeviceAddress ref = 0;
    VkDeviceAddress matches = 0;  // batch * group (group-strided)
    VkDeviceAddress counts = 0;   // batch
    // batch * bm3d_scratch_floats(block, group, wiener) floats (shapes whose
    // cubes do not fit shared memory), else unused.
    VkDeviceAddress scratch = 0;
    std::int64_t plane_offset = 0;  // floats from a table's plane to the plane filtered (a channel of a unit)
    int pitch = 0;                  // floats; also the cells per row of `fused`
    int batch = 0;
    int block = 0;
    int group = 0;
    float sigma = 0.f;  // effective (profile-scaled)
    FixedTarget fused{};
};

// Floats of scratch per group (0 for the shapes that keep their cubes on chip).
std::size_t bm3d_scratch_floats(int block, int group, bool wiener);

class Bm3dKernels {
public:
    explicit Bm3dKernels(std::shared_ptr<Device> device) : device_(std::move(device)) {}

    // Whether a block / group shape has kernels (BM3D's blocks 1, 2, 4, 8,
    // 12, 16, 32 and groups 1, 2, 4, ..., 64).
    static bool supports(int block, int group);
    // Creates the pipeline of a shape and stage (at filter creation).
    void prepare(int block, int group, bool wiener);
    // Records the filtering of args.batch groups into `commands` (compute).
    void filter(VkCommandBuffer commands, const Bm3dGroupArgs& args);
    // Tests: the emulated warp layer on any device.
    void force_emulation(bool on) noexcept { force_emulation_ = on; }
    // Tests: rows of at most `x` workgroups (default 65535), so that small
    // batches take the multi-row dispatch.
    void limit_groups_x(std::uint32_t x) noexcept { max_groups_x_ = x; }
    // Whether the warp kernel (else the block kernel) serves a shape and stage.
    static bool warp_kernel(int block, int group, bool wiener);
    const Device& device() const noexcept { return *device_; }

private:
    const Pipeline& pipeline(int block, int group, bool wiener, bool& warp);

    std::shared_ptr<Device> device_;
    bool force_emulation_ = false;
    std::uint32_t max_groups_x_ = 65535;  // the smallest maxComputeWorkGroupCount Vulkan allows
    std::mutex mutex_;
    std::map<std::string, Pipeline> pipelines_;
};

}  // namespace nss_vk
