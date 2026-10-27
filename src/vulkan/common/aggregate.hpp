// SPDX-License-Identifier: GPL-2.0-only
// Deterministic aggregation by fixed point for the patch-group filters
// (docs/vulkan-plan.md V3, D17 / D26), as nss_cuda's FixedTarget,
// fixed_clear and fixed_finish (src/cuda/common/aggregate.hpp). Kernels that
// aggregate their own output add weight * value at every pixel of a patch to
// num and the weight once, at the patch's top-left pixel, to den
// (shaders/common/fixed.glsl): int64 cells of value * 2^32 whose sums do not
// depend on the order of the atomics. fixed_finish divides one slice;
// fixed_slices turns it into the legacy output's float num / den.
#pragma once

#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/pipeline.hpp"

#include <cstdint>
#include <memory>

namespace nss_vk {

// As shaders/common/fixed.glsl's FixedTarget (std430). Slice s of a plane
// starts slice_step cells after slice s - 1; a patch of window slot t goes to
// logical slice t + slice_base and is dropped outside [0, slices); the slices
// are a ring of slice_ring cells starting at slice_first (the rolling driver
// keeps one per frame from its chunk's first on).
struct FixedTarget {
    VkDeviceAddress num = 0;
    VkDeviceAddress den = 0;
    std::int64_t slice_step = 0;
    std::int64_t channel_step = 0;  // cells from one channel's slices to the next
    std::int32_t pitch = 0;         // cells per row
    std::int32_t slice_base = 0;
    std::int32_t slices = 1;
    std::int32_t slice_first = 0;
    std::int32_t slice_ring = 1;
    std::int32_t pad[3]{};
};
static_assert(sizeof(FixedTarget) == 64, "matches shaders/common/fixed.glsl");

// Zeroes `bytes` of `buffer` from `offset` (transfer fill), after the compute
// work recorded before it in `commands` and visible to the compute work
// recorded after it.
void fixed_clear(const Device& device, VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset,
                 VkDeviceSize bytes);

class Aggregator {
public:
    explicit Aggregator(std::shared_ptr<Device> device);

    // out = den > 1e-12 ? num / den : src over width x height, for patches of
    // block x block pixels: num and den are one slice's cells (pitch cells a
    // row; src and out pitch floats a row), rows scratch of height * pitch
    // cells. Recorded into `commands` (compute), with the barriers between
    // its two kernels; the caller orders what writes num / den before it.
    void fixed_finish(VkCommandBuffer commands, VkDeviceAddress num, VkDeviceAddress den, int pitch, int block,
                      VkDeviceAddress src, int width, int height, VkDeviceAddress rows, VkDeviceAddress out) const;
    // The legacy output's float slices of one slice's cells: num / 2^32 at
    // `out` and the covering den cells / 2^32 height * pitch floats after it
    // (the fat frame's num and den rows of the slice). As fixed_finish.
    void fixed_slices(VkCommandBuffer commands, VkDeviceAddress num, VkDeviceAddress den, int pitch, int block,
                      int width, int height, VkDeviceAddress rows, VkDeviceAddress out) const;

private:
    void run(VkCommandBuffer commands, const Pipeline& last, VkDeviceAddress num, VkDeviceAddress den, int pitch,
             int block, VkDeviceAddress src, int width, int height, VkDeviceAddress rows, VkDeviceAddress out) const;

    std::shared_ptr<Device> device_;
    Pipeline rows_;
    Pipeline finish_;
    Pipeline slices_;
};

}  // namespace nss_vk
