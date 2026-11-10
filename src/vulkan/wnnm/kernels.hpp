// SPDX-License-Identifier: GPL-2.0-only
// WNNM group shrinkage on the device (docs/vulkan-plan.md V7): the Gram group
// kernels of shaders/wnnm, as nss_cuda's wnnm_filter_groups (gram_group.cuh
// with the WNNM model): up to 8 patches three kernels (rows, solve, apply),
// 9 to 16 the block kernel with the parallel Jacobi, 17 to 32 the block
// kernel with Householder and divide and conquer, then an apply kernel. Every group adds its
// weighted patches to fixed-point sums itself (common/aggregate.hpp).
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

struct WnnmGroupArgs {
    VkDeviceAddress src = 0;      // table of plane addresses indexed by DeviceMatch::t
    VkDeviceAddress matches = 0;  // batch * group (group-strided)
    VkDeviceAddress counts = 0;   // batch
    VkDeviceAddress scratch = 0;  // batch * wnnm_scratch_floats(group) floats
    int pitch = 0;                // floats; also the cells per row of `fused`
    int batch = 0;
    int block = 0;
    int group = 0;                // <= 32 (nss::kWnnmMaxGroup)
    float sigma = 0.f;            // user sigma / 255
    bool residual = false;
    bool adaptive = false;
    FixedTarget fused{};
};

// Floats of scratch per group: the Gram matrix, M and the weight up to 8
// patches, the divide and conquer's two matrices, M and the weight from 17.
std::size_t wnnm_scratch_floats(int group);

class WnnmKernels {
public:
    explicit WnnmKernels(std::shared_ptr<Device> device) : device_(std::move(device)) {}

    // Creates the pipelines of a group size (at filter creation).
    void prepare(int group);
    // Records the filtering of args.batch groups into `commands` (compute),
    // with the barriers between its kernels.
    void filter(VkCommandBuffer commands, const WnnmGroupArgs& args);
    // Tests: the emulated warp layer on any device.
    void force_emulation(bool on) noexcept { force_emulation_ = on; }
    // Tests: rows of at most `x` workgroups (default 65535), so that small
    // batches take the multi-row dispatch.
    void limit_groups_x(std::uint32_t x) noexcept { max_groups_x_ = x; }
    const Device& device() const noexcept { return *device_; }

private:
    // A kernel by name; `lanes`: the warp layer's segment (0: no warp
    // operations, the driver's subgroup size).
    const Pipeline& pipeline(const std::string& name, int lanes);
    void dispatch(VkCommandBuffer commands, const Pipeline& p, const void* push, std::uint32_t groups) const;

    std::shared_ptr<Device> device_;
    bool force_emulation_ = false;
    std::uint32_t max_groups_x_ = 65535;
    std::mutex mutex_;
    std::map<std::string, Pipeline> pipelines_;
};

}  // namespace nss_vk
