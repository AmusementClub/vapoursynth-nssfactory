// SPDX-License-Identifier: GPL-2.0-only
// MCWNNM group filter on the device (docs/vulkan-plan.md V8): the kernels of
// shaders/mcwnnm, as nss_cuda's mcwnnm_filter_groups (src/cuda/mcwnnm): up
// to 8 patches three kernels (the channel classes' Gram matrices, the ADMM
// one invocation a group, the output), 9 to 32 the block kernel with the
// parallel Jacobi (N = 16 or 32). Every group adds its weighted patches to
// the fixed-point sums of every channel itself (common/aggregate.hpp).
#pragma once

#include "vulkan/common/aggregate.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace nss_vk {

// The model of a filter (nss::McwnnmParams without the matching).
struct McwnnmModel {
    float sigma[3]{};  // user sigma / 255 per channel
    bool residual = false;
    bool adaptive = true;
    int admm_iter = 10;
    float rho = 3.f;
    float mu = 1.001f;
};

struct McwnnmGroupArgs {
    VkDeviceAddress src = 0;      // table of the unit's channel 0 addresses indexed by DeviceMatch::t
    VkDeviceAddress matches = 0;  // batch * group (group-strided)
    VkDeviceAddress counts = 0;   // batch
    VkDeviceAddress scratch = 0;  // batch * mcwnnm_scratch_floats(group) floats
    long long channel_step = 0;   // floats between the channels of a source frame
    int pitch = 0;                // floats; also the cells per row of `fused`
    int batch = 0;
    int block = 0;
    int group = 0;                // <= 32 (nss::kWnnmMaxGroup)
    FixedTarget fused{};          // channel c at + c * fused.channel_step cells
};

// Floats of state per group: per channel class the Gram matrix, P and Q,
// the last eigenvectors and the aggregation weight (group x group each).
std::size_t mcwnnm_scratch_floats(int group);

class McwnnmKernels {
public:
    // Channels of equal positive sigma form a class (one P, Q, Z) as in
    // nss_cuda; the model goes to a small device buffer once.
    McwnnmKernels(std::shared_ptr<Device> device, const McwnnmModel& model);

    // Creates the pipelines of a group size (at filter creation).
    void prepare(int group);
    // Records the filtering of args.batch groups into `commands` (compute),
    // with the barriers between its kernels.
    void filter(VkCommandBuffer commands, const McwnnmGroupArgs& args);
    // Tests: the emulated warp layer on any device.
    void force_emulation(bool on) noexcept { force_emulation_ = on; }
    // Tests: rows of at most `x` workgroups (default 65535).
    void limit_groups_x(std::uint32_t x) noexcept { max_groups_x_ = x; }
    int classes() const noexcept { return classes_; }
    const Device& device() const noexcept { return *device_; }

private:
    const Pipeline& pipeline(const std::string& name, int lanes);
    void dispatch(VkCommandBuffer commands, const Pipeline& p, const void* push, std::uint32_t groups) const;

    std::shared_ptr<Device> device_;
    Buffer params_;
    int classes_ = 3;
    bool force_emulation_ = false;
    std::uint32_t max_groups_x_ = 65535;
    std::mutex mutex_;
    std::map<std::string, Pipeline> pipelines_;
};

}  // namespace nss_vk
