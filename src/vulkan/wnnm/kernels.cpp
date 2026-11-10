// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/wnnm/kernels.hpp"

#include <algorithm>
#include <stdexcept>

namespace nss_vk {
namespace {

struct Push {
    VkDeviceAddress src, matches, counts, scratch;
    FixedTarget fused;
    std::int32_t pitch, batch, block, group;
    float sigma;
    std::int32_t flags;  // 1 residual, 2 adaptive
};
static_assert(sizeof(Push) == 120, "matches the push block of shaders/wnnm/common.glsl");

// shaders/wnnm/gram8.glsl's kSplitFloats; from 17 patches the divide and
// conquer's two 32 x 32 matrices (shaders/wnnm/block_eigen.glsl dc_floats;
// its vectors are in shared memory), then M and the weight for the apply
// kernel (group_floats).
constexpr std::size_t kSplitFloats = 36 + 64 + 1;
constexpr std::size_t kDcFloats = 2 * 32 * 32;
constexpr std::size_t kBlockFloats = kDcFloats + 32 * 32 + 1;

// Workgroups: up to 8 patches, the rows and apply kernels 16 groups (128
// invocations, 8 a group) and the solve kernel 64; then one workgroup of 32
// takes two groups of 16 or one of 32.
constexpr std::uint32_t kGroupsPerSplitWorkgroup = 16, kSolveWorkgroup = 64;
// From 17 patches the apply kernel: 128 invocations, one a patch of 4 groups.
constexpr std::uint32_t kGroupsPerApplyWorkgroup = 4;

}  // namespace

std::size_t wnnm_scratch_floats(int group) {
    if (group <= 8) return kSplitFloats;
    return group > 16 ? kBlockFloats : 0;
}

const Pipeline& WnnmKernels::pipeline(const std::string& name, int lanes) {
    // A warp kernel runs natively on the widest selectable subgroup up to 32
    // where its segments are native (D23), else on the emulated layer.
    const DeviceCaps& caps = device_->caps();
    const std::uint32_t width = std::clamp<std::uint32_t>(32, caps.min_subgroup_size, caps.max_subgroup_size);
    const bool native = lanes > 0 && !force_emulation_ && width <= 32 &&
                        device_->support().native_lanes >= static_cast<std::uint32_t>(lanes);
    std::string id = name;
    if (lanes > 0) id += native ? "_n" : "_e";
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) {
        it = pipelines_.emplace(id, Pipeline(device_, {id, sizeof(Push), native ? width : 0u, {}})).first;
    }
    return it->second;
}

void WnnmKernels::prepare(int group) {
    if (group < 1 || group > 32) throw std::logic_error("nss_vk: unsupported WNNM group size");
    if (group <= 8) {
        pipeline("wnnm_gram_rows", 8);
        pipeline("wnnm_gram_solve", 0);
        pipeline("wnnm_gram_apply", 0);
    } else {
        pipeline(group <= 16 ? "wnnm_gram_block_n16" : "wnnm_gram_block_n32", group <= 16 ? 16 : 32);
        if (group > 16) pipeline("wnnm_gram_apply_n32", 0);
    }
}

void WnnmKernels::dispatch(VkCommandBuffer commands, const Pipeline& p, const void* push, std::uint32_t groups) const {
    const std::uint32_t x = std::min(groups, max_groups_x_);
    p.dispatch(commands, push, x, (groups + x - 1) / x);
}

void WnnmKernels::filter(VkCommandBuffer commands, const WnnmGroupArgs& a) {
    if (a.batch <= 0) return;
    if (a.group < 1 || a.group > 32) throw std::logic_error("nss_vk: unsupported WNNM group size");
    Push push{};
    push.src = a.src;
    push.matches = a.matches;
    push.counts = a.counts;
    push.scratch = a.scratch;
    push.fused = a.fused;
    push.pitch = a.pitch;
    push.batch = a.batch;
    push.block = a.block;
    push.group = a.group;
    push.sigma = a.sigma;
    push.flags = (a.residual ? 1 : 0) | (a.adaptive ? 2 : 0);
    const std::uint32_t batch = static_cast<std::uint32_t>(a.batch);
    if (a.group <= 8) {
        if (!a.scratch) throw std::logic_error("nss_vk: WNNM groups of up to 8 need the Gram scratch");
        const std::uint32_t split = (batch + kGroupsPerSplitWorkgroup - 1) / kGroupsPerSplitWorkgroup;
        dispatch(commands, pipeline("wnnm_gram_rows", 8), &push, split);
        compute_barrier(*device_, commands);
        dispatch(commands, pipeline("wnnm_gram_solve", 0), &push, (batch + kSolveWorkgroup - 1) / kSolveWorkgroup);
        compute_barrier(*device_, commands);
        dispatch(commands, pipeline("wnnm_gram_apply", 0), &push, split);
    } else if (a.group <= 16) {
        dispatch(commands, pipeline("wnnm_gram_block_n16", 16), &push, (batch + 1) / 2);
    } else {
        if (!a.scratch) throw std::logic_error("nss_vk: WNNM groups above 16 need the eigen scratch");
        dispatch(commands, pipeline("wnnm_gram_block_n32", 32), &push, batch);
        compute_barrier(*device_, commands);
        dispatch(commands, pipeline("wnnm_gram_apply_n32", 0), &push, (batch + kGroupsPerApplyWorkgroup - 1) / kGroupsPerApplyWorkgroup);
    }
}

}  // namespace nss_vk
