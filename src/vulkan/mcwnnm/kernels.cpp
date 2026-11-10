// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/mcwnnm/kernels.hpp"

#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace nss_vk {
namespace {

struct Push {
    VkDeviceAddress src, matches, counts, scratch, params;
    FixedTarget fused;
    std::int32_t channel_step, pitch, batch, block, group, pad;
};
static_assert(sizeof(Push) == 128, "matches the push block of shaders/mcwnnm/common.glsl");

// shaders/mcwnnm/common.glsl Params (std430).
struct ParamsBlock {
    float weight[3];
    float smin;
    float rho;
    float mu;
    std::int32_t admm_iter;
    std::int32_t flags;  // 1 residual, 2 adaptive
    std::int32_t classes;
    std::int32_t class_of[3];
    std::int32_t channel_of[3];
    std::int32_t pad;
};
static_assert(sizeof(ParamsBlock) == 64, "matches Params in shaders/mcwnnm/common.glsl");

// Workgroups: up to 8 patches, the rows and apply kernels 16 groups (128
// invocations, 8 a group) and the ADMM kernel 64; then one workgroup of 32
// takes two groups of 16 or one of 32.
constexpr std::uint32_t kGroupsPerLaneWorkgroup = 16, kAdmmWorkgroup = 64;

void upload_once(const std::shared_ptr<Device>& device, const Buffer& target, const void* data, std::size_t bytes) {
    Buffer staging(device, bytes, Memory::Staging);
    std::memcpy(staging.mapped(), data, bytes);
    staging.flush();
    Slot slot(device);
    const VkBufferCopy copy{0, 0, bytes};
    device->vk().vkCmdCopyBuffer(slot.commands(Slot::Upload), staging.get(), target.get(), 1, &copy);
    slot.submit();
    slot.wait();
}

}  // namespace

std::size_t mcwnnm_scratch_floats(int group) {
    return 10u * static_cast<std::size_t>(group) * static_cast<std::size_t>(group) + 1u;
}

McwnnmKernels::McwnnmKernels(std::shared_ptr<Device> device, const McwnnmModel& model) : device_(std::move(device)) {
    ParamsBlock b{};
    // Channel weights (nss::channel_weight_diag): (smin / sigma_c)^2 with the
    // smallest positive sigma; a zero-sigma channel is pinned to its input.
    float smin = 0.f;
    for (const float s : model.sigma) {
        if (s > 0.f) smin = smin == 0.f ? s : std::min(smin, s);
    }
    for (int c = 0; c < 3; ++c) {
        const float sc = model.sigma[c] > 0.f ? model.sigma[c] : smin * 1e-6f;
        b.weight[c] = (smin / sc) * (smin / sc);
    }
    b.smin = smin;
    b.rho = model.rho;
    b.mu = model.mu;
    b.admm_iter = model.admm_iter;
    b.flags = (model.residual ? 1 : 0) | (model.adaptive ? 2 : 0);
    // Classes, as nss_cuda: channels of equal positive sigma; a channel without
    // noise (pinned by a weight of 1e12) stays a class of its own.
    b.classes = 0;
    for (int c = 0; c < 3; ++c) {
        int cls = 0;
        while (cls < b.classes && !(model.sigma[c] > 0.f && model.sigma[b.channel_of[cls]] == model.sigma[c])) ++cls;
        if (cls == b.classes) b.channel_of[b.classes++] = c;
        b.class_of[c] = cls;
    }
    classes_ = b.classes;
    params_ = Buffer(device_, sizeof(ParamsBlock), Memory::Device);
    upload_once(device_, params_, &b, sizeof(b));
}

const Pipeline& McwnnmKernels::pipeline(const std::string& name, int lanes) {
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

void McwnnmKernels::prepare(int group) {
    if (group < 1 || group > 32) throw std::logic_error("nss_vk: unsupported MCWNNM group size");
    if (group <= 8) {
        pipeline("mcwnnm_rows8", 8);
        pipeline("mcwnnm_admm8", 0);
        pipeline("mcwnnm_apply8", 0);
    } else {
        const std::string u = classes_ == 1 ? "_u0" : "_u1";
        pipeline(std::string(group <= 16 ? "mcwnnm_block_n16" : "mcwnnm_block_n32") + u, group <= 16 ? 16 : 32);
    }
}

void McwnnmKernels::dispatch(VkCommandBuffer commands, const Pipeline& p, const void* push, std::uint32_t groups) const {
    const std::uint32_t x = std::min(groups, max_groups_x_);
    p.dispatch(commands, push, x, (groups + x - 1) / x);
}

void McwnnmKernels::filter(VkCommandBuffer commands, const McwnnmGroupArgs& a) {
    if (a.batch <= 0) return;
    if (a.group < 1 || a.group > 32) throw std::logic_error("nss_vk: unsupported MCWNNM group size");
    if (!a.scratch) throw std::logic_error("nss_vk: MCWNNM needs its state scratch");
    if (a.channel_step > 0x7fffffff) throw std::logic_error("nss_vk: MCWNNM channel step beyond 32 bits");
    Push push{};
    push.src = a.src;
    push.matches = a.matches;
    push.counts = a.counts;
    push.scratch = a.scratch;
    push.params = params_.address();
    push.fused = a.fused;
    push.channel_step = static_cast<std::int32_t>(a.channel_step);
    push.pitch = a.pitch;
    push.batch = a.batch;
    push.block = a.block;
    push.group = a.group;
    const std::uint32_t batch = static_cast<std::uint32_t>(a.batch);
    if (a.group <= 8) {
        const std::uint32_t lanes = (batch + kGroupsPerLaneWorkgroup - 1) / kGroupsPerLaneWorkgroup;
        dispatch(commands, pipeline("mcwnnm_rows8", 8), &push, lanes);
        compute_barrier(*device_, commands);
        dispatch(commands, pipeline("mcwnnm_admm8", 0), &push, (batch + kAdmmWorkgroup - 1) / kAdmmWorkgroup);
        compute_barrier(*device_, commands);
        dispatch(commands, pipeline("mcwnnm_apply8", 0), &push, lanes);
    } else {
        const std::string u = classes_ == 1 ? "_u0" : "_u1";
        if (a.group <= 16) {
            dispatch(commands, pipeline("mcwnnm_block_n16" + u, 16), &push, (batch + 1) / 2);
        } else {
            dispatch(commands, pipeline("mcwnnm_block_n32" + u, 32), &push, batch);
        }
    }
}

}  // namespace nss_vk
