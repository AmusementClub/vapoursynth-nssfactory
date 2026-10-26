// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/bm3d/kernels.hpp"

#include <algorithm>
#include <stdexcept>

namespace nss_vk {
namespace {

struct Push {
    VkDeviceAddress src, ref, matches, counts, scratch;
    std::int64_t plane_offset;
    FixedTarget fused;
    std::int32_t pitch, batch;
    float sigma;
    std::int32_t pad;
};
static_assert(sizeof(Push) == 128, "matches the push block of shaders/bm3d/common.glsl");

// The shapes the warp kernel serves, as nss_cuda's warp_serves (admitted
// there on paired measurements against the block kernel): up to 128 samples
// a lane for both stages, 256 for the hard threshold only, and 16 / 16.
bool warp_serves(int block, int group, bool wiener) {
    switch (block * 100 + group) {
        case 402: case 404: case 408: case 416: case 432:
        case 802: case 804: case 808: case 816:
        case 1602: case 1604: case 1608: case 1616:
            return true;
        case 464: case 832: case 1632:
            return !wiener;
        default:
            return false;
    }
}

// nss_cuda's kCubeSharedBytes: the block kernel keeps larger cubes in scratch.
constexpr std::size_t kCubeSharedBytes = 24 * 1024;

std::size_t cube_bytes(int block, int group, bool wiener) {
    return static_cast<std::size_t>(group) * block * block * sizeof(float) * (wiener ? 2 : 1);
}

constexpr std::uint32_t kMaxGroupsX = 65535;

}  // namespace

std::size_t bm3d_scratch_floats(int block, int group, bool wiener) {
    if (warp_serves(block, group, wiener) || cube_bytes(block, group, wiener) <= kCubeSharedBytes) return 0;
    return static_cast<std::size_t>(group) * block * block * (wiener ? 2 : 1);
}

bool Bm3dKernels::supports(int block, int group) {
    const bool b = block == 1 || block == 2 || block == 4 || block == 8 || block == 12 || block == 16 || block == 32;
    const bool g = group >= 1 && group <= 64 && (group & (group - 1)) == 0;
    return b && g;
}

const Pipeline& Bm3dKernels::pipeline(int block, int group, bool wiener, bool& warp) {
    if (!supports(block, group)) throw std::logic_error("nss_vk: unsupported BM3D block or group size");
    warp = warp_serves(block, group, wiener);
    // The warp kernel (a 32-invocation workgroup) runs natively on the
    // widest selectable subgroup up to 32 where segments of B lanes are
    // native (D23), else on the emulated layer.
    const DeviceCaps& caps = device_->caps();
    const std::uint32_t width = std::clamp<std::uint32_t>(32, caps.min_subgroup_size, caps.max_subgroup_size);
    const bool native = warp && !force_emulation_ && width <= 32 &&
                        device_->support().native_lanes >= static_cast<std::uint32_t>(block);
    std::string id = std::string(warp ? "bm3d_warp" : "bm3d_shape") + "_b" + std::to_string(block) + "_g" +
                     std::to_string(group) + (wiener ? "_w" : "_h");
    if (warp) id += native ? "_n" : "_e";
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) {
        it = pipelines_.emplace(id, Pipeline(device_, {id, sizeof(Push), native ? width : 0u, {}})).first;
    }
    return it->second;
}

void Bm3dKernels::prepare(int block, int group, bool wiener) {
    bool warp = false;
    pipeline(block, group, wiener, warp);
}

void Bm3dKernels::filter(VkCommandBuffer commands, const Bm3dGroupArgs& a) {
    if (a.batch <= 0) return;
    bool warp = false;
    const Pipeline& p = pipeline(a.block, a.group, a.ref != 0, warp);
    Push push{};
    push.src = a.src;
    push.ref = a.ref;
    push.matches = a.matches;
    push.counts = a.counts;
    push.scratch = a.scratch;
    push.plane_offset = a.plane_offset;
    push.fused = a.fused;
    push.pitch = a.pitch;
    push.batch = a.batch;
    push.sigma = a.sigma;
    // Warp kernel: 32 / block groups a workgroup; block kernel: one.
    const std::uint32_t per = warp ? static_cast<std::uint32_t>(32 / a.block) : 1u;
    const std::uint32_t groups = (static_cast<std::uint32_t>(a.batch) + per - 1) / per;
    const std::uint32_t x = std::min(groups, kMaxGroupsX);
    p.dispatch(commands, &push, x, (groups + x - 1) / x);
}

}  // namespace nss_vk
