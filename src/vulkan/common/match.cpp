// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/common/match.hpp"

#include <algorithm>
#include <stdexcept>

namespace nss_vk {
namespace {

constexpr int kLaneWarps = 4;  // 32-lane warps a workgroup (match_lane.comp)
constexpr int kLaneMaxCoordinate = 0xffff;

struct Push {
    VkDeviceAddress source, matches, counts;
    std::int64_t channel_step;
    std::int32_t width, height, pitch, group;
    std::int32_t lo, hi;
    std::int32_t nx, step, max_x, max_y;
    std::int32_t ref_begin, ref_count;
    std::int32_t t0, radius, ntemp, valid_begin, valid_end, ps_num, ps_range;
    std::int32_t block, channels;
};
static_assert(sizeof(Push) == 120, "matches the push blocks of shaders/common/match_lane.comp and match_block.comp");

// The block fallback's bound on the group and ps_num (match_block.comp's K).
constexpr int kBlockMaxK = 256;
int block_k(const MatchGeometry& g, int ps_num) { return g.group <= 64 && ps_num <= 64 ? 64 : kBlockMaxK; }

// Workgroups of the block fallback: one per reference, in rows of at most
// 65535 (the smallest maxComputeWorkGroupCount Vulkan allows).
constexpr int kMaxGroupsX = 65535;

}  // namespace

int lane_group(const MatchGeometry& g, int ps_num) {
    if ((g.channels != 1 && g.channels != 3) || g.group < 2) return 0;
    if (g.block < 1 || (g.block > 16 && g.block != 32)) return 0;
    if (g.channels != 1 && g.block > 8) return 0;
    if (g.width > kLaneMaxCoordinate || g.height > kLaneMaxCoordinate) return 0;
    const int need = std::max(g.group - 1, ps_num);
    if (need > 32) return 0;
    return g.block == 32 ? 32 : need <= 8 ? 8 : need <= 16 ? 16 : 32;
}

bool Matcher::supports(const MatchGeometry& g, int ps_num) const {
    // Plane indices are 32-bit in the shaders (channels move the address).
    if (static_cast<long long>(g.pitch) * g.height >= (1LL << 31) || g.block < 1 || g.group < 1 || g.channels < 1 || g.channels > 3) return false;
    if (g.block > g.width || g.block > g.height) return false;
    return lanes(g, ps_num) || (g.group <= kBlockMaxK && ps_num <= kBlockMaxK);
}

bool Matcher::lanes(const MatchGeometry& g, int ps_num) const { return lane_group(g, ps_num) != 0; }

bool Matcher::emulated(const MatchGeometry& g, int ps_num) const {
    if (!lanes(g, ps_num)) return false;  // the block fallback has no subgroup operations
    const std::uint32_t width = static_cast<std::uint32_t>(lane_group(g, ps_num));
    return force_emulation_ || device_->support().native_lanes < width;
}

const Pipeline& Matcher::pipeline(const MatchGeometry& g, bool predictive, int ps_num) {
    if (!supports(g, ps_num)) throw std::logic_error("nss_vk: no matcher for this search");
    const bool lane = lanes(g, ps_num);
    const bool emulate = emulated(g, ps_num);
    const std::string id =
        lane ? "match_lane_b" + std::to_string(g.block) + "_l" + std::to_string(lane_group(g, ps_num)) + "_c" +
                   std::to_string(g.channels) + (predictive ? "_p" : "_s") + (emulate ? "_e" : "_n")
             : "match_block_k" + std::to_string(block_k(g, ps_num)) + (predictive ? "_p" : "_s");
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) {
        // Native lanes: the widest selectable subgroup up to 32 (lanes divides it).
        const DeviceCaps& caps = device_->caps();
        const std::uint32_t width = !lane || emulate ? 0 : std::clamp<std::uint32_t>(32, caps.min_subgroup_size,
                                                                                    caps.max_subgroup_size);
        it = pipelines_.emplace(id, Pipeline(device_, {id, sizeof(Push), width, {}})).first;
    }
    return it->second;
}

void Matcher::prepare(const MatchGeometry& g, bool predictive, int ps_num) {
    pipeline(g, predictive, ps_num);
}

namespace {

Push push_of(const MatchGeometry& g, const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
             VkDeviceAddress counts) {
    Push p{};
    p.matches = matches;
    p.counts = counts;
    p.channel_step = g.channel_step;
    p.width = g.width;
    p.height = g.height;
    p.pitch = g.pitch;
    p.group = g.group;
    p.lo = g.lo();
    p.hi = g.hi();
    p.nx = grid.nx;
    p.step = grid.step;
    p.max_x = grid.max_x;
    p.max_y = grid.max_y;
    p.ref_begin = ref_begin;
    p.ref_count = ref_count;
    p.block = g.block;
    p.channels = g.channels;
    return p;
}

}  // namespace

void Matcher::dispatch(VkCommandBuffer commands, const MatchGeometry& g, bool predictive, int ps_num, const void* push,
                       int ref_count) {
    const Pipeline& p = pipeline(g, predictive, ps_num);
    if (lanes(g, ps_num)) {
        const int per_group = kLaneWarps * (32 / lane_group(g, ps_num));
        p.dispatch(commands, push, static_cast<std::uint32_t>((ref_count + per_group - 1) / per_group));
    } else {
        const int x = std::min(ref_count, kMaxGroupsX);
        p.dispatch(commands, push, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>((ref_count + x - 1) / x));
    }
}

void Matcher::spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g, const RasterGrid& grid,
                      int ref_begin, int ref_count, VkDeviceAddress matches, VkDeviceAddress counts) {
    if (ref_count <= 0) return;
    Push p = push_of(g, grid, ref_begin, ref_count, matches, counts);
    p.source = plane;
    dispatch(commands, g, false, 0, &p, ref_count);
}

void Matcher::predictive(VkCommandBuffer commands, const MatchGeometry& g, const TemporalWindow& window,
                         const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                         VkDeviceAddress counts) {
    if (ref_count <= 0) return;
    Push p = push_of(g, grid, ref_begin, ref_count, matches, counts);
    p.source = window.frames;
    p.t0 = window.t0;
    p.radius = window.radius;
    p.ntemp = window.ntemp;
    p.valid_begin = window.valid_begin;
    p.valid_end = window.valid_end;
    p.ps_num = window.ps_num;
    p.ps_range = window.ps_range;
    dispatch(commands, g, true, window.ps_num, &p, ref_count);
}

}  // namespace nss_vk
