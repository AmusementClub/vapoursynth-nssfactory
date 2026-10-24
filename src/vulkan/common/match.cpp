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
};
static_assert(sizeof(Push) == 112, "matches the push block of shaders/common/match_lane.comp");

bool built_block(int block) {
    switch (block) {
        case 1: case 2: case 4: case 8: case 12: case 16: case 32: return true;
        default: return false;
    }
}

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
    const bool fits = static_cast<long long>(g.pitch) * g.height < (1LL << 31);
    return fits && g.channels == 1 && built_block(g.block) && lane_group(g, ps_num) != 0;
}

bool Matcher::emulated(const MatchGeometry& g, int ps_num) const {
    const std::uint32_t lanes = static_cast<std::uint32_t>(lane_group(g, ps_num));
    return force_emulation_ || device_->support().native_lanes < lanes;
}

const Pipeline& Matcher::pipeline(const MatchGeometry& g, bool predictive, int ps_num) {
    if (!supports(g, ps_num)) throw std::logic_error("nss_vk: no lane matcher for this search");
    const int lanes = lane_group(g, ps_num);
    const bool emulate = emulated(g, ps_num);
    const std::string id = "match_lane_b" + std::to_string(g.block) + "_l" + std::to_string(lanes) + "_c" +
                           std::to_string(g.channels) + (predictive ? "_p" : "_s") + (emulate ? "_e" : "_n");
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) {
        // Native: the widest selectable subgroup up to 32 (lanes divides it).
        const DeviceCaps& caps = device_->caps();
        const std::uint32_t width = emulate ? 0 : std::clamp<std::uint32_t>(32, caps.min_subgroup_size,
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
    return p;
}

std::uint32_t groups_for(int ref_count, int lanes) {
    const int per_group = kLaneWarps * (32 / lanes);
    return static_cast<std::uint32_t>((ref_count + per_group - 1) / per_group);
}

}  // namespace

void Matcher::spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g, const RasterGrid& grid,
                      int ref_begin, int ref_count, VkDeviceAddress matches, VkDeviceAddress counts) {
    if (ref_count <= 0) return;
    Push p = push_of(g, grid, ref_begin, ref_count, matches, counts);
    p.source = plane;
    pipeline(g, false, 0).dispatch(commands, &p, groups_for(ref_count, lane_group(g, 0)));
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
    pipeline(g, true, window.ps_num).dispatch(commands, &p, groups_for(ref_count, lane_group(g, window.ps_num)));
}

}  // namespace nss_vk
