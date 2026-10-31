// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/common/match.hpp"

#include <algorithm>

namespace nss_vk {
namespace {

struct ColumnPush {
    VkDeviceAddress source, matches, counts;
    std::int32_t width, height, pitch, group;
    std::int32_t lo, hi, nx, ny;
    std::int32_t step, max_x, max_y, ref_begin;
    std::int32_t ref_count, row_first, segs, reserved;
};
static_assert(sizeof(ColumnPush) == 88, "column matcher push block");

}  // namespace

bool spatial_match_column_takes(const MatchGeometry& g, const RasterGrid& grid) {
    const bool group_ok = g.group == 8 || g.group == 16 || g.group == 32;
    const bool shape_ok = g.block == 8 || (g.block == 4 && grid.step != 2) || (g.block == 12 && grid.step != 3) ||
                          (g.block == 16 && g.group != 32 && !(g.group == 8 && grid.step == 3));
    return g.channels == 1 && shape_ok && group_ok && g.range_hi < 0 && g.width >= g.block && g.height >= g.block &&
           g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 && grid.ny >= 1 && grid.step >= 1 && grid.step <= 3;
}

const Pipeline& ColumnMatcher::pipeline(int block, int group, int step) {
    const bool pair = group == 8 && (block == 12 || block == 16);
    const std::string id = pair ? "match_column_pair8_b" + std::to_string(block) + "_c" + std::to_string(step + 1)
                                : "match_column_b" + std::to_string(block) + "_k" + std::to_string(group - 1) + "_c" +
                                      std::to_string(step + 1);
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) it = pipelines_.emplace(id, Pipeline(device_, {id, sizeof(ColumnPush), 0, {}})).first;
    return it->second;
}

void ColumnMatcher::prepare(const MatchGeometry& g, const RasterGrid& grid) {
    if (spatial_match_column_takes(g, grid)) (void)pipeline(g.block, g.group, grid.step);
}

void ColumnMatcher::spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g,
                            const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                            VkDeviceAddress counts) {
    if (ref_count <= 0 || !spatial_match_column_takes(g, grid)) return;
    const int row_first = ref_begin / grid.nx;
    const int row_last = (ref_begin + ref_count - 1) / grid.nx;
    const int segs = (grid.nx + 127) / 128;
    const int blocks = (row_last - row_first + 1) * segs;
    ColumnPush p{};
    p.source = plane;
    p.matches = matches;
    p.counts = counts;
    p.width = g.width;
    p.height = g.height;
    p.pitch = g.pitch;
    p.group = g.group;
    p.lo = g.lo();
    p.hi = g.hi();
    p.nx = grid.nx;
    p.ny = grid.ny;
    p.step = grid.step;
    p.max_x = grid.max_x;
    p.max_y = grid.max_y;
    p.ref_begin = ref_begin;
    p.ref_count = ref_count;
    p.row_first = row_first;
    p.segs = segs;
    const Pipeline& pipe = pipeline(g.block, g.group, grid.step);
    const int x = std::min(blocks, 65535);
    pipe.dispatch(commands, &p, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>((blocks + x - 1) / x));
}

}  // namespace nss_vk
