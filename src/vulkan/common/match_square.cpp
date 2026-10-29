// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/common/match.hpp"

#include <algorithm>
#include <stdexcept>

namespace nss_vk {
namespace {

constexpr int kSide = 16;
constexpr int kRefs = kSide - 1;

struct SquarePush {
    VkDeviceAddress source, matches, counts;
    std::int32_t width, height, pitch, group_range;
    std::int32_t nx_ny, step_rows, max_x, max_y, ref_begin, ref_count;
    std::int32_t pass[16];
};
static_assert(sizeof(SquarePush) == 128, "square matcher push block");

struct Passes {
    int pass[4][4]{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}};
    int count = 0;
    int blocks = 0;
    int rows = 1;
};

Passes plan_square(int q, const RasterGrid& grid, int ref_begin, int ref_count) {
    const int row_first = ref_begin / grid.nx, row_last = (ref_begin + ref_count - 1) / grid.nx;
    const bool clamp_x = grid.max_x % q != 0, clamp_y = grid.max_y % q != 0;
    const int lattice_cols = clamp_x ? grid.nx - 1 : grid.nx;
    const int lattice_rows = clamp_y ? grid.ny - 1 : grid.ny;
    const int blocks_x = (lattice_cols + kRefs - 1) / kRefs;
    const int last_lattice = std::min(row_last, lattice_rows - 1);
    const int span = last_lattice - row_first + 1;
    const int bands = span > 0 ? (span + kRefs - 1) / kRefs : 0;
    Passes plan{};
    plan.rows = bands > 0 ? (span + bands - 1) / bands : 1;
    const int shift_x = grid.max_x - q * (kRefs - 1), shift_y = grid.max_y - q * (plan.rows - 1);
    const bool last_row = clamp_y && row_last == grid.ny - 1;
    const auto add = [&](int ox, int oy, int bx, int nbands) {
        if (bx <= 0 || nbands <= 0) return;
        int* pass = plan.pass[plan.count++];
        pass[0] = ox;
        pass[1] = oy;
        pass[2] = bx;
        pass[3] = plan.blocks;
        plan.blocks += bx * nbands;
    };
    if (blocks_x > 0) add(0, q * row_first, blocks_x, bands);
    if (clamp_x) add(shift_x, q * row_first, 1, bands);
    if (last_row) {
        if (blocks_x > 0) add(0, shift_y, blocks_x, 1);
        if (clamp_x) add(shift_x, shift_y, 1, 1);
    }
    return plan;
}

}  // namespace

bool spatial_match_square_shape(const MatchGeometry& g, const RasterGrid& grid) {
    const int q = grid.step;
    return (q == 1 || q == 2 || q == 4 || q == 6 || q == 8) && g.block == 2 * q && g.channels == 1 &&
           g.range_hi < 0 && g.group >= 2 && g.group <= 32 && g.width >= g.block && g.height >= g.block &&
           g.width <= 0xffff && g.height <= 0xffff && grid.nx >= 1 && grid.ny >= 1;
}

bool spatial_match_square_takes(const MatchGeometry& g, const RasterGrid& grid) {
    if (!spatial_match_square_shape(g, grid)) return false;
    const int r = g.lo();
    const bool lanes = grid.step != 6 && g.group >= 4 && g.group <= 16 && (r == 1 || (r == 2 && g.group >= 8));
    return !lanes;
}

const Pipeline& SquareMatcher::pipeline(int q, int group) {
    const int wanted = group - 1;
    const int k = wanted <= 1 ? 1 : wanted <= 3 ? 3 : wanted <= 7 ? 7 : wanted <= 15 ? 15 : 31;
    const std::string id = "match_square_q" + std::to_string(q) + "_k" + std::to_string(k);
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(id);
    if (it == pipelines_.end()) it = pipelines_.emplace(id, Pipeline(device_, {id, sizeof(SquarePush), 0, {}})).first;
    return it->second;
}

void SquareMatcher::prepare(const MatchGeometry& g, const RasterGrid& grid) {
    if (spatial_match_square_shape(g, grid)) pipeline(grid.step, g.group);
}

void SquareMatcher::spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g,
                            const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                            VkDeviceAddress counts) {
    if (ref_count <= 0 || !spatial_match_square_takes(g, grid)) return;
    const int q = grid.step;
    const Passes plan = plan_square(q, grid, ref_begin, ref_count);
    if (plan.blocks == 0) return;
    SquarePush p{};
    p.source = plane;
    p.matches = matches;
    p.counts = counts;
    p.width = g.width;
    p.height = g.height;
    p.pitch = g.pitch;
    p.group_range = (g.group << 16) | g.bm_range;
    p.nx_ny = (grid.ny << 16) | grid.nx;
    p.step_rows = (plan.rows << 16) | q;
    p.max_x = grid.max_x;
    p.max_y = grid.max_y;
    p.ref_begin = ref_begin;
    p.ref_count = ref_count;
    std::copy(&plan.pass[0][0], &plan.pass[0][0] + 16, p.pass);
    const Pipeline& pipe = pipeline(q, g.group);
    const int x = std::min(plan.blocks, 65535);
    pipe.dispatch(commands, &p, static_cast<std::uint32_t>(x),
                  static_cast<std::uint32_t>((plan.blocks + x - 1) / x));
}

}  // namespace nss_vk
