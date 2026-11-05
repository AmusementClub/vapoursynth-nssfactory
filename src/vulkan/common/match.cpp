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
bool ps2_fast(const MatchGeometry& g, int ps_num) {
    return ps_num == 2 && g.block == 16 && g.channels == 1 && lane_group(g, ps_num) == 16;
}
bool pair2_fast(const MatchGeometry& g, int ps_num) {
    return ps_num == 2 && g.block == 8 && g.channels == 1 && lane_group(g, ps_num) == 8;
}

// The window kernel (match_window.comp): four 32-invocation subgroups a
// workgroup, a reference each, the window in tiles of at most 33 x 33
// candidates. Modes as nss_cuda's match_window.cu: tree (blocks up to 8,
// groups up to 33: the lane kernels' arithmetic), chain (groups 34 to 64:
// match_block.comp's), split (blocks 12 and 16, groups up to 33).
struct WindowPush {
    VkDeviceAddress source, matches, counts;
    std::int32_t width, height, pitch, group;
    std::int32_t lo, hi, nx, step;
    std::int32_t max_x, max_y, ref_begin, ref_count;
    std::int32_t frame, tile_w, tile_h, tile_cols;
};
static_assert(sizeof(WindowPush) == 88, "matches the push block of shaders/common/match_window.comp");
constexpr int kWindowWarps = 4, kWindowTile = 33, kWindowMaxSide = 129;
enum class WindowMode { none, tree, chain, split };

// The mode of a search, among the variants this build has (CMakeLists.txt).
WindowMode window_mode(const MatchGeometry& g) {
    if (g.channels != 1 || g.group < 2 || g.group > 64) return WindowMode::none;
    if (g.group <= 33 && (g.block == 4 || g.block == 8)) return WindowMode::tree;
    if (g.group > 33 && (g.block == 2 || g.block == 4 || g.block == 8)) return WindowMode::chain;
    if ((g.block == 12 || g.block == 16) && g.group <= 33) return WindowMode::split;
    return WindowMode::none;
}

// The smallest window side where the kernel beat the kernel it replaces on
// NVIDIA (resident 1080p gray BM3D, RTX 5080, against the plugin without it):
// - tree: block 8 from side 15, block 4 from 25 (below 0.71-0.91x; 1.2-1.8x
//   from them up to side 97), block 2 0.60-0.95x up to side 33; blocks 3, 5,
//   6 and 7 (no ported filter takes them) have no variant;
// - chain: 1.7-3.3x at every side measured (9 to 65, blocks 4 and 8);
// - split: blocks 12 and 16, both layouts, 0.82-0.87x at sides 9 to 13,
//   0.99-1.03x at 19, 1.03-1.09x at 25, 1.12-1.32x at 33 and 49.
// NSS_VK_WINDOW_ALL (a measurement build) takes every side.
int window_min_side(WindowMode m, int block) {
#ifdef NSS_VK_WINDOW_ALL
    (void)m, (void)block;
    return 1;
#else
    constexpr int kOff = 1 << 30;
    switch (m) {
    case WindowMode::tree: return block == 8 ? 15 : block == 4 ? 25 : kOff;
    case WindowMode::chain: return 1;
    case WindowMode::split: return 25;
    default: return kOff;
    }
#endif
}
// Seeding a temporal search pays from a larger window (the extra dispatch and
// barrier): four more than the spatial minimum; tree mode only (the split
// layout follows the predictive lane kernels' lane count, which ps_num can
// raise).
constexpr int kWindowSeedMargin = 4;

// The tile side for a window side: the fewest columns of at most kWindowTile,
// then more while a workgroup's regions (kWindowWarps of (tile + B - 1) rows
// of pitch (tile + B - 1) | 1 floats) exceed the device's shared memory (a
// 32 KiB device at blocks 12 and 16). Tiles split the search, not its result:
// the matches do not depend on them.
int window_tile(int side, int block, std::uint32_t shared) {
    for (int cols = (side + kWindowTile - 1) / kWindowTile;; ++cols) {
        const int tile = (side + cols - 1) / cols;
        const long long span = tile + block - 1;
        if (tile == 1 || kWindowWarps * span * (span | 1) * static_cast<long long>(sizeof(float)) <= shared) return tile;
    }
}

// Rows an item takes for a tile side, as nss_cuda's match_window.cu: the
// fewest instructions by a rough count an invocation makes.
int window_rows(int tile, int block, int lanes_per, int max_ny) {
    const int per_round = 32 / lanes_per, cols = block / lanes_per;
    int best = 2;
    long long best_cost = -1;
    for (int ny = 2; ny <= max_ny; ++ny) {
        const long long rounds = (static_cast<long long>(tile) * ((tile + ny - 1) / ny) + per_round - 1) / per_round;
        const long long cost = rounds * (ny * 2 * block * cols + (ny + block - 1) * cols + 40);
        if (best_cost < 0 || cost < best_cost) best = ny, best_cost = cost;
    }
    return best;
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

const Pipeline& Matcher::pipeline(const MatchGeometry& g, bool predictive, int ps_num, bool seeded, bool pair2) {
    if (!supports(g, ps_num)) throw std::logic_error("nss_vk: no matcher for this search");
    const bool lane = lanes(g, ps_num);
    const bool emulate = emulated(g, ps_num);
    const bool paired = predictive && pair2 && pair2_fast(g, ps_num);
    const std::string id =
        lane ? "match_lane_b" + std::to_string(g.block) + "_l" + std::to_string(lane_group(g, ps_num)) + "_c" +
                   std::to_string(g.channels) + (predictive ? "_p" : "_s") + (emulate ? "_e" : "_n") +
                   (predictive && ps_num == 1 ? "_ps1" : paired ? "_pair2" : predictive && ps2_fast(g, ps_num) ? "_ps2" : "") +
                   (seeded ? "_seed" : "")
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

void Matcher::prepare(const MatchGeometry& g, bool predictive, int ps_num, bool seeded, int radius) {
    if (!predictive && window(g)) {
        window_pipeline(g, false);
        return;
    }
    if (predictive && seeded && window_seeds(g, ps_num)) window_pipeline(g, true);
    pipeline(g, predictive, ps_num, seeded, predictive && radius == 1 && pair2_fast(g, ps_num));
}

bool Matcher::window(const MatchGeometry& g) const {
    const WindowMode m = window_mode(g);
    const int side = g.lo() + g.hi() + 1;
    const DeviceCaps& caps = device_->caps();
    return m != WindowMode::none && !force_emulation_ && !disable_window_ &&
           side >= window_min_side(m, g.block) && side <= kWindowMaxSide && g.width >= g.block &&
           g.height >= g.block && g.width <= kLaneMaxCoordinate && g.height <= kLaneMaxCoordinate &&
           static_cast<long long>(g.pitch) * g.height < (1LL << 31) && device_->support().native_lanes >= 32 &&
           caps.min_subgroup_size <= 32 && caps.max_subgroup_size >= 32;
}

bool Matcher::window_seeds(const MatchGeometry& g, int ps_num) const {
    const WindowMode m = window_mode(g);
    return m == WindowMode::tree && window(g) && lanes(g, ps_num) &&
           g.lo() + g.hi() + 1 >= window_min_side(m, g.block) + kWindowSeedMargin;
}

const Pipeline& Matcher::window_pipeline(const MatchGeometry& g, bool temporal) {
    const WindowMode m = window_mode(g);
    const int tile = window_tile(g.lo() + g.hi() + 1, g.block, device_->caps().shared_memory);
    const bool split = m == WindowMode::split;
    const std::string id = "match_window_" +
                           std::string(m == WindowMode::tree ? "tree" : m == WindowMode::chain ? "chain" : "split") +
                           "_b" + std::to_string(g.block) +
                           (split ? std::string(g.group <= 9 ? "_pair" : "_cols") : std::string()) + "_ny" +
                           std::to_string(window_rows(tile, g.block, split ? 2 : 1, split ? 4 : 6)) +
                           (temporal ? "_t" : "");
    // The shared region is sized for the tile (specialization constant 0).
    const int span = tile + g.block - 1;
    const auto region_floats = static_cast<std::uint32_t>(span * (span | 1));
    const std::string key = id + "_r" + std::to_string(region_floats);
    std::lock_guard lock(mutex_);
    auto it = pipelines_.find(key);
    if (it == pipelines_.end()) {
        it = pipelines_.emplace(key, Pipeline(device_, {id, sizeof(WindowPush), 32, {region_floats}})).first;
    }
    return it->second;
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
                       int ref_count, bool seeded, bool pair2) {
    if (static_cast<long long>(ref_count) * g.group >= (1LL << 31)) {
        throw std::logic_error("nss_vk: match batch beyond 32-bit indices");
    }
    const Pipeline& p = pipeline(g, predictive, ps_num, seeded, pair2);
    // Lanes: 32 / L references a warp. The PS2 and paired variants use eight
    // warps, the other lane kernels four. Block: one reference a workgroup.
    const int warps = predictive && (ps2_fast(g, ps_num) || pair2) ? 8 : kLaneWarps;
    const int per_group = lanes(g, ps_num) ? warps * (32 / lane_group(g, ps_num)) : 1;
    const int groups = (ref_count + per_group - 1) / per_group;
    const int x = std::min(groups, max_groups_x_);
    p.dispatch(commands, push, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>((groups + x - 1) / x));
}

void Matcher::spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g, const RasterGrid& grid,
                      int ref_begin, int ref_count, VkDeviceAddress matches, VkDeviceAddress counts) {
    if (ref_count <= 0) return;
    if (window(g)) {
        dispatch_window(commands, plane, -1, g, grid, ref_begin, ref_count, matches, counts);
        return;
    }
    Push p = push_of(g, grid, ref_begin, ref_count, matches, counts);
    p.source = plane;
    dispatch(commands, g, false, 0, &p, ref_count);
}

void Matcher::dispatch_window(VkCommandBuffer commands, VkDeviceAddress source, int frame, const MatchGeometry& g,
                              const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                              VkDeviceAddress counts) {
    if (static_cast<long long>(ref_count) * g.group >= (1LL << 31)) {
        throw std::logic_error("nss_vk: match batch beyond 32-bit indices");
    }
    const int side = g.lo() + g.hi() + 1, tile = window_tile(side, g.block, device_->caps().shared_memory);
    const WindowPush p{source,     matches,   counts,    g.width,   g.height,           g.pitch,
                       g.group,    g.lo(),    g.hi(),    grid.nx,   grid.step,          grid.max_x,
                       grid.max_y, ref_begin, ref_count, std::max(frame, 0), tile,     tile,
                       (side + tile - 1) / tile};
    const int groups = (ref_count + kWindowWarps - 1) / kWindowWarps;
    const int x = std::min(groups, max_groups_x_);
    window_pipeline(g, frame >= 0).dispatch(commands, &p, static_cast<std::uint32_t>(x),
                                            static_cast<std::uint32_t>((groups + x - 1) / x));
}

bool Matcher::predictive_window(VkCommandBuffer commands, const MatchGeometry& g, const TemporalWindow& window,
                                const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                                VkDeviceAddress counts) {
    if (!window_seeds(g, window.ps_num)) return false;
    if (ref_count <= 0) return true;
    dispatch_window(commands, window.frames, window.t0, g, grid, ref_begin, ref_count, matches, counts);
    compute_barrier(*device_, commands);
    predictive(commands, g, window, grid, ref_begin, ref_count, matches, counts, true);
    return true;
}

void Matcher::predictive(VkCommandBuffer commands, const MatchGeometry& g, const TemporalWindow& window,
                         const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                         VkDeviceAddress counts, bool seeded) {
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
    dispatch(commands, g, true, window.ps_num, &p, ref_count, seeded, window.radius == 1 && pair2_fast(g, window.ps_num));
}

}  // namespace nss_vk
