// SPDX-License-Identifier: GPL-2.0-only
// Device block matching for the patch-group filters (docs/vulkan-plan.md V3):
// the lane kernels of shaders/common/match_lane.comp and, for the searches
// they do not take, the block fallback of match_block.comp, with nss_cuda's
// semantics (src/cuda/common/match.hpp: reference first, then (finite,
// distance, t, y, x); distances summed per lane then across lanes, so near
// ties may resolve differently from the CPU). Geometry is gpuhost's.
#pragma once

#include "gpuhost/geometry.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/pipeline.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace nss_vk {

using nss::gpuhost::MatchGeometry;
using nss::gpuhost::RasterGrid;
using nss::gpuhost::make_raster_grid;

// A match as the shaders write it (as nss_cuda's DeviceMatch).
struct DeviceMatch {
    int x;
    int y;
    int t;
    float dist;
};

// The predictive search's window (nss_cuda's TemporalWindow, with a device
// table of ntemp plane addresses instead of a pointer array).
struct TemporalWindow {
    VkDeviceAddress frames;  // ntemp VkDeviceAddress of the planes (same pitch)
    int ntemp;
    int t0;
    int radius;
    int valid_begin;  // first usable index (clip boundary)
    int valid_end;    // one past the last usable index (< 0: ntemp)
    int ps_num;
    int ps_range;
};

// Lanes per reference the lane kernels use for this search (0: none), as
// nss_cuda's lane_group: the fewest of 8, 16 and 32 that hold the matches and
// the layer's best; block 32 takes 32.
int lane_group(const MatchGeometry& g, int ps_num);

class Matcher {
public:
    explicit Matcher(std::shared_ptr<Device> device) : device_(std::move(device)) {}

    // Whether this build has kernels for the search: the lane kernels
    // (lane_group: blocks 1-16 and 32 for one channel, 1-8 for three, groups
    // up to 33 incl. the reference), else the block fallback (1 to 3
    // channels, any block, groups and ps_num up to 256).
    bool supports(const MatchGeometry& g, int ps_num) const;
    // Whether the lane kernels serve the search (else the block fallback).
    bool lanes(const MatchGeometry& g, int ps_num) const;
    // Creates the pipeline a search will use (at filter creation).
    void prepare(const MatchGeometry& g, bool predictive, int ps_num);

    // References [ref_begin, ref_begin + ref_count) of the grid; `matches`
    // holds ref_count * g.group DeviceMatch (group-strided), `counts`
    // ref_count ints. Recorded into `commands` (compute).
    void spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g, const RasterGrid& grid,
                 int ref_begin, int ref_count, VkDeviceAddress matches, VkDeviceAddress counts);
    void predictive(VkCommandBuffer commands, const MatchGeometry& g, const TemporalWindow& window,
                    const RasterGrid& grid, int ref_begin, int ref_count, VkDeviceAddress matches,
                    VkDeviceAddress counts);
    // Whether a search of this shape runs on the emulated warp layer here (D23;
    // never the block fallback, which has no subgroup operations).
    bool emulated(const MatchGeometry& g, int ps_num) const;
    // Tests: the emulated layer on any device (it gives the native bits).
    void force_emulation(bool on) noexcept { force_emulation_ = on; }
    // Tests: rows of at most `x` workgroups (default 65535), so that small
    // searches take the multi-row dispatch.
    void limit_groups_x(int x) noexcept { max_groups_x_ = x; }

private:
    const Pipeline& pipeline(const MatchGeometry& g, bool predictive, int ps_num);
    void dispatch(VkCommandBuffer commands, const MatchGeometry& g, bool predictive, int ps_num, const void* push,
                  int ref_count);

    std::shared_ptr<Device> device_;
    bool force_emulation_ = false;
    // Dispatches wider than this go in rows (common/warp.glsl's w_workgroup,
    // match_block.comp's main): 65535 is the smallest
    // maxComputeWorkGroupCount Vulkan allows.
    int max_groups_x_ = 65535;
    std::mutex mutex_;
    std::map<std::string, Pipeline> pipelines_;
};

// Square matching for BM3D's block = 2 * step shapes. The quarter-distance
// arithmetic differs from the lane kernels in the last bits, so near ties may
// resolve differently. spatial() declines a shape that admission rejects and
// leaves it to Matcher's existing path.
bool spatial_match_square_shape(const MatchGeometry& g, const RasterGrid& grid);
bool spatial_match_square_takes(const MatchGeometry& g, const RasterGrid& grid);

class SquareMatcher {
public:
    explicit SquareMatcher(std::shared_ptr<Device> device) : device_(std::move(device)) {}
    void prepare(const MatchGeometry& g, const RasterGrid& grid);
    void spatial(VkCommandBuffer commands, VkDeviceAddress plane, const MatchGeometry& g, const RasterGrid& grid,
                 int ref_begin, int ref_count, VkDeviceAddress matches, VkDeviceAddress counts);

private:
    const Pipeline& pipeline(int q, int group);
    std::shared_ptr<Device> device_;
    std::mutex mutex_;
    std::map<std::string, Pipeline> pipelines_;
};

}  // namespace nss_vk
