// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for patch-group filters (BM3D, WNNM, MCWNNM, NCSR): the
// D19 temporal driver plus everything else a filter node needs around its
// group kernel. A filter parses its arguments, fills a GroupFilterConfig and
// supplies `launch`, which turns one batch of matched groups into filtered
// patches and aggregation metadata; the driver owns
//   - memory planning against memory_limit_mb (slots, batches, default streams),
//   - uploads/downloads through pinned staging,
//   - matching (spatial or predictive, on the guide clip when present),
//   - deterministic ordered aggregation,
//   - the three output modes: spatial, legacy fat intermediate, rolling.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "cuda/runtime/context.hpp"
#include "frontend/ownership.hpp"
#include "nss/contracts.hpp"
#include "nss/params/temporal.hpp"

#include <VapourSynth4.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace nss_cuda {

enum class GroupMode { Spatial, Legacy, Rolling };

struct GroupPlane {
    // Filled by the filter:
    bool active = false;  // false copies the plane (temporal: identity slices)
    int block = 0, group = 0, step = 0, range = 0, ps_num = 0, ps_range = 0;
    float sigma = 0.f;    // filter-defined scale, passed through to launch
    // The filter's kernel aggregates this plane's groups itself, into
    // GroupLaunch::fused; the driver then keeps no values, patch records or
    // float sums for it. Single-channel, single-round filters only, and
    // spatial output only: the driver clears the flag for radius > 0.
    bool fused = false;
    // Filled by the driver:
    int width = 0, height = 0;
    std::size_t floats = 0;
    RasterGrid grid{};
    int batch = 0;
};

// One batch of matched groups. Patch id = group index * plane->group + j.
struct GroupLaunch {
    const float* const* src;    // device array: window slot t -> source plane
    const float* const* guide;  // same for the guide clip, nullptr without one
    int pitch;                  // floats
    const DeviceMatch* matches; // batch * plane->group (group-strided)
    const int* counts;          // batch
    int batch;
    const GroupPlane* plane;
    int channels;               // 1, or 3 for joint groups
    long long channel_step;     // floats from channel c to c + 1 inside a src/guide frame
    float* values;              // channels * batch * group * block^2, channel-major: filtered patches
    float* scratch;             // batch * scratch_floats(plane) extra workspace, or nullptr
    AggregatePatch* patches;    // batch * group: position, slice = match t, weight; unused slots slice -1
    // plane->fused: the accumulators of the plane (slice = match t); values
    // and patches are then null.
    FixedTarget fused{};
    cudaStream_t stream;
};

struct GroupFilterConfig {
    std::string name;          // "BM3D": node name and error prefix nss_cuda.<name>
    nss::Model model = nss::Model::BM3D;
    nss::NodeRef node, guide;  // guide: matching clip (ref / rclip), may be null
    VSVideoInfo vi{}, vi_out{};
    int radius = 0;
    GroupMode mode = GroupMode::Legacy;  // for radius > 0; radius 0 is always spatial
    nss::RollingParams rolling{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    DeviceInfo device;
    GroupPlane planes[3];
    // Joint groups (MCWNNM): with channels == 3 the three planes are matched
    // and filtered as one unit using planes[0]'s geometry; planes[c].active
    // then only says whether channel c is written or copied from the source.
    int channels = 1;
    // Outer rounds (MCWNNM, NCSR): every round after the first relaxes the
    // estimate toward the input by delta, re-matches on the estimate (the
    // guide only drives the first round) and filters the estimate again.
    // Rolling mode requires channels == 1 and iters == 1.
    int iters = 1;
    float delta = 0.f;
    // Extra device floats per reference group (second argument: guide present).
    std::function<std::size_t(const GroupPlane&, bool)> scratch_floats;
    std::function<void(const GroupLaunch&)> launch;
};

// Plans memory, builds the stream slots and creates the VapourSynth node.
void group_filter_install(GroupFilterConfig&& config, VSMap* out, VSCore* core, const VSAPI* vsapi);

}  // namespace nss_cuda
