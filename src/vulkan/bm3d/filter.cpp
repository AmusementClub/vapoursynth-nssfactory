// SPDX-License-Identifier: GPL-2.0-only
// nss_vk.BM3D (docs/vulkan-plan.md V5, V6). Arguments, defaults and errors
// come from the shared frontend (parse_bm3d / parse_temporal_mode /
// parse_rolling with the "nss_vk" prefix); the node is the shared group
// driver (common/group_driver.hpp) around the BM3D group kernels, as
// nss_cuda.BM3D. `ref` is the guide clip: it drives matching and supplies the
// Wiener reference cube.
#include "frontend/bm3d_args.hpp"
#include "frontend/temporal_args.hpp"
#include "frontend/validate.hpp"
#include "vulkan/bm3d/kernels.hpp"
#include "vulkan/common/group_driver.hpp"

#include <memory>
#include <string>

namespace nss_vk {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order as nss.BM3D so shared errors match.
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    const bool rolling =
        nss::frontend::parse_temporal_mode(vsapi, in, radius, "BM3D", "nss_vk", kDefaultTemporalMode) ==
        nss::TemporalMode::Rolling;
    GroupFilterConfig config;
    config.name = "BM3D";
    config.model = nss::Model::BM3D;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "ref", 0, &err);
    if (err) config.guide = nullptr;
    const VSVideoInfo* ref_vi = config.guide ? vsapi->getVideoInfo(config.guide) : nullptr;
    const nss::Bm3dParams p = nss::frontend::parse_bm3d(vsapi, in, config.vi, ref_vi, "nss_vk");
    if (rolling) config.rolling = nss::frontend::parse_rolling(vsapi, in, p.radius, "BM3D", "nss_vk", kRollingCache);
    // final: the driver keeps the first stage's estimate on the device for
    // spatial and rolling output. The legacy intermediate chains two nodes.
    if (p.final && p.radius > 0 && !rolling) {
        (void)parse_backend_args(vsapi, in, "BM3D");
        nss::frontend::bm3d_two_nodes(vsapi, in, out, core, "nss_vk");
        return;
    }
    config.radius = p.radius;
    config.mode = rolling ? GroupMode::Rolling : GroupMode::Legacy;
    config.backend = parse_backend_args(vsapi, in, "BM3D");
    config.device = acquire_device(config.backend.device_id, "BM3D", core, vsapi);
    for (int plane = 0; plane < config.vi.format.numPlanes; ++plane) {
        GroupPlane& g = config.planes[plane];
        g.sigma = p.sigma[plane];
        g.active = g.sigma != 0.f;
        g.block = p.block_size[plane];
        g.group = p.group_size[plane];
        g.step = p.block_step[plane];
        g.range = p.bm_range[plane];
        g.ps_num = p.ps_num[plane];
        g.ps_range = p.ps_range[plane];
        if (!p.final) continue;
        GroupPlane& b = config.basic[plane];
        b = g;
        b.sigma = p.sigma_basic[plane];
        b.active = b.sigma != 0.f;
        b.block = p.block_size_basic[plane];
        b.group = p.group_size_basic[plane];
        b.step = p.block_step_basic[plane];
        b.ps_num = p.ps_num_basic[plane];
    }
    config.two_stage = p.final;
    config.scratch_floats = [](const GroupPlane& g, bool wiener) { return bm3d_scratch_floats(g.block, g.group, wiener); };
    if (p.chroma) {
        // CBM3D: one unit of three planes, groups matched on plane 0. Plane 0
        // carries the geometry, which the frontend made the same for all.
        config.channels = 3;
        config.shared_match = true;
    }
    auto kernels = std::make_shared<Bm3dKernels>(config.device);
    config.prepare = [kernels](const GroupPlane& g, bool wiener) { kernels->prepare(g.block, g.group, wiener); };
    // One plane, or with chroma each filtered plane of the unit in turn: its
    // own sigma (l.plane[c] is planes[c]) and sums.
    config.launch = [kernels](const GroupLaunch& l) {
        Bm3dGroupArgs args{};
        args.src = l.src;
        args.ref = l.guide;
        args.pitch = l.pitch;
        args.matches = l.matches;
        args.counts = l.counts;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.scratch = l.scratch;
        for (int c = 0; c < l.channels; ++c) {
            if (l.channels > 1 && !l.channel_active[c]) continue;
            args.plane_offset = c * l.channel_step;
            args.sigma = l.plane[c].sigma;
            args.fused = l.fused;
            args.fused.num += static_cast<VkDeviceAddress>(c) * static_cast<VkDeviceAddress>(l.fused.channel_step) * 8;
            args.fused.den += static_cast<VkDeviceAddress>(c) * static_cast<VkDeviceAddress>(l.fused.channel_step) * 8;
            kernels->filter(l.commands, args);
            // Channels share the scratch.
            if (c + 1 < l.channels && args.scratch) compute_barrier(kernels->device(), l.commands);
        }
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

}  // namespace

void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kBm3dSignature);
    vspapi->registerFunction("BM3D", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>,
                             nullptr, plugin);
}

}  // namespace nss_vk
