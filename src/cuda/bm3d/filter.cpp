// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM3D. Arguments, defaults and errors come from the shared
// frontend (parse_bm3d / parse_temporal_mode / parse_rolling with the
// "nss_cuda" prefix, D14); the node itself is the shared group driver
// (common/group_driver.hpp) around the BM3D group kernel. `ref` is the guide
// clip: it drives matching and supplies the Wiener reference cube.
#include "cuda/bm3d/kernels.hpp"
#include "cuda/common/group_driver.hpp"
#include "frontend/bm3d_args.hpp"
#include "frontend/temporal_args.hpp"
#include "frontend/validate.hpp"

#include <string>

namespace nss_cuda {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order as nss.BM3D so shared errors match (D14).
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    const bool rolling =
        nss::frontend::parse_temporal_mode(vsapi, in, radius, "BM3D", "nss_cuda", kDefaultTemporalMode) ==
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
    const nss::Bm3dParams p = nss::frontend::parse_bm3d(vsapi, in, config.vi, ref_vi, "nss_cuda");
    if (rolling) config.rolling = nss::frontend::parse_rolling(vsapi, in, p.radius, "BM3D", "nss_cuda");
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
        g.fused = bm3d_fuses(g.block, g.group, config.guide != nullptr);
    }
    // Only the shapes whose Wiener stage keeps its reference cube in device memory need scratch.
    config.scratch_floats = [](const GroupPlane& g, bool guide) { return bm3d_scratch_floats(g.block, g.group, guide); };
    config.launch = [](const GroupLaunch& l) {
        Bm3dGroupArgs args{};
        args.src = l.src;
        args.ref = l.guide;
        args.pitch = l.pitch;
        args.matches = l.matches;
        args.counts = l.counts;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.sigma = l.plane->sigma;
        args.values = l.values;
        args.ref_cube = l.scratch;
        args.patches = l.patches;
        args.fused = l.fused;
        bm3d_filter_groups(args, l.stream);
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

}  // namespace

void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kBm3dSignature);
    vspapi->registerFunction("BM3D", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
