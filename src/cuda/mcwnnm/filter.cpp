// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.MCWNNM: parse_mcwnnm with the "nss_cuda" prefix (D14) around the
// shared group driver in joint three-channel mode and the MCWNNM ADMM group
// kernel. rclip guides matching in the first outer round only.
#include "cuda/common/group_driver.hpp"
#include "cuda/mcwnnm/kernels.hpp"
#include "frontend/mcwnnm_args.hpp"
#include "frontend/validate.hpp"

#include <stdexcept>
#include <string>

namespace nss_cuda {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.MCWNNM (D14).
    GroupFilterConfig config;
    config.name = "MCWNNM";
    config.model = nss::Model::MCWNNM;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    const nss::McwnnmParams p = nss::frontend::parse_mcwnnm(vsapi, in, config.vi, "nss_cuda");
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        config.guide = nullptr;
    } else if (!nss::same_video(config.vi, *vsapi->getVideoInfo(config.guide))) {
        throw std::invalid_argument("nss_cuda.MCWNNM: rclip must match clip");
    }
    nss::validate_group_planes(config.vi, p.sigma, p.block_size);
    config.radius = p.radius;
    config.mode = GroupMode::Legacy;
    config.channels = 3;
    config.iters = p.iters;
    config.delta = p.delta;
    config.backend = parse_backend_args(vsapi, in, "MCWNNM");
    config.device = acquire_device(config.backend.device_id, "MCWNNM", core, vsapi);
    for (int plane = 0; plane < 3; ++plane) {
        GroupPlane& g = config.planes[plane];
        g.sigma = p.sigma[plane] / 255.f;
        g.active = p.sigma[plane] != 0.f;
        g.block = p.block_size;
        g.group = p.group_size;
        g.step = p.block_step;
        g.range = p.bm_range;
        g.ps_num = p.ps_num;
        g.ps_range = p.ps_range;
    }
    config.scratch_floats = [](const GroupPlane& g, bool) { return mcwnnm_scratch_floats(g.block, g.group); };
    McwnnmGroupArgs base{};
    for (int c = 0; c < 3; ++c) base.sigma[c] = p.sigma[c] / 255.f;
    base.residual = p.residual;
    base.adaptive = p.adaptive;
    base.admm_iter = p.admm_iter;
    base.rho = p.rho;
    base.mu = p.mu;
    config.launch = [base](const GroupLaunch& l) {
        McwnnmGroupArgs args = base;
        args.src = l.src;
        args.pitch = l.pitch;
        args.channel_step = l.channel_step;
        args.matches = l.matches;
        args.counts = l.counts;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.values = l.values;
        args.scratch = l.scratch;
        args.patches = l.patches;
        mcwnnm_filter_groups(args, l.stream);
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

void VS_CC create_temporal(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    nss::frontend::create_temporal(create, in, out, core, vsapi, nss::kWnnmDefaultRadius, "MCWNNM", "nss_cuda",
                                   kDefaultTemporalMode, kRollingCache);
}

}  // namespace

void register_mcwnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kMcwnnmSignature);
    vspapi->registerFunction("MCWNNM", args.c_str(), "clip:vnode;", nss::checked_create<create_temporal, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
