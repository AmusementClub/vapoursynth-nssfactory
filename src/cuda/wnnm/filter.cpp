// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.WNNM: parse_wnnm with the "nss_cuda" prefix (D14) around the
// shared group driver and the WNNM group kernel. rclip guides matching only.
#include "cuda/common/group_driver.hpp"
#include "cuda/wnnm/kernels.hpp"
#include "frontend/validate.hpp"
#include "frontend/wnnm_args.hpp"

#include <stdexcept>
#include <string>

namespace nss_cuda {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.WNNM (D14).
    const auto temporal = nss::frontend::parse_temporal(vsapi, in, nss::kWnnmDefaultRadius, "WNNM", "nss_cuda", kDefaultTemporalMode);
    GroupFilterConfig config;
    config.name = "WNNM";
    config.model = nss::Model::WNNM;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    const nss::WnnmParams p = nss::frontend::parse_wnnm(vsapi, in, config.vi, "nss_cuda");
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        config.guide = nullptr;
    } else if (!nss::same_video(config.vi, *vsapi->getVideoInfo(config.guide))) {
        throw std::invalid_argument("nss_cuda.WNNM: rclip must match clip");
    }
    nss::validate_group_planes(config.vi, p.sigma, p.block_size);
    config.radius = p.radius;
    config.mode = temporal.rolling ? GroupMode::Rolling : GroupMode::Legacy;
    config.rolling = temporal.params;
    config.backend = parse_backend_args(vsapi, in, "WNNM");
    config.device = acquire_device(config.backend.device_id, "WNNM", core, vsapi);
    for (int plane = 0; plane < config.vi.format.numPlanes; ++plane) {
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
    const int residual = p.residual, adaptive = p.adaptive;
    config.launch = [residual, adaptive](const GroupLaunch& l) {
        WnnmGroupArgs args{};
        args.src = l.src;
        args.pitch = l.pitch;
        args.matches = l.matches;
        args.counts = l.counts;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.sigma = l.plane->sigma;
        args.residual = residual;
        args.adaptive = adaptive;
        args.values = l.values;
        args.patches = l.patches;
        wnnm_filter_groups(args, l.stream);
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

}  // namespace

void register_wnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kWnnmSignature);
    vspapi->registerFunction("WNNM", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
