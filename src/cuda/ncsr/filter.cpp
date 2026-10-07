// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NCSR: parse_ncsr with the "nss_cuda" prefix (D14) around the
// shared group driver (outer rounds with delta relaxation) and the NCSR group
// kernel. rclip guides matching in the first round only.
#include "cuda/common/gram_group.hpp"
#include "cuda/common/group_driver.hpp"
#include "cuda/ncsr/kernels.hpp"
#include "frontend/validate.hpp"
#include "frontend/ncsr_args.hpp"

#include <stdexcept>
#include <string>

namespace nss_cuda {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.NCSR (D14).
    const auto temporal = nss::frontend::parse_temporal(vsapi, in, nss::kWnnmDefaultRadius, "NCSR", "nss_cuda", kDefaultTemporalMode, kRollingCache);
    GroupFilterConfig config;
    config.name = "NCSR";
    config.model = nss::Model::NCSR;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    const nss::NcsrParams p = nss::frontend::parse_ncsr(vsapi, in, config.vi, "nss_cuda");
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        config.guide = nullptr;
    } else if (!nss::same_video(config.vi, *vsapi->getVideoInfo(config.guide))) {
        throw std::invalid_argument("nss_cuda.NCSR: rclip must match clip");
    }
    nss::validate_group_planes(config.vi, p.sigma, p.block_size);
    config.radius = p.radius;
    // The driver's rolling mode covers a single round; more rounds aggregate
    // the fat intermediate with nss_cuda.VAggregate instead.
    const bool device_rolling = temporal.rolling && p.iters == 1;
    config.mode = device_rolling ? GroupMode::Rolling : GroupMode::Legacy;
    config.rolling = temporal.params;
    config.iters = p.iters;
    config.delta = p.delta;
    config.backend = parse_backend_args(vsapi, in, "NCSR");
    config.device = acquire_device(config.backend.device_id, "NCSR", core, vsapi);
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
        g.fused = true;
    }
    config.launch = [](const GroupLaunch& l) {
        NcsrGroupArgs args{};
        args.src = l.src;
        args.pitch = l.pitch;
        args.matches = l.matches;
        args.counts = l.counts;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.sigma = l.plane->sigma;
        args.values = l.values;
        args.patches = l.patches;
        args.fused = l.fused;
        args.scratch = l.scratch;
        ncsr_filter_groups(args, l.stream);
    };
    config.scratch_floats = [](const GroupPlane& g, bool) { return gram_scratch_floats(g.group); };
    group_filter_install(std::move(config), out, core, vsapi);
    if (temporal.rolling && !device_rolling) {
        nss::frontend::aggregate_rolling(vsapi, in, out, core, nss::kWnnmDefaultRadius, "NCSR", "nss_cuda");
    }
}

}  // namespace

void register_ncsr(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNcsrSignature);
    vspapi->registerFunction("NCSR", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
