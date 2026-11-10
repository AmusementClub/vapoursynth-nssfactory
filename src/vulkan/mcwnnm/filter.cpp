// SPDX-License-Identifier: GPL-2.0-only
// nss_vk.MCWNNM (docs/vulkan-plan.md V8): parse_mcwnnm and parse_temporal
// with the "nss_vk" prefix around the shared group driver
// (common/group_driver.hpp) with one joint unit of the three channels
// (matched on all three) and its outer rounds, and the MCWNNM kernels, as
// nss_cuda.MCWNNM. rclip guides matching only.
#include "frontend/mcwnnm_args.hpp"
#include "frontend/validate.hpp"
#include "vulkan/common/group_driver.hpp"
#include "vulkan/mcwnnm/kernels.hpp"
#include "vulkan/runtime/gpu_frames.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace nss_vk {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.MCWNNM.
    const auto temporal = nss::frontend::parse_temporal(vsapi, in, nss::kWnnmDefaultRadius, "MCWNNM", "nss_vk",
                                                        kDefaultTemporalMode, kRollingCache);
    GroupFilterConfig config;
    config.name = "MCWNNM";
    config.model = nss::Model::MCWNNM;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    const nss::McwnnmParams p = nss::frontend::parse_mcwnnm(vsapi, in, config.vi, "nss_vk");
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        config.guide = nullptr;
    } else if (!nss::same_video(config.vi, *vsapi->getVideoInfo(config.guide))) {
        throw std::invalid_argument("nss_vk.MCWNNM: rclip must match clip");
    }
    nss::validate_group_planes(config.vi, p.sigma, p.block_size);
    config.radius = p.radius;
    config.mode = temporal.rolling ? GroupMode::Rolling : GroupMode::Legacy;
    config.rolling = temporal.params;
    config.channels = 3;  // one joint unit, matched on all three channels
    config.iters = p.iters;
    config.delta = p.delta;
    config.backend = parse_backend_args(vsapi, in, "MCWNNM");
    // GPU clips run on VapourSynth's device (or are downloaded, take_clips).
    config.gpu_api = take_clips("MCWNNM", {{"clip", &config.node}, {"rclip", &config.guide}}, core, vsapi);
    config.device = config.gpu_api ? acquire_core_device(config.backend, "MCWNNM", core, vsapi, config.gpu_api)
                                   : acquire_device(config.backend.device_id, "MCWNNM", core, vsapi);
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
    config.scratch_floats = [](const GroupPlane& g, bool) { return mcwnnm_scratch_floats(g.group); };
    McwnnmModel model;
    for (int c = 0; c < 3; ++c) model.sigma[c] = p.sigma[c] / 255.f;
    model.residual = p.residual != 0;
    model.adaptive = p.adaptive != 0;
    model.admm_iter = p.admm_iter;
    model.rho = p.rho;
    model.mu = p.mu;
    auto kernels = std::make_shared<McwnnmKernels>(config.device, model);
    // Tests: the emulated warp layer's kernels on a native device.
    const char* force = std::getenv("NSS_VK_FORCE_EMULATED_WARP");
    if (force && force[0] == '1') kernels->force_emulation(true);
    config.prepare = [kernels](const GroupPlane& g, bool) { kernels->prepare(g.group); };
    config.launch = [kernels](const GroupLaunch& l) {
        McwnnmGroupArgs args{};
        args.src = l.src;
        args.matches = l.matches;
        args.counts = l.counts;
        args.scratch = l.scratch;
        args.channel_step = l.channel_step;
        args.pitch = l.pitch;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.fused = l.fused;
        kernels->filter(l.commands, args);
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

}  // namespace

void register_mcwnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi, bool gpu_frames) {
    const std::string args = residency_args(signature(nss::frontend::kMcwnnmSignature), gpu_frames);
    vspapi->registerFunction("MCWNNM", args.c_str(), clip_return(gpu_frames), nss::checked_create<create, kDefaultMemoryLimitMb>,
                             nullptr, plugin);
}

}  // namespace nss_vk
