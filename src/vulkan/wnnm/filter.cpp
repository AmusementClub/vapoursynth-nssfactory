// SPDX-License-Identifier: GPL-2.0-only
// nss_vk.WNNM (docs/vulkan-plan.md V7): parse_wnnm and parse_temporal with
// the "nss_vk" prefix around the shared group driver
// (common/group_driver.hpp) and the WNNM Gram kernels, as nss_cuda.WNNM.
// rclip guides matching only.
#include "frontend/validate.hpp"
#include "frontend/wnnm_args.hpp"
#include "vulkan/common/group_driver.hpp"
#include "vulkan/runtime/gpu_frames.hpp"
#include "vulkan/wnnm/kernels.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace nss_vk {
namespace {

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.WNNM.
    const auto temporal =
        nss::frontend::parse_temporal(vsapi, in, nss::kWnnmDefaultRadius, "WNNM", "nss_vk", kDefaultTemporalMode, kRollingCache);
    GroupFilterConfig config;
    config.name = "WNNM";
    config.model = nss::Model::WNNM;
    config.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    config.vi = *vsapi->getVideoInfo(config.node);
    const nss::WnnmParams p = nss::frontend::parse_wnnm(vsapi, in, config.vi, "nss_vk");
    int err = 0;
    config.guide = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        config.guide = nullptr;
    } else if (!nss::same_video(config.vi, *vsapi->getVideoInfo(config.guide))) {
        throw std::invalid_argument("nss_vk.WNNM: rclip must match clip");
    }
    nss::validate_group_planes(config.vi, p.sigma, p.block_size);
    config.radius = p.radius;
    config.mode = temporal.rolling ? GroupMode::Rolling : GroupMode::Legacy;
    config.rolling = temporal.params;
    config.backend = parse_backend_args(vsapi, in, "WNNM");
    // GPU clips run on VapourSynth's device (or are downloaded, take_clips).
    config.gpu_api = take_clips("WNNM", {{"clip", &config.node}, {"rclip", &config.guide}}, core, vsapi);
    config.device = config.gpu_api ? acquire_core_device(config.backend, "WNNM", core, vsapi, config.gpu_api)
                                   : acquire_device(config.backend.device_id, "WNNM", core, vsapi);
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
    // Spatial matching by the column or square matcher where it applies; the
    // square matcher also takes the centre frame of a temporal search.
    config.column_match = true;
    config.square_match = true;
    config.scratch_floats = [](const GroupPlane& g, bool) { return wnnm_scratch_floats(g.group); };
    auto kernels = std::make_shared<WnnmKernels>(config.device);
    // Tests: the emulated warp layer's kernels on a native device.
    const char* force = std::getenv("NSS_VK_FORCE_EMULATED_WARP");
    if (force && force[0] == '1') kernels->force_emulation(true);
    config.prepare = [kernels](const GroupPlane& g, bool) { kernels->prepare(g.group); };
    const bool residual = p.residual != 0, adaptive = p.adaptive != 0;
    config.launch = [kernels, residual, adaptive](const GroupLaunch& l) {
        WnnmGroupArgs args{};
        args.src = l.src;
        args.matches = l.matches;
        args.counts = l.counts;
        args.scratch = l.scratch;
        args.pitch = l.pitch;
        args.batch = l.batch;
        args.block = l.plane->block;
        args.group = l.plane->group;
        args.sigma = l.plane->sigma;
        args.residual = residual;
        args.adaptive = adaptive;
        args.fused = l.fused;
        kernels->filter(l.commands, args);
    };
    group_filter_install(std::move(config), out, core, vsapi);
}

}  // namespace

void register_wnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi, bool gpu_frames) {
    const std::string args = residency_args(signature(nss::frontend::kWnnmSignature), gpu_frames);
    vspapi->registerFunction("WNNM", args.c_str(), clip_return(gpu_frames), nss::checked_create<create, kDefaultMemoryLimitMb>,
                             nullptr, plugin);
}

}  // namespace nss_vk
