// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/validate.hpp"
#include "frontend/lssc_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct LsscData : nss::LsscParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    nss::Workspace ws;
};

const VSFrame* VS_CC lsscGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<LsscData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrame* src0 = frames_owned.getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);

    const int block = d->block_size;
    const int step = d->block_step;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(frames_owned.getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(frames_owned.getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp + y * dstride, srcp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
            }
            continue;
        }

        const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);
        const int denoise_n = nss::lssc_denoise_work_floats(pw, ph, block, step);
        const std::size_t need = plane_sz * 2 + static_cast<std::size_t>(denoise_n) + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = scratch + plane_sz;
        float* denoise_work = den + plane_sz;
        const float sigma = d->sigma[plane] / 255.f;
        nss::lssc_denoise_plane(srcp, pw, ph, sstride, num, den, pw, block, step, sigma, denoise_work, denoise_n);
        nss::aggregate_finish(outp, num, den, srcp, pw, ph, dstride, pw, sstride);
    }

    frames_owned.freeFrame(src0);
    return frames_owned.keep(dst);
}

void VS_CC lsscFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<LsscData*>(instanceData);
    d->node.reset();
    delete d;
}

void VS_CC lsscCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(out, "nss.LSSC: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return;
    }
    auto d = std::make_unique<LsscData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) {
        vsapi->mapSetError(out, msg);
        d->node.reset();
    };
    static_cast<nss::LsscParams&>(*d) = nss::frontend::parse_lssc(vsapi, in, d->vi, "nss");
    nss::validate_group_planes(d->vi, d->sigma, d->block_size);
    d->vi_out = d->vi;
    VSFilterDependency deps[1]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    LsscData* raw = d.get();
    VSNode* node =
        vsapi->createVideoFilter2("LSSC", &raw->vi_out, nss::checked_frame<lsscGetFrame>, lsscFree, fmParallel, deps, 1, raw, core);
    if (!node) {
        fail("nss.LSSC: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_lssc(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("LSSC", nss::frontend::kLsscSignature, "clip:vnode;", nss::checked_create<lsscCreate>, nullptr, plugin);
}
