// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/validate.hpp"
#include "frontend/pnlm_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_pnlm.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <cstring>
#include <memory>
#include <vector>

namespace {

struct PnlmData : nss::PnlmParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    VSVideoInfo vi{};
    nss::Workspace ws;
};

const VSFrame* VS_CC pnlmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<PnlmData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    const int first = nss::frontend::pnlm_first_frame(n, d->d);
    const int last = nss::frontend::pnlm_last_frame(n, d->d, d->vi.numFrames);
    if (activationReason == arInitial) {
        for (int i = first; i <= last; ++i) vsapi->requestFrameFilter(i, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    std::vector<const VSFrame*> window;
    for (int i = first; i <= last; ++i) window.push_back(frames_owned.getFrameFilter(i, d->node, frameCtx));
    const VSFrame* src = window[static_cast<std::size_t>(n - first)];
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    std::vector<const float*> planes(window.size());
    std::vector<std::ptrdiff_t> strides(window.size());
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int w = nss::plane_width(d->vi, plane), h = nss::plane_height(d->vi, plane);
        float* out = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const std::ptrdiff_t out_stride = frames_owned.getStride(dst, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
        if (d->sigma[plane] == 0.f) {  // unprocessed: the source plane
            const float* in = reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane));
            const std::ptrdiff_t in_stride = frames_owned.getStride(src, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
            for (int y = 0; y < h; ++y)
                std::memcpy(out + y * out_stride, in + y * in_stride, static_cast<std::size_t>(w) * sizeof(float));
            continue;
        }
        for (std::size_t t = 0; t < window.size(); ++t) {
            planes[t] = reinterpret_cast<const float*>(vsapi->getReadPtr(window[t], plane));
            strides[t] = frames_owned.getStride(window[t], plane) / static_cast<std::ptrdiff_t>(sizeof(float));
        }
        nss::PnlmImage img;
        img.width = w;
        img.height = h;
        img.frames = static_cast<int>(window.size());
        img.center = n - first;
        img.src = planes.data();
        img.src_stride = strides.data();
        img.out = out;
        img.out_stride = out_stride;
        img.sigma = d->sigma[plane];
        img.a = d->a;
        img.block = d->block;
        img.group = d->group;
        const std::size_t need = nss::pnlm_work_floats(w, d->block);
        nss::pnlm_denoise(img, d->ws.get(need), need);
    }
    for (const VSFrame* f : window) frames_owned.freeFrame(f);
    return frames_owned.keep(dst);
}

void VS_CC pnlmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    (void)vsapi;
    auto* d = static_cast<PnlmData*>(instanceData);
    d->node.reset();
    delete d;
}

void VS_CC pnlmCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(out, "nss.PNLM: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return;
    }
    auto d = std::make_unique<PnlmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::PnlmParams&>(*d) = nss::frontend::parse_pnlm(vsapi, in, d->vi, "nss");
    VSFilterDependency deps[1]{{d->node, d->d == 0 ? rpStrictSpatial : rpGeneral}};
    PnlmData* raw = d.get();
    VSNode* node =
        vsapi->createVideoFilter2("PNLM", &raw->vi, nss::checked_frame<pnlmGetFrame>, pnlmFree, fmParallel, deps, 1, raw, core);
    if (!node) {
        vsapi->mapSetError(out, "nss.PNLM: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_pnlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("PNLM", nss::frontend::kPnlmSignature, "clip:vnode;", nss::checked_create<pnlmCreate>, nullptr,
                             plugin);
}
