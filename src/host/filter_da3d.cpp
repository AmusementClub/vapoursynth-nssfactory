// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/validate.hpp"
#include "frontend/da3d_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_da3d.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <cstring>
#include <memory>

namespace {

struct Da3dData : nss::Da3dParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    nss::NodeRef ref;
    VSVideoInfo vi{};
    bool joint_rgb = false;
    nss::Workspace ws;
};

const VSFrame* VS_CC da3dGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<Da3dData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->ref, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    const VSFrame* noisy = frames_owned.getFrameFilter(n, d->node, frameCtx);
    const VSFrame* guide = frames_owned.getFrameFilter(n, d->ref, frameCtx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, noisy, core);
    const int planes = d->vi.format.numPlanes;

    auto view = [&](int plane, nss::Da3dImage& img, int c) {
        img.noisy[c] = reinterpret_cast<const float*>(vsapi->getReadPtr(noisy, plane));
        img.guide[c] = reinterpret_cast<const float*>(vsapi->getReadPtr(guide, plane));
        img.out[c] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        img.noisy_stride[c] = frames_owned.getStride(noisy, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
        img.guide_stride[c] = frames_owned.getStride(guide, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
        img.out_stride[c] = frames_owned.getStride(dst, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
    };
    nss::Da3dStats stats;
    auto run = [&](nss::Da3dImage& img) {
        const std::size_t need = nss::da3d_work_floats(img.width, img.height, img.channels);
        nss::da3d_denoise(img, d->ws.get(need), need, &stats);
    };

    if (d->joint_rgb) {
        nss::Da3dImage img;
        img.channels = 3;
        img.width = d->vi.width;
        img.height = d->vi.height;
        img.rgb = true;
        img.sigma = d->sigma[0] / 255.f;
        for (int plane = 0; plane < 3; ++plane) view(plane, img, plane);
        run(img);
    } else {
        for (int plane = 0; plane < planes; ++plane) {
            nss::Da3dImage img;
            img.width = nss::plane_width(d->vi, plane);
            img.height = nss::plane_height(d->vi, plane);
            view(plane, img, 0);
            if (d->sigma[plane] == 0.f) {  // unprocessed: the guide's plane
                const auto bytes = static_cast<std::size_t>(img.width) * sizeof(float);
                for (int y = 0; y < img.height; ++y)
                    std::memcpy(img.out[0] + y * img.out_stride[0], img.guide[0] + y * img.guide_stride[0], bytes);
                continue;
            }
            img.sigma = d->sigma[plane] / 255.f;
            run(img);
        }
    }

    VSMap* props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_NSSDA3DPatches", stats.patches, maReplace);
    vsapi->mapSetInt(props, "_NSSDA3DGuideOnly", stats.guide_only, maReplace);
    frames_owned.freeFrame(noisy);
    frames_owned.freeFrame(guide);
    return frames_owned.keep(dst);
}

void VS_CC da3dFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    (void)vsapi;
    auto* d = static_cast<Da3dData*>(instanceData);
    d->node.reset();
    d->ref.reset();
    delete d;
}

void VS_CC da3dCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(out, "nss.DA3D: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return;
    }
    auto d = std::make_unique<Da3dData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::Da3dParams&>(*d) = nss::frontend::parse_da3d(vsapi, in, d->vi, "nss");
    d->ref = nss::get_node(vsapi, in, "ref", 0, nullptr);
    if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->ref)))
        throw std::invalid_argument("nss.DA3D: ref must have the clip's format, size and length");
    d->joint_rgb = nss::frontend::da3d_joint_rgb(d->vi, *d);
    VSFilterDependency deps[2]{{d->node, rpStrictSpatial}, {d->ref, rpStrictSpatial}};
    Da3dData* raw = d.get();
    VSNode* node =
        vsapi->createVideoFilter2("DA3D", &raw->vi, nss::checked_frame<da3dGetFrame>, da3dFree, fmParallel, deps, 2, raw, core);
    if (!node) {
        vsapi->mapSetError(out, "nss.DA3D: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_da3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("DA3D", nss::frontend::kDa3dSignature, "clip:vnode;", nss::checked_create<da3dCreate>, nullptr,
                             plugin);
}
