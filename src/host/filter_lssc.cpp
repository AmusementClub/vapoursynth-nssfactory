#include "host/validate.hpp"
#include "nss/avx2.hpp"
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

struct LsscData {
    VSNode* node = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kLsscDefaultSigma, nss::kLsscDefaultSigma, nss::kLsscDefaultSigma};
    int block_size = nss::kLsscDefaultBlock;
    int block_step = nss::kLsscDefaultStep;
    int radius = 0;
    nss::Workspace ws;
};

const VSFrame* VS_CC lsscGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<LsscData*>(instanceData);
    (void)frameData;
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrame* src0 = vsapi->getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);

    const int block = d->block_size;
    const int step = d->block_step;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
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

    vsapi->freeFrame(src0);
    return dst;
}

void VS_CC lsscFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<LsscData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC lsscCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(out, "nss.LSSC: AVX2 is required");
        return;
    }
    auto d = std::make_unique<LsscData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) {
        vsapi->mapSetError(out, msg);
        vsapi->freeNode(d->node);
    };
    if (!nss::is_const_32f(d->vi)) {
        fail("nss.LSSC: constant Gray/YUV/RGB 32-bit float required");
        return;
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kLsscDefaultSigma);
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kLsscDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kLsscDefaultStep);
    d->radius = nss::map_int(vsapi, in, "radius", 0);
    if (!nss::bm_allowed_block(d->block_size) || d->block_size > 16 || d->block_step < 1 ||
        d->block_step > d->block_size) {
        fail("nss.LSSC: block_size must be 1, 2, 4, 8, or 16");
        return;
    }
    if (d->radius != 0) {
        fail("nss.LSSC: radius>0 is not implemented (no temporal clustering)");
        return;
    }
    const int np0 = nss::lssc_grid_count(d->vi.width, d->vi.height, d->block_size, d->block_step);
    const int area = d->block_size * d->block_size;
    if (area > 0 && np0 > 16000000 / area) {
        fail("nss.LSSC: grid too large; increase block_step");
        return;
    }
    d->vi_out = d->vi;
    VSFilterDependency deps[1]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    LsscData* raw = d.get();
    VSNode* node =
        vsapi->createVideoFilter2("LSSC", &raw->vi_out, lsscGetFrame, lsscFree, fmParallel, deps, 1, raw, core);
    if (!node) {
        fail("nss.LSSC: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_lssc(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("LSSC",
                             "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;radius:int:opt;",
                             "clip:vnode;", lsscCreate, nullptr, plugin);
}
