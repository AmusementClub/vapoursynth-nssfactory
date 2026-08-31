#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct NcsrData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kNcsrDefaultSigma, nss::kNcsrDefaultSigma, nss::kNcsrDefaultSigma};
    int block_size = nss::kNcsrDefaultBlock;
    int block_step = nss::kNcsrDefaultStep;
    int group_size = nss::kNcsrDefaultGroup;
    int bm_range = nss::kNcsrDefaultRange;
    int radius = nss::kWnnmDefaultRadius;
    int ps_num = nss::kWnnmDefaultPsNum;
    int ps_range = nss::kWnnmDefaultPsRange;
    int iters = nss::kNcsrDefaultIters;
    float delta = nss::kNcsrDefaultDelta;
    nss::Workspace ws;
};

const VSFrame* VS_CC ncsrGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<NcsrData*>(instanceData);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->radius);
        const int end = std::min(n + d->radius, d->vi.numFrames - 1);
        for (int i = start; i <= end; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
            if (d->rclip) {
                vsapi->requestFrameFilter(i, d->rclip, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const bool fat = d->radius > 0;
    const int ntemp = 2 * d->radius + 1;
    const int t0 = d->radius;
    std::vector<const VSFrame*> srcf(static_cast<std::size_t>(ntemp));
    std::vector<const VSFrame*> reff(static_cast<std::size_t>(ntemp));
    for (int t = 0; t < ntemp; ++t) {
        const int fn = std::clamp(n - d->radius + t, 0, d->vi.numFrames - 1);
        srcf[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, d->node, frameCtx);
        reff[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, d->rclip ? d->rclip : d->node, frameCtx);
    }
    const VSFrame* src0 = srcf[static_cast<std::size_t>(t0)];
    VSFrame* dst = vsapi->newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);

    const int block = d->block_size;
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const int group = d->group_size;
    const int iters = d->iters;
    const float delta = d->delta;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            if (!fat) {
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(outp + y * dstride, srcp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
                }
            } else {
                for (int sl = 0; sl < ntemp; ++sl) {
                    const float* sp = reinterpret_cast<const float*>(
                        vsapi->getReadPtr(srcf[static_cast<std::size_t>(sl)], plane));
                    float* on = outp + (sl * 2) * ph * dstride;
                    float* od = outp + (sl * 2 + 1) * ph * dstride;
                    for (int y = 0; y < ph; ++y) {
                        std::memcpy(on + y * dstride, sp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
                        for (int x = 0; x < pw; ++x) {
                            od[y * dstride + x] = 1.f;
                        }
                    }
                }
            }
            continue;
        }

        std::vector<const float*> srcs(static_cast<std::size_t>(ntemp));
        std::vector<const float*> refs(static_cast<std::size_t>(ntemp));
        int src_strides[nss::kBmMaxRadius * 2 + 1];
        for (int t = 0; t < ntemp; ++t) {
            srcs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            refs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
            src_strides[t] = sstride;
        }

        const int slices = ntemp;
        const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);
        const std::size_t need = plane_sz * static_cast<std::size_t>(slices) * 4 +
                                 static_cast<std::size_t>(lda * group) +
                                 static_cast<std::size_t>(nss::ncsr_filter_work_floats(m, group)) + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = num + plane_sz * static_cast<std::size_t>(slices);
        float* est = den + plane_sz * static_cast<std::size_t>(slices);
        float* noisy = est + plane_sz * static_cast<std::size_t>(slices);
        float* patches = noisy + plane_sz * static_cast<std::size_t>(slices);
        float* shrink_work = patches + static_cast<std::size_t>(lda * group);
        const float sigma = d->sigma[plane] / 255.f;

        for (int t = 0; t < ntemp; ++t) {
            float* e = est + static_cast<std::size_t>(t) * plane_sz;
            float* y = noisy + static_cast<std::size_t>(t) * plane_sz;
            const float* s = srcs[static_cast<std::size_t>(t)];
            for (int row = 0; row < ph; ++row) {
                std::memcpy(e + static_cast<std::size_t>(row * pw), s + row * sstride,
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
            std::memcpy(y, e, plane_sz * sizeof(float));
        }

        nss::SearchConfig cfg;
        cfg.block = block;
        cfg.step = d->block_step;
        cfg.group = group;
        cfg.bm_range = d->bm_range;
        cfg.radius = d->radius;
        cfg.ps_num = d->ps_num;
        cfg.ps_range = d->ps_range;

        std::vector<const float*> est_refs(static_cast<std::size_t>(ntemp));
        int est_strides[nss::kBmMaxRadius * 2 + 1];
        for (int t = 0; t < ntemp; ++t) {
            est_refs[static_cast<std::size_t>(t)] = est + static_cast<std::size_t>(t) * plane_sz;
            est_strides[t] = pw;
        }

        for (int iter = 0; iter < iters; ++iter) {
            if (iter > 0) {
                for (int sl = 0; sl < slices; ++sl) {
                    nss::iter_regularize(est + static_cast<std::size_t>(sl) * plane_sz,
                                         noisy + static_cast<std::size_t>(sl) * plane_sz, pw * ph, delta);
                }
            }
            std::memset(num, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
            std::memset(den, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
            const bool use_rclip = (iter == 0 && d->rclip != nullptr);
            const float* const* match_refs = use_rclip ? refs.data() : est_refs.data();
            const int* match_strides = use_rclip ? src_strides : est_strides;
            nss::ncsr_run_groups(match_refs, match_strides, est_refs.data(), est_strides, ntemp, t0, pw, ph, cfg, sigma,
                                 num, den, patches, shrink_work);
            const bool last = (iter == iters - 1);
            if (last && fat) {
                break;
            }
            for (int sl = 0; sl < slices; ++sl) {
                float* e = est + static_cast<std::size_t>(sl) * plane_sz;
                nss::aggregate_finish(e, num + static_cast<std::size_t>(sl) * plane_sz,
                                      den + static_cast<std::size_t>(sl) * plane_sz, e, pw, ph, pw, pw);
            }
        }
        if (fat) {
            for (int sl = 0; sl < slices; ++sl) {
                const float* np = num + static_cast<std::size_t>(sl) * plane_sz;
                const float* dp = den + static_cast<std::size_t>(sl) * plane_sz;
                float* on = outp + (sl * 2) * ph * dstride;
                float* od = outp + (sl * 2 + 1) * ph * dstride;
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(on + y * dstride, np + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                    std::memcpy(od + y * dstride, dp + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                }
            }
        } else {
            const float* e = est + static_cast<std::size_t>(t0) * plane_sz;
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp + y * dstride, e + static_cast<std::size_t>(y * pw),
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
        vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    return dst;
}

void VS_CC ncsrFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<NcsrData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

}  // namespace

void VS_CC ncsrCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(out, "nss.NCSR: AVX2 is required");
        return;
    }
    auto d = std::make_unique<NcsrData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) {
        vsapi->mapSetError(out, msg);
        vsapi->freeNode(d->node);
        if (d->rclip) {
            vsapi->freeNode(d->rclip);
        }
    };
    if (!nss::is_const_32f(d->vi)) {
        fail("nss.NCSR: constant Gray/YUV/RGB 32-bit float required");
        return;
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kNcsrDefaultSigma);
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kNcsrDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kNcsrDefaultStep);
    d->group_size = nss::map_int(vsapi, in, "group_size", nss::kNcsrDefaultGroup);
    d->bm_range = nss::map_int(vsapi, in, "bm_range", nss::kNcsrDefaultRange);
    d->radius = nss::map_int(vsapi, in, "radius", nss::kWnnmDefaultRadius);
    d->ps_num = nss::map_int(vsapi, in, "ps_num", nss::kWnnmDefaultPsNum);
    d->ps_range = nss::map_int(vsapi, in, "ps_range", nss::kWnnmDefaultPsRange);
    d->iters = nss::map_int(vsapi, in, "iters", nss::kNcsrDefaultIters);
    d->delta = nss::map_float(vsapi, in, "delta", nss::kNcsrDefaultDelta);
    if (d->block_size < 1 || d->block_size > nss::kWnnmMaxBlock || d->group_size < 1 ||
        d->group_size > nss::kWnnmMaxGroup || d->block_step < 1 || d->block_step > d->block_size || d->radius < 0 ||
        d->radius > nss::kBmMaxRadius) {
        fail("nss.NCSR: invalid block_size/group_size/block_step/radius");
        return;
    }
    if (d->iters < 1 || !std::isfinite(d->delta)) {
        fail("nss.NCSR: invalid iters/delta");
        return;
    }
    if (d->ps_num < 1 || d->ps_num > d->group_size || d->ps_range < 1 || d->ps_range > nss::kBmMaxRange) {
        fail("nss.NCSR: invalid ps_num/ps_range");
        return;
    }
    if (d->bm_range < 1 || d->bm_range > nss::kBmMaxRange) {
        fail("nss.NCSR: bm_range must be in [1, 64]");
        return;
    }
    int e = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        fail("nss.NCSR: rclip must match clip");
        return;
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    NcsrData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("NCSR", &raw->vi_out, ncsrGetFrame, ncsrFree, fmParallel, deps, ndeps, raw,
                                             core);
    if (!node) {
        fail("nss.NCSR: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

void register_ncsr(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;rclip:vnode:opt;"
        "iters:int:opt;delta:float:opt;";
    vspapi->registerFunction("NCSR", args, "clip:vnode;", ncsrCreate, nullptr, plugin);
}
