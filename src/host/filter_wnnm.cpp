#include "host/filters.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct WnnmData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kWnnmDefaultSigma, nss::kWnnmDefaultSigma, nss::kWnnmDefaultSigma};
    int block_size = nss::kWnnmDefaultBlock;
    int block_step = nss::kWnnmDefaultStep;
    int group_size = nss::kWnnmDefaultGroup;
    int bm_range = nss::kWnnmDefaultRange;
    int radius = nss::kWnnmDefaultRadius;
    int ps_num = nss::kWnnmDefaultPsNum;
    int ps_range = nss::kWnnmDefaultPsRange;
    int residual = nss::kWnnmDefaultResidual;
    int adaptive = nss::kWnnmDefaultAdaptive;
    nss::Workspace ws;
};

const VSFrame* VS_CC wnnmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<WnnmData*>(instanceData);
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

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            for (int y = 0; y < (fat ? ph * ntemp * 2 : ph); ++y) {
                const float* row = srcp + (y % ph) * sstride;
                std::memcpy(outp + y * dstride, row, static_cast<std::size_t>(pw) * sizeof(float));
            }
            continue;
        }

        std::vector<const float*> srcs(static_cast<std::size_t>(ntemp));
        std::vector<const float*> refs(static_cast<std::size_t>(ntemp));
        int strides[nss::kBmMaxRadius * 2 + 1];
        for (int t = 0; t < ntemp; ++t) {
            srcs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            refs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
            strides[t] = sstride;
        }

        const int slices = ntemp;
        const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);
        const std::size_t need = plane_sz * static_cast<std::size_t>(slices) * 2 +
                                 static_cast<std::size_t>(lda * group) +
                                 static_cast<std::size_t>(nss::wnnm_shrink_work_floats(m, group)) + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = scratch + plane_sz * static_cast<std::size_t>(slices);
        float* patches = den + plane_sz * static_cast<std::size_t>(slices);
        float* shrink_work = patches + static_cast<std::size_t>(lda * group);
        std::memset(num, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
        std::memset(den, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));

        nss::SearchConfig cfg;
        cfg.block = block;
        cfg.step = d->block_step;
        cfg.group = group;
        cfg.bm_range = d->bm_range;
        cfg.radius = d->radius;
        cfg.ps_num = d->ps_num;
        cfg.ps_range = d->ps_range;

        nss::Match matches[nss::kWnnmMaxGroup];
        const float sigma = d->sigma[plane] / 255.f;

        for (int by0 = 0; by0 < ph - block + d->block_step; by0 += d->block_step) {
            const int by = std::min(by0, std::max(0, ph - block));
            for (int bx0 = 0; bx0 < pw - block + d->block_step; bx0 += d->block_step) {
                const int bx = std::min(bx0, std::max(0, pw - block));
                const int k = (d->radius > 0)
                                  ? nss::predictive_match(refs.data(), strides, ntemp, pw, ph, bx, by, t0, cfg, matches)
                                  : nss::spatial_match(refs[static_cast<std::size_t>(t0)], sstride, pw, ph, bx, by, block,
                                                       d->bm_range, group, matches);
                if (k <= 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    const int t = (d->radius > 0) ? matches[i].t : t0;
                    nss::pack_patch(patches + i * lda, lda, srcs[static_cast<std::size_t>(t)], sstride, matches[i].x,
                                    matches[i].y, block, pw, ph);
                }
                float aw = 1.f;
                if (nss::wnnm_shrink(patches, m, k, lda, sigma, d->residual, d->adaptive, &aw, shrink_work,
                                     nss::wnnm_shrink_work_floats(m, group)) != 0) {
                    continue;
                }
                for (int i = 0; i < k; ++i) {
                    int sl = 0;
                    if (d->radius > 0) {
                        sl = matches[i].t - t0 + d->radius;
                        sl = std::clamp(sl, 0, slices - 1);
                    }
                    nss::aggregate_add(num + static_cast<std::size_t>(sl) * plane_sz,
                                       den + static_cast<std::size_t>(sl) * plane_sz, pw, matches[i].x, matches[i].y,
                                       patches + i * lda, block, pw, ph, aw);
                }
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
            nss::aggregate_finish(outp, num, den, srcp, pw, ph, dstride, pw);
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
        vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    return dst;
}

void VS_CC wnnmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<WnnmData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

}  // namespace

VSNode* nss_create_wnnm(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.WNNM: AVX2 is required");
        return nullptr;
    }
    auto d = std::make_unique<WnnmData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        vsapi->freeNode(d->node);
        if (d->rclip) {
            vsapi->freeNode(d->rclip);
        }
        return nullptr;
    };
    if (!nss::is_const_32f(d->vi)) {
        return fail("nss.WNNM: constant Gray/YUV/RGB 32-bit float required");
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kWnnmDefaultSigma);
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kWnnmDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kWnnmDefaultStep);
    d->group_size = nss::map_int(vsapi, in, "group_size", nss::kWnnmDefaultGroup);
    d->bm_range = nss::map_int(vsapi, in, "bm_range", nss::kWnnmDefaultRange);
    d->radius = nss::map_int(vsapi, in, "radius", nss::kWnnmDefaultRadius);
    d->ps_num = nss::map_int(vsapi, in, "ps_num", nss::kWnnmDefaultPsNum);
    d->ps_range = nss::map_int(vsapi, in, "ps_range", nss::kWnnmDefaultPsRange);
    d->residual = nss::map_int(vsapi, in, "residual", nss::kWnnmDefaultResidual);
    d->adaptive = nss::map_int(vsapi, in, "adaptive_aggregation", nss::kWnnmDefaultAdaptive);
    if (d->block_size < 1 || d->block_size > nss::kWnnmMaxBlock || d->group_size < 1 ||
        d->group_size > nss::kWnnmMaxGroup || d->block_step < 1 || d->radius < 0 ||
        d->radius > nss::kBmMaxRadius) {
        return fail("nss.WNNM: invalid block_size/group_size/block_step/radius");
    }
    if (d->bm_range < 1 || d->bm_range > nss::kBmMaxRange) {
        return fail("nss.WNNM: bm_range must be in [1, 64]");
    }
    int e = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        return fail("nss.WNNM: rclip must match clip");
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    WnnmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("WNNM", &raw->vi_out, wnnmGetFrame, wnnmFree, fmParallel, deps, ndeps, raw,
                                            core);
    if (!node) {
        return fail("nss.WNNM: failed to create filter");
    }
    d.release();
    return node;
}

void VS_CC wnnmCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    VSNode* node = nss_create_wnnm(in, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void VS_CC wnnmv2Create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    VSNode* wn = nss_create_wnnm(in, core, vsapi, out);
    if (!wn) {
        return;
    }
    if (radius <= 0) {
        vsapi->mapConsumeNode(out, "clip", wn, maAppend);
        return;
    }
    VSNode* src = vsapi->mapGetNode(in, "clip", 0, nullptr);
    VSNode* vagg = nss_create_vaggregate(wn, src, radius, nullptr, core, vsapi, out);
    if (!vagg) {
        return;
    }
    vsapi->mapConsumeNode(out, "clip", vagg, maAppend);
}

void register_wnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;residual:int:opt;"
        "adaptive_aggregation:int:opt;rclip:vnode:opt;";
    vspapi->registerFunction("WNNM", args, "clip:vnode;", wnnmCreate, nullptr, plugin);
    vspapi->registerFunction("WNNMv2", args, "clip:vnode;", wnnmv2Create, nullptr, plugin);
}
