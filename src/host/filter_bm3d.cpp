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

struct Bm3dData {
    VSNode* node = nullptr;
    VSNode* ref = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kBmDefaultSigma, nss::kBmDefaultSigma, nss::kBmDefaultSigma};
    int block_size[3]{nss::kBmBlock, nss::kBmBlock, nss::kBmBlock};
    int group_size[3]{nss::kBmGroup, nss::kBmGroup, nss::kBmGroup};
    int block_step[3]{nss::kBmDefaultStep, nss::kBmDefaultStep, nss::kBmDefaultStep};
    int bm_range[3]{nss::kBmDefaultRange, nss::kBmDefaultRange, nss::kBmDefaultRange};
    int ps_num[3]{nss::kBmDefaultPsNum, nss::kBmDefaultPsNum, nss::kBmDefaultPsNum};
    int ps_range[3]{nss::kBmDefaultPsRange, nss::kBmDefaultPsRange, nss::kBmDefaultPsRange};
    int radius = 0;
    nss::Workspace ws;
};

void process_plane(const float* const* srcs, const float* const* refs, int ntemp, int t0,
                   float* dst, int width, int height, int sstride, int dstride, int fat_stride,
                   float sigma, int block, int group, int step, int bm_range, int ps_num, int ps_range,
                   int radius, bool wiener, bool emit_fat, float* scratch) {
    const int slices = 2 * radius + 1;
    const int plane = width * height;
    const int area = block * block;
    const int lda = area;
    float* num0 = scratch;
    float* den0 = scratch + static_cast<std::size_t>(slices) * static_cast<std::size_t>(plane);
    float* work = den0 + static_cast<std::size_t>(slices) * static_cast<std::size_t>(plane);
    float* patches = work + static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block));
    float* ref_patches = patches + static_cast<std::size_t>(group) * static_cast<std::size_t>(area);
    std::memset(num0, 0, static_cast<std::size_t>(slices) * static_cast<std::size_t>(plane) * sizeof(float));
    std::memset(den0, 0, static_cast<std::size_t>(slices) * static_cast<std::size_t>(plane) * sizeof(float));

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = bm_range;
    cfg.radius = radius;
    cfg.ps_num = ps_num;
    cfg.ps_range = ps_range;

    int strides[nss::kBmMaxRadius * 2 + 1];
    for (int t = 0; t < ntemp; ++t) {
        strides[t] = sstride;
    }
    nss::Match matches[nss::kBmMaxGroup];

    for (int by0 = 0; by0 < height - block + step; by0 += step) {
        const int by = std::min(by0, height - block);
        for (int bx0 = 0; bx0 < width - block + step; bx0 += step) {
            const int bx = std::min(bx0, width - block);
            const int k = (radius > 0)
                              ? nss::predictive_match(refs, strides, ntemp, width, height, bx, by, t0, cfg, matches)
                              : nss::spatial_match(refs[t0], sstride, width, height, bx, by, block, bm_range, group,
                                                   matches);
            if (k <= 0) {
                continue;
            }
            if (block == 8 && group == 8 && radius == 0) {
                nss::bm3d_filter8(srcs[t0], sstride, matches, k, sigma, wiener, wiener ? refs[t0] : nullptr, sstride,
                                  num0, den0, width, width, height);
                continue;
            }
            for (int i = 0; i < k; ++i) {
                const int t = (radius > 0) ? matches[i].t : t0;
                nss::pack_patch(patches + static_cast<std::size_t>(i) * lda, lda, srcs[t], sstride, matches[i].x,
                                matches[i].y, block, width, height);
                if (wiener) {
                    nss::pack_patch(ref_patches + static_cast<std::size_t>(i) * lda, lda, refs[t], sstride,
                                    matches[i].x, matches[i].y, block, width, height);
                }
            }
            float w = 1.f;
            nss::bm3d_filter_group(patches, lda, group, k, block, sigma, wiener, wiener ? ref_patches : nullptr, &w,
                                   work);
            for (int i = 0; i < k; ++i) {
                int sl = 0;
                if (radius > 0) {
                    sl = matches[i].t - t0 + radius;
                    sl = std::clamp(sl, 0, slices - 1);
                }
                nss::aggregate_add(num0 + sl * plane, den0 + sl * plane, width, matches[i].x, matches[i].y,
                                   patches + static_cast<std::size_t>(i) * lda, block, width, height, w);
            }
        }
    }

    if (emit_fat) {
        for (int sl = 0; sl < slices; ++sl) {
            const float* np = num0 + sl * plane;
            const float* dp = den0 + sl * plane;
            float* on = dst + (sl * 2) * height * fat_stride;
            float* od = dst + (sl * 2 + 1) * height * fat_stride;
            for (int y = 0; y < height; ++y) {
                std::memcpy(on + y * fat_stride, np + y * width, static_cast<std::size_t>(width) * sizeof(float));
                std::memcpy(od + y * fat_stride, dp + y * width, static_cast<std::size_t>(width) * sizeof(float));
            }
        }
    } else {
        nss::aggregate_finish(dst, num0, den0, srcs[t0], width, height, dstride, width);
    }
}

const VSFrame* VS_CC bm3dGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<Bm3dData*>(instanceData);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->radius);
        const int end = std::min(n + d->radius, d->vi.numFrames - 1);
        for (int i = start; i <= end; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
            if (d->ref) {
                vsapi->requestFrameFilter(i, d->ref, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const bool fat = d->radius > 0;
    const VSFrame* src0 = vsapi->getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);

    const int ntemp = 2 * d->radius + 1;
    std::vector<const VSFrame*> srcf(static_cast<std::size_t>(ntemp));
    std::vector<const VSFrame*> reff(static_cast<std::size_t>(ntemp));
    for (int t = 0; t < ntemp; ++t) {
        const int fn = std::clamp(n - d->radius + t, 0, d->vi.numFrames - 1);
        srcf[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, d->node, frameCtx);
        reff[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, d->ref ? d->ref : d->node, frameCtx);
    }
    const int t0 = d->radius;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            for (int y = 0; y < (fat ? ph * (2 * d->radius + 1) * 2 : ph); ++y) {
                const float* row = srcp + (y % ph) * sstride;
                std::memcpy(outp + y * dstride, row, static_cast<std::size_t>(pw) * sizeof(float));
            }
            continue;
        }
        std::vector<const float*> srcs(static_cast<std::size_t>(ntemp));
        std::vector<const float*> refs(static_cast<std::size_t>(ntemp));
        for (int t = 0; t < ntemp; ++t) {
            srcs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            refs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
        }
        const int slices = 2 * d->radius + 1;
        const int block = d->block_size[plane];
        const int group = d->group_size[plane];
        const int area = block * block;
        const std::size_t need = static_cast<std::size_t>(slices) * 2 * static_cast<std::size_t>(pw * ph) +
                                 static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)) +
                                 static_cast<std::size_t>(group) * static_cast<std::size_t>(area) *
                                     (d->ref != nullptr ? 2u : 1u) +
                                 64;
        float* scratch = d->ws.get(need);
        process_plane(srcs.data(), refs.data(), ntemp, t0, outp, pw, ph, sstride, dstride, dstride, d->sigma[plane],
                      d->block_size[plane], d->group_size[plane], d->block_step[plane], d->bm_range[plane],
                      d->ps_num[plane], d->ps_range[plane], d->radius, d->ref != nullptr, fat, scratch);
        if (!fat && d->sigma[plane] != 0.f) {
            (void)srcp;
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
        vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    vsapi->freeFrame(src0);
    return dst;
}

void VS_CC bm3dFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<Bm3dData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->ref) {
        vsapi->freeNode(d->ref);
    }
    delete d;
}

}  // namespace

VSNode* nss_create_bm3d(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.BM3D: AVX2 is required");
        return nullptr;
    }
    auto d = std::make_unique<Bm3dData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    if (!nss::is_const_32f(d->vi)) {
        vsapi->mapSetError(err, "nss.BM3D: constant Gray/YUV/RGB 32-bit float required");
        vsapi->freeNode(d->node);
        return nullptr;
    }
    int e = 0;
    d->ref = vsapi->mapGetNode(in, "ref", 0, &e);
    if (e) {
        d->ref = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->ref))) {
        vsapi->mapSetError(err, "nss.BM3D: ref must match clip");
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref);
        return nullptr;
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kBmDefaultSigma);
    // User sigma is 8-bit units, same as bm3dcpu. Clip is 32f in 0-1.
    // General sizes: orthonormal DCT-II, only /255. Hard lambda 2.7 in the kernel.
    // Fused 8x8x8 radius=0 copies bm3dcpu FFTW scale inside bm3d_filter8: extra (3/4)*64 and 1/4096.
    for (int i = 0; i < np; ++i) {
        if (d->sigma[i] != 0.f) {
            d->sigma[i] *= (1.f / 255.f);
        }
    }
    nss::map_inherit_int(vsapi, in, "block_size", d->block_size, np, nss::kBmBlock);
    nss::map_inherit_int(vsapi, in, "group_size", d->group_size, np, nss::kBmGroup);
    {
        const int nstep = vsapi->mapNumElements(in, "block_step");
        if (nstep <= 0) {
            for (int i = 0; i < np; ++i) {
                d->block_step[i] = std::min(nss::kBmDefaultStep, d->block_size[i]);
            }
        } else {
            nss::map_inherit_int(vsapi, in, "block_step", d->block_step, np, nss::kBmDefaultStep);
        }
    }
    nss::map_int_array(vsapi, in, "bm_range", d->bm_range, np, nss::kBmDefaultRange);
    {
        const int nps = vsapi->mapNumElements(in, "ps_num");
        if (nps <= 0) {
            for (int i = 0; i < np; ++i) {
                d->ps_num[i] = std::min(nss::kBmDefaultPsNum, d->group_size[i]);
            }
        } else {
            nss::map_inherit_int(vsapi, in, "ps_num", d->ps_num, np, nss::kBmDefaultPsNum);
        }
    }
    nss::map_int_array(vsapi, in, "ps_range", d->ps_range, np, nss::kBmDefaultPsRange);
    d->radius = nss::map_int(vsapi, in, "radius", 0);
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        vsapi->freeNode(d->node);
        if (d->ref) {
            vsapi->freeNode(d->ref);
        }
        return nullptr;
    };
    if (d->radius < 0 || d->radius > nss::kBmMaxRadius) {
        return fail("nss.BM3D: radius must be in [0, 16]");
    }
    for (int i = 0; i < np; ++i) {
        if (!nss::bm_allowed_block(d->block_size[i])) {
            return fail("nss.BM3D: block_size must be one of 1, 2, 4, 8, 16, 32");
        }
        if (!nss::bm_allowed_group(d->group_size[i])) {
            return fail("nss.BM3D: group_size must be one of 1, 2, 4, 8, 16, 32, 64");
        }
        if (d->block_size[i] > nss::plane_width(d->vi, i) || d->block_size[i] > nss::plane_height(d->vi, i)) {
            return fail("nss.BM3D: block_size must not exceed plane dimensions");
        }
        if (d->block_step[i] < 1 || d->block_step[i] > d->block_size[i]) {
            return fail("nss.BM3D: block_step must be in [1, block_size]");
        }
        if (d->ps_num[i] < 1 || d->ps_num[i] > d->group_size[i]) {
            return fail("nss.BM3D: ps_num must be in [1, group_size]");
        }
        if (d->bm_range[i] < 1 || d->bm_range[i] > nss::kBmMaxRange) {
            return fail("nss.BM3D: bm_range must be in [1, 64]");
        }
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->ref, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->ref ? 2 : 1;
    Bm3dData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("BM3D", &raw->vi_out, bm3dGetFrame, bm3dFree, fmParallel, deps, ndeps, raw,
                                            core);
    if (!node) {
        vsapi->mapSetError(err, "nss.BM3D: failed to create filter");
        vsapi->freeNode(d->node);
        if (d->ref) {
            vsapi->freeNode(d->ref);
        }
        return nullptr;
    }
    d.release();
    return node;
}

void VS_CC bm3dCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    VSNode* node = nss_create_bm3d(in, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void VS_CC bm3dv2Create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    VSNode* bm = nss_create_bm3d(in, core, vsapi, out);
    if (!bm) {
        return;
    }
    if (radius <= 0) {
        vsapi->mapConsumeNode(out, "clip", bm, maAppend);
        return;
    }
    VSNode* src = vsapi->mapGetNode(in, "clip", 0, nullptr);
    VSNode* vagg = nss_create_vaggregate(bm, src, radius, nullptr, core, vsapi, out);
    if (!vagg) {
        return;
    }
    vsapi->mapConsumeNode(out, "clip", vagg, maAppend);
}

void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_size:int[]:opt;group_size:int[]:opt;"
        "block_step:int[]:opt;bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;";
    vspapi->registerFunction("BM3D", args, "clip:vnode;", bm3dCreate, nullptr, plugin);
    vspapi->registerFunction("BM3Dv2", args, "clip:vnode;", bm3dv2Create, nullptr, plugin);
}
