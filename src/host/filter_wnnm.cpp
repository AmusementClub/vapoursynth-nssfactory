#include "host/filters.hpp"
#include "host/batch_runner.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
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

struct WnnmBatchResult {
    bool valid = false;
    int k = 0;
    int slice[nss::kWnnmMaxGroup]{};
    nss::Match matches[nss::kWnnmMaxGroup]{};
    const float* patches = nullptr;
    float weight = 1.f;
};

void process_plane_batched(const float* const* srcs, const float* const* refs, int ntemp, int t0, float* dst,
                           int width, int height, int sstride, int dstride, float sigma, int block, int group, int step,
                           int bm_range, int ps_num, int ps_range, int radius, int residual, int adaptive,
                           bool emit_fat, float* scratch) {
    const int slices = 2 * radius + 1;
    const std::size_t plane_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    float* num = scratch;
    float* den = num + static_cast<std::size_t>(slices) * plane_size;
    std::memset(num, 0, static_cast<std::size_t>(slices) * plane_size * sizeof(float));
    std::memset(den, 0, static_cast<std::size_t>(slices) * plane_size * sizeof(float));

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = bm_range;
    cfg.radius = radius;
    cfg.ps_num = ps_num;
    cfg.ps_range = ps_range;
    int strides[nss::kBmMaxRadius * 2 + 1]{};
    for (int t = 0; t < ntemp; ++t) {
        strides[t] = sstride;
    }

    std::vector<nss::GroupJob> jobs;
    nss::host_detail::append_raster_jobs(
        jobs, width, height, block, step,
        nss::GroupKey{block * block, group, 1, nss::GroupAlgorithm::WNNM, false, residual != 0}, t0);
    if (jobs.empty()) {
        return;
    }

    const int area = block * block;
    const int work_floats = nss::wnnm_shrink_work_floats(area, group);
    const std::size_t batch_capacity = std::min(jobs.size(), nss::host_detail::kGroupBatchWindow);
    const std::size_t patch_stride = static_cast<std::size_t>(group) * static_cast<std::size_t>(area);
    std::vector<float> patches(batch_capacity * patch_stride, 0.f);
    std::vector<float> shrink_work(batch_capacity * static_cast<std::size_t>(work_floats), 0.f);
    for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
        const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
        const int count = static_cast<int>(end - begin);
        std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
        std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
        std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kWnnmMaxGroup> match_storage{};
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] = nss::MatchBatchItem{job.x, job.y, block, bm_range, group};
        }
        const int match_rc = radius > 0
                                 ? nss::predictive_match_batch(refs, strides, ntemp, width, height, t0, cfg,
                                                               match_items.data(), count, match_storage.data(),
                                                               nss::kWnnmMaxGroup, counts.data())
                                 : nss::spatial_match_batch(refs[t0], sstride, width, height, match_items.data(), count,
                                                            match_storage.data(), nss::kWnnmMaxGroup, counts.data());
        if (match_rc < 0) {
            continue;
        }

        std::fill_n(patches.data(), static_cast<std::size_t>(count) * patch_stride, 0.f);
        std::fill_n(shrink_work.data(), static_cast<std::size_t>(count) * static_cast<std::size_t>(work_floats), 0.f);
        std::array<float, nss::host_detail::kGroupBatchWindow> weights{};
        std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
        std::array<nss::WnnmShrinkBatchItem, nss::host_detail::kGroupBatchWindow> shrink_items{};
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* patch = patches.data() + static_cast<std::size_t>(i) * patch_stride;
            for (int j = 0; j < k; ++j) {
                const auto& m = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                const int t = radius > 0 ? m.t : t0;
                nss::pack_patch(patch + static_cast<std::size_t>(j) * area, area, srcs[t], sstride, m.x, m.y, block,
                                width, height);
            }
            weights[static_cast<std::size_t>(i)] = 1.f;
            shrink_items[static_cast<std::size_t>(i)] = nss::WnnmShrinkBatchItem{
                patch, area, k, area, sigma, residual, adaptive, &weights[static_cast<std::size_t>(i)],
                shrink_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(work_floats), work_floats,
                &filter_status[static_cast<std::size_t>(i)]};
        }
        (void)nss::wnnm_shrink_batch(shrink_items.data(), count);

        auto prepare = [&](const nss::GroupJob& job, WnnmBatchResult& result) {
            const std::size_t i = static_cast<std::size_t>(job.ordinal - jobs[begin].ordinal);
            if (i >= static_cast<std::size_t>(count) || counts[i] <= 0 || filter_status[i] == 0) {
                return false;
            }
            result.valid = true;
            result.k = counts[i];
            result.patches = patches.data() + i * patch_stride;
            result.weight = weights[i];
            for (int j = 0; j < result.k; ++j) {
                result.matches[j] = match_storage[i * nss::kWnnmMaxGroup + j];
                result.slice[j] = radius > 0 ? std::clamp(result.matches[j].t - t0 + radius, 0, slices - 1) : 0;
            }
            return true;
        };
        auto commit = [&](const WnnmBatchResult& result) {
            if (!result.valid) {
                return;
            }
            for (int j = 0; j < result.k; ++j) {
                nss::aggregate_add(num + static_cast<std::size_t>(result.slice[j]) * plane_size,
                                   den + static_cast<std::size_t>(result.slice[j]) * plane_size, width,
                                   result.matches[j].x, result.matches[j].y,
                                   result.patches + static_cast<std::size_t>(j) * area, block, width, height,
                                   result.weight);
            }
        };
        (void)nss::host_detail::commit_prepared_chunk<WnnmBatchResult>(jobs, begin, end, prepare, commit);
    }

    if (emit_fat) {
        for (int sl = 0; sl < slices; ++sl) {
            const float* np = num + static_cast<std::size_t>(sl) * plane_size;
            const float* dp = den + static_cast<std::size_t>(sl) * plane_size;
            float* on = dst + static_cast<std::size_t>(sl * 2) * static_cast<std::size_t>(height) * dstride;
            float* od = dst + static_cast<std::size_t>(sl * 2 + 1) * static_cast<std::size_t>(height) * dstride;
            for (int y = 0; y < height; ++y) {
                std::memcpy(on + y * dstride, np + static_cast<std::size_t>(y) * width,
                            static_cast<std::size_t>(width) * sizeof(float));
                std::memcpy(od + y * dstride, dp + static_cast<std::size_t>(y) * width,
                            static_cast<std::size_t>(width) * sizeof(float));
            }
        }
    } else {
        nss::aggregate_finish(dst, num, den, srcs[t0], width, height, dstride, width, sstride);
    }
}

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
        const std::size_t need = plane_sz * static_cast<std::size_t>(slices) * 2 + 64;
        float* scratch = d->ws.get(need);
        process_plane_batched(srcs.data(), refs.data(), ntemp, t0, outp, pw, ph, sstride, dstride,
                              d->sigma[plane] / 255.f, block, group, d->block_step, d->bm_range, d->ps_num,
                              d->ps_range, d->radius, d->residual, d->adaptive, fat, scratch);
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
        d->group_size > nss::kWnnmMaxGroup || d->block_step < 1 || d->block_step > d->block_size || d->radius < 0 ||
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
