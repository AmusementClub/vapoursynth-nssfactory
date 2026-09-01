#include "host/filters.hpp"
#include "host/batch_runner.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_nlh.hpp"
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

struct NlhData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kNlhDefaultSigma, nss::kNlhDefaultSigma, nss::kNlhDefaultSigma};
    int block_size = nss::kNlhDefaultBlock;
    int block_step = nss::kNlhDefaultStep;
    int group_size = nss::kNlhDefaultGroup;
    int bm_range = nss::kNlhDefaultRange;
    int radius = 0;
    int ps_num = nss::kBmDefaultPsNum;
    int ps_range = nss::kBmDefaultPsRange;
    int q = nss::kNlhDefaultQ;
    nss::Workspace ws;
};

void run_groups(const float* const* match_refs, const int* match_strides, const float* const* noisy,
                const int* noisy_strides, int ntemp, int t0, int pw, int ph, int block, int step, int group,
                int bm_range, int radius, int ps_num, int ps_range, int q, float sigma, bool wiener, float* num,
                float* den, float* patches, float* ref_patches, float* work) {
    const int m = block * block;
    const int lda = (m + 15) & ~15;
    const int slices = 2 * radius + 1;
    const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);
    std::memset(num, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
    std::memset(den, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = bm_range;
    cfg.radius = radius;
    cfg.ps_num = ps_num;
    cfg.ps_range = ps_range;

    std::vector<nss::GroupJob> jobs;
    nss::host_detail::append_raster_jobs(
        jobs, pw, ph, block, step,
        nss::GroupKey{m, group, 1, nss::GroupAlgorithm::NLH, !wiener, false}, t0);
    if (jobs.empty()) {
        return;
    }
    const int group_work = nss::nlh_filter_work_floats(m, group, q, lda);
    for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
        const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
        const int count = static_cast<int>(end - begin);
        std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
        std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
        std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kBmMaxGroup> match_storage{};
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] = nss::MatchBatchItem{job.x, job.y, block, bm_range, group};
        }
        const int match_rc = radius > 0
                                 ? nss::predictive_match_batch(match_refs, match_strides, ntemp, pw, ph, t0, cfg,
                                                               match_items.data(), count, match_storage.data(),
                                                               nss::kBmMaxGroup, counts.data())
                                 : nss::spatial_match_batch(match_refs[t0], match_strides[t0], pw, ph,
                                                            match_items.data(), count, match_storage.data(),
                                                            nss::kBmMaxGroup, counts.data());
        if (match_rc < 0) {
            continue;
        }
        const std::size_t group_storage = static_cast<std::size_t>(group) * static_cast<std::size_t>(lda);
        std::vector<float> batch_patches(static_cast<std::size_t>(count) * group_storage, 0.f);
        std::vector<float> batch_refs;
        if (wiener) {
            batch_refs.resize(batch_patches.size(), 0.f);
        }
        std::vector<float> batch_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(group_work), 0.f);
        std::array<float, nss::host_detail::kGroupBatchWindow> weights{};
        std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
        std::array<nss::NlhFilterBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* p = batch_patches.data() + static_cast<std::size_t>(i) * group_storage;
            float* rp = wiener ? batch_refs.data() + static_cast<std::size_t>(i) * group_storage : nullptr;
            for (int j = 0; j < k; ++j) {
                const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kBmMaxGroup + j];
                const int t = radius > 0 ? mm.t : t0;
                nss::pack_patch(p + static_cast<std::size_t>(j) * lda, lda, noisy[t], noisy_strides[t], mm.x, mm.y,
                                block, pw, ph);
                if (rp) {
                    nss::pack_patch(rp + static_cast<std::size_t>(j) * lda, lda, match_refs[t], match_strides[t], mm.x,
                                    mm.y, block, pw, ph);
                }
            }
            weights[static_cast<std::size_t>(i)] = 1.f;
            filter_items[static_cast<std::size_t>(i)] = nss::NlhFilterBatchItem{
                p, m, k, lda, q, sigma, wiener, rp, &weights[static_cast<std::size_t>(i)],
                batch_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group_work), group_work,
                &filter_status[static_cast<std::size_t>(i)]};
        }
        (void)nss::nlh_filter_group_batch(filter_items.data(), count);

        struct Result {
            bool valid = false;
            int k = 0;
            nss::Match matches[nss::kBmMaxGroup]{};
            int slices[nss::kBmMaxGroup]{};
            const float* patches = nullptr;
            float weight = 1.f;
        };
        auto prepare = [&](const nss::GroupJob& job, Result& result) {
            const std::size_t i = static_cast<std::size_t>(job.ordinal - jobs[begin].ordinal);
            if (i >= static_cast<std::size_t>(count) || counts[i] <= 0 || filter_status[i] == 0) {
                return false;
            }
            result.valid = true;
            result.k = counts[i];
            result.patches = batch_patches.data() + i * group_storage;
            result.weight = weights[i];
            for (int j = 0; j < result.k; ++j) {
                result.matches[j] = match_storage[i * nss::kBmMaxGroup + j];
                result.slices[j] = radius > 0 ? std::clamp(result.matches[j].t - t0 + radius, 0, slices - 1) : 0;
            }
            return true;
        };
        auto commit = [&](const Result& result) {
            if (!result.valid) {
                return;
            }
            for (int j = 0; j < result.k; ++j) {
                nss::aggregate_add(num + static_cast<std::size_t>(result.slices[j]) * plane_sz,
                                   den + static_cast<std::size_t>(result.slices[j]) * plane_sz, pw,
                                   result.matches[j].x, result.matches[j].y,
                                   result.patches + static_cast<std::size_t>(j) * lda, block, pw, ph, result.weight);
            }
        };
        (void)nss::host_detail::execute_ordered_chunk<Result>(jobs, begin, end, prepare, commit);
    }
}

const VSFrame* VS_CC nlhGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                 VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<NlhData*>(instanceData);
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
    const int q = d->q;

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
        const int work_n = nss::nlh_filter_work_floats(m, group, q, lda);
        const std::size_t need = plane_sz * static_cast<std::size_t>(slices) * 3 +
                                 static_cast<std::size_t>(lda * group) * 2 + static_cast<std::size_t>(work_n) + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = scratch + plane_sz * static_cast<std::size_t>(slices);
        float* basic = den + plane_sz * static_cast<std::size_t>(slices);
        float* patches = basic + plane_sz * static_cast<std::size_t>(slices);
        float* ref_patches = patches + static_cast<std::size_t>(lda * group);
        float* work = ref_patches + static_cast<std::size_t>(lda * group);
        const float sigma = d->sigma[plane] / 255.f;

        run_groups(refs.data(), strides, srcs.data(), strides, ntemp, t0, pw, ph, block, d->block_step, group,
                   d->bm_range, d->radius, d->ps_num, d->ps_range, q, sigma, false, num, den, patches, ref_patches,
                   work);
        for (int sl = 0; sl < slices; ++sl) {
            nss::aggregate_finish(basic + static_cast<std::size_t>(sl) * plane_sz,
                                  num + static_cast<std::size_t>(sl) * plane_sz,
                                  den + static_cast<std::size_t>(sl) * plane_sz, srcs[static_cast<std::size_t>(sl)], pw,
                                  ph, pw, pw, strides[sl]);
        }

        std::vector<const float*> basic_refs(static_cast<std::size_t>(ntemp));
        int basic_strides[nss::kBmMaxRadius * 2 + 1];
        for (int t = 0; t < ntemp; ++t) {
            basic_refs[static_cast<std::size_t>(t)] = basic + static_cast<std::size_t>(t) * plane_sz;
            basic_strides[t] = pw;
        }
        run_groups(basic_refs.data(), basic_strides, srcs.data(), strides, ntemp, t0, pw, ph, block, d->block_step,
                   group, d->bm_range, d->radius, d->ps_num, d->ps_range, q, sigma, true, num, den, patches, ref_patches,
                   work);

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
            nss::aggregate_finish(outp, num, den, srcp, pw, ph, dstride, pw, sstride);
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
        vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    return dst;
}

void VS_CC nlhFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<NlhData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

}  // namespace

VSNode* nss_create_nlh(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.NLH: AVX2 is required");
        return nullptr;
    }
    auto d = std::make_unique<NlhData>();
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
        return fail("nss.NLH: constant Gray/YUV/RGB 32-bit float required");
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kNlhDefaultSigma);
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kNlhDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kNlhDefaultStep);
    d->group_size = nss::map_int(vsapi, in, "group_size", nss::kNlhDefaultGroup);
    d->bm_range = nss::map_int(vsapi, in, "bm_range", nss::kNlhDefaultRange);
    d->radius = nss::map_int(vsapi, in, "radius", 0);
    d->ps_num = nss::map_int(vsapi, in, "ps_num", nss::kBmDefaultPsNum);
    d->ps_range = nss::map_int(vsapi, in, "ps_range", nss::kBmDefaultPsRange);
    d->q = nss::map_int(vsapi, in, "q", nss::kNlhDefaultQ);
    if (d->block_size != 4 && d->block_size != 8) {
        return fail("nss.NLH: block_size must be 4 or 8");
    }
    if (d->group_size != 2 && d->group_size != 4 && d->group_size != 8 && d->group_size != 16) {
        return fail("nss.NLH: group_size must be 2, 4, 8, or 16");
    }
    if (d->block_size > 16 || d->block_step < 1 || d->block_step > d->block_size || d->radius < 0 ||
        d->radius > nss::kBmMaxRadius) {
        return fail("nss.NLH: invalid block_size/block_step/radius");
    }
    if (d->ps_num < 1 || d->ps_num > d->group_size || d->ps_range < 1 || d->ps_range > nss::kBmMaxRange) {
        return fail("nss.NLH: invalid ps_num/ps_range");
    }
    if (d->q != 2 && d->q != 4 && d->q != 8) {
        return fail("nss.NLH: q must be 2, 4, or 8");
    }
    if (d->bm_range < 1 || d->bm_range > nss::kBmMaxRange) {
        return fail("nss.NLH: bm_range must be in [1, 64]");
    }
    int e = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        return fail("nss.NLH: rclip must match clip");
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    NlhData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("NLH", &raw->vi_out, nlhGetFrame, nlhFree, fmParallel, deps, ndeps, raw,
                                             core);
    if (!node) {
        return fail("nss.NLH: failed to create filter");
    }
    d.release();
    return node;
}

void VS_CC nlhCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    VSNode* node = nss_create_nlh(in, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void VS_CC nlhv2Create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    VSNode* wn = nss_create_nlh(in, core, vsapi, out);
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

void register_nlh(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;q:int:opt;rclip:vnode:opt;";
    vspapi->registerFunction("NLH", args, "clip:vnode;", nlhCreate, nullptr, plugin);
    vspapi->registerFunction("NLHv2", args, "clip:vnode;", nlhv2Create, nullptr, plugin);
}
