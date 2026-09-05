#include "host/filters.hpp"
#include "host/batch_runner.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct McwnnmData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kMcwnnmDefaultSigma, nss::kMcwnnmDefaultSigma, nss::kMcwnnmDefaultSigma};
    int block_size = nss::kMcwnnmDefaultBlock;
    int block_step = nss::kMcwnnmDefaultStep;
    int group_size = nss::kMcwnnmDefaultGroup;
    int bm_range = nss::kMcwnnmDefaultRange;
    int radius = nss::kWnnmDefaultRadius;
    int ps_num = nss::kWnnmDefaultPsNum;
    int ps_range = nss::kWnnmDefaultPsRange;
    int residual = nss::kMcwnnmDefaultResidual;
    int adaptive = nss::kMcwnnmDefaultAdaptive;
    int admm_iter = nss::kMcwnnmDefaultAdmmIter;
    float rho = nss::kMcwnnmDefaultRho;
    float mu = nss::kMcwnnmDefaultMu;
    int iters = nss::kMcwnnmDefaultIters;
    float delta = nss::kMcwnnmDefaultDelta;
    nss::Workspace ws;
};

const VSFrame* VS_CC mcwnnmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<McwnnmData*>(instanceData);
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
    const int nch = 3;
    const int m = nch * block * block;
    const int lda = (m + 15) & ~15;
    const int group = d->group_size;
    const int pw = nss::plane_width(d->vi, 0);
    const int ph = nss::plane_height(d->vi, 0);
    const int slices = ntemp;
    const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);

    float sig[3]{d->sigma[0] / 255.f, d->sigma[1] / 255.f, d->sigma[2] / 255.f};
    const bool all_zero = d->sigma[0] == 0.f && d->sigma[1] == 0.f && d->sigma[2] == 0.f;

    if (all_zero) {
        for (int plane = 0; plane < nch; ++plane) {
            const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
            const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
            float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
            const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
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
        }
        for (int t = 0; t < ntemp; ++t) {
            vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
            vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
        }
        return dst;
    }

    int sstride[3];
    float* outp[3];
    std::vector<const float*> src_planes(static_cast<std::size_t>(nch * ntemp));
    std::vector<const float*> ref_planes(static_cast<std::size_t>(nch * ntemp));
    for (int plane = 0; plane < nch; ++plane) {
        sstride[plane] = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        outp[plane] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        for (int t = 0; t < ntemp; ++t) {
            src_planes[static_cast<std::size_t>(plane * ntemp + t)] = reinterpret_cast<const float*>(
                vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            ref_planes[static_cast<std::size_t>(plane * ntemp + t)] = reinterpret_cast<const float*>(
                vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
        }
    }
    int ch_strides[3]{sstride[0], sstride[1], sstride[2]};

    const std::size_t need = plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * 4 +
                             64;
    float* scratch = d->ws.get(need);
    float* num = scratch;
    float* den = num + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);
    float* est = den + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);
    float* noisy = est + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = d->block_step;
    cfg.group = group;
    cfg.bm_range = d->bm_range;
    cfg.radius = d->radius;
    cfg.ps_num = d->ps_num;
    cfg.ps_range = d->ps_range;

    int agg_strides[3]{pw, pw, pw};
    int est_strides[3]{pw, pw, pw};
    std::vector<const float*> est_planes(static_cast<std::size_t>(nch * ntemp));
    for (int plane = 0; plane < nch; ++plane) {
        for (int t = 0; t < ntemp; ++t) {
            float* e = est + (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                              static_cast<std::size_t>(t)) *
                                 plane_sz;
            float* y = noisy + (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                                static_cast<std::size_t>(t)) *
                                   plane_sz;
            const float* s = src_planes[static_cast<std::size_t>(plane * ntemp + t)];
            for (int row = 0; row < ph; ++row) {
                std::memcpy(e + static_cast<std::size_t>(row * pw), s + row * sstride[plane],
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
            std::memcpy(y, e, plane_sz * sizeof(float));
            est_planes[static_cast<std::size_t>(plane * ntemp + t)] = e;
        }
    }

    const int niter = d->iters < 1 ? 1 : d->iters;
    std::vector<nss::GroupJob> jobs;
    nss::host_detail::append_raster_jobs(
        jobs, pw, ph, block, d->block_step,
        nss::GroupKey{m, group, nch, nss::GroupAlgorithm::MCWNNM, false, d->residual != 0}, t0);
    for (int iter = 0; iter < niter; ++iter) {
        if (iter > 0) {
            for (int plane = 0; plane < nch; ++plane) {
                for (int sl = 0; sl < slices; ++sl) {
                    const std::size_t off =
                        (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) + static_cast<std::size_t>(sl)) *
                        plane_sz;
                    nss::iter_regularize(est + off, noisy + off, pw * ph, d->delta);
                }
            }
        }
        std::memset(num, 0, plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * sizeof(float));
        std::memset(den, 0, plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * sizeof(float));
        const bool use_rclip = (iter == 0 && d->rclip != nullptr);
        const float* const* match_refs = use_rclip ? ref_planes.data() : est_planes.data();
        const int* match_st = use_rclip ? ch_strides : est_strides;
        const float* match_cur[3] = {match_refs[0 * ntemp + t0], match_refs[1 * ntemp + t0], match_refs[2 * ntemp + t0]};

        for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
            const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
            const int count = static_cast<int>(end - begin);
            std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
            std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
            std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kWnnmMaxGroup> match_storage{};
            for (int i = 0; i < count; ++i) {
                const auto& job = jobs[begin + static_cast<std::size_t>(i)];
                match_items[static_cast<std::size_t>(i)] = nss::MatchBatchItem{job.x, job.y, block, d->bm_range, group};
            }
            const int match_rc = d->radius > 0
                                     ? nss::predictive_match_nch_batch(match_refs, match_st, nch, ntemp, pw, ph, t0,
                                                                        cfg, match_items.data(), count,
                                                                        match_storage.data(), nss::kWnnmMaxGroup,
                                                                        counts.data())
                                     : nss::spatial_match_nch_batch(match_cur, match_st, nch, pw, ph, match_items.data(),
                                                                     count, match_storage.data(), nss::kWnnmMaxGroup,
                                                                     counts.data());
            if (match_rc < 0) {
                continue;
            }
            for (int i = 0; i < count; ++i) {
                jobs[begin + static_cast<std::size_t>(i)].key.k = counts[static_cast<std::size_t>(i)];
            }
            const std::size_t group_storage = static_cast<std::size_t>(group) * static_cast<std::size_t>(lda);
            const int filter_work_floats = nss::mcwnnm_filter_work_floats(m, group);
            std::vector<float> batch_patches(static_cast<std::size_t>(count) * group_storage, 0.f);
            std::vector<float> batch_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(filter_work_floats), 0.f);
            std::array<float, nss::host_detail::kGroupBatchWindow> weights{};
            std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
            std::array<nss::McwnnmFilterBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
            for (int i = 0; i < count; ++i) {
                const int k = counts[static_cast<std::size_t>(i)];
                float* p = batch_patches.data() + static_cast<std::size_t>(i) * group_storage;
                for (int j = 0; j < k; ++j) {
                    const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                    const int t = d->radius > 0 ? mm.t : t0;
                    const float* pack_src[3] = {est_planes[static_cast<std::size_t>(t)],
                                                est_planes[static_cast<std::size_t>(ntemp + t)],
                                                est_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                    nss::pack_patch_nch(p + static_cast<std::size_t>(j) * lda, lda, pack_src, est_strides, nch, mm.x,
                                        mm.y, block, pw, ph);
                }
                weights[static_cast<std::size_t>(i)] = 1.f;
                filter_items[static_cast<std::size_t>(i)] = nss::McwnnmFilterBatchItem{
                    p, m, k, lda, nch, sig, d->admm_iter, d->rho, d->mu, d->residual, d->adaptive,
                    &weights[static_cast<std::size_t>(i)],
                    batch_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(filter_work_floats),
                    filter_work_floats, &filter_status[static_cast<std::size_t>(i)]};
            }
            (void)nss::mcwnnm_filter_group_batch(filter_items.data(), count);

            struct Result {
                bool valid = false;
                int k = 0;
                nss::Match matches[nss::kWnnmMaxGroup]{};
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
                    result.matches[j] = match_storage[i * nss::kWnnmMaxGroup + j];
                }
                return true;
            };
            auto commit = [&](const Result& result) {
                if (!result.valid) {
                    return;
                }
                for (int j = 0; j < result.k; ++j) {
                    const int sl = d->radius > 0
                                       ? std::clamp(result.matches[j].t - t0 + d->radius, 0, slices - 1)
                                       : 0;
                    float* nums[3];
                    float* dens[3];
                    for (int c = 0; c < nch; ++c) {
                        const std::size_t off =
                            (static_cast<std::size_t>(c) * static_cast<std::size_t>(slices) + sl) * plane_sz;
                        nums[c] = num + off;
                        dens[c] = den + off;
                    }
                    nss::unpack_patch_nch(nums, dens, agg_strides, nch, result.matches[j].x, result.matches[j].y,
                                          result.patches + static_cast<std::size_t>(j) * lda, block, pw, ph,
                                          result.weight);
                }
            };
            (void)nss::host_detail::commit_prepared_chunk<Result>(jobs, begin, end, prepare, commit);
        }
        const bool last = (iter == niter - 1);
        if (last && fat) {
            break;
        }
        for (int plane = 0; plane < nch; ++plane) {
            for (int sl = 0; sl < slices; ++sl) {
                const std::size_t off =
                    (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) + static_cast<std::size_t>(sl)) *
                    plane_sz;
                nss::aggregate_finish(est + off, num + off, den + off, est + off, pw, ph, pw, pw);
            }
        }
    }

    for (int plane = 0; plane < nch; ++plane) {
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* plane_num = num + static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) * plane_sz;
        float* plane_den = den + static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) * plane_sz;
        if (fat) {
            for (int sl = 0; sl < slices; ++sl) {
                const float* np = plane_num + static_cast<std::size_t>(sl) * plane_sz;
                const float* dp = plane_den + static_cast<std::size_t>(sl) * plane_sz;
                float* on = outp[plane] + (sl * 2) * ph * dstride;
                float* od = outp[plane] + (sl * 2 + 1) * ph * dstride;
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(on + y * dstride, np + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                    std::memcpy(od + y * dstride, dp + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                }
            }
        } else {
            const float* e = est + static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) * plane_sz +
                             static_cast<std::size_t>(t0) * plane_sz;
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp[plane] + y * dstride, e + static_cast<std::size_t>(y * pw),
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

void VS_CC mcwnnmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<McwnnmData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

}  // namespace

static VSNode* nss_create_mcwnnm(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.MCWNNM: AVX2 is required");
        return nullptr;
    }
    auto d = std::make_unique<McwnnmData>();
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
        return fail("nss.MCWNNM: constant RGBS or YUV444PS required");
    }
    if (d->vi.format.numPlanes != 3 || d->vi.format.subSamplingW != 0 || d->vi.format.subSamplingH != 0 ||
        (d->vi.format.colorFamily != cfRGB && d->vi.format.colorFamily != cfYUV)) {
        return fail("nss.MCWNNM: constant RGBS or YUV444PS required");
    }
    nss::map_float_array(vsapi, in, "sigma", d->sigma, 3, nss::kMcwnnmDefaultSigma);
    for (int c = 0; c < 3; ++c) {
        if (!nss::is_finite_bits(d->sigma[c])) {
            return fail("nss.MCWNNM: sigma must be finite");
        }
    }
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kMcwnnmDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kMcwnnmDefaultStep);
    d->group_size = nss::map_int(vsapi, in, "group_size", nss::kMcwnnmDefaultGroup);
    d->bm_range = nss::map_int(vsapi, in, "bm_range", nss::kMcwnnmDefaultRange);
    d->radius = nss::map_int(vsapi, in, "radius", nss::kWnnmDefaultRadius);
    d->ps_num = nss::map_int(vsapi, in, "ps_num", nss::kWnnmDefaultPsNum);
    d->ps_range = nss::map_int(vsapi, in, "ps_range", nss::kWnnmDefaultPsRange);
    d->residual = nss::map_int(vsapi, in, "residual", nss::kMcwnnmDefaultResidual);
    d->adaptive = nss::map_int(vsapi, in, "adaptive_aggregation", nss::kMcwnnmDefaultAdaptive);
    d->admm_iter = nss::map_int(vsapi, in, "admm_iter", nss::kMcwnnmDefaultAdmmIter);
    d->rho = nss::map_float(vsapi, in, "rho", nss::kMcwnnmDefaultRho);
    d->mu = nss::map_float(vsapi, in, "mu", nss::kMcwnnmDefaultMu);
    d->iters = nss::map_int(vsapi, in, "iters", nss::kMcwnnmDefaultIters);
    d->delta = nss::map_float(vsapi, in, "delta", nss::kMcwnnmDefaultDelta);
    if (d->block_size < 1 || d->block_size > nss::kWnnmMaxBlock || d->group_size < 1 ||
        d->group_size > nss::kWnnmMaxGroup || d->block_step < 1 || d->block_step > d->block_size || d->radius < 0 ||
        d->radius > nss::kBmMaxRadius) {
        return fail("nss.MCWNNM: invalid block_size/group_size/block_step/radius");
    }
    if (3 * d->block_size * d->block_size > nss::kSvdMaxM) {
        return fail("nss.MCWNNM: 3*block_size*block_size exceeds SVD limit");
    }
    if (d->bm_range < 1 || d->bm_range > nss::kBmMaxRange) {
        return fail("nss.MCWNNM: bm_range must be in [1, 64]");
    }
    if (d->admm_iter < 1 || !(d->rho > 0.f) || d->mu < 1.f || !nss::is_finite_bits(d->rho) ||
        !nss::is_finite_bits(d->mu)) {
        return fail("nss.MCWNNM: invalid admm_iter/rho/mu (mu >= 1)");
    }
    if (d->iters < 1 || !nss::is_finite_bits(d->delta)) {
        return fail("nss.MCWNNM: invalid iters/delta");
    }
    if (d->ps_num < 1 || d->ps_num > d->group_size || d->ps_range < 1 || d->ps_range > nss::kBmMaxRange) {
        return fail("nss.MCWNNM: invalid ps_num/ps_range");
    }
    int e = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        return fail("nss.MCWNNM: rclip must match clip");
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    McwnnmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("MCWNNM", &raw->vi_out, mcwnnmGetFrame, mcwnnmFree, fmParallel, deps, ndeps,
                                            raw, core);
    if (!node) {
        return fail("nss.MCWNNM: failed to create filter");
    }
    d.release();
    return node;
}

void VS_CC mcwnnmCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    VSNode* node = nss_create_mcwnnm(in, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void register_mcwnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;residual:int:opt;"
        "adaptive_aggregation:int:opt;rclip:vnode:opt;admm_iter:int:opt;rho:float:opt;mu:float:opt;"
        "iters:int:opt;delta:float:opt;";
    vspapi->registerFunction("MCWNNM", args, "clip:vnode;", mcwnnmCreate, nullptr, plugin);
}
