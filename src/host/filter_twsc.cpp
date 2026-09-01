#include "host/batch_runner.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/cpu_twsc.hpp"
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

struct TwscData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    float sigma[3]{nss::kTwscDefaultSigma, nss::kTwscDefaultSigma, nss::kTwscDefaultSigma};
    int block_size = nss::kTwscDefaultBlock;
    int block_step = nss::kTwscDefaultStep;
    int group_size = nss::kTwscDefaultGroup;
    int bm_range = nss::kTwscDefaultRange;
    int radius = nss::kWnnmDefaultRadius;
    int ps_num = nss::kWnnmDefaultPsNum;
    int ps_range = nss::kWnnmDefaultPsRange;
    float lambda2 = nss::kTwscDefaultLambda2;
    int iters = nss::kTwscDefaultIters;
    float delta = nss::kTwscDefaultDelta;
    nss::Workspace ws;
};

const VSFrame* VS_CC twscGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<TwscData*>(instanceData);
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
    const bool joint = d->vi.format.numPlanes == 3 && d->vi.format.subSamplingW == 0 && d->vi.format.subSamplingH == 0;

    auto write_fat_identity = [&](int plane, const float* srcp, float* outp, int sstride, int dstride, int pw, int ph) {
        if (!fat) {
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp + y * dstride, srcp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
            }
            return;
        }
        for (int sl = 0; sl < ntemp; ++sl) {
            const float* sp = reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(sl)], plane));
            float* on = outp + (sl * 2) * ph * dstride;
            float* od = outp + (sl * 2 + 1) * ph * dstride;
            for (int y = 0; y < ph; ++y) {
                std::memcpy(on + y * dstride, sp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
                for (int x = 0; x < pw; ++x) {
                    od[y * dstride + x] = 1.f;
                }
            }
        }
    };

    if (joint) {
        const int nch = 3;
        const int m = nch * block * block;
        const int lda = (m + 15) & ~15;
        const int pw = nss::plane_width(d->vi, 0);
        const int ph = nss::plane_height(d->vi, 0);
        const int slices = ntemp;
        const std::size_t plane_sz = static_cast<std::size_t>(pw * ph);
        float sig[3]{d->sigma[0] / 255.f, d->sigma[1] / 255.f, d->sigma[2] / 255.f};
        const bool all_zero = d->sigma[0] == 0.f && d->sigma[1] == 0.f && d->sigma[2] == 0.f;
        int sstride[3];
        float* outp[3];
        const float* srcp0[3];
        std::vector<const float*> src_planes(static_cast<std::size_t>(nch * ntemp));
        std::vector<const float*> ref_planes(static_cast<std::size_t>(nch * ntemp));
        for (int plane = 0; plane < nch; ++plane) {
            sstride[plane] = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
            outp[plane] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
            srcp0[plane] = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
            for (int t = 0; t < ntemp; ++t) {
                src_planes[static_cast<std::size_t>(plane * ntemp + t)] =
                    reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
                ref_planes[static_cast<std::size_t>(plane * ntemp + t)] =
                    reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
            }
        }
        if (all_zero) {
            for (int plane = 0; plane < nch; ++plane) {
                write_fat_identity(plane, srcp0[plane], outp[plane], sstride[plane],
                                   static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float)), pw, ph);
            }
            for (int t = 0; t < ntemp; ++t) {
                vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
                vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
            }
            return dst;
        }
        int ch_strides[3]{sstride[0], sstride[1], sstride[2]};
        const std::size_t need = plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * 4 + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = num + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);
        float* est = den + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);
        float* noisyb = est + plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices);
        nss::SearchConfig cfg;
        cfg.block = block;
        cfg.step = d->block_step;
        cfg.group = group;
        cfg.bm_range = d->bm_range;
        cfg.radius = d->radius;
        cfg.ps_num = d->ps_num;
        cfg.ps_range = d->ps_range;
        float sigma = sig[0];
        for (int c = 1; c < nch; ++c) {
            if (sig[c] > 0.f) {
                sigma = (sigma > 0.f) ? std::min(sigma, sig[c]) : sig[c];
            }
        }
        int agg_strides[3]{pw, pw, pw};
        int est_strides[3]{pw, pw, pw};
        std::vector<const float*> est_planes(static_cast<std::size_t>(nch * ntemp));
        std::vector<const float*> noisy_planes(static_cast<std::size_t>(nch * ntemp));
        for (int plane = 0; plane < nch; ++plane) {
            for (int t = 0; t < ntemp; ++t) {
                float* e = est + (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                                  static_cast<std::size_t>(t)) *
                                     plane_sz;
                float* y = noisyb + (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                                     static_cast<std::size_t>(t)) *
                                        plane_sz;
                const float* s = src_planes[static_cast<std::size_t>(plane * ntemp + t)];
                for (int row = 0; row < ph; ++row) {
                    std::memcpy(e + static_cast<std::size_t>(row * pw), s + row * sstride[plane],
                                static_cast<std::size_t>(pw) * sizeof(float));
                }
                std::memcpy(y, e, plane_sz * sizeof(float));
                est_planes[static_cast<std::size_t>(plane * ntemp + t)] = e;
                noisy_planes[static_cast<std::size_t>(plane * ntemp + t)] = y;
            }
        }
        const int niter = d->iters < 1 ? 1 : d->iters;
        float col_sigma[nss::kWnnmMaxGroup];
        float col_w[nss::kWnnmMaxGroup];
        float row_w[nss::kSvdMaxM];
        {
            const int area = block * block;
            for (int c = 0; c < nch; ++c) {
                float wc = 1.f;
                if (sig[c] > 0.f && sigma > 0.f) {
                    wc = sigma / sig[c];
                }
                for (int i = 0; i < area && c * area + i < nss::kSvdMaxM; ++i) {
                    row_w[c * area + i] = wc;
                }
            }
        }
        const bool use_lane_batch = group == 8 && m >= 8 && m <= nss::kSvdBatch8MaxM;
        const std::size_t group_storage = static_cast<std::size_t>(group) * static_cast<std::size_t>(lda);
        const int work_n = nss::twsc_pca_soft_work_floats(m, group);
        std::vector<float> scalar_patches;
        std::vector<float> scalar_noisy;
        std::vector<float> scalar_work;
        if (!use_lane_batch) {
            scalar_patches.resize(group_storage);
            scalar_noisy.resize(group_storage);
            scalar_work.resize(static_cast<std::size_t>(work_n));
        }
        for (int iter = 0; iter < niter; ++iter) {
            if (iter > 0) {
                for (int plane = 0; plane < nch; ++plane) {
                    for (int sl = 0; sl < slices; ++sl) {
                        const std::size_t off = (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                                                 static_cast<std::size_t>(sl)) *
                                                plane_sz;
                        nss::iter_regularize(est + off, noisyb + off, pw * ph, d->delta);
                    }
                }
            }
            std::memset(num, 0,
                        plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * sizeof(float));
            std::memset(den, 0,
                        plane_sz * static_cast<std::size_t>(nch) * static_cast<std::size_t>(slices) * sizeof(float));
            const bool use_rclip = (iter == 0 && d->rclip != nullptr);
            const float* const* match_refs = use_rclip ? ref_planes.data() : est_planes.data();
            const int* match_st = use_rclip ? ch_strides : est_strides;
            const float* match_cur[3] = {match_refs[0 * ntemp + t0], match_refs[1 * ntemp + t0],
                                         match_refs[2 * ntemp + t0]};
            std::vector<nss::GroupJob> jobs;
            nss::host_detail::append_raster_jobs(
                jobs, pw, ph, block, d->block_step,
                nss::GroupKey{m, group, nch, nss::GroupAlgorithm::TWSC, false, false}, t0);
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
                                         : nss::spatial_match_nch_batch(match_cur, match_st, nch, pw, ph,
                                                                         match_items.data(), count, match_storage.data(),
                                                                         nss::kWnnmMaxGroup, counts.data());
                if (match_rc < 0) {
                    continue;
                }
                if (!use_lane_batch) {
                    for (int i = 0; i < count; ++i) {
                        const int k = counts[static_cast<std::size_t>(i)];
                        if (k <= 0) {
                            continue;
                        }
                        std::fill(col_sigma, col_sigma + k, sigma);
                        std::fill(col_w, col_w + k, 1.f);
                        for (int j = 0; j < k; ++j) {
                            const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                            const int t = d->radius > 0 ? mm.t : t0;
                            const float* pack_src[3] = {est_planes[static_cast<std::size_t>(t)],
                                                        est_planes[static_cast<std::size_t>(ntemp + t)],
                                                        est_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                            nss::pack_patch_nch(scalar_patches.data() + static_cast<std::size_t>(j) * lda, lda,
                                                pack_src, est_strides, nch, mm.x, mm.y, block, pw, ph);
                            if (iter > 0) {
                                const float* noisy_src[3] = {noisy_planes[static_cast<std::size_t>(t)],
                                                             noisy_planes[static_cast<std::size_t>(ntemp + t)],
                                                             noisy_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                                nss::pack_patch_nch(scalar_noisy.data() + static_cast<std::size_t>(j) * lda, lda,
                                                    noisy_src, est_strides, nch, mm.x, mm.y, block, pw, ph);
                                const float ms = nss::ssd_vec(
                                                     scalar_noisy.data() + static_cast<std::size_t>(j) * lda,
                                                     scalar_patches.data() + static_cast<std::size_t>(j) * lda, m) /
                                                 static_cast<float>(m);
                                col_sigma[j] = d->lambda2 * std::sqrt(std::fabs(sigma * sigma - ms));
                            }
                        }
                        if (nss::twsc_pca_soft(scalar_patches.data(), m, k, lda, sigma, scalar_work.data(), work_n,
                                               col_sigma, col_w, row_w) != 0) {
                            continue;
                        }
                        for (int j = 0; j < k; ++j) {
                            const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                            const int sl = d->radius > 0 ? std::clamp(mm.t - t0 + d->radius, 0, slices - 1) : 0;
                            float* nums[3];
                            float* dens[3];
                            for (int c = 0; c < nch; ++c) {
                                const std::size_t off =
                                    (static_cast<std::size_t>(c) * static_cast<std::size_t>(slices) + sl) * plane_sz;
                                nums[c] = num + off;
                                dens[c] = den + off;
                            }
                            nss::unpack_patch_nch(nums, dens, agg_strides, nch, mm.x, mm.y,
                                                  scalar_patches.data() + static_cast<std::size_t>(j) * lda, block, pw,
                                                  ph, col_w[j]);
                        }
                    }
                    continue;
                }
                std::vector<float> batch_patches(static_cast<std::size_t>(count) * group_storage, 0.f);
                std::vector<float> batch_noisy(static_cast<std::size_t>(count) * group_storage, 0.f);
                std::vector<float> batch_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(work_n), 0.f);
                std::vector<float> batch_col_sigma(static_cast<std::size_t>(count) * static_cast<std::size_t>(group), sigma);
                std::vector<float> batch_col_w(static_cast<std::size_t>(count) * static_cast<std::size_t>(group), 1.f);
                std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
                std::array<nss::TwscPcaBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
                for (int i = 0; i < count; ++i) {
                    const int k = counts[static_cast<std::size_t>(i)];
                    float* p = batch_patches.data() + static_cast<std::size_t>(i) * group_storage;
                    float* np = batch_noisy.data() + static_cast<std::size_t>(i) * group_storage;
                    float* cs = batch_col_sigma.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group);
                    float* cw = batch_col_w.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group);
                    for (int j = 0; j < k; ++j) {
                        const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                        const int t = d->radius > 0 ? mm.t : t0;
                        const float* pack_src[3] = {est_planes[static_cast<std::size_t>(t)],
                                                    est_planes[static_cast<std::size_t>(ntemp + t)],
                                                    est_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                        nss::pack_patch_nch(p + static_cast<std::size_t>(j) * lda, lda, pack_src, est_strides, nch,
                                            mm.x, mm.y, block, pw, ph);
                        if (iter > 0) {
                            const float* noisy_src[3] = {noisy_planes[static_cast<std::size_t>(t)],
                                                         noisy_planes[static_cast<std::size_t>(ntemp + t)],
                                                         noisy_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                            nss::pack_patch_nch(np + static_cast<std::size_t>(j) * lda, lda, noisy_src, est_strides, nch,
                                                mm.x, mm.y, block, pw, ph);
                            const float ms = nss::ssd_vec(np + static_cast<std::size_t>(j) * lda,
                                                          p + static_cast<std::size_t>(j) * lda, m) /
                                             static_cast<float>(m);
                            cs[j] = d->lambda2 * std::sqrt(std::fabs(sigma * sigma - ms));
                        }
                    }
                    filter_items[static_cast<std::size_t>(i)] = nss::TwscPcaBatchItem{
                        p, m, k, lda, sigma,
                        batch_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(work_n), work_n,
                        cs, cw, row_w, &filter_status[static_cast<std::size_t>(i)]};
                }
                (void)nss::twsc_pca_soft_batch(filter_items.data(), count);

                struct Result {
                    bool valid = false;
                    int k = 0;
                    nss::Match matches[nss::kWnnmMaxGroup]{};
                    const float* patches = nullptr;
                    const float* weights = nullptr;
                };
                auto prepare = [&](const nss::GroupJob& job, Result& result) {
                    const std::size_t i = static_cast<std::size_t>(job.ordinal - jobs[begin].ordinal);
                    if (i >= static_cast<std::size_t>(count) || counts[i] <= 0 || filter_status[i] == 0) {
                        return false;
                    }
                    result.valid = true;
                    result.k = counts[i];
                    result.patches = batch_patches.data() + i * group_storage;
                    result.weights = batch_col_w.data() + i * static_cast<std::size_t>(group);
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
                                              result.weights[j]);
                    }
                };
                (void)nss::host_detail::execute_ordered_chunk<Result>(jobs, begin, end, prepare, commit);
            }
            const bool last = (iter == niter - 1);
            if (last && fat) {
                break;
            }
            for (int plane = 0; plane < nch; ++plane) {
                for (int sl = 0; sl < slices; ++sl) {
                    const std::size_t off = (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) +
                                             static_cast<std::size_t>(sl)) *
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

    const int m = block * block;
    const int lda = (m + 15) & ~15;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            write_fat_identity(plane, srcp, outp, sstride, dstride, pw, ph);
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
        const std::size_t need = plane_sz * static_cast<std::size_t>(slices) * 4 + 64;
        float* scratch = d->ws.get(need);
        float* num = scratch;
        float* den = scratch + plane_sz * static_cast<std::size_t>(slices);
        float* estp = den + plane_sz * static_cast<std::size_t>(slices);
        float* noisyp = estp + plane_sz * static_cast<std::size_t>(slices);

        nss::SearchConfig cfg;
        cfg.block = block;
        cfg.step = d->block_step;
        cfg.group = group;
        cfg.bm_range = d->bm_range;
        cfg.radius = d->radius;
        cfg.ps_num = d->ps_num;
        cfg.ps_range = d->ps_range;

        const float sigma = d->sigma[plane] / 255.f;
        std::vector<const float*> est_refs(static_cast<std::size_t>(ntemp));
        std::vector<const float*> noisy_refs(static_cast<std::size_t>(ntemp));
        int est_strides[nss::kBmMaxRadius * 2 + 1];
        for (int t = 0; t < ntemp; ++t) {
            float* e = estp + static_cast<std::size_t>(t) * plane_sz;
            float* y = noisyp + static_cast<std::size_t>(t) * plane_sz;
            const float* s = srcs[static_cast<std::size_t>(t)];
            for (int row = 0; row < ph; ++row) {
                std::memcpy(e + static_cast<std::size_t>(row * pw), s + row * sstride,
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
            std::memcpy(y, e, plane_sz * sizeof(float));
            est_refs[static_cast<std::size_t>(t)] = e;
            noisy_refs[static_cast<std::size_t>(t)] = y;
            est_strides[t] = pw;
        }
        const int niter = d->iters < 1 ? 1 : d->iters;
        std::vector<nss::GroupJob> jobs;
        nss::host_detail::append_raster_jobs(
            jobs, pw, ph, block, d->block_step,
            nss::GroupKey{m, group, 1, nss::GroupAlgorithm::TWSC, false, false}, t0);
        const bool use_lane_batch = group == 8 && m >= 8 && m <= nss::kSvdBatch8MaxM;
        const std::size_t group_storage = static_cast<std::size_t>(group) * static_cast<std::size_t>(lda);
        const int work_n = nss::twsc_pca_soft_work_floats(m, group);
        std::vector<float> scalar_patches;
        std::vector<float> scalar_noisy;
        std::vector<float> scalar_work;
        std::vector<float> scalar_col_sigma;
        std::vector<float> scalar_col_w;
        if (!use_lane_batch) {
            scalar_patches.resize(group_storage);
            scalar_noisy.resize(group_storage);
            scalar_work.resize(static_cast<std::size_t>(work_n));
            scalar_col_sigma.resize(static_cast<std::size_t>(group));
            scalar_col_w.resize(static_cast<std::size_t>(group));
        }
        for (int iter = 0; iter < niter; ++iter) {
            if (iter > 0) {
                for (int sl = 0; sl < slices; ++sl) {
                    nss::iter_regularize(estp + static_cast<std::size_t>(sl) * plane_sz,
                                         noisyp + static_cast<std::size_t>(sl) * plane_sz, pw * ph, d->delta);
                }
            }
            std::memset(num, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
            std::memset(den, 0, plane_sz * static_cast<std::size_t>(slices) * sizeof(float));
            const bool use_rclip = (iter == 0 && d->rclip != nullptr);
            const float* const* match_refs = use_rclip ? refs.data() : est_refs.data();
            const int* match_st = use_rclip ? strides : est_strides;
            for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
                const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
                const int count = static_cast<int>(end - begin);
                std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
                std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
                std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kWnnmMaxGroup> match_storage{};
                for (int i = 0; i < count; ++i) {
                    const auto& job = jobs[begin + static_cast<std::size_t>(i)];
                    match_items[static_cast<std::size_t>(i)] =
                        nss::MatchBatchItem{job.x, job.y, block, d->bm_range, group};
                }
                const int match_rc = d->radius > 0
                                         ? nss::predictive_match_batch(match_refs, match_st, ntemp, pw, ph, t0, cfg,
                                                                       match_items.data(), count, match_storage.data(),
                                                                       nss::kWnnmMaxGroup, counts.data())
                                         : nss::spatial_match_batch(match_refs[t0], match_st[t0], pw, ph,
                                                                    match_items.data(), count, match_storage.data(),
                                                                    nss::kWnnmMaxGroup, counts.data());
                if (match_rc < 0) {
                    continue;
                }
                if (!use_lane_batch) {
                    for (int i = 0; i < count; ++i) {
                        const int k = counts[static_cast<std::size_t>(i)];
                        if (k <= 0) {
                            continue;
                        }
                        std::fill(scalar_col_sigma.begin(), scalar_col_sigma.begin() + k, sigma);
                        std::fill(scalar_col_w.begin(), scalar_col_w.begin() + k, 1.f);
                        for (int j = 0; j < k; ++j) {
                            const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                            const int t = d->radius > 0 ? mm.t : t0;
                            nss::pack_patch(scalar_patches.data() + static_cast<std::size_t>(j) * lda, lda,
                                            est_refs[static_cast<std::size_t>(t)], pw, mm.x, mm.y, block, pw, ph);
                            if (iter > 0) {
                                nss::pack_patch(scalar_noisy.data() + static_cast<std::size_t>(j) * lda, lda,
                                                noisy_refs[static_cast<std::size_t>(t)], pw, mm.x, mm.y, block, pw, ph);
                                const float ms = nss::ssd_vec(
                                                     scalar_noisy.data() + static_cast<std::size_t>(j) * lda,
                                                     scalar_patches.data() + static_cast<std::size_t>(j) * lda, m) /
                                                 static_cast<float>(m);
                                scalar_col_sigma[static_cast<std::size_t>(j)] =
                                    d->lambda2 * std::sqrt(std::fabs(sigma * sigma - ms));
                            }
                        }
                        if (nss::twsc_pca_soft(scalar_patches.data(), m, k, lda, sigma, scalar_work.data(), work_n,
                                               scalar_col_sigma.data(), scalar_col_w.data()) != 0) {
                            continue;
                        }
                        for (int j = 0; j < k; ++j) {
                            const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                            const int sl = d->radius > 0 ? std::clamp(mm.t - t0 + d->radius, 0, slices - 1) : 0;
                            nss::aggregate_add(num + static_cast<std::size_t>(sl) * plane_sz,
                                               den + static_cast<std::size_t>(sl) * plane_sz, pw, mm.x, mm.y,
                                               scalar_patches.data() + static_cast<std::size_t>(j) * lda, block, pw, ph,
                                               scalar_col_w[static_cast<std::size_t>(j)]);
                        }
                    }
                    continue;
                }
                std::vector<float> batch_patches(static_cast<std::size_t>(count) * group_storage, 0.f);
                std::vector<float> batch_noisy;
                if (iter > 0) {
                    batch_noisy.resize(batch_patches.size(), 0.f);
                }
                std::vector<float> batch_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(work_n), 0.f);
                std::vector<float> batch_col_sigma(static_cast<std::size_t>(count) * static_cast<std::size_t>(group), sigma);
                std::vector<float> batch_col_w(static_cast<std::size_t>(count) * static_cast<std::size_t>(group), 1.f);
                std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
                std::array<nss::TwscPcaBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
                for (int i = 0; i < count; ++i) {
                    const int k = counts[static_cast<std::size_t>(i)];
                    float* p = batch_patches.data() + static_cast<std::size_t>(i) * group_storage;
                    float* np = iter > 0 ? batch_noisy.data() + static_cast<std::size_t>(i) * group_storage : nullptr;
                    float* cs = batch_col_sigma.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group);
                    float* cw = batch_col_w.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group);
                    for (int j = 0; j < k; ++j) {
                        const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                        const int t = d->radius > 0 ? mm.t : t0;
                        nss::pack_patch(p + static_cast<std::size_t>(j) * lda, lda,
                                        est_refs[static_cast<std::size_t>(t)], pw, mm.x, mm.y, block, pw, ph);
                        if (np) {
                            nss::pack_patch(np + static_cast<std::size_t>(j) * lda, lda,
                                            noisy_refs[static_cast<std::size_t>(t)], pw, mm.x, mm.y, block, pw, ph);
                            const float ms = nss::ssd_vec(np + static_cast<std::size_t>(j) * lda,
                                                          p + static_cast<std::size_t>(j) * lda, m) /
                                             static_cast<float>(m);
                            cs[j] = d->lambda2 * std::sqrt(std::fabs(sigma * sigma - ms));
                        }
                    }
                    filter_items[static_cast<std::size_t>(i)] = nss::TwscPcaBatchItem{
                        p, m, k, lda, sigma,
                        batch_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(work_n), work_n,
                        cs, cw, nullptr, &filter_status[static_cast<std::size_t>(i)]};
                }
                (void)nss::twsc_pca_soft_batch(filter_items.data(), count);

                struct Result {
                    bool valid = false;
                    int k = 0;
                    nss::Match matches[nss::kWnnmMaxGroup]{};
                    const float* patches = nullptr;
                    const float* weights = nullptr;
                };
                auto prepare = [&](const nss::GroupJob& job, Result& result) {
                    const std::size_t i = static_cast<std::size_t>(job.ordinal - jobs[begin].ordinal);
                    if (i >= static_cast<std::size_t>(count) || counts[i] <= 0 || filter_status[i] == 0) {
                        return false;
                    }
                    result.valid = true;
                    result.k = counts[i];
                    result.patches = batch_patches.data() + i * group_storage;
                    result.weights = batch_col_w.data() + i * static_cast<std::size_t>(group);
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
                        nss::aggregate_add(num + static_cast<std::size_t>(sl) * plane_sz,
                                           den + static_cast<std::size_t>(sl) * plane_sz, pw,
                                           result.matches[j].x, result.matches[j].y,
                                           result.patches + static_cast<std::size_t>(j) * lda, block, pw, ph,
                                           result.weights[j]);
                    }
                };
                (void)nss::host_detail::execute_ordered_chunk<Result>(jobs, begin, end, prepare, commit);
            }
            const bool last = (iter == niter - 1);
            if (last && fat) {
                break;
            }
            for (int sl = 0; sl < slices; ++sl) {
                nss::aggregate_finish(estp + static_cast<std::size_t>(sl) * plane_sz,
                                      num + static_cast<std::size_t>(sl) * plane_sz,
                                      den + static_cast<std::size_t>(sl) * plane_sz,
                                      estp + static_cast<std::size_t>(sl) * plane_sz, pw, ph, pw, pw);
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
            const float* e = estp + static_cast<std::size_t>(t0) * plane_sz;
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

void VS_CC twscFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<TwscData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

}  // namespace

void VS_CC twscCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(out, "nss.TWSC: AVX2 is required");
        return;
    }
    auto d = std::make_unique<TwscData>();
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
        fail("nss.TWSC: constant Gray/YUV/RGB 32-bit float required");
        return;
    }
    const int np = d->vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d->sigma, np, nss::kTwscDefaultSigma);
    d->block_size = nss::map_int(vsapi, in, "block_size", nss::kTwscDefaultBlock);
    d->block_step = nss::map_int(vsapi, in, "block_step", nss::kTwscDefaultStep);
    d->group_size = nss::map_int(vsapi, in, "group_size", nss::kTwscDefaultGroup);
    d->bm_range = nss::map_int(vsapi, in, "bm_range", nss::kTwscDefaultRange);
    d->radius = nss::map_int(vsapi, in, "radius", nss::kWnnmDefaultRadius);
    d->ps_num = nss::map_int(vsapi, in, "ps_num", nss::kWnnmDefaultPsNum);
    d->ps_range = nss::map_int(vsapi, in, "ps_range", nss::kWnnmDefaultPsRange);
    const float lambda1 = nss::map_float(vsapi, in, "lambda1", nss::kTwscDefaultLambda1);
    d->lambda2 = nss::map_float(vsapi, in, "lambda2", nss::kTwscDefaultLambda2);
    d->iters = nss::map_int(vsapi, in, "iters", nss::kTwscDefaultIters);
    d->delta = nss::map_float(vsapi, in, "delta", nss::kTwscDefaultDelta);
    if (lambda1 != 0.f) {
        fail("nss.TWSC: only lambda1=0");
        return;
    }
    if (d->lambda2 < 0.f || !std::isfinite(d->lambda2)) {
        fail("nss.TWSC: lambda2 must be non-negative");
        return;
    }
    if (d->iters < 1 || !std::isfinite(d->delta)) {
        fail("nss.TWSC: invalid iters/delta");
        return;
    }
    if (d->block_size < 1 || d->block_size > nss::kWnnmMaxBlock || d->group_size < 1 ||
        d->group_size > nss::kWnnmMaxGroup || d->block_step < 1 || d->block_step > d->block_size || d->radius < 0 ||
        d->radius > nss::kBmMaxRadius) {
        fail("nss.TWSC: invalid block_size/group_size/block_step/radius");
        return;
    }
    if (d->vi.format.numPlanes == 3 && d->vi.format.subSamplingW == 0 && d->vi.format.subSamplingH == 0 &&
        3 * d->block_size * d->block_size > nss::kSvdMaxM) {
        fail("nss.TWSC: 3*block_size*block_size exceeds SVD limit");
        return;
    }
    if (d->ps_num < 1 || d->ps_num > d->group_size || d->ps_range < 1 || d->ps_range > nss::kBmMaxRange) {
        fail("nss.TWSC: invalid ps_num/ps_range");
        return;
    }
    if (d->bm_range < 1 || d->bm_range > nss::kBmMaxRange) {
        fail("nss.TWSC: bm_range must be in [1, 64]");
        return;
    }
    int e = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        fail("nss.TWSC: rclip must match clip");
        return;
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = d->vi.height * (2 * d->radius + 1) * 2;
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    TwscData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("TWSC", &raw->vi_out, twscGetFrame, twscFree, fmParallel, deps, ndeps, raw,
                                             core);
    if (!node) {
        fail("nss.TWSC: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

void register_twsc(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args =
        "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
        "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;"
        "lambda1:float:opt;lambda2:float:opt;rclip:vnode:opt;iters:int:opt;delta:float:opt;";
    vspapi->registerFunction("TWSC", args, "clip:vnode;", twscCreate, nullptr, plugin);
}
