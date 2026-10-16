// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#include "host/filters.hpp"
#include "frontend/temporal.hpp"
#include "host/batch_runner.hpp"
#include "frontend/validate.hpp"
#include "frontend/mcwnnm_args.hpp"
#include "frontend/contribution.hpp"
#include "nss/backend.hpp"
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
#include <stdexcept>
#include <vector>

namespace {

struct McwnnmData : nss::McwnnmParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    nss::NodeRef rclip;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    nss::Workspace ws;
};

constexpr int kChannels = 3;

// A plane copied unfiltered: as is, or as the identity slices of the fat output.
void write_identity(const McwnnmData* d, float* out, int out_stride, const float* src, int src_stride, int pw, int ph) {
    if (d->radius > 0) {
        nss::host_detail::temporal_identity(out, out_stride, src, src_stride, pw, ph, d->radius);
        return;
    }
    for (int y = 0; y < ph; ++y) {
        std::memcpy(out + y * out_stride, src + y * src_stride, static_cast<std::size_t>(pw) * sizeof(float));
    }
}

// The frame's scratch: num, den, the estimate and the noisy input, each one
// plane per (channel, window slice), channel-major.
struct McwnnmBuffers {
    std::size_t plane_sz = 0;
    int slices = 0;
    float* num = nullptr;
    float* den = nullptr;
    float* est = nullptr;
    float* noisy = nullptr;

    std::size_t floats() const { return plane_sz * static_cast<std::size_t>(kChannels) * static_cast<std::size_t>(slices); }
    std::size_t offset(int plane, int slice) const {
        return (static_cast<std::size_t>(plane) * static_cast<std::size_t>(slices) + static_cast<std::size_t>(slice)) *
               plane_sz;
    }
};

// One frame's rounds: match on the estimates (on rclip in the first round),
// filter each group, and add the groups to num/den in raster order.
struct McwnnmFrame {
    const McwnnmData* d;
    int pw, ph, ntemp, t0, block, group, m, lda;
    float sig[3];
    McwnnmBuffers buf;
    const float* const* est_planes;  // channel-major, ntemp per channel
    const float* const* ref_planes;  // rclip's (or the source's)
    const int* ref_strides;
    int plane_strides[3];            // of the estimates and the sums: tight rows
    nss::SearchConfig cfg;
    std::size_t group_storage;
    int filter_work_floats;
    float* batch_patches;
    float* batch_work;

    // The jobs [begin, end) of one batch window.
    void batch(nss::ResourceVector<nss::GroupJob>& jobs, std::size_t begin, std::size_t end,
               const float* const* match_refs, const int* match_st) const {
        constexpr int kWindow = nss::host_detail::kGroupBatchWindow;
        const int count = static_cast<int>(end - begin);
        const float* match_cur[3] = {match_refs[0 * ntemp + t0], match_refs[1 * ntemp + t0],
                                     match_refs[2 * ntemp + t0]};
        std::array<nss::MatchBatchItem, kWindow> match_items{};
        std::array<int, kWindow> counts{};
        std::array<nss::Match, kWindow * nss::kWnnmMaxGroup> match_storage;
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] = nss::MatchBatchItem{job.x, job.y, block, d->bm_range, group};
        }
        const int match_rc = d->radius > 0
                                 ? nss::predictive_match_nch_batch(match_refs, match_st, kChannels, ntemp, pw, ph, t0,
                                                                    cfg, match_items.data(), count,
                                                                    match_storage.data(), nss::kWnnmMaxGroup,
                                                                    counts.data())
                                 : nss::spatial_match_nch_batch(match_cur, match_st, kChannels, pw, ph,
                                                                 match_items.data(), count, match_storage.data(),
                                                                 nss::kWnnmMaxGroup, counts.data());
        if (match_rc != 0) {
            throw std::runtime_error("nss: matching failed for an active group");
        }
        for (int i = 0; i < count; ++i) {
            jobs[begin + static_cast<std::size_t>(i)].key.k = counts[static_cast<std::size_t>(i)];
        }
        std::array<float, kWindow> weights{};
        std::array<int, kWindow> filter_status{};
        std::array<nss::McwnnmFilterBatchItem, kWindow> filter_items{};
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* p = batch_patches + static_cast<std::size_t>(i) * group_storage;
            for (int j = 0; j < k; ++j) {
                const auto& mm = match_storage[static_cast<std::size_t>(i) * nss::kWnnmMaxGroup + j];
                const int t = d->radius > 0 ? mm.t : t0;
                const float* pack_src[3] = {est_planes[static_cast<std::size_t>(t)],
                                            est_planes[static_cast<std::size_t>(ntemp + t)],
                                            est_planes[static_cast<std::size_t>(2 * ntemp + t)]};
                nss::pack_patch_nch(p + static_cast<std::size_t>(j) * lda, lda, pack_src, plane_strides, kChannels,
                                    mm.x, mm.y, block, pw, ph);
            }
            weights[static_cast<std::size_t>(i)] = 1.f;
            filter_items[static_cast<std::size_t>(i)] = nss::McwnnmFilterBatchItem{
                p, m, k, lda, kChannels, sig, d->admm_iter, d->rho, d->mu, d->residual, d->adaptive,
                &weights[static_cast<std::size_t>(i)],
                batch_work + static_cast<std::size_t>(i) * static_cast<std::size_t>(filter_work_floats),
                filter_work_floats, &filter_status[static_cast<std::size_t>(i)],
                NSS_AVX2_DEFAULTS && block == 8 && group == 8 && d->radius == 0};
        }
        if (nss::mcwnnm_filter_group_batch(filter_items.data(), count) != 0) {
            throw std::runtime_error("nss: numerical group processing failed");
        }

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
            result.patches = batch_patches + i * group_storage;
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
                const int sl = d->radius > 0 ? std::clamp(result.matches[j].t - t0 + d->radius, 0, buf.slices - 1) : 0;
                float* nums[3];
                float* dens[3];
                for (int c = 0; c < kChannels; ++c) {
                    nums[c] = buf.num + buf.offset(c, sl);
                    dens[c] = buf.den + buf.offset(c, sl);
                }
                nss::unpack_patch_nch(nums, dens, plane_strides, kChannels, result.matches[j].x, result.matches[j].y,
                                      result.patches + static_cast<std::size_t>(j) * lda, block, pw, ph,
                                      result.weight);
            }
        };
        (void)nss::host_detail::commit_prepared_chunk<Result>(jobs, begin, end, prepare, commit);
    }

    // Round `iter` into num/den; rounds after the first start from the
    // estimates relaxed towards the noisy input.
    void round(nss::ResourceVector<nss::GroupJob>& jobs, int iter) const {
        if (iter > 0) {
            for (int plane = 0; plane < kChannels; ++plane) {
                for (int sl = 0; sl < buf.slices; ++sl) {
                    const std::size_t off = buf.offset(plane, sl);
                    nss::iter_regularize(buf.est + off, buf.noisy + off, pw * ph, d->delta);
                }
            }
        }
        std::memset(buf.num, 0, buf.floats() * sizeof(float));
        std::memset(buf.den, 0, buf.floats() * sizeof(float));
        const bool use_rclip = iter == 0 && d->rclip != nullptr;
        const float* const* match_refs = use_rclip ? ref_planes : est_planes;
        const int* match_st = use_rclip ? ref_strides : plane_strides;
        for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
            batch(jobs, begin, std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow), match_refs,
                  match_st);
        }
    }

    // The round's result becomes the estimate of every slice.
    void finish_round() const {
        for (int plane = 0; plane < kChannels; ++plane) {
            for (int sl = 0; sl < buf.slices; ++sl) {
                const std::size_t off = buf.offset(plane, sl);
                nss::aggregate_finish(buf.est + off, buf.num + off, buf.den + off, buf.est + off, pw, ph, pw, pw);
            }
        }
    }
};

const VSFrame* VS_CC mcwnnmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<McwnnmData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->radius);
        const int end = nss::host_detail::temporal_last(n,d->radius,d->vi.numFrames);
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
    nss::ResourceVector<const VSFrame*> srcf(static_cast<std::size_t>(ntemp));
    nss::ResourceVector<const VSFrame*> reff(static_cast<std::size_t>(ntemp));
    for (int t = 0; t < ntemp; ++t) {
        const int fn = nss::host_detail::temporal_slot_frame(n,t,d->radius,d->vi.numFrames);
        srcf[static_cast<std::size_t>(t)] = frames_owned.getFrameFilter(fn, d->node, frameCtx);
        reff[static_cast<std::size_t>(t)] = frames_owned.getFrameFilter(fn, d->rclip ? d->rclip : d->node, frameCtx);
    }
    const VSFrame* src0 = srcf[static_cast<std::size_t>(t0)];
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);
    nss::stamp_contribution(dst, d->radius, n, nss::Model::MCWNNM, vsapi);

    const int pw = nss::plane_width(d->vi, 0);
    const int ph = nss::plane_height(d->vi, 0);
    int sstride[3];
    float* outp[3];
    int dstride[3];
    for (int plane = 0; plane < kChannels; ++plane) {
        sstride[plane] = static_cast<int>(frames_owned.getStride(src0, plane) / sizeof(float));
        dstride[plane] = static_cast<int>(frames_owned.getStride(dst, plane) / sizeof(float));
        outp[plane] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
    }
    if (d->sigma[0] == 0.f && d->sigma[1] == 0.f && d->sigma[2] == 0.f) {
        for (int plane = 0; plane < kChannels; ++plane) {
            write_identity(d, outp[plane], dstride[plane],
                           reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane)), sstride[plane], pw, ph);
        }
        for (int t = 0; t < ntemp; ++t) {
            frames_owned.freeFrame(srcf[static_cast<std::size_t>(t)]);
            frames_owned.freeFrame(reff[static_cast<std::size_t>(t)]);
        }
        return frames_owned.keep(dst);
    }

    nss::ResourceVector<const float*> src_planes(static_cast<std::size_t>(kChannels * ntemp));
    nss::ResourceVector<const float*> ref_planes(static_cast<std::size_t>(kChannels * ntemp));
    for (int plane = 0; plane < kChannels; ++plane) {
        for (int t = 0; t < ntemp; ++t) {
            src_planes[static_cast<std::size_t>(plane * ntemp + t)] = frames_owned.readPlane(srcf[static_cast<std::size_t>(t)], plane, sstride[plane]);
            ref_planes[static_cast<std::size_t>(plane * ntemp + t)] = frames_owned.readPlane(reff[static_cast<std::size_t>(t)], plane, sstride[plane]);
        }
    }

    McwnnmBuffers buf;
    buf.plane_sz = static_cast<std::size_t>(pw * ph);
    buf.slices = ntemp;
    float* scratch = d->ws.get(buf.floats() * 4 + 64);
    buf.num = scratch;
    buf.den = buf.num + buf.floats();
    buf.est = buf.den + buf.floats();
    buf.noisy = buf.est + buf.floats();

    // The window's planes as the first estimates (and the noisy input).
    nss::ResourceVector<const float*> est_planes(static_cast<std::size_t>(kChannels * ntemp));
    for (int plane = 0; plane < kChannels; ++plane) {
        for (int t = 0; t < ntemp; ++t) {
            float* e = buf.est + buf.offset(plane, t);
            const float* s = src_planes[static_cast<std::size_t>(plane * ntemp + t)];
            for (int row = 0; row < ph; ++row) {
                std::memcpy(e + static_cast<std::size_t>(row * pw), s + row * sstride[plane],
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
            std::memcpy(buf.noisy + buf.offset(plane, t), e, buf.plane_sz * sizeof(float));
            est_planes[static_cast<std::size_t>(plane * ntemp + t)] = e;
        }
    }

    const int block = d->block_size;
    const int m = kChannels * block * block;
    McwnnmFrame frame{d, pw, ph, ntemp, t0, block, d->group_size, m, (m + 15) & ~15,
                      {d->sigma[0] / 255.f, d->sigma[1] / 255.f, d->sigma[2] / 255.f}, buf, est_planes.data(),
                      ref_planes.data(), sstride, {pw, pw, pw}, {}, 0, 0, nullptr, nullptr};
    frame.cfg.block = block;
    frame.cfg.step = d->block_step;
    frame.cfg.group = d->group_size;
    frame.cfg.bm_range = d->bm_range;
    frame.cfg.radius = d->radius;
    frame.cfg.valid_t_begin = std::max(0, d->radius - n);
    frame.cfg.valid_t_end = d->radius + std::min(d->radius + 1, d->vi.numFrames - n);
    frame.cfg.ps_num = d->ps_num;
    frame.cfg.ps_range = d->ps_range;

    nss::ResourceVector<nss::GroupJob> jobs;
    nss::host_detail::append_raster_jobs(
        jobs, pw, ph, block, d->block_step,
        nss::GroupKey{m, d->group_size, kChannels, nss::GroupAlgorithm::MCWNNM, false, d->residual != 0}, t0);
    // Per-window batch buffers are loop-invariant in size and are fully
    // rewritten before every read (pack fills each used patch column;
    // mcwnnm_admm memsets Z/A and writes X/Temp/SVD outputs before reading
    // them), so allocate once per frame instead of allocating and zeroing
    // ~2.5 MB per 32-group window.
    frame.group_storage = static_cast<std::size_t>(frame.group) * static_cast<std::size_t>(frame.lda);
    frame.filter_work_floats = nss::mcwnnm_filter_work_floats(m, frame.group);
    nss::ResourceVector<float> batch_patches(
        static_cast<std::size_t>(nss::host_detail::kGroupBatchWindow) * frame.group_storage);
    nss::ResourceVector<float> batch_work(static_cast<std::size_t>(nss::host_detail::kGroupBatchWindow) *
                                          static_cast<std::size_t>(frame.filter_work_floats));
    frame.batch_patches = batch_patches.data();
    frame.batch_work = batch_work.data();

    // The fat output takes the last round's sums; otherwise its estimate.
    const int niter = d->iters < 1 ? 1 : d->iters;
    for (int iter = 0; iter < niter; ++iter) {
        frame.round(jobs, iter);
        if (iter == niter - 1 && fat) break;
        frame.finish_round();
    }

    for (int plane = 0; plane < kChannels; ++plane) {
        if (d->sigma[plane] == 0.f) {
            write_identity(d, outp[plane], dstride[plane], src_planes[static_cast<std::size_t>(plane) * ntemp + t0],
                           sstride[plane], pw, ph);
            continue;
        }
        if (fat) {
            for (int sl = 0; sl < ntemp; ++sl) {
                const float* np = buf.num + buf.offset(plane, sl);
                const float* dp = buf.den + buf.offset(plane, sl);
                float* on = outp[plane] + (sl * 2) * ph * dstride[plane];
                float* od = outp[plane] + (sl * 2 + 1) * ph * dstride[plane];
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(on + y * dstride[plane], np + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                    std::memcpy(od + y * dstride[plane], dp + y * pw, static_cast<std::size_t>(pw) * sizeof(float));
                }
            }
        } else {
            const float* e = buf.est + buf.offset(plane, t0);
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp[plane] + y * dstride[plane], e + static_cast<std::size_t>(y * pw),
                            static_cast<std::size_t>(pw) * sizeof(float));
            }
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        frames_owned.freeFrame(srcf[static_cast<std::size_t>(t)]);
        frames_owned.freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    return frames_owned.keep(dst);
}

void VS_CC mcwnnmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<McwnnmData*>(instanceData);
    d->node.reset();
    if (d->rclip) {
        d->rclip.reset();
    }
    delete d;
}

}  // namespace

static VSNode* nss_create_mcwnnm(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(err, "nss.MCWNNM: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return nullptr;
    }
    auto d = std::make_unique<McwnnmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        d->node.reset();
        if (d->rclip) {
            d->rclip.reset();
        }
        return nullptr;
    };
    static_cast<nss::McwnnmParams&>(*d) = nss::frontend::parse_mcwnnm(vsapi, in, d->vi, "nss");
    int e = 0;
    d->rclip = nss::get_node(vsapi, in, "rclip", 0, &e);
    if (e) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        return fail("nss.MCWNNM: rclip must match clip");
    }
    nss::validate_group_planes(d->vi, d->sigma, d->block_size);
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = nss::checked_fat_height(d->vi.height, d->radius);
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->rclip, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->rclip ? 2 : 1;
    McwnnmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("MCWNNM", &raw->vi_out, nss::checked_frame<mcwnnmGetFrame>, mcwnnmFree, fmParallel, deps, ndeps,
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

void VS_CC mcwnnmCreateTemporal(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    nss::frontend::create_temporal(mcwnnmCreate, in, out, core, vsapi, nss::kWnnmDefaultRadius, "MCWNNM", "nss");
}

void register_mcwnnm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args = nss::frontend::kMcwnnmSignature;
    vspapi->registerFunction("MCWNNM", args, "clip:vnode;", nss::checked_create<mcwnnmCreateTemporal>, nullptr, plugin);
}
