// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#if NSS_BM_EXPERIMENT & 64
#include "cpu/bm/sliding-batch.hpp"
#endif
#include "host/filters.hpp"
#include "frontend/temporal.hpp"
#include "host/batch_runner.hpp"
#include "frontend/validate.hpp"
#include "frontend/contribution.hpp"
#include "frontend/bm3d_args.hpp"
#include "frontend/temporal_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Bm3dData : nss::Bm3dParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    nss::NodeRef ref;
    VSVideoInfo vi{};
    VSVideoInfo vi_out{};
    nss::Workspace ws;
};

struct Bm3dBatchResult {
    bool valid = false;
    int k = 0;
    int slice[ nss::kBmMaxGroup ]{};
    nss::Match matches[nss::kBmMaxGroup]{};
    const float* patches = nullptr;
    float weight = 1.f;
};

// One plane's share of a pass over the groups: its frames, its Wiener
// reference, and where its sums and its result go. scratch holds the plane's
// num and den slices.
struct Bm3dChannel {
    const float* const* srcs;
    const int* src_strides;
    const float* const* refs;
    const int* ref_strides;
    float* dst;
    int dstride;
    int fat_stride;
    float sigma;
    float* scratch;
};

// Matches every group once on `match` and filters each channel with those
// groups. Without chroma there is one channel and `match` is its reference;
// with chroma (CBM3D) the channels are the planes of a 4:4:4 frame and
// `match` is plane 0.
void process_planes_batched(const float* const* match, const int* match_strides, const Bm3dChannel* channels,
                            int nch, int ntemp, int t0, int width, int height,
                            int block, int group, int step, int bm_range, int ps_num, int ps_range,
                            int radius, bool wiener, bool emit_fat, int center, int frame_count
#if NSS_BM_SCRATCH
                            , bool scratch_only=false
#endif
                            ) {
#if NSS_BM_EXPERIMENT & 128
    nss::bm3d_cache_epoch();
#endif
    const int slices = 2 * radius + 1;
    const std::size_t plane_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (int c = 0; c < nch; ++c) {
        std::memset(channels[c].scratch, 0, static_cast<std::size_t>(slices) * 2 * plane_size * sizeof(float));
    }

    nss::SearchConfig cfg;
    cfg.block = block;
    cfg.step = step;
    cfg.group = group;
    cfg.bm_range = bm_range;
    cfg.radius = radius;
    cfg.valid_t_begin = std::max(0, radius - center);
    cfg.valid_t_end = radius + std::min(radius + 1, frame_count - center);
    cfg.ps_num = ps_num;
    cfg.ps_range = ps_range;

#if NSS_BM_RASTER
    const nss::GroupKey key{block*block,group,1,nss::GroupAlgorithm::BM3D,!wiener,false};
    nss::host_detail::RasterJobs jobs(width,height,block,step,key,t0);
#else
    nss::ResourceVector<nss::GroupJob> jobs;
    jobs.reserve(nss::checked_product({
        (static_cast<std::size_t>(width - block) + step - 1) / step + 1,
        (static_cast<std::size_t>(height - block) + step - 1) / step + 1}));
    const nss::GroupKey key{block * block, group, 1, nss::GroupAlgorithm::BM3D, !wiener, false};
    nss::host_detail::append_raster_jobs(jobs, width, height, block, step, key, t0);
#endif
    if (jobs.empty()) {
        return;
    }

    const bool fused = block == 8 && group == 8 && radius == 0;
    // The fused template serves the other shapes that fit vectors, spatial and temporal
    // (NSS_BM_FUSED_TEMPLATE; false on targets narrower than 256 bits).
    const bool fused_template = !fused && NSS_BM_FUSED_TEMPLATE &&
                                nss::bm3d_filter_fused(block, group, nullptr, nullptr, nullptr, 0, 0.f, nullptr, nullptr,
                                                       nullptr, nullptr, 0, 0, 0, 0, 0, 0);
    const bool direct = radius == 0 && (block == 4 || block == 8 || block == 12 || block == 16) && !fused &&
                        !fused_template;
    const int area = block * block;
    nss::ResourceVector<float> direct_cube;
    nss::ResourceVector<float> direct_work;
    if (direct) {
        direct_cube.resize(static_cast<std::size_t>(group) * static_cast<std::size_t>(area) * (wiener ? 2u : 1u), 0.f);
        direct_work.resize(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)), 0.f);
    }
#if NSS_BM_REUSE
    nss::ResourceVector<float> reused_patches, reused_refs, reused_work;
#endif
    for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
        const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
        const int count = static_cast<int>(end - begin);
        std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
        std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
        // Match storage is written [0, counts[i]) by the matcher before any
        // read; value-initializing 160 KB per window is dead work.
        std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kBmMaxGroup> match_storage;
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] =
                nss::MatchBatchItem{job.x, job.y, block, bm_range, group, nss::detail::avx2_policy(nss::detail::Avx2Algorithm::BM3D, block, group, radius, wiener, 0)};
        }
        int match_rc=-2;
#if NSS_BM_EXPERIMENT & 64
        if(radius==0&&block>=8&&step<=4)match_rc=nss::detail::sliding_batch(match[t0],match_strides[t0],width,height,match_items.data(),count,match_storage.data(),nss::kBmMaxGroup,counts.data());
#endif
        if(match_rc==-2)match_rc = radius > 0
                                 ? nss::predictive_match_batch(match, match_strides, ntemp, width, height, t0, cfg,
                                                               match_items.data(), count, match_storage.data(),
                                                               nss::kBmMaxGroup, counts.data())
                                 : nss::spatial_match_batch(match[t0], match_strides[t0], width, height,
                                                            match_items.data(), count, match_storage.data(),
                                                            nss::kBmMaxGroup, counts.data());
        // Matching failures are fatal for the frame: a nonzero code identifies
        // the first failed job in the window.
        if (match_rc != 0) {
            throw std::runtime_error("nss: matching failed for an active group");
        }
#if !NSS_BM_RASTER
        for (int i = 0; i < count; ++i) {
            jobs[begin + static_cast<std::size_t>(i)].key.k = counts[static_cast<std::size_t>(i)];
        }
#endif

        for (int channel = 0; channel < nch; ++channel) {
        // The names the single-plane code below uses, for this channel.
        const Bm3dChannel& ch = channels[channel];
        const float* const* srcs = ch.srcs;
        const float* const* refs = ch.refs;
        const int* src_strides = ch.src_strides;
        const int* ref_strides = ch.ref_strides;
        const float sigma = ch.sigma;
        float* num = ch.scratch;
        float* den = num + static_cast<std::size_t>(slices) * plane_size;
        if (fused_template) {
            for (int i = 0; i < count; ++i) {
                const int k = counts[static_cast<std::size_t>(i)];
                if (k <= 0) continue;
                const nss::Match* matches = match_storage.data() + static_cast<std::size_t>(i) * nss::kBmMaxGroup;
                (void)nss::bm3d_filter_fused(block, group, srcs, src_strides, matches, k, sigma, wiener ? refs : nullptr,
                                             ref_strides, num, den, width, width, height, t0, radius, plane_size);
            }
            continue;
        }
        if (fused) {
            for (int i = 0; i < count; ++i) {
                const int k = counts[static_cast<std::size_t>(i)];
                if (k <= 0) {
                    continue;
                }
                const nss::Match* matches = match_storage.data() +
                                             static_cast<std::size_t>(i) * nss::kBmMaxGroup;
                nss::bm3d_filter8(srcs[t0], src_strides[t0], matches, k, sigma, wiener,
                                  wiener ? refs[t0] : nullptr, ref_strides[t0], num, den, width, width, height);
            }
            continue;
        }
        if (direct) {
            for (int i = 0; i < count; ++i) {
                const int k = counts[static_cast<std::size_t>(i)];
                if (k <= 0) {
                    continue;
                }
                const nss::Match* matches = match_storage.data() +
                                             static_cast<std::size_t>(i) * nss::kBmMaxGroup;
                nss::bm3d_filter_direct(srcs[t0], src_strides[t0], matches, k, block, group, sigma, wiener,
                                        wiener ? refs[t0] : nullptr, ref_strides[t0], num, den, width, width, height,
                                        direct_cube.data(), direct_work.data(),
                                        nss::detail::avx2_policy(nss::detail::Avx2Algorithm::BM3D, block, group, radius, wiener));
            }
            continue;
        }

        const int area = block * block;
        const int work_floats = nss::bm3d_filter_work_floats(group, block);
#if NSS_BM_REUSE
        auto& patches=reused_patches;auto& ref_patches=reused_refs;auto& filter_work=reused_work;
        // Each real patch is overwritten by pack_patch. The group kernel
        // explicitly zeroes k..group before either transform.
        patches.resize(static_cast<std::size_t>(count)*group*area);
        filter_work.resize(static_cast<std::size_t>(count)*work_floats);
#else
        nss::ResourceVector<float> patches(static_cast<std::size_t>(count) * static_cast<std::size_t>(group) * area, 0.f);
        nss::ResourceVector<float> ref_patches;
#endif
        if (wiener) {
            ref_patches.resize(patches.size(), 0.f);
        }
#if !NSS_BM_REUSE
        nss::ResourceVector<float> filter_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(work_floats), 0.f);
#endif
        std::array<float, nss::host_detail::kGroupBatchWindow> weights{};
        std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
        std::array<nss::Bm3dFilterBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
#if NSS_BM_EXPERIMENT & 128
        std::array<nss::Bm3dPatchKey,nss::host_detail::kGroupBatchWindow*nss::kBmMaxGroup> keys{},rkeys{};
#endif
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* patch = patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group) * area;
            if (k > 0) {
                for (int j = 0; j < k; ++j) {
                    const int t = radius > 0 ? match_storage[static_cast<std::size_t>(i) * nss::kBmMaxGroup + j].t : t0;
                    const auto& m = match_storage[static_cast<std::size_t>(i) * nss::kBmMaxGroup + j];
#if NSS_BM_EXPERIMENT & 128
                    keys[i*nss::kBmMaxGroup+j]={srcs[t],src_strides[t],m.x,m.y};
                    rkeys[i*nss::kBmMaxGroup+j]={refs[t],ref_strides[t],m.x,m.y};
#endif
                    nss::pack_patch(patch + static_cast<std::size_t>(j) * area, area, srcs[t], src_strides[t], m.x,
                                    m.y, block, width, height);
                    if (wiener) {
                        float* rp = ref_patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group) * area;
                        nss::pack_patch(rp + static_cast<std::size_t>(j) * area, area, refs[t], ref_strides[t], m.x,
                                        m.y, block, width, height);
                    }
                }
            }
            weights[static_cast<std::size_t>(i)] = 1.f;
            filter_items[static_cast<std::size_t>(i)] = nss::Bm3dFilterBatchItem{
                patch, area, group, k, block, sigma, wiener,
                wiener ? ref_patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group) * area : nullptr,
                &weights[static_cast<std::size_t>(i)],
                filter_work.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(work_floats),
                &filter_status[static_cast<std::size_t>(i)]};
#if NSS_BM_EXPERIMENT & 128
            filter_items[i].keys=keys.data()+i*nss::kBmMaxGroup;
            filter_items[i].ref_keys=rkeys.data()+i*nss::kBmMaxGroup;
#endif
        }
        if (nss::bm3d_filter_group_batch(filter_items.data(), count) != 0) throw std::runtime_error("nss: numerical group processing failed");

        auto prepare = [&](const nss::GroupJob& job, Bm3dBatchResult& result) {
            const std::size_t i = static_cast<std::size_t>(job.ordinal - jobs[begin].ordinal);
            if (i >= static_cast<std::size_t>(count) || counts[i] <= 0 || filter_status[i] == 0) {
                return false;
            }
            result.valid = true;
            result.k = counts[i];
            result.patches = patches.data() + i * static_cast<std::size_t>(group) * area;
            result.weight = weights[i];
            for (int j = 0; j < result.k; ++j) {
                result.matches[j] = match_storage[i * nss::kBmMaxGroup + j];
                int sl = 0;
                if (radius > 0) {
                    sl = std::clamp(result.matches[j].t - t0 + radius, 0, slices - 1);
                }
                result.slice[j] = sl;
            }
            return true;
        };
        auto commit = [&](const Bm3dBatchResult& result) {
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
        (void)nss::host_detail::commit_prepared_chunk<Bm3dBatchResult>(jobs, begin, end, prepare, commit);
        }
    }

#if NSS_BM_SCRATCH
    if (scratch_only) return;
#endif
    for (int channel = 0; channel < nch; ++channel) {
    const Bm3dChannel& ch = channels[channel];
    const float* const* srcs = ch.srcs;
    const int* src_strides = ch.src_strides;
    float* dst = ch.dst;
    const int dstride = ch.dstride, fat_stride = ch.fat_stride;
    const float* num = ch.scratch;
    const float* den = num + static_cast<std::size_t>(slices) * plane_size;
    if (emit_fat) {
        for (int sl = 0; sl < slices; ++sl) {
            const float* np = num + static_cast<std::size_t>(sl) * plane_size;
            const float* dp = den + static_cast<std::size_t>(sl) * plane_size;
            float* on = dst + static_cast<std::size_t>(sl * 2) * static_cast<std::size_t>(height) * fat_stride;
            float* od = dst + static_cast<std::size_t>(sl * 2 + 1) * static_cast<std::size_t>(height) * fat_stride;
            for (int y = 0; y < height; ++y) {
                std::memcpy(on + y * fat_stride, np + static_cast<std::size_t>(y) * width,
                            static_cast<std::size_t>(width) * sizeof(float));
                std::memcpy(od + y * fat_stride, dp + static_cast<std::size_t>(y) * width,
                            static_cast<std::size_t>(width) * sizeof(float));
            }
        }
    } else {
        nss::aggregate_finish(dst, num, den, srcs[t0], width, height, dstride, width, src_strides[t0]);
    }
    }
}

void process_plane_batched(const float* const* srcs, const float* const* refs, int ntemp, int t0,
                           const int* src_strides, const int* ref_strides, float* dst, int width, int height,
                           int dstride, int fat_stride,
                           float sigma, int block, int group, int step, int bm_range, int ps_num, int ps_range,
                           int radius, bool wiener, bool emit_fat, float* scratch, int center, int frame_count
#if NSS_BM_SCRATCH
                           , bool scratch_only=false
#endif
                           ) {
    const Bm3dChannel channel{srcs, src_strides, refs, ref_strides, dst, dstride, fat_stride, sigma, scratch};
    process_planes_batched(refs, ref_strides, &channel, 1, ntemp, t0, width, height, block, group, step, bm_range,
                           ps_num, ps_range, radius, wiener, emit_fat, center, frame_count
#if NSS_BM_SCRATCH
                           , scratch_only
#endif
                           );
}

const VSFrame* VS_CC bm3dGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<Bm3dData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->radius);
        const int end = nss::host_detail::temporal_last(n,d->radius,d->vi.numFrames);
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
    const VSFrame* src0 = frames_owned.getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);
    nss::stamp_contribution(dst, d->radius, n, nss::Model::BM3D, vsapi);

    const int ntemp = 2 * d->radius + 1;
    nss::ResourceVector<const VSFrame*> srcf(static_cast<std::size_t>(ntemp));
    nss::ResourceVector<const VSFrame*> reff(static_cast<std::size_t>(ntemp));
    for (int t = 0; t < ntemp; ++t) {
        const int fn = nss::host_detail::temporal_slot_frame(n,t,d->radius,d->vi.numFrames);
        srcf[static_cast<std::size_t>(t)] = frames_owned.getFrameFilter(fn, d->node, frameCtx);
        reff[static_cast<std::size_t>(t)] = frames_owned.getFrameFilter(fn, d->ref ? d->ref : d->node, frameCtx);
    }
    const int t0 = d->radius;

    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const int sstride = static_cast<int>(frames_owned.getStride(src0, plane) / sizeof(float));
        const int dstride = static_cast<int>(frames_owned.getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src0, plane));
        if (d->sigma[plane] == 0.f) {
            if (!fat) {
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(outp + y * dstride, srcp + y * sstride,
                                static_cast<std::size_t>(pw) * sizeof(float));
                }
                continue;
            }
            const int slices = 2 * d->radius + 1;
            nss::host_detail::temporal_identity(outp, dstride, srcp, sstride, pw, ph, d->radius);
            continue;
        }
        if (d->chroma) continue;  // the filtered planes share one pass, below
        nss::ResourceVector<const float*> srcs(static_cast<std::size_t>(ntemp));
        nss::ResourceVector<const float*> refs(static_cast<std::size_t>(ntemp));
        nss::ResourceVector<int> src_strides(static_cast<std::size_t>(ntemp));
        nss::ResourceVector<int> ref_strides(static_cast<std::size_t>(ntemp));
        for (int t = 0; t < ntemp; ++t) {
            srcs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            refs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
            src_strides[static_cast<std::size_t>(t)] =
                static_cast<int>(frames_owned.getStride(srcf[static_cast<std::size_t>(t)], plane) / sizeof(float));
            ref_strides[static_cast<std::size_t>(t)] =
                static_cast<int>(frames_owned.getStride(reff[static_cast<std::size_t>(t)], plane) / sizeof(float));
        }
        const int slices = 2 * d->radius + 1;
        const int block = d->block_size[plane];
        const int group = d->group_size[plane];
        const int area = block * block;
        const std::size_t need = static_cast<std::size_t>(slices) * 2 * static_cast<std::size_t>(pw) * ph
#if !NSS_BM_REUSE
                                 +
                                 static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)) +
                                 static_cast<std::size_t>(group) * static_cast<std::size_t>(area) *
                                     (d->ref != nullptr ? 2u : 1u) +
                                 64
#endif
                                 ;
        float* scratch = d->ws.get(need);
        process_plane_batched(srcs.data(), refs.data(), ntemp, t0, src_strides.data(), ref_strides.data(), outp, pw,
                              ph, dstride, dstride,
                              d->sigma[plane], d->block_size[plane], d->group_size[plane], d->block_step[plane],
                              d->bm_range[plane], d->ps_num[plane], d->ps_range[plane], d->radius, d->ref != nullptr,
                              fat, scratch, n, d->vi.numFrames);
        if (!fat && d->sigma[plane] != 0.f) {
            (void)srcp;
        }
    }

    if (d->chroma) {
        // CBM3D: groups matched on plane 0 (of ref when given), every filtered plane with those groups.
        const std::size_t nt = static_cast<std::size_t>(ntemp);
        nss::ResourceVector<const float*> srcs(3 * nt), refs(3 * nt);
        nss::ResourceVector<int> src_strides(3 * nt), ref_strides(3 * nt);
        for (int plane = 0; plane < 3; ++plane) {
            for (std::size_t t = 0; t < nt; ++t) {
                srcs[plane * nt + t] = reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[t], plane));
                refs[plane * nt + t] = reinterpret_cast<const float*>(vsapi->getReadPtr(reff[t], plane));
                src_strides[plane * nt + t] = static_cast<int>(frames_owned.getStride(srcf[t], plane) / sizeof(float));
                ref_strides[plane * nt + t] = static_cast<int>(frames_owned.getStride(reff[t], plane) / sizeof(float));
            }
        }
        const int pw = d->vi.width, ph = d->vi.height;
        const std::size_t need = nt * 2 * static_cast<std::size_t>(pw) * ph;
        int nch = 0;
        for (int plane = 0; plane < 3; ++plane) nch += d->sigma[plane] != 0.f;
        float* scratch = nch ? d->ws.get(need * static_cast<std::size_t>(nch)) : nullptr;
        Bm3dChannel channels[3];
        nch = 0;
        for (int plane = 0; plane < 3; ++plane) {
            if (d->sigma[plane] == 0.f) continue;
            const int dstride = static_cast<int>(frames_owned.getStride(dst, plane) / sizeof(float));
            channels[nch] = Bm3dChannel{srcs.data() + plane * nt, src_strides.data() + plane * nt,
                                        refs.data() + plane * nt, ref_strides.data() + plane * nt,
                                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane)), dstride, dstride,
                                        d->sigma[plane], scratch + need * static_cast<std::size_t>(nch)};
            ++nch;
        }
        if (nch) {
            process_planes_batched(refs.data(), ref_strides.data(), channels, nch, ntemp, t0, pw, ph,
                                   d->block_size[0], d->group_size[0], d->block_step[0], d->bm_range[0],
                                   d->ps_num[0], d->ps_range[0], d->radius, d->ref != nullptr, fat, n,
                                   d->vi.numFrames);
        }
    }

    for (int t = 0; t < ntemp; ++t) {
        frames_owned.freeFrame(srcf[static_cast<std::size_t>(t)]);
        frames_owned.freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    frames_owned.freeFrame(src0);
    return frames_owned.keep(dst);
}

void VS_CC bm3dFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<Bm3dData*>(instanceData);
    d->node.reset();
    if (d->ref) {
        d->ref.reset();
    }
    delete d;
}

struct RollingPlane {
    nss::ResourceVector<float> data;
    int width = 0;
    int height = 0;
};

struct RollingFrameStore {
    nss::ResourceVector<RollingPlane> planes;
};

struct RollingChunkStore {
    std::shared_ptr<nss::ResourceAccount> account = nss::current_account();
    int start = 0;
    int count = 0;
    nss::ResourceVector<RollingFrameStore> frames;
};

struct RollingData {
    Bm3dData bm;
    int rolling_chunk = 4;
    int cache_limit = 1;
    std::mutex cache_mu;
    std::mutex compute_mu;
    std::list<std::shared_ptr<const RollingChunkStore>> cache;
};

int rolling_chunk_start(int n, int chunk) {
    return (n / chunk) * chunk;
}

std::shared_ptr<const RollingChunkStore> rolling_find_chunk(RollingData* d, int start) {
    for (const auto& chunk : d->cache) {
        if (chunk->start == start) {
            return chunk;
        }
    }
    return {};
}

void rolling_touch_chunk(RollingData* d, int start) {
    auto it = std::find_if(d->cache.begin(), d->cache.end(),
                           [start](const auto& c) { return c->start == start; });
    if (it != d->cache.end() && it != d->cache.begin()) {
        d->cache.splice(d->cache.begin(), d->cache, it);
    }
}

void rolling_store_plane(RollingPlane& plane, const float* src, int width, int height, int stride) {
    plane.width = width;
    plane.height = height;
    plane.data.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        std::memcpy(plane.data.data() + static_cast<std::size_t>(y) * width, src + y * stride,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
}

void rolling_write_plane(float* dst, int dstride, const RollingPlane& plane) {
    for (int y = 0; y < plane.height; ++y) {
        std::memcpy(dst + y * dstride, plane.data.data() + static_cast<std::size_t>(y) * plane.width,
                    static_cast<std::size_t>(plane.width) * sizeof(float));
    }
}

// CBM3D chunk: per center one pass that matches on plane 0 and filters the
// active planes, each plane's contributions added to its target frames in
// ascending center order (as legacy + VAggregate).
void rolling_fill_chunk_chroma(RollingData* d, RollingChunkStore& store, int start, int count,
                               VSFrameContext* frameCtx, const VSAPI* vsapi) {
    nss::FrameScope frames_owned(vsapi);
    auto& bm = d->bm;
    const int r = bm.radius, ntemp = 2 * r + 1, nframes = bm.vi.numFrames;
    const int w = bm.vi.width, h = bm.vi.height;
    const std::size_t size = static_cast<std::size_t>(w) * h, nt = static_cast<std::size_t>(ntemp);
    int planes[3], nch = 0;
    for (int plane = 0; plane < 3; ++plane) {
        if (bm.sigma[plane] != 0.f) planes[nch++] = plane;
    }
    const std::size_t fat_size = nt * 2 * size;
    nss::ResourceVector<float> sums(static_cast<std::size_t>(nch) * count * 2 * size, 0.f);
#if !NSS_BM_SCRATCH
    nss::ResourceVector<float> fat(static_cast<std::size_t>(nch) * fat_size);
#endif
    float* scratch = nch ? bm.ws.get(static_cast<std::size_t>(nch) * fat_size) : nullptr;
    for (int center = std::max(0, start - r);
         nch && center <= nss::host_detail::temporal_last(start + count - 1, r, nframes); ++center) {
        nss::ResourceVector<const VSFrame*> sf(nt), rf(nt);
        nss::ResourceVector<const float*> sp(3 * nt), rp(3 * nt);
        nss::ResourceVector<int> ss(3 * nt), rs(3 * nt);
        for (std::size_t t = 0; t < nt; ++t) {
            const int fn = nss::host_detail::temporal_slot_frame(center, static_cast<int>(t), r, nframes);
            sf[t] = frames_owned.getFrameFilter(fn, bm.node, frameCtx);
            rf[t] = frames_owned.getFrameFilter(fn, bm.ref ? bm.ref : bm.node, frameCtx);
            for (int plane = 0; plane < 3; ++plane) {
                sp[plane * nt + t] = reinterpret_cast<const float*>(vsapi->getReadPtr(sf[t], plane));
                rp[plane * nt + t] = reinterpret_cast<const float*>(vsapi->getReadPtr(rf[t], plane));
                ss[plane * nt + t] = static_cast<int>(frames_owned.getStride(sf[t], plane) / sizeof(float));
                rs[plane * nt + t] = static_cast<int>(frames_owned.getStride(rf[t], plane) / sizeof(float));
            }
        }
        Bm3dChannel channels[3];
        for (int c = 0; c < nch; ++c) {
            const int plane = planes[c];
#if NSS_BM_SCRATCH
            float* out = nullptr;
#else
            float* out = fat.data() + c * fat_size;
#endif
            channels[c] = Bm3dChannel{sp.data() + plane * nt, ss.data() + plane * nt, rp.data() + plane * nt,
                                      rs.data() + plane * nt, out, w, w, bm.sigma[plane], scratch + c * fat_size};
        }
        process_planes_batched(rp.data(), rs.data(), channels, nch, ntemp, r, w, h, bm.block_size[0], bm.group_size[0],
                               bm.block_step[0], bm.bm_range[0], bm.ps_num[0], bm.ps_range[0], r, bm.ref != nullptr,
                               true, center, nframes
#if NSS_BM_SCRATCH
                               , true
#endif
                               );
        for (int target = std::max(start, center - r);
             target <= std::min(start + count - 1, nss::host_detail::temporal_last(center, r, nframes)); ++target) {
            const std::size_t slice = static_cast<std::size_t>(target - center + r);
            for (int c = 0; c < nch; ++c) {
                float* num = sums.data() + (static_cast<std::size_t>(c) * count + (target - start)) * 2 * size;
#if NSS_BM_SCRATCH
                const float* plane_scratch = scratch + c * fat_size;
                for (std::size_t i = 0; i < size; ++i) {
                    num[i] += plane_scratch[slice * size + i];
                    num[size + i] += plane_scratch[(nt + slice) * size + i];
                }
#else
                const float* contribution = fat.data() + c * fat_size + slice * 2 * size;
                for (std::size_t i = 0; i < 2 * size; ++i) num[i] += contribution[i];
#endif
            }
        }
        for (std::size_t t = 0; t < nt; ++t) {
            frames_owned.freeFrame(sf[t]);
            frames_owned.freeFrame(rf[t]);
        }
    }
    for (int i = 0; i < count; ++i) {
        const VSFrame* frame = frames_owned.getFrameFilter(start + i, bm.node, frameCtx);
        for (int plane = 0; plane < 3; ++plane) {
            const float* src = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, plane));
            const int stride = static_cast<int>(frames_owned.getStride(frame, plane) / sizeof(float));
            auto& output = store.frames[i].planes[plane];
            int c = 0;
            while (c < nch && planes[c] != plane) ++c;
            if (c == nch) {
                rolling_store_plane(output, src, w, h, stride);
                continue;
            }
            output.width = w;
            output.height = h;
            output.data.resize(size);
            const float* num = sums.data() + (static_cast<std::size_t>(c) * count + i) * 2 * size;
            nss::aggregate_finish(output.data.data(), num, num + size, src, w, h, w, w, stride);
        }
        frames_owned.freeFrame(frame);
    }
}

bool rolling_fill_chunk(RollingData* d, RollingChunkStore& store, int start, int count, VSFrameContext* frameCtx,
                        VSCore* core, const VSAPI* vsapi) {
    (void)core;
    nss::FrameScope frames_owned(vsapi);
    auto& bm=d->bm;
    const int r=bm.radius, ntemp=2*r+1, nframes=bm.vi.numFrames, np=bm.vi.format.numPlanes;
    store.start=start; store.count=count;
    store.frames.assign(count,RollingFrameStore{});
    for(auto& frame:store.frames) frame.planes.resize(np);
    if (bm.chroma) {
        rolling_fill_chunk_chroma(d, store, start, count, frameCtx, vsapi);
        return true;
    }
    // One center's fat plus target chunk accumulators, never all centers' fats.
    for(int plane=0;plane<np;++plane) {
        const int w=nss::plane_width(bm.vi,plane), h=nss::plane_height(bm.vi,plane);
        const std::size_t size=static_cast<std::size_t>(w)*h;
#if NSS_BM_RING
        const int slots=std::min(count,ntemp);
#else
        const int slots=count;
#endif
        nss::ResourceVector<float> sums(bm.sigma[plane] ? slots*2*size : 0,0.f);
        if(bm.sigma[plane]) {
#if NSS_BM_SCRATCH
            nss::ResourceVector<float> fat;
#else
            nss::ResourceVector<float> fat(ntemp*2*size);
#endif
#if NSS_BM_RING
            int next_output=start;
#endif
            const int block=bm.block_size[plane], group=bm.group_size[plane];
            float* scratch=bm.ws.get(ntemp*2*size
#if !NSS_BM_REUSE
                                     +nss::bm3d_filter_work_floats(group,block)+
                                     group*block*block*(bm.ref ? 2u : 1u)+64
#endif
                                     );
            for(int center=std::max(0,start-r);center<=nss::host_detail::temporal_last(start+count-1,r,nframes);++center) {
                nss::ResourceVector<const VSFrame*> sf(ntemp),rf(ntemp);
                nss::ResourceVector<const float*> sp(ntemp),rp(ntemp);
                nss::ResourceVector<int> ss(ntemp),rs(ntemp);
                for(int t=0;t<ntemp;++t) {
                    int fn=nss::host_detail::temporal_slot_frame(center,t,r,nframes);
                    sf[t]=frames_owned.getFrameFilter(fn,bm.node,frameCtx);
                    rf[t]=frames_owned.getFrameFilter(fn,bm.ref ? bm.ref : bm.node,frameCtx);
                    sp[t]=reinterpret_cast<const float*>(vsapi->getReadPtr(sf[t],plane));
                    rp[t]=reinterpret_cast<const float*>(vsapi->getReadPtr(rf[t],plane));
                    ss[t]=static_cast<int>(frames_owned.getStride(sf[t],plane)/sizeof(float));
                    rs[t]=static_cast<int>(frames_owned.getStride(rf[t],plane)/sizeof(float));
                }
                process_plane_batched(sp.data(),rp.data(),ntemp,r,ss.data(),rs.data(),fat.data(),w,h,w,w,
                                      bm.sigma[plane],block,group,bm.block_step[plane],bm.bm_range[plane],
                                      bm.ps_num[plane],bm.ps_range[plane],r,bm.ref!=nullptr,true,scratch,center,nframes
#if NSS_BM_SCRATCH
                                      , true
#endif
                                      );
                for(int target=std::max(start,center-r);target<=std::min(start+count-1,nss::host_detail::temporal_last(center,r,nframes));++target) {
                    const int slice=target-center+r;
#if NSS_BM_RING
                    float* num=sums.data()+((target-start)%slots)*2*size;
#else
                    float* num=sums.data()+(target-start)*2*size;
#endif
#if NSS_BM_SCRATCH
                    for(std::size_t i=0;i<size;++i){num[i]+=scratch[slice*size+i];num[size+i]+=scratch[(ntemp+slice)*size+i];}
#else
                    const float* contribution=fat.data()+slice*2*size;
                    for(std::size_t i=0;i<2*size;++i)num[i]+=contribution[i];
#endif
                }
                for(int t=0;t<ntemp;++t){frames_owned.freeFrame(sf[t]);frames_owned.freeFrame(rf[t]);}
#if NSS_BM_RING
                while(next_output<start+count && nss::host_detail::temporal_last(next_output,r,nframes)<=center){
                    const auto* frame=frames_owned.getFrameFilter(next_output,bm.node,frameCtx);
                    const float* src=reinterpret_cast<const float*>(vsapi->getReadPtr(frame,plane));
                    int stride=static_cast<int>(frames_owned.getStride(frame,plane)/sizeof(float));
                    auto& output=store.frames[next_output-start].planes[plane];
                    output.width=w;output.height=h;output.data.resize(size);
                    float* num=sums.data()+((next_output-start)%slots)*2*size;
                    nss::aggregate_finish(output.data.data(),num,num+size,src,w,h,w,w,stride);
                    std::fill_n(num,2*size,0.f);frames_owned.freeFrame(frame);++next_output;
                }
#endif
            }
        }
#if NSS_BM_RING
        if(bm.sigma[plane])continue;
#endif
        for(int i=0;i<count;++i) {
            const VSFrame* frame=frames_owned.getFrameFilter(start+i,bm.node,frameCtx);
            const float* src=reinterpret_cast<const float*>(vsapi->getReadPtr(frame,plane));
            int stride=static_cast<int>(frames_owned.getStride(frame,plane)/sizeof(float));
            auto& output=store.frames[i].planes[plane];
            output.width=w;output.height=h;output.data.resize(size);
            if(bm.sigma[plane]) {
                const float* num=sums.data()+i*2*size;
                nss::aggregate_finish(output.data.data(),num,num+size,src,w,h,w,w,stride);
            } else rolling_store_plane(output,src,w,h,stride);
            frames_owned.freeFrame(frame);
        }
    }
    return true;
}

const VSFrame* VS_CC rollingGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                     VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<RollingData*>(instanceData);
    nss::ResourceScope resource_scope(d->bm.budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    const int chunk = d->rolling_chunk;
    const int start = rolling_chunk_start(n, chunk);
    const int count = std::min(chunk, d->bm.vi.numFrames - start);
    if (activationReason == arInitial) {
        // A cache hit observed here is not pinned until arAllFramesReady and can
        // be evicted by another request. Always declare the full dependency
        // window so a later miss can safely recompute the chunk.
        const int radius = d->bm.radius;
        const int first = std::max(0, start - 2 * radius);
        const int last = nss::host_detail::temporal_last(start + count - 1,2*radius,d->bm.vi.numFrames);
        for (int i = first; i <= last; ++i) {
            vsapi->requestFrameFilter(i, d->bm.node, frameCtx);
            if (d->bm.ref) {
                vsapi->requestFrameFilter(i, d->bm.ref, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    const VSFrame* srcn = frames_owned.getFrameFilter(n, d->bm.node, frameCtx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->bm.vi.format, d->bm.vi.width, d->bm.vi.height, srcn, core);
    nss::stamp_contribution(dst, 0, n, nss::Model::BM3D, vsapi);
    frames_owned.freeFrame(srcn);

    std::shared_ptr<const RollingChunkStore> result;
    {
        std::lock_guard<std::mutex> guard(d->cache_mu);
        if (auto hit = rolling_find_chunk(d, start)) {
            rolling_touch_chunk(d, start);
            result = std::move(hit);
        }
    }
    if (!result) {
        std::lock_guard<std::mutex> compute(d->compute_mu);
        {
            std::lock_guard<std::mutex> guard(d->cache_mu);
            if (auto hit = rolling_find_chunk(d, start)) {
                rolling_touch_chunk(d, start);
                result = std::move(hit);
            }
        }
        if (!result) {
            auto chunk_account = nss::make_resource_account(nss::ResourceKind::Inflight);
            nss::ResourceScope chunk_scope(d->bm.budget, chunk_account);
            auto computed = std::allocate_shared<RollingChunkStore>(nss::ResourceAllocator<RollingChunkStore>{});
            rolling_fill_chunk(d, *computed, start, count, frameCtx, core, vsapi);
            std::lock_guard<std::mutex> guard(d->cache_mu);
            d->cache.push_front(computed);
            if (computed->account) computed->account->retag(nss::ResourceKind::Cached);
            result = std::move(computed);
            while (static_cast<int>(d->cache.size()) > d->cache_limit) {
                if (d->cache.back()->account) d->cache.back()->account->retag(nss::ResourceKind::Pinned);
                d->cache.pop_back();
            }
        }
    }

    const RollingFrameStore& local = result->frames[static_cast<std::size_t>(n - result->start)];
    for (int plane = 0; plane < d->bm.vi.format.numPlanes; ++plane) {
        const int dstride = static_cast<int>(frames_owned.getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        rolling_write_plane(outp, dstride, local.planes[static_cast<std::size_t>(plane)]);
    }
    return frames_owned.keep(dst);
}

void VS_CC rollingFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<RollingData*>(instanceData);
    d->bm.node.reset();
    if (d->bm.ref) {
        d->bm.ref.reset();
    }
    delete d;
}

}  // namespace

const char* fill_bm3d_data(Bm3dData& d, const VSMap* in, const VSAPI* vsapi) {
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    int e = 0;
    d.ref = nss::get_node(vsapi, in, "ref", 0, &e);
    if (e) {
        d.ref = nullptr;
    }
    const VSVideoInfo* ref_vi = d.ref ? vsapi->getVideoInfo(d.ref) : nullptr;
    static_cast<nss::Bm3dParams&>(d) = nss::frontend::parse_bm3d(vsapi, in, d.vi, ref_vi, "nss");
    d.vi_out = d.vi;
    return nullptr;
}

void release_bm3d_nodes(Bm3dData& d, const VSAPI* vsapi) {
    if (d.node) {
        d.node.reset();
        d.node = nullptr;
    }
    if (d.ref) {
        d.ref.reset();
        d.ref = nullptr;
    }
}

VSNode* create_rolling_bm3d(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(err, "nss.BM3D: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return nullptr;
    }
    auto d = std::make_unique<RollingData>();
    if (const char* msg = fill_bm3d_data(d->bm, in, vsapi)) {
        vsapi->mapSetError(err, msg);
        release_bm3d_nodes(d->bm, vsapi);
        return nullptr;
    }
    const auto rolling = nss::frontend::parse_rolling(vsapi, in, d->bm.radius, "BM3D", "nss");
    d->rolling_chunk = rolling.rolling_chunk;
    d->cache_limit = rolling.cache_limit;
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        release_bm3d_nodes(d->bm, vsapi);
        return nullptr;
    };
    d->bm.ws.set_serial();
    d->bm.vi_out = d->bm.vi;
    VSFilterDependency deps[2]{{d->bm.node, rpGeneral}, {d->bm.ref, rpGeneral}};
    const int ndeps = d->bm.ref ? 2 : 1;
    RollingData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("BM3D", &raw->bm.vi_out, nss::checked_frame<rollingGetFrame>, rollingFree, fmParallel, deps,
                                            ndeps, raw, core);
    if (!node) {
        return fail("nss.BM3D: failed to create rolling filter");
    }
    d.release();
    return node;
}

VSNode* nss_create_bm3d(const VSMap* in, VSCore* core, const VSAPI* vsapi, VSMap* err) {
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(err, "nss.BM3D: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return nullptr;
    }
    auto d = std::make_unique<Bm3dData>();
    if (const char* msg = fill_bm3d_data(*d, in, vsapi)) {
        vsapi->mapSetError(err, msg);
        release_bm3d_nodes(*d, vsapi);
        return nullptr;
    }
    d->vi_out = d->vi;
    if (d->radius > 0) {
        d->vi_out.height = nss::checked_fat_height(d->vi.height, d->radius);
    }
    VSFilterDependency deps[2]{{d->node, d->radius == 0 ? rpStrictSpatial : rpGeneral},
                               {d->ref, d->radius == 0 ? rpStrictSpatial : rpGeneral}};
    const int ndeps = d->ref ? 2 : 1;
    Bm3dData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("BM3D", &raw->vi_out, nss::checked_frame<bm3dGetFrame>, bm3dFree, fmParallel, deps, ndeps, raw,
                                            core);
    if (!node) {
        vsapi->mapSetError(err, "nss.BM3D: failed to create filter");
        release_bm3d_nodes(*d, vsapi);
        return nullptr;
    }
    d.release();
    return node;
}

void VS_CC bm3dCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    const bool rolling =
        nss::frontend::parse_temporal_mode(vsapi, in, radius, "BM3D", "nss") == nss::TemporalMode::Rolling;
    if (nss::map_int(vsapi, in, "final", 0) != 0) {
        // Both stages: validate the whole call, then chain two nodes (the CPU has no transfer to save).
        const nss::NodeRef clip = nss::get_node(vsapi, in, "clip", 0, nullptr);
        int no_ref = 0;
        const nss::NodeRef ref = nss::get_node(vsapi, in, "ref", 0, &no_ref);
        (void)nss::frontend::parse_bm3d(vsapi, in, *vsapi->getVideoInfo(clip),
                                        no_ref ? nullptr : vsapi->getVideoInfo(ref), "nss");
        if (rolling) (void)nss::frontend::parse_rolling(vsapi, in, radius, "BM3D", "nss");
        nss::frontend::bm3d_two_nodes(vsapi, in, out, core, "nss");
        return;
    }
    if (rolling) {
        VSNode* node = create_rolling_bm3d(in, core, vsapi, out);
        if (node) {
            vsapi->mapConsumeNode(out, "clip", node, maAppend);
        }
        return;
    }
    VSNode* node = nss_create_bm3d(in, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    const char* args = nss::frontend::kBm3dSignature;
    vspapi->registerFunction("BM3D", args, "clip:vnode;", nss::checked_create<bm3dCreate>, nullptr, plugin);
}
