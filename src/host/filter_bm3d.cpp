#include "nss/resources.hpp"
#include "nss/avx2_policy.hpp"
#if NSS_BM_EXPERIMENT & 64
#include "cpu/bm/sliding-batch.hpp"
#endif
#include "host/filters.hpp"
#include "host/temporal.hpp"
#include "host/batch_runner.hpp"
#include "host/validate.hpp"
#include "host/contribution.hpp"
#include "nss/avx2.hpp"
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

struct Bm3dData {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    nss::NodeRef ref;
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

struct Bm3dBatchResult {
    bool valid = false;
    int k = 0;
    int slice[ nss::kBmMaxGroup ]{};
    nss::Match matches[nss::kBmMaxGroup]{};
    const float* patches = nullptr;
    float weight = 1.f;
};

void process_plane_batched(const float* const* srcs, const float* const* refs, int ntemp, int t0,
                           const int* src_strides, const int* ref_strides, float* dst, int width, int height,
                           int dstride, int fat_stride,
                           float sigma, int block, int group, int step, int bm_range, int ps_num, int ps_range,
                           int radius, bool wiener, bool emit_fat, float* scratch, int center, int frame_count
#if NSS_BM_SCRATCH
                           , bool scratch_only=false
#endif
                           ) {
#if NSS_BM_EXPERIMENT & 128
    nss::bm3d_cache_epoch();
#endif
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
    const bool direct = radius == 0 && (block == 4 || block == 8 || block == 12 || block == 16) && !fused;
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
        std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kBmMaxGroup> match_storage{};
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] =
                nss::MatchBatchItem{job.x, job.y, block, bm_range, group, nss::detail::avx2_policy(nss::detail::Avx2Algorithm::BM3D, block, group, radius, wiener, 0)};
        }
        int match_rc=-2;
#if NSS_BM_EXPERIMENT & 64
        if(radius==0&&block>=8&&step<=4)match_rc=nss::detail::sliding_batch(refs[t0],ref_strides[t0],width,height,match_items.data(),count,match_storage.data(),nss::kBmMaxGroup,counts.data());
#endif
        if(match_rc==-2)match_rc = radius > 0
                                 ? nss::predictive_match_batch(refs, ref_strides, ntemp, width, height, t0, cfg,
                                                               match_items.data(), count, match_storage.data(),
                                                               nss::kBmMaxGroup, counts.data())
                                 : nss::spatial_match_batch(refs[t0], ref_strides[t0], width, height,
                                                            match_items.data(), count, match_storage.data(),
                                                            nss::kBmMaxGroup, counts.data());
        // A nonzero positive code identifies an individual failed job; keep
        // the other jobs in the window and let their zero count skip itself.
        if (match_rc != 0) {
            throw std::runtime_error("nss: matching failed for an active group");
        }
#if !NSS_BM_RASTER
        for (int i = 0; i < count; ++i) {
            jobs[begin + static_cast<std::size_t>(i)].key.k = counts[static_cast<std::size_t>(i)];
        }
#endif

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

#if NSS_BM_SCRATCH
    if (scratch_only) return;
#endif
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

bool rolling_fill_chunk(RollingData* d, RollingChunkStore& store, int start, int count, VSFrameContext* frameCtx,
                        VSCore* core, const VSAPI* vsapi) {
    (void)core;
    nss::FrameScope frames_owned(vsapi);
    auto& bm=d->bm;
    const int r=bm.radius, ntemp=2*r+1, nframes=bm.vi.numFrames, np=bm.vi.format.numPlanes;
    store.start=start; store.count=count;
    store.frames.assign(count,RollingFrameStore{});
    for(auto& frame:store.frames) frame.planes.resize(np);
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
    if (!nss::is_const_32f(d.vi)) {
        return "nss.BM3D: constant Gray/YUV/RGB 32-bit float required";
    }
    int e = 0;
    d.ref = nss::get_node(vsapi, in, "ref", 0, &e);
    if (e) {
        d.ref = nullptr;
    } else if (!nss::same_video(d.vi, *vsapi->getVideoInfo(d.ref))) {
        return "nss.BM3D: ref must match clip";
    }
    const int np = d.vi.format.numPlanes;
    nss::map_float_array(vsapi, in, "sigma", d.sigma, np, nss::kBmDefaultSigma);
    for (int i = 0; i < np; ++i) {
        if (!nss::is_finite_bits(d.sigma[i]) || d.sigma[i] < 0.f) {
            return "nss.BM3D: sigma must be finite and non-negative";
        }
        if (d.sigma[i] != 0.f) {
            d.sigma[i] = nss::NoiseProfile::bm3d_effective(d.sigma[i]);
        }
    }
    nss::map_inherit_int(vsapi, in, "block_size", d.block_size, np, nss::kBmBlock);
    nss::map_inherit_int(vsapi, in, "group_size", d.group_size, np, nss::kBmGroup);
    {
        const int nstep = vsapi->mapNumElements(in, "block_step");
        if (nstep <= 0) {
            for (int i = 0; i < np; ++i) {
                d.block_step[i] = std::min(nss::kBmDefaultStep, d.block_size[i]);
            }
        } else {
            nss::map_inherit_int(vsapi, in, "block_step", d.block_step, np, nss::kBmDefaultStep);
        }
    }
    nss::map_int_array(vsapi, in, "bm_range", d.bm_range, np, nss::kBmDefaultRange);
    {
        const int nps = vsapi->mapNumElements(in, "ps_num");
        if (nps <= 0) {
            for (int i = 0; i < np; ++i) {
                d.ps_num[i] = std::min(nss::kBmDefaultPsNum, d.group_size[i]);
            }
        } else {
            nss::map_inherit_int(vsapi, in, "ps_num", d.ps_num, np, nss::kBmDefaultPsNum);
        }
    }
    nss::map_int_array(vsapi, in, "ps_range", d.ps_range, np, nss::kBmDefaultPsRange);
    d.radius = nss::map_int(vsapi, in, "radius", 0);
    if (d.radius < 0 || d.radius > nss::kBmMaxRadius) {
        return "nss.BM3D: radius must be in [0, 16]";
    }
    for (int i = 0; i < np; ++i) {
        if (!nss::bm_allowed_block(d.block_size[i])) {
            return "nss.BM3D: block_size must be one of 1, 2, 4, 8, 12, 16, 32";
        }
        if (!nss::bm_allowed_group(d.group_size[i])) {
            return "nss.BM3D: group_size must be one of 1, 2, 4, 8, 16, 32, 64";
        }
        if (d.block_size[i] > nss::plane_width(d.vi, i) || d.block_size[i] > nss::plane_height(d.vi, i)) {
            return "nss.BM3D: block_size must not exceed plane dimensions";
        }
        if (d.block_step[i] < 1 || d.block_step[i] > d.block_size[i]) {
            return "nss.BM3D: block_step must be in [1, block_size]";
        }
        if (d.ps_num[i] < 1 || d.ps_num[i] > d.group_size[i]) {
            return "nss.BM3D: ps_num must be in [1, group_size]";
        }
        if (d.bm_range[i] < 1 || d.bm_range[i] > nss::kBmMaxRange) {
            return "nss.BM3D: bm_range must be in [1, 64]";
        }
        if (d.ps_range[i] < 0 || d.ps_range[i] > nss::kBmMaxRange) {
            return "nss.BM3D: ps_range must be in [0, 64]";
        }
    }
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
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.BM3D: AVX2 is required");
        return nullptr;
    }
    auto d = std::make_unique<RollingData>();
    if (const char* msg = fill_bm3d_data(d->bm, in, vsapi)) {
        vsapi->mapSetError(err, msg);
        release_bm3d_nodes(d->bm, vsapi);
        return nullptr;
    }
    d->rolling_chunk = nss::map_int(vsapi, in, "rolling_chunk", 4);
    int cache_limit_err = 0;
    d->cache_limit = nss::map_int(vsapi, in, "rolling_cache_limit", 1, &cache_limit_err);
    int cache_chunks_err = 0;
    const int cache_chunks = nss::map_int(vsapi, in, "rolling_cache_chunks", 1, &cache_chunks_err);
    if (!cache_chunks_err) {
        d->cache_limit = cache_chunks;
    }
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        release_bm3d_nodes(d->bm, vsapi);
        return nullptr;
    };
    if (d->bm.radius < 1) {
        return fail("nss.BM3D: rolling mode requires radius > 0");
    }
    if (d->rolling_chunk < 1 || d->rolling_chunk > 64) {
        return fail("nss.BM3D: rolling_chunk must be in [1, 64]");
    }
    if (!cache_limit_err && !cache_chunks_err) {
        return fail("nss.BM3D: use only one of rolling_cache_limit and rolling_cache_chunks");
    }
    if (d->cache_limit < 1 || d->cache_limit > 64) {
        return fail("nss.BM3D: rolling_cache_limit must be in [1, 64]");
    }
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
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(err, "nss.BM3D: AVX2 is required");
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
    int mode_err = 0;
    const char* mode = vsapi->mapGetData(in, "temporal_mode", 0, &mode_err);
    std::string mode_s;
    if (!mode_err && mode) {
        mode_s = mode;
        for (char& c : mode_s) {
            if (c >= 'A' && c <= 'Z') {
                c = static_cast<char>(c - 'A' + 'a');
            }
        }
    }
    if (mode_s == "fused") {
        vsapi->mapSetError(out, "nss.BM3D: temporal_mode=fused is not supported; use rolling or legacy");
        return;
    }
    if (!mode_s.empty() && mode_s != "rolling" && mode_s != "legacy") {
        vsapi->mapSetError(out, "nss.BM3D: temporal_mode must be rolling or legacy");
        return;
    }
    const bool rolling = radius > 0 && mode_s == "rolling";
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
    const char* args =
        "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_size:int[]:opt;group_size:int[]:opt;"
        "block_step:int[]:opt;bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;"
        "temporal_mode:data:opt;rolling_chunk:int:opt;rolling_cache_chunks:int:opt;rolling_cache_limit:int:opt;memory_limit_mb:int:opt;";
    vspapi->registerFunction("BM3D", args, "clip:vnode;", nss::checked_create<bm3dCreate>, nullptr, plugin);
}
