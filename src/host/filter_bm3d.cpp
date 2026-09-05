#include "host/filters.hpp"
#include "host/batch_runner.hpp"
#include "host/validate.hpp"
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
                           int radius, bool wiener, bool emit_fat, float* scratch) {
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

    std::vector<nss::GroupJob> jobs;
    jobs.reserve(static_cast<std::size_t>(std::max(1, ((width - block + step - 1) / step) *
                                                        ((height - block + step - 1) / step))));
    const nss::GroupKey key{block * block, group, 1, nss::GroupAlgorithm::BM3D, !wiener, false};
    nss::host_detail::append_raster_jobs(jobs, width, height, block, step, key, t0);
    if (jobs.empty()) {
        return;
    }

    const bool fused = block == 8 && group == 8 && radius == 0;
    const bool direct = radius == 0 && (block == 4 || block == 8 || block == 12 || block == 16) && !fused;
    const int area = block * block;
    std::vector<float> direct_cube;
    std::vector<float> direct_work;
    if (direct) {
        direct_cube.resize(static_cast<std::size_t>(group) * static_cast<std::size_t>(area) * (wiener ? 2u : 1u), 0.f);
        direct_work.resize(static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)), 0.f);
    }
    for (std::size_t begin = 0; begin < jobs.size(); begin += nss::host_detail::kGroupBatchWindow) {
        const std::size_t end = std::min(jobs.size(), begin + nss::host_detail::kGroupBatchWindow);
        const int count = static_cast<int>(end - begin);
        std::array<nss::MatchBatchItem, nss::host_detail::kGroupBatchWindow> match_items{};
        std::array<int, nss::host_detail::kGroupBatchWindow> counts{};
        std::array<nss::Match, nss::host_detail::kGroupBatchWindow * nss::kBmMaxGroup> match_storage{};
        for (int i = 0; i < count; ++i) {
            const auto& job = jobs[begin + static_cast<std::size_t>(i)];
            match_items[static_cast<std::size_t>(i)] =
                nss::MatchBatchItem{job.x, job.y, block, bm_range, group};
        }
        const int match_rc = radius > 0
                                 ? nss::predictive_match_batch(refs, ref_strides, ntemp, width, height, t0, cfg,
                                                               match_items.data(), count, match_storage.data(),
                                                               nss::kBmMaxGroup, counts.data())
                                 : nss::spatial_match_batch(refs[t0], ref_strides[t0], width, height,
                                                            match_items.data(), count, match_storage.data(),
                                                            nss::kBmMaxGroup, counts.data());
        // A nonzero positive code identifies an individual failed job; keep
        // the other jobs in the window and let their zero count skip itself.
        if (match_rc < 0) {
            continue;
        }
        for (int i = 0; i < count; ++i) {
            jobs[begin + static_cast<std::size_t>(i)].key.k = counts[static_cast<std::size_t>(i)];
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
                                        direct_cube.data(), direct_work.data());
            }
            continue;
        }

        const int area = block * block;
        const int work_floats = nss::bm3d_filter_work_floats(group, block);
        std::vector<float> patches(static_cast<std::size_t>(count) * static_cast<std::size_t>(group) * area, 0.f);
        std::vector<float> ref_patches;
        if (wiener) {
            ref_patches.resize(patches.size(), 0.f);
        }
        std::vector<float> filter_work(static_cast<std::size_t>(count) * static_cast<std::size_t>(work_floats), 0.f);
        std::array<float, nss::host_detail::kGroupBatchWindow> weights{};
        std::array<int, nss::host_detail::kGroupBatchWindow> filter_status{};
        std::array<nss::Bm3dFilterBatchItem, nss::host_detail::kGroupBatchWindow> filter_items{};
        for (int i = 0; i < count; ++i) {
            const int k = counts[static_cast<std::size_t>(i)];
            float* patch = patches.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(group) * area;
            if (k > 0) {
                for (int j = 0; j < k; ++j) {
                    const int t = radius > 0 ? match_storage[static_cast<std::size_t>(i) * nss::kBmMaxGroup + j].t : t0;
                    const auto& m = match_storage[static_cast<std::size_t>(i) * nss::kBmMaxGroup + j];
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
        }
        (void)nss::bm3d_filter_group_batch(filter_items.data(), count);

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
            if (!fat) {
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(outp + y * dstride, srcp + y * sstride,
                                static_cast<std::size_t>(pw) * sizeof(float));
                }
                continue;
            }
            const int slices = 2 * d->radius + 1;
            for (int sl = 0; sl < slices; ++sl) {
                float* num = outp + static_cast<std::size_t>(sl * 2) * ph * dstride;
                float* den = outp + static_cast<std::size_t>(sl * 2 + 1) * ph * dstride;
                for (int y = 0; y < ph; ++y) {
                    std::memcpy(num + y * dstride, srcp + y * sstride,
                                static_cast<std::size_t>(pw) * sizeof(float));
                    std::fill_n(den + y * dstride, pw, 1.f);
                }
            }
            continue;
        }
        std::vector<const float*> srcs(static_cast<std::size_t>(ntemp));
        std::vector<const float*> refs(static_cast<std::size_t>(ntemp));
        std::vector<int> src_strides(static_cast<std::size_t>(ntemp));
        std::vector<int> ref_strides(static_cast<std::size_t>(ntemp));
        for (int t = 0; t < ntemp; ++t) {
            srcs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
            refs[static_cast<std::size_t>(t)] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
            src_strides[static_cast<std::size_t>(t)] =
                static_cast<int>(vsapi->getStride(srcf[static_cast<std::size_t>(t)], plane) / sizeof(float));
            ref_strides[static_cast<std::size_t>(t)] =
                static_cast<int>(vsapi->getStride(reff[static_cast<std::size_t>(t)], plane) / sizeof(float));
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
        process_plane_batched(srcs.data(), refs.data(), ntemp, t0, src_strides.data(), ref_strides.data(), outp, pw,
                              ph, dstride, dstride,
                              d->sigma[plane], d->block_size[plane], d->group_size[plane], d->block_step[plane],
                              d->bm_range[plane], d->ps_num[plane], d->ps_range[plane], d->radius, d->ref != nullptr,
                              fat, scratch);
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

struct RollingPlane {
    std::vector<float> data;
    int width = 0;
    int height = 0;
};

struct RollingFrameStore {
    std::vector<RollingPlane> planes;
};

struct RollingChunkStore {
    int start = 0;
    int count = 0;
    std::vector<RollingFrameStore> frames;
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
    auto& bm = d->bm;
    const int radius = bm.radius;
    const int ntemp = 2 * radius + 1;
    const int nframes = bm.vi.numFrames;
    const int np = bm.vi.format.numPlanes;
    store.start = start;
    store.count = count;
    store.frames.assign(static_cast<std::size_t>(count), RollingFrameStore{});
    for (int i = 0; i < count; ++i) {
        store.frames[static_cast<std::size_t>(i)].planes.resize(static_cast<std::size_t>(np));
    }

    for (int local = 0; local < count; ++local) {
        const int center = start + local;
        std::vector<const VSFrame*> srcf(static_cast<std::size_t>(ntemp));
        std::vector<const VSFrame*> reff(static_cast<std::size_t>(ntemp));
        for (int t = 0; t < ntemp; ++t) {
            const int fn = std::clamp(center - radius + t, 0, nframes - 1);
            srcf[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, bm.node, frameCtx);
            reff[static_cast<std::size_t>(t)] =
                vsapi->getFrameFilter(fn, bm.ref ? bm.ref : bm.node, frameCtx);
        }
        const int t0 = radius;
        for (int plane = 0; plane < np; ++plane) {
            const int pw = nss::plane_width(bm.vi, plane);
            const int ph = nss::plane_height(bm.vi, plane);
            const int sstride =
                static_cast<int>(vsapi->getStride(srcf[static_cast<std::size_t>(t0)], plane) / sizeof(float));
            const float* srcp =
                reinterpret_cast<const float*>(vsapi->getReadPtr(srcf[static_cast<std::size_t>(t0)], plane));
            if (bm.sigma[plane] == 0.f) {
                rolling_store_plane(store.frames[static_cast<std::size_t>(local)].planes[static_cast<std::size_t>(plane)],
                                    srcp, pw, ph, sstride);
                continue;
            }
            std::vector<const float*> srcs(static_cast<std::size_t>(ntemp));
            std::vector<const float*> refs(static_cast<std::size_t>(ntemp));
            std::vector<int> src_strides(static_cast<std::size_t>(ntemp));
            std::vector<int> ref_strides(static_cast<std::size_t>(ntemp));
            for (int t = 0; t < ntemp; ++t) {
                srcs[static_cast<std::size_t>(t)] = reinterpret_cast<const float*>(
                    vsapi->getReadPtr(srcf[static_cast<std::size_t>(t)], plane));
                refs[static_cast<std::size_t>(t)] = reinterpret_cast<const float*>(
                    vsapi->getReadPtr(reff[static_cast<std::size_t>(t)], plane));
                src_strides[static_cast<std::size_t>(t)] =
                    static_cast<int>(vsapi->getStride(srcf[static_cast<std::size_t>(t)], plane) / sizeof(float));
                ref_strides[static_cast<std::size_t>(t)] =
                    static_cast<int>(vsapi->getStride(reff[static_cast<std::size_t>(t)], plane) / sizeof(float));
            }
            const int slices = ntemp;
            const int block = bm.block_size[plane];
            const int group = bm.group_size[plane];
            const int area = block * block;
            const std::size_t fat_n =
                static_cast<std::size_t>(slices) * 2 * static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph);
            const std::size_t need = fat_n + static_cast<std::size_t>(nss::bm3d_filter_work_floats(group, block)) +
                                     static_cast<std::size_t>(group) * static_cast<std::size_t>(area) *
                                         (bm.ref != nullptr ? 2u : 1u) +
                                     64;
            float* scratch = bm.ws.get(need);
            std::vector<float> fat(fat_n, 0.f);
            process_plane_batched(srcs.data(), refs.data(), ntemp, t0, src_strides.data(), ref_strides.data(),
                                  fat.data(), pw, ph, pw, pw,
                                  bm.sigma[plane], bm.block_size[plane], bm.group_size[plane], bm.block_step[plane],
                                  bm.bm_range[plane], bm.ps_num[plane], bm.ps_range[plane], bm.radius,
                                  bm.ref != nullptr, true, scratch);
            std::vector<float> out(static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph), 0.f);
            nss::vaggregate_reduce(out.data(), fat.data(), srcp, pw, ph, pw, pw, sstride, radius);
            RollingPlane& stored =
                store.frames[static_cast<std::size_t>(local)].planes[static_cast<std::size_t>(plane)];
            stored.width = pw;
            stored.height = ph;
            stored.data = std::move(out);
        }
        for (int t = 0; t < ntemp; ++t) {
            vsapi->freeFrame(srcf[static_cast<std::size_t>(t)]);
            vsapi->freeFrame(reff[static_cast<std::size_t>(t)]);
        }
    }
    return true;
}

const VSFrame* VS_CC rollingGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                     VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<RollingData*>(instanceData);
    (void)frameData;
    const int chunk = d->rolling_chunk;
    const int start = rolling_chunk_start(n, chunk);
    const int count = std::min(chunk, d->bm.vi.numFrames - start);
    if (activationReason == arInitial) {
        // A cache hit observed here is not pinned until arAllFramesReady and can
        // be evicted by another request. Always declare the full dependency
        // window so a later miss can safely recompute the chunk.
        const int radius = d->bm.radius;
        const int first = std::max(0, start - radius);
        const int last = std::min(d->bm.vi.numFrames - 1, start + count - 1 + radius);
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

    const VSFrame* srcn = vsapi->getFrameFilter(n, d->bm.node, frameCtx);
    VSFrame* dst = vsapi->newVideoFrame(&d->bm.vi.format, d->bm.vi.width, d->bm.vi.height, srcn, core);
    vsapi->freeFrame(srcn);

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
            auto computed = std::make_shared<RollingChunkStore>();
            rolling_fill_chunk(d, *computed, start, count, frameCtx, core, vsapi);
            std::lock_guard<std::mutex> guard(d->cache_mu);
            d->cache.push_front(computed);
            result = std::move(computed);
            while (static_cast<int>(d->cache.size()) > d->cache_limit) {
                d->cache.pop_back();
            }
        }
    }

    const RollingFrameStore& local = result->frames[static_cast<std::size_t>(n - result->start)];
    for (int plane = 0; plane < d->bm.vi.format.numPlanes; ++plane) {
        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        rolling_write_plane(outp, dstride, local.planes[static_cast<std::size_t>(plane)]);
    }
    return dst;
}

void VS_CC rollingFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<RollingData*>(instanceData);
    vsapi->freeNode(d->bm.node);
    if (d->bm.ref) {
        vsapi->freeNode(d->bm.ref);
    }
    delete d;
}

}  // namespace

const char* fill_bm3d_data(Bm3dData& d, const VSMap* in, const VSAPI* vsapi) {
    d.node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    if (!nss::is_const_32f(d.vi)) {
        return "nss.BM3D: constant Gray/YUV/RGB 32-bit float required";
    }
    int e = 0;
    d.ref = vsapi->mapGetNode(in, "ref", 0, &e);
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
            d.sigma[i] *= (1.f / 255.f);
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
        vsapi->freeNode(d.node);
        d.node = nullptr;
    }
    if (d.ref) {
        vsapi->freeNode(d.ref);
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
    d->bm.vi_out = d->bm.vi;
    VSFilterDependency deps[2]{{d->bm.node, rpGeneral}, {d->bm.ref, rpGeneral}};
    const int ndeps = d->bm.ref ? 2 : 1;
    RollingData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("BM3D", &raw->bm.vi_out, rollingGetFrame, rollingFree, fmParallel, deps,
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
        "temporal_mode:data:opt;rolling_chunk:int:opt;rolling_cache_chunks:int:opt;rolling_cache_limit:int:opt;";
    vspapi->registerFunction("BM3D", args, "clip:vnode;", bm3dCreate, nullptr, plugin);
}
