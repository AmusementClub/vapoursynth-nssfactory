// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/validate.hpp"
#include "frontend/bm4d_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_bm4d.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

// The finished frames of one chunk (model in include/nss/params/bm4d.hpp).
struct ChunkStore {
    int first = 0;
    int count = 0;
    nss::ResourceVector<float> data;  // frame-major, then plane, rows of the plane's width
};

struct Bm4dData : nss::Bm4dParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    VSVideoInfo vi{};
    int threads = 1;
    std::size_t plane_offset[3]{};
    std::size_t frame_size = 0;
    static constexpr int kCacheLimit = 2;
    std::mutex mu;  // guards cache and computing
    std::condition_variable computed;
    std::set<int> computing;
    std::list<std::shared_ptr<const ChunkStore>> cache;
};

void chunk_window(const Bm4dData* d, int n, int* first, int* count, int* v0, int* v1) {
    *first = nss::bm4d_chunk_first(n);
    *count = std::min(nss::kBm4dChunk, d->vi.numFrames - *first);
    *v0 = std::max(0, *first - nss::kBm4dChunkPad);
    *v1 = std::min(d->vi.numFrames, *first + *count + nss::kBm4dChunkPad);
}

struct TileJob {
    int plane, y0, y1, x0, x1;
};

std::shared_ptr<ChunkStore> compute_chunk(Bm4dData* d, int first, int count, int v0, int v1,
                                          const std::vector<const VSFrame*>& frames, const VSAPI* vsapi) {
    auto store = std::allocate_shared<ChunkStore>(nss::ResourceAllocator<ChunkStore>{});
    store->first = first;
    store->count = count;
    store->data.resize(d->frame_size * static_cast<std::size_t>(count));
    const int planes = d->vi.format.numPlanes;
    const int depth = v1 - v0;

    std::vector<TileJob> jobs;
    for (int plane = 0; plane < planes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane), ph = nss::plane_height(d->vi, plane);
        if (d->sigma[plane] == 0.f) {
            for (int f = 0; f < count; ++f) {
                const VSFrame* src = frames[static_cast<std::size_t>(first + f - v0)];
                const auto* sp = reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane));
                const std::ptrdiff_t ss = vsapi->getStride(src, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
                float* dp = store->data.data() + d->frame_size * f + d->plane_offset[plane];
                for (int y = 0; y < ph; ++y) std::memcpy(dp + static_cast<std::size_t>(y) * pw, sp + y * ss, pw * sizeof(float));
            }
            continue;
        }
        const int ty = nss::bm4d_tile_count(ph), tx = nss::bm4d_tile_count(pw);
        for (int i = 0; i < ty; ++i)
            for (int j = 0; j < tx; ++j)
                jobs.push_back({plane, nss::bm4d_tile_start(ph, ty, i), nss::bm4d_tile_start(ph, ty, i + 1),
                                nss::bm4d_tile_start(pw, tx, j), nss::bm4d_tile_start(pw, tx, j + 1)});
    }

    std::atomic<std::size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mu;
    const auto account = nss::current_account();
    auto worker = [&]() {
        try {
            nss::ResourceScope scope(d->budget, account);
            nss::ResourceVector<float> volume, out, work;
            for (std::size_t k; (k = next.fetch_add(1)) < jobs.size();) {
                const TileJob& job = jobs[k];
                const int pw = nss::plane_width(d->vi, job.plane), ph = nss::plane_height(d->vi, job.plane);
                const int ry0 = std::max(0, job.y0 - nss::kBm4dTilePad), ry1 = std::min(ph, job.y1 + nss::kBm4dTilePad);
                const int rx0 = std::max(0, job.x0 - nss::kBm4dTilePad), rx1 = std::min(pw, job.x1 + nss::kBm4dTilePad);
                const int h = ry1 - ry0, w = rx1 - rx0;
                const std::size_t n = static_cast<std::size_t>(depth) * h * w;
                volume.resize(n);
                out.resize(n);
                const std::size_t need = nss::bm4d_work_floats(depth, h, w);
                if (work.size() < need) work.resize(need);
                for (int z = 0; z < depth; ++z) {
                    const VSFrame* src = frames[static_cast<std::size_t>(z)];
                    const auto* sp = reinterpret_cast<const float*>(vsapi->getReadPtr(src, job.plane));
                    const std::ptrdiff_t ss = vsapi->getStride(src, job.plane) / static_cast<std::ptrdiff_t>(sizeof(float));
                    for (int y = 0; y < h; ++y)
                        std::memcpy(volume.data() + (static_cast<std::size_t>(z) * h + y) * w, sp + (ry0 + y) * ss + rx0,
                                    w * sizeof(float));
                }
                nss::bm4d_denoise_volume(volume.data(), depth, h, w, d->sigma[job.plane] / 255.f, out.data(), work.data(),
                                         work.size());
                for (int f = 0; f < count; ++f) {
                    const int z = first + f - v0;
                    float* dp = store->data.data() + d->frame_size * f + d->plane_offset[job.plane];
                    for (int y = job.y0; y < job.y1; ++y)
                        std::memcpy(dp + static_cast<std::size_t>(y) * pw + job.x0,
                                    out.data() + (static_cast<std::size_t>(z) * h + (y - ry0)) * w + (job.x0 - rx0),
                                    (job.x1 - job.x0) * sizeof(float));
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(failure_mu);
            if (!failure) failure = std::current_exception();
            next = jobs.size();
        }
    };
    const int helpers = std::min<int>(d->threads, static_cast<int>(jobs.size())) - 1;
    std::vector<std::thread> pool;
    for (int i = 0; i < helpers; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (failure) std::rethrow_exception(failure);
    return store;
}

const VSFrame* VS_CC bm4dGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<Bm4dData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    int first, count, v0, v1;
    chunk_window(d, n, &first, &count, &v0, &v1);
    if (activationReason == arInitial) {
        for (int i = v0; i < v1; ++i) vsapi->requestFrameFilter(i, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    std::shared_ptr<const ChunkStore> result;
    {
        std::unique_lock<std::mutex> lock(d->mu);
        for (;;) {
            for (auto it = d->cache.begin(); it != d->cache.end(); ++it) {
                if ((*it)->first == first) {
                    result = *it;
                    d->cache.splice(d->cache.begin(), d->cache, it);
                    break;
                }
            }
            if (result || !d->computing.count(first)) break;
            d->computed.wait(lock);  // another thread is computing this chunk
        }
        if (!result) d->computing.insert(first);
    }
    if (!result) {
        struct Computing {
            Bm4dData* d;
            int first;
            ~Computing() {
                {
                    std::lock_guard<std::mutex> guard(d->mu);
                    d->computing.erase(first);
                }
                d->computed.notify_all();
            }
        } computing{d, first};
        std::vector<const VSFrame*> frames;
        for (int i = v0; i < v1; ++i) frames.push_back(frames_owned.getFrameFilter(i, d->node, frameCtx));
        auto chunk_account = nss::make_resource_account(nss::ResourceKind::Inflight);
        std::shared_ptr<ChunkStore> store;
        {
            nss::ResourceScope chunk_scope(d->budget, chunk_account);
            store = compute_chunk(d, first, count, v0, v1, frames, vsapi);
        }
        if (chunk_account) chunk_account->retag(nss::ResourceKind::Cached);
        for (const VSFrame* f : frames) frames_owned.freeFrame(f);
        std::lock_guard<std::mutex> guard(d->mu);
        d->cache.push_front(store);
        while (static_cast<int>(d->cache.size()) > Bm4dData::kCacheLimit) d->cache.pop_back();
        result = std::move(store);
    }

    const VSFrame* src = frames_owned.getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    frames_owned.freeFrame(src);
    const float* local = result->data.data() + d->frame_size * static_cast<std::size_t>(n - result->first);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane), ph = nss::plane_height(d->vi, plane);
        const std::ptrdiff_t ds = frames_owned.getStride(dst, plane) / static_cast<std::ptrdiff_t>(sizeof(float));
        float* dp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        for (int y = 0; y < ph; ++y)
            std::memcpy(dp + y * ds, local + d->plane_offset[plane] + static_cast<std::size_t>(y) * pw, pw * sizeof(float));
    }
    return frames_owned.keep(dst);
}

void VS_CC bm4dFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    (void)vsapi;
    auto* d = static_cast<Bm4dData*>(instanceData);
    d->node.reset();
    delete d;
}

void VS_CC bm4dCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(out, "nss.BM4D: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return;
    }
    auto d = std::make_unique<Bm4dData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::Bm4dParams&>(*d) = nss::frontend::parse_bm4d(vsapi, in, d->vi, "nss");
    VSCoreInfo info{};
    vsapi->getCoreInfo(core, &info);
    d->threads = std::max(1, info.numThreads);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        d->plane_offset[plane] = d->frame_size;
        d->frame_size += static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * nss::plane_height(d->vi, plane);
    }
    VSFilterDependency deps[1]{{d->node, rpGeneral}};
    Bm4dData* raw = d.get();
    VSNode* node =
        vsapi->createVideoFilter2("BM4D", &raw->vi, nss::checked_frame<bm4dGetFrame>, bm4dFree, fmParallel, deps, 1, raw, core);
    if (!node) {
        vsapi->mapSetError(out, "nss.BM4D: failed to create filter");
        return;
    }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_bm4d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("BM4D", nss::frontend::kBm4dSignature, "clip:vnode;", nss::checked_create<bm4dCreate>, nullptr,
                             plugin);
}
