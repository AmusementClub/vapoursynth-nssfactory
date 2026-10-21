// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM4D: parse_bm4d with the "nss_cuda" prefix around the device
// pipeline of bm4d/kernels.hpp, on the CPU node's volumes (32-frame chunks
// with 8 frames of padding, tiles of about 256 with 16 pixels of overlap).
// A chunk's frames are uploaded once; each tile volume is gathered, filtered
// and its kept part scattered into the chunk's output frames on the device;
// the finished chunk is downloaded and cached for its other frames.
#include "cuda/bm4d/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/bm4d_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

constexpr int kVolumeFrames = nss::kBm4dChunk + 2 * nss::kBm4dChunkPad;

struct ChunkStore {
    int first = 0, count = 0;
    nss::ResourceVector<float> data;  // frame-major, then plane, rows of the plane's width
};

struct Bm4dData : nss::Bm4dParams {
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node;
    VSVideoInfo vi{};
    BackendArgs backend;
    DeviceInfo device;
    std::size_t plane_offset[3]{};
    std::size_t frame_size = 0;
    std::size_t max_plane = 0;
    int max_pw = 0;
    // device state, used by one chunk at a time
    std::mutex device_mu;
    std::unique_ptr<Stream> stream;
    DeviceBuffer frames;      // kVolumeFrames planes of max_plane floats
    DeviceBuffer outputs;     // kBm4dChunk planes
    DeviceBuffer in_ptrs, out_ptrs;
    DeviceBuffer volume, filtered, work, tables;
    PinnedBuffer staging;     // one plane
    // chunk cache
    static constexpr int kCacheLimit = 2;
    std::mutex mu;
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

std::shared_ptr<ChunkStore> compute_chunk(Bm4dData* d, int first, int count, int v0, int v1,
                                          const std::vector<const VSFrame*>& frames, const VSAPI* vsapi) {
    NSS_CUDA_RANGE("bm4d.chunk");
    auto store = std::allocate_shared<ChunkStore>(nss::ResourceAllocator<ChunkStore>{});
    store->first = first;
    store->count = count;
    store->data.resize(d->frame_size * static_cast<std::size_t>(count));
    const int depth = v1 - v0;
    std::lock_guard<std::mutex> lock(d->device_mu);
    cudaStream_t stream = *d->stream;
    auto* staging = d->staging.as<std::uint8_t>();
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane), ph = nss::plane_height(d->vi, plane);
        const std::size_t row = static_cast<std::size_t>(pw) * sizeof(float);
        if (d->sigma[plane] == 0.f) {
            for (int f = 0; f < count; ++f) {
                const VSFrame* src = frames[static_cast<std::size_t>(first + f - v0)];
                vsh::bitblt(store->data.data() + d->frame_size * f + d->plane_offset[plane], row, vsapi->getReadPtr(src, plane),
                            vsapi->getStride(src, plane), row, ph);
            }
            continue;
        }
        // upload the volume's frames (pitch: the plane's width)
        for (int z = 0; z < depth; ++z) {
            upload_plane(vsapi->getReadPtr(frames[static_cast<std::size_t>(z)], plane),
                         vsapi->getStride(frames[static_cast<std::size_t>(z)], plane), row, ph, staging,
                         d->frames.as<float>() + static_cast<std::size_t>(z) * d->max_plane, row, stream);
            d->stream->synchronize();  // the staging region is reused
        }
        const int ty = nss::bm4d_tile_count(ph), tx = nss::bm4d_tile_count(pw);
        for (int i = 0; i < ty; ++i) {
            for (int j = 0; j < tx; ++j) {
                const int y0 = nss::bm4d_tile_start(ph, ty, i), y1 = nss::bm4d_tile_start(ph, ty, i + 1);
                const int x0 = nss::bm4d_tile_start(pw, tx, j), x1 = nss::bm4d_tile_start(pw, tx, j + 1);
                const int ry0 = std::max(0, y0 - nss::kBm4dTilePad), ry1 = std::min(ph, y1 + nss::kBm4dTilePad);
                const int rx0 = std::max(0, x0 - nss::kBm4dTilePad), rx1 = std::min(pw, x1 + nss::kBm4dTilePad);
                const int h = ry1 - ry0, w = rx1 - rx0;
                bm4d_gather(d->in_ptrs.as<const float*>(), static_cast<std::size_t>(pw), depth, ry0, rx0, h, w,
                            d->volume.as<float>(), stream);
                bm4d_run_volume(d->volume.as<float>(), depth, h, w, d->sigma[plane] / 255.f, d->filtered.as<float>(),
                                d->tables.as<float>(), d->work.get(), d->work.bytes(), stream);
                bm4d_scatter(d->filtered.as<float>(), h, w, ry0, rx0, first - v0, count, y0, y1, x0, x1,
                             d->out_ptrs.as<float*>(), static_cast<std::size_t>(pw), stream);
            }
        }
        for (int f = 0; f < count; ++f) {
            begin_download(d->outputs.as<float>() + static_cast<std::size_t>(f) * d->max_plane, row, row, ph, staging, stream);
            d->stream->synchronize();
            finish_download(staging, row, ph, store->data.data() + d->frame_size * f + d->plane_offset[plane],
                            static_cast<std::ptrdiff_t>(row));
        }
    }
    return store;
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Bm4dData*>(instance);
    int first, count, v0, v1;
    chunk_window(d, n, &first, &count, &v0, &v1);
    if (activation == arInitial) {
        for (int i = v0; i < v1; ++i) vsapi->requestFrameFilter(i, d->node, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
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
            d->computed.wait(lock);
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
        for (int i = v0; i < v1; ++i) frames.push_back(frames_owned.getFrameFilter(i, d->node, ctx));
        auto store = compute_chunk(d, first, count, v0, v1, frames, vsapi);
        for (const VSFrame* f : frames) frames_owned.freeFrame(f);
        std::lock_guard<std::mutex> guard(d->mu);
        d->cache.push_front(store);
        while (static_cast<int>(d->cache.size()) > Bm4dData::kCacheLimit) d->cache.pop_back();
        result = std::move(store);
    }
    const VSFrame* src = frames_owned.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = frames_owned.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    frames_owned.freeFrame(src);
    const float* local = result->data.data() + d->frame_size * static_cast<std::size_t>(n - result->first);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane), ph = nss::plane_height(d->vi, plane);
        vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), local + d->plane_offset[plane],
                    static_cast<std::ptrdiff_t>(pw) * sizeof(float), static_cast<std::size_t>(pw) * sizeof(float), ph);
    }
    return frames_owned.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Bm4dData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->stream.reset();
        d->frames.reset();
        d->outputs.reset();
        d->in_ptrs.reset();
        d->out_ptrs.reset();
        d->volume.reset();
        d->filtered.reset();
        d->work.reset();
        d->tables.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<Bm4dData>();
    Bm4dData& d = *data;
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    static_cast<nss::Bm4dParams&>(d) = nss::frontend::parse_bm4d(vsapi, in, d.vi, "nss_cuda");
    d.backend = parse_backend_args(vsapi, in, "BM4D");
    d.device = acquire_device(d.backend.device_id, "BM4D", core, vsapi);
    d.budget = nss::current_budget();
    std::size_t max_volume = 0, work = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d.vi, plane), ph = nss::plane_height(d.vi, plane);
        d.plane_offset[plane] = d.frame_size;
        d.frame_size += static_cast<std::size_t>(pw) * ph;
        d.max_plane = std::max(d.max_plane, static_cast<std::size_t>(pw) * ph);
        d.max_pw = std::max(d.max_pw, pw);
        if (d.sigma[plane] == 0.f) continue;
        const int h = std::min(ph, nss::bm4d_tile_max(ph) + 2 * nss::kBm4dTilePad);
        const int w = std::min(pw, nss::bm4d_tile_max(pw) + 2 * nss::kBm4dTilePad);
        const int t = std::min(d.vi.numFrames, kVolumeFrames);
        max_volume = std::max(max_volume, static_cast<std::size_t>(t) * h * w);
        work = std::max(work, bm4d_volume_work_bytes(t, h, w));
    }
    const std::size_t plane_bytes = d.max_plane * sizeof(float);
    const std::size_t need = (kVolumeFrames + nss::kBm4dChunk) * plane_bytes + 2 * max_volume * sizeof(float) + work;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (limit < need)
        throw std::invalid_argument("nss_cuda.BM4D: memory_limit_mb is too small for this clip: it needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB");
    {
        DeviceGuard guard(d.device.index);
        d.stream = std::make_unique<Stream>();
        if (max_volume) {
            d.frames = DeviceBuffer(kVolumeFrames * plane_bytes, d.budget);
            d.outputs = DeviceBuffer(nss::kBm4dChunk * plane_bytes, d.budget);
            d.volume = DeviceBuffer(max_volume * sizeof(float), d.budget);
            d.filtered = DeviceBuffer(max_volume * sizeof(float), d.budget);
            d.work = DeviceBuffer(work, d.budget);
            std::vector<const float*> in_ptrs(kVolumeFrames);
            std::vector<float*> out_ptrs(nss::kBm4dChunk);
            for (int z = 0; z < kVolumeFrames; ++z) in_ptrs[z] = d.frames.as<float>() + static_cast<std::size_t>(z) * d.max_plane;
            for (int f = 0; f < nss::kBm4dChunk; ++f) out_ptrs[f] = d.outputs.as<float>() + static_cast<std::size_t>(f) * d.max_plane;
            d.in_ptrs = DeviceBuffer(kVolumeFrames * sizeof(float*), d.budget);
            d.out_ptrs = DeviceBuffer(nss::kBm4dChunk * sizeof(float*), d.budget);
            NSS_CUDA_CHECK(cudaMemcpy(d.in_ptrs.get(), in_ptrs.data(), kVolumeFrames * sizeof(float*), cudaMemcpyHostToDevice));
            NSS_CUDA_CHECK(cudaMemcpy(d.out_ptrs.get(), out_ptrs.data(), nss::kBm4dChunk * sizeof(float*), cudaMemcpyHostToDevice));
            std::vector<float> tables(bm4d_table_floats());
            bm4d_fill_tables(tables.data());
            d.tables = DeviceBuffer(tables.size() * sizeof(float), d.budget);
            NSS_CUDA_CHECK(cudaMemcpy(d.tables.get(), tables.data(), tables.size() * sizeof(float), cudaMemcpyHostToDevice));
            d.staging = PinnedBuffer(plane_bytes, d.budget);
        }
    }
    VSFilterDependency deps[1]{{d.node, rpGeneral}};
    VSNode* node =
        vsapi->createVideoFilter2("BM4D", &d.vi, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps, 1, &d, core);
    if (!node) {
        freeFilter(data.release(), core, vsapi);
        throw std::runtime_error("nss_cuda.BM4D: failed to create filter");
    }
    data.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_bm4d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kBm4dSignature);
    vspapi->registerFunction("BM4D", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr,
                             plugin);
}

}  // namespace nss_cuda
