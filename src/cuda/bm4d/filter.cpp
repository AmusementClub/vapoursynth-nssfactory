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
#include "cuda/runtime/frame_memory.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/bm4d_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
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
    // Two slots of input frames: a chunk uploads into its slot (chunk index
    // parity) while the other slot's chunk runs its tiles. The tiles and the
    // download run one chunk at a time (device_mu) on the shared work and
    // output buffers. A clip of one chunk never prefetches and allocates only
    // slot 0.
    struct Slot {
        std::mutex mu;  // held by the chunk using the slot
        std::unique_ptr<Stream> stream;  // uploads and downloads
        DeviceBuffer frames;   // kVolumeFrames planes of max_plane floats (the clip's frames if fewer)
        DeviceBuffer in_ptrs;
        PinnedBuffer staging;  // one plane
    };
    Slot slots[2];
    std::mutex device_mu;
    std::unique_ptr<Stream> stream;  // tiles
    DeviceBuffer volume, filtered, work;
    DeviceBuffer outputs;  // kBm4dChunk planes (the clip's frames if fewer)
    std::unique_ptr<FrameMemory> frame_memory;  // direct DMA (vapoursynth-classic)
    DeviceBuffer out_ptrs;
    // chunk cache
    static constexpr int kCacheLimit = 2;
    std::mutex mu;
    std::condition_variable computed;
    std::set<int> computing;
    std::list<std::shared_ptr<const ChunkStore>> cache;
    // Stores of chunks dropped from the cache, reused: a 1080p chunk is 265 MB,
    // and a new one is zeroed and faulted in (about 300 ms).
    std::vector<std::shared_ptr<ChunkStore>> spare;
    // Lookahead: a frame past a chunk's first quarter also asks for the next
    // chunk's frames (while that chunk is neither cached nor claimed) and
    // computes it on a thread of its own: its frames are decoded and uploaded
    // while this chunk runs its tiles, and its tiles follow in chunk order.
    // At most two prefetch threads at a time.
    std::mutex prefetch_mu;  // guards prefetchers
    std::vector<std::thread> prefetchers;
    std::atomic<int> prefetching{0};
    static constexpr int kPrefetchers = 2;
    // A request that computes its chunk in place holds a compute permit until
    // the chunk is in the cache (a store besides the cache's), and a request
    // holds an output permit, taken once it holds its chunk, until its frame
    // is copied out (an output frame): the memory plan counts kComputers
    // stores and kOutputs frames. The cache does not drop a chunk a request
    // holds (it waits for it): a request holding a chunk waits only for an
    // output permit, whose holders only copy their frame out. A finished
    // chunk's requests copy their frames out together: with two output
    // permits 1080p gray ran 10% slower.
    static constexpr int kComputers = 2, kOutputs = 8;
    struct Permit {};
    std::unique_ptr<SlotPool<Permit>> computers, permits;
};

void chunk_window(const Bm4dData* d, int n, int* first, int* count, int* v0, int* v1) {
    *first = nss::bm4d_chunk_first(n);
    *count = std::min(nss::kBm4dChunk, d->vi.numFrames - *first);
    *v0 = std::max(0, *first - nss::kBm4dChunkPad);
    *v1 = std::min(d->vi.numFrames, *first + *count + nss::kBm4dChunkPad);
}

bool chunk_known(const Bm4dData* d, int first) {  // cached or being computed; d->mu held
    for (const auto& c : d->cache)
        if (c->first == first) return true;
    return d->computing.count(first) != 0;
}

// Marks the chunk as being computed unless it is cached or being computed.
bool claim_chunk(Bm4dData* d, int first) {
    std::lock_guard<std::mutex> lock(d->mu);
    if (chunk_known(d, first)) return false;
    d->computing.insert(first);
    return true;
}

void finish_chunk(Bm4dData* d, int first, std::shared_ptr<const ChunkStore> store) {
    {
        std::lock_guard<std::mutex> guard(d->mu);
        if (store) {
            d->cache.push_front(std::move(store));
            d->computing.erase(first);
        } else {
            d->computing.erase(first);
        }
    }
    d->computed.notify_all();
    // Over the limit: the oldest chunk no request holds goes (see Bm4dData::computers).
    std::unique_lock<std::mutex> lock(d->mu);
    while (static_cast<int>(d->cache.size()) > Bm4dData::kCacheLimit) {
        auto victim = d->cache.end();
        for (auto it = std::next(d->cache.begin()); it != d->cache.end(); ++it) {
            if (it->use_count() == 1) victim = it;  // only the cache holds it
        }
        if (victim == d->cache.end()) {
            d->computed.wait_for(lock, std::chrono::milliseconds(1));  // a request lets go
            continue;
        }
        d->spare.push_back(std::const_pointer_cast<ChunkStore>(*victim));
        d->cache.erase(victim);
    }
}

std::shared_ptr<ChunkStore> compute_chunk(Bm4dData* d, int first, int count, int v0, int v1,
                                          const std::vector<const VSFrame*>& frames, const VSAPI* vsapi) {
    NSS_CUDA_RANGE("bm4d.chunk");
    std::shared_ptr<ChunkStore> store;
    {
        std::lock_guard<std::mutex> lock(d->mu);
        if (!d->spare.empty()) {
            store = std::move(d->spare.back());
            d->spare.pop_back();
        }
    }
    if (!store) store = std::allocate_shared<ChunkStore>(nss::ResourceAllocator<ChunkStore>{});
    store->first = first;
    store->count = count;
    if (store->data.size() < d->frame_size * static_cast<std::size_t>(count))
        store->data.resize(d->frame_size * static_cast<std::size_t>(std::min(nss::kBm4dChunk, d->vi.numFrames)));
    const int depth = v1 - v0;
    Bm4dData::Slot& sl = d->slots[(first / nss::kBm4dChunk) & 1];
    std::lock_guard<std::mutex> slot_lock(sl.mu);
    cudaStream_t io = *sl.stream;
    // On an error the uploads must not outlive the frames (direct DMA reads
    // them) or the slot: the caller frees the frames when this throws.
    struct Drain {
        cudaStream_t stream;
        bool armed = true;
        ~Drain() {
            if (armed) cudaStreamSynchronize(stream);
        }
    } drain{io};
    auto* staging = sl.staging.as<std::uint8_t>();
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
        {
            NSS_CUDA_RANGE("bm4d.upload");  // the volume's frames (pitch: the plane's width)
            // Page-locked planes go straight to the device (direct DMA); the
            // others share the one staging region, which each waits for.
            bool staged = false;
            for (int z = 0; z < depth; ++z) {
                const VSFrame* f = frames[static_cast<std::size_t>(z)];
                float* device = sl.frames.as<float>() + static_cast<std::size_t>(z) * d->max_plane;
                if (d->frame_memory->pinned(f, plane)) {
                    upload_direct(vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane), row, ph, device, row, io);
                    continue;
                }
                if (staged) sl.stream->synchronize();  // the staging region is reused
                upload_plane(vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane), row, ph, staging, device, row, io);
                staged = true;
            }
            sl.stream->synchronize();  // the tiles read the frames on another stream
        }
        {
            // the tiles run in chunk order: a prefetched chunk's upload
            // overlaps the earlier chunk, its tiles wait for it
            std::unique_lock<std::mutex> order(d->mu);
            d->computed.wait(order, [&] { return *d->computing.begin() >= first; });
        }
        {
            std::lock_guard<std::mutex> lock(d->device_mu);  // the shared work and output buffers
            {
                NSS_CUDA_RANGE("bm4d.tiles");
                cudaStream_t stream = *d->stream;
                const int ty = nss::bm4d_tile_count(ph), tx = nss::bm4d_tile_count(pw);
                for (int i = 0; i < ty; ++i) {
                    for (int j = 0; j < tx; ++j) {
                        const int y0 = nss::bm4d_tile_start(ph, ty, i), y1 = nss::bm4d_tile_start(ph, ty, i + 1);
                        const int x0 = nss::bm4d_tile_start(pw, tx, j), x1 = nss::bm4d_tile_start(pw, tx, j + 1);
                        const int ry0 = std::max(0, y0 - nss::kBm4dTilePad), ry1 = std::min(ph, y1 + nss::kBm4dTilePad);
                        const int rx0 = std::max(0, x0 - nss::kBm4dTilePad), rx1 = std::min(pw, x1 + nss::kBm4dTilePad);
                        const int h = ry1 - ry0, w = rx1 - rx0;
                        bm4d_gather(sl.in_ptrs.as<const float*>(), static_cast<std::size_t>(pw), depth, ry0, rx0, h, w,
                                    d->volume.as<float>(), stream);
                        const Bm4dKeep keep{first - v0, first - v0 + count, y0 - ry0, y1 - ry0, x0 - rx0, x1 - rx0};
                        bm4d_run_volume(d->volume.as<float>(), depth, h, w, d->sigma[plane] / 255.f, keep,
                                        d->filtered.as<float>(), d->work.get(), d->work.bytes(), stream);
                        bm4d_scatter(d->filtered.as<float>(), h, w, ry0, rx0, first - v0, count, y0, y1, x0, x1,
                                     d->out_ptrs.as<float*>(), static_cast<std::size_t>(pw), stream);
                    }
                }
                d->stream->synchronize();
            }
            NSS_CUDA_RANGE("bm4d.download");
            for (int f = 0; f < count; ++f) {
                begin_download(d->outputs.as<float>() + static_cast<std::size_t>(f) * d->max_plane, row, row, ph, staging, io);
                sl.stream->synchronize();
                finish_download(staging, row, ph, store->data.data() + d->frame_size * f + d->plane_offset[plane],
                                static_cast<std::ptrdiff_t>(row));
            }
        }
    }
    drain.armed = false;
    return store;
}

const VSFrame* getFrame(int n, int activation, void* instance, void** frame_data, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Bm4dData*>(instance);
    int first, count, v0, v1;
    chunk_window(d, n, &first, &count, &v0, &v1);
    bool lookahead = n - first >= nss::kBm4dChunk / 4 && first + nss::kBm4dChunk < d->vi.numFrames;
    int nfirst = 0, ncount = 0, nv0 = 0, nv1 = 0;
    if (lookahead) chunk_window(d, first + nss::kBm4dChunk, &nfirst, &ncount, &nv0, &nv1);
    if (activation == arInitial) {
        for (int i = v0; i < v1; ++i) vsapi->requestFrameFilter(i, d->node, ctx);
        if (lookahead) {  // only while the next chunk is not cached or claimed (the frames may be decoded again)
            std::lock_guard<std::mutex> lock(d->mu);
            lookahead = !chunk_known(d, nfirst);
        }
        if (lookahead)
            for (int i = std::max(nv0, v1); i < nv1; ++i) vsapi->requestFrameFilter(i, d->node, ctx);
        *frame_data = lookahead ? d : nullptr;  // whether the next chunk's frames were asked for
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    lookahead = *frame_data != nullptr;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    if (lookahead && d->prefetching.load() < Bm4dData::kPrefetchers && claim_chunk(d, nfirst)) {
        // the next chunk, on a thread of its own (before this frame waits for
        // its own chunk); its frames were asked for above
        std::vector<const VSFrame*> next;
        for (int i = nv0; i < nv1; ++i) next.push_back(vsapi->getFrameFilter(i, d->node, ctx));
        std::lock_guard<std::mutex> launch(d->prefetch_mu);
        ++d->prefetching;
        d->prefetchers.emplace_back([d, nfirst, ncount, nv0, nv1, next, vsapi]() {
            std::shared_ptr<ChunkStore> store;
            try {
                DeviceGuard guard(d->device.index);
                nss::ResourceScope scope(d->budget);
                store = compute_chunk(d, nfirst, ncount, nv0, nv1, next, vsapi);
            } catch (...) {
                store.reset();  // asked for again, the chunk is computed in place and reports the error
            }
            for (const VSFrame* f : next) vsapi->freeFrame(f);
            finish_chunk(d, nfirst, std::move(store));
            --d->prefetching;
        });
    }
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
        auto compute = d->computers->acquire();  // held until the chunk is in the cache (Finish)
        std::shared_ptr<ChunkStore> store;
        struct Finish {
            Bm4dData* d;
            int first;
            std::shared_ptr<ChunkStore>& store;
            ~Finish() { finish_chunk(d, first, store); }
        } finish{d, first, store};
        std::vector<const VSFrame*> frames;
        for (int i = v0; i < v1; ++i) frames.push_back(frames_owned.getFrameFilter(i, d->node, ctx));
        store = compute_chunk(d, first, count, v0, v1, frames, vsapi);
        for (const VSFrame* f : frames) frames_owned.freeFrame(f);
        result = store;
    }
    auto permit = d->permits->acquire();  // for the output frame, until it is returned
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
    for (auto& t : d->prefetchers)
        if (t.joinable()) t.join();
    {
        DeviceGuard guard(d->device.index);
        d->stream.reset();
        for (auto& sl : d->slots) {
            sl.stream.reset();
            sl.frames.reset();
            sl.in_ptrs.reset();
            sl.staging.reset();
        }
        d->volume.reset();
        d->filtered.reset();
        d->work.reset();
        d->outputs.reset();
        d->out_ptrs.reset();
        d->frame_memory.reset();
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
    d.frame_memory = std::make_unique<FrameMemory>(core, vsapi);
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
    const int slot_count = d.vi.numFrames > nss::kBm4dChunk ? 2 : 1;
    // Host chunk stores: stores are kept for reuse, so as many as were ever
    // in use at once: the cache's, the prefetched ones and one per compute
    // permit (and never more than the clip has chunks), each of a chunk of
    // the clip's frames at most; the output permits' frames (rows at most
    // 64-byte aligned) and the slots' pinned staging.
    const int chunk_frames = std::min(nss::kBm4dChunk, d.vi.numFrames);
    const std::size_t store_bytes = d.frame_size * chunk_frames * sizeof(float);
    const std::size_t stores = std::min<std::size_t>(Bm4dData::kCacheLimit + Bm4dData::kPrefetchers + Bm4dData::kComputers,
                                                     (d.vi.numFrames + nss::kBm4dChunk - 1) / nss::kBm4dChunk);
    std::size_t output_bytes = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const std::size_t row = (static_cast<std::size_t>(nss::plane_width(d.vi, plane)) * sizeof(float) + 63) & ~std::size_t{63};
        output_bytes += row * static_cast<std::size_t>(nss::plane_height(d.vi, plane));
    }
    const std::size_t host = stores * store_bytes +
                             Bm4dData::kOutputs * output_bytes + slot_count * plane_bytes;
    // A clip shorter than a volume or a chunk holds only its frames.
    const int volume_frames = std::min(kVolumeFrames, d.vi.numFrames), output_frames = std::min(nss::kBm4dChunk, d.vi.numFrames);
    const std::size_t need = (slot_count * volume_frames + output_frames) * plane_bytes + 2 * max_volume * sizeof(float) +
                             work + host;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (limit < need)
        throw std::invalid_argument("nss_cuda.BM4D: memory_limit_mb is too small for this clip: it needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB");
    {
        DeviceGuard guard(d.device.index);
        d.stream = std::make_unique<Stream>();
        const auto pool = [](int count) {
            std::vector<std::unique_ptr<Bm4dData::Permit>> permits;
            for (int i = 0; i < count; ++i) permits.push_back(std::make_unique<Bm4dData::Permit>());
            return std::make_unique<SlotPool<Bm4dData::Permit>>(std::move(permits));
        };
        d.computers = pool(Bm4dData::kComputers);
        d.permits = pool(Bm4dData::kOutputs);
        if (max_volume) {
            d.volume = DeviceBuffer(max_volume * sizeof(float), d.budget);
            d.filtered = DeviceBuffer(max_volume * sizeof(float), d.budget);
            d.work = DeviceBuffer(work, d.budget);
            d.outputs = DeviceBuffer(output_frames * plane_bytes, d.budget);
            std::vector<float*> out_ptrs(output_frames);
            for (int f = 0; f < output_frames; ++f) out_ptrs[f] = d.outputs.as<float>() + static_cast<std::size_t>(f) * d.max_plane;
            d.out_ptrs = DeviceBuffer(output_frames * sizeof(float*), d.budget);
            NSS_CUDA_CHECK(cudaMemcpy(d.out_ptrs.get(), out_ptrs.data(), output_frames * sizeof(float*), cudaMemcpyHostToDevice));
            for (int i = 0; i < slot_count; ++i) {
                Bm4dData::Slot& sl = d.slots[i];
                sl.stream = std::make_unique<Stream>();
                sl.frames = DeviceBuffer(volume_frames * plane_bytes, d.budget);
                std::vector<const float*> in_ptrs(volume_frames);
                for (int z = 0; z < volume_frames; ++z) in_ptrs[z] = sl.frames.as<float>() + static_cast<std::size_t>(z) * d.max_plane;
                sl.in_ptrs = DeviceBuffer(volume_frames * sizeof(float*), d.budget);
                NSS_CUDA_CHECK(cudaMemcpy(sl.in_ptrs.get(), in_ptrs.data(), volume_frames * sizeof(float*), cudaMemcpyHostToDevice));
                sl.staging = PinnedBuffer(plane_bytes, d.budget);
            }
            std::vector<float> tables(bm4d_table_floats());
            bm4d_fill_tables(tables.data());
            bm4d_upload_constants(tables.data());
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
