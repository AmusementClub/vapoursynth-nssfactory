// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.PNLM: parse_pnlm with the "nss_cuda" prefix around the device
// pipeline of pnlm/kernels.hpp. Each plane with a sigma is filtered on its
// own (a plane with sigma 0 is the source's).
//
// The frames a request searches (the distinct frames n - d .. n + d) stay on
// the device in a cache shared by the streams, so that a frame is uploaded
// once however many windows it is part of: a request pins its whole window
// at once, stages the frames it is the first to need before it takes a
// stream, uploads them on its stream, and waits on the device for the ones
// other requests upload (it waits only for requests that pinned before it,
// so requests cannot wait for one another in a circle). Page-locked
// VapourSynth planes (FrameMemory, vapoursynth-classic) go to and from the
// device directly; others through pinned staging.
#include "cuda/pnlm/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/frame_memory.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/ownership.hpp"
#include "frontend/pnlm_args.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

struct Event {
    cudaEvent_t event = nullptr;
    explicit Event(unsigned flags = kStreamEvent) { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event) cudaEventDestroy(event);
    }
};

// One frame of the clip on the device (its filtered planes side by side), and
// the pinned block its upload is staged in. The fields below the buffers are
// guarded by FrameCache::mu.
struct CachedFrame {
    DeviceBuffer planes;
    PinnedBuffer staging;
    Event ready;         // behind the upload
    int n = -1;          // frame number, -1: none
    int pins = 0;        // requests reading it; it is not replaced while pinned
    bool queued = false; // the upload is queued and ready is recorded
    bool failed = false; // the request that was to upload it gave up
    std::uint64_t used = 0;
};

struct FrameCache {
    std::mutex mu;
    std::condition_variable changed;
    std::vector<std::unique_ptr<CachedFrame>> frames;
    std::uint64_t clock = 0;
};

// The window of one request: pins its frames in the cache and takes the
// cache's free frames for those that are not there. Unpins on destruction
// (after the request's stream has finished with them); a frame it did not
// get to upload is given up.
class Window {
public:
    Window(FrameCache& cache, int first, int last) : cache_(cache) {
        std::unique_lock lock(cache_.mu);
        const std::size_t count = static_cast<std::size_t>(last - first + 1);
        std::vector<CachedFrame*> found(count);
        // All of the window at once: a request never waits while it holds pins.
        cache_.changed.wait(lock, [&] {
            std::size_t missing = 0, free = 0;
            for (std::size_t i = 0; i < count; ++i) {
                found[i] = nullptr;
                for (auto& f : cache_.frames) {
                    if (f->n == first + static_cast<int>(i)) found[i] = f.get();
                }
                missing += found[i] == nullptr;
            }
            for (auto& f : cache_.frames) {
                free += f->pins == 0 && std::find(found.begin(), found.end(), f.get()) == found.end();
            }
            return missing <= free;
        });
        for (std::size_t i = 0; i < count; ++i) {
            CachedFrame* f = found[i];
            if (!f) {
                // The free frame that was used longest ago.
                for (auto& other : cache_.frames) {
                    if (other->pins != 0 || std::find(found.begin(), found.end(), other.get()) != found.end()) continue;
                    if (!f || other->used < f->used) f = other.get();
                }
                f->n = first + static_cast<int>(i);
                f->queued = false;
                f->failed = false;
                found[i] = f;
                uploads_.push_back(f);
            }
            ++f->pins;
            f->used = ++cache_.clock;
        }
        frames_ = std::move(found);
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    ~Window() {
        {
            std::lock_guard lock(cache_.mu);
            for (CachedFrame* f : uploads_) {
                if (f->queued) continue;
                f->n = -1;
                f->failed = true;
            }
            for (CachedFrame* f : frames_) --f->pins;
        }
        cache_.changed.notify_all();
    }

    // The window's frames in clip order.
    const std::vector<CachedFrame*>& frames() const { return frames_; }
    bool uploads(const CachedFrame* f) const { return std::find(uploads_.begin(), uploads_.end(), f) != uploads_.end(); }

    // Returns once the frames other requests upload are queued, so that their
    // events can be waited for on the device.
    void wait_queued() {
        std::unique_lock lock(cache_.mu);
        bool failed = false;
        cache_.changed.wait(lock, [&] {
            bool all = true;
            for (CachedFrame* f : frames_) {
                failed = failed || f->failed;
                all = all && (f->queued || uploads(f));
            }
            return failed || all;
        });
        if (failed) throw std::runtime_error("nss_cuda.PNLM: a frame of the window was not uploaded");
    }
    void mark_queued() {
        {
            std::lock_guard lock(cache_.mu);
            for (CachedFrame* f : uploads_) f->queued = true;
        }
        cache_.changed.notify_all();
    }

private:
    FrameCache& cache_;
    std::vector<CachedFrame*> frames_, uploads_;
};

struct Slot {
    Stream stream;
    DeviceBuffer out, est, keys;
};

// Where a frame in flight has its result downloaded to.
struct Stage {
    PinnedBuffer bytes;              // frame_floats
    Event done{kHostWaitEvent};      // behind the last download; the request waits for it
};
// Stages beyond the streams: requests that stage their uploads while the
// streams are busy.
constexpr std::size_t kExtraStages = 2;

struct PnlmData : nss::PnlmParams {
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node;
    VSVideoInfo vi{};
    BackendArgs backend;
    DeviceInfo device;
    std::size_t offset[4] = {};  // the filtered planes' offsets in a frame, in floats
    int planes[3] = {};
    int count = 0;               // filtered planes
    std::size_t frame_floats = 0;
    FrameCache cache;
    std::unique_ptr<SlotPool<Stage>> stages;
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<FrameMemory> frame_memory;  // direct DMA (vapoursynth-classic)
};

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<PnlmData*>(instance);
    const int first = nss::frontend::pnlm_first_frame(n, d->d);
    const int last = nss::frontend::pnlm_last_frame(n, d->d, d->vi.numFrames);
    if (activation == arInitial) {
        for (int i = first; i <= last; ++i) vsapi->requestFrameFilter(i, d->node, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("pnlm.frame");
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    std::vector<const VSFrame*> sources;
    for (int i = first; i <= last; ++i) sources.push_back(frames.getFrameFilter(i, d->node, ctx));
    const int count = static_cast<int>(sources.size()), center = n - first;
    const VSFrame* src = sources[static_cast<std::size_t>(center)];
    // Stages and streams go to the waiting requests in frame order: windows
    // then move through the cache as the clip does, whatever order
    // VapourSynth's threads bring the requests in, and at most the stages'
    // count of requests hold pins.
    auto stage = d->stages->acquire(n);
    // Unpinned last: after the wait below, when the device is done with the frames.
    Window window(d->cache, first, last);
    // Frames this request uploads: staged now, or direct (page-locked) later.
    std::vector<std::vector<bool>> direct_in(static_cast<std::size_t>(count), std::vector<bool>(d->count));
    {
        NSS_CUDA_RANGE("pnlm.stage_in");
        for (int t = 0; t < count; ++t) {
            CachedFrame* f = window.frames()[static_cast<std::size_t>(t)];
            if (!window.uploads(f)) continue;
            for (int c = 0; c < d->count; ++c) {
                const int plane = d->planes[c];
                const VSFrame* frame = sources[static_cast<std::size_t>(t)];
                if (d->frame_memory->pinned(frame, plane)) {
                    direct_in[static_cast<std::size_t>(t)][static_cast<std::size_t>(c)] = true;
                    continue;
                }
                stage_plane(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane),
                            static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float),
                            nss::plane_height(d->vi, plane), f->staging.as<float>() + d->offset[c]);
            }
        }
    }
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        if (d->sigma[plane] != 0.f) continue;
        vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                    vsapi->getStride(src, plane), static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float),
                    nss::plane_height(d->vi, plane));
    }
    bool direct_out[3]{};
    for (int c = 0; c < d->count; ++c) direct_out[c] = d->frame_memory->pinned(dst, d->planes[c]);
    window.wait_queued();
    auto* out_bytes = stage->bytes.as<std::uint8_t>();
    // On an error the queued copies must not outlive the pins, the frames and the stage.
    struct Drain {
        cudaEvent_t done;
        bool armed = false;
        ~Drain() {
            if (armed) cudaEventSynchronize(done);
        }
    } drain{stage->done.event};
    {
        auto slot = d->pool->acquire(n);
        Slot& s = *slot;
        for (int t = 0; t < count; ++t) {
            CachedFrame* f = window.frames()[static_cast<std::size_t>(t)];
            if (!window.uploads(f)) continue;
            for (int c = 0; c < d->count; ++c) {
                const int plane = d->planes[c];
                const int w = nss::plane_width(d->vi, plane), h = nss::plane_height(d->vi, plane);
                const std::size_t row = static_cast<std::size_t>(w) * sizeof(float);
                float* device = f->planes.as<float>() + d->offset[c];
                if (direct_in[static_cast<std::size_t>(t)][static_cast<std::size_t>(c)]) {
                    const VSFrame* frame = sources[static_cast<std::size_t>(t)];
                    upload_direct(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row, h, device, row,
                                  s.stream);
                } else {
                    upload_staged(f->staging.as<float>() + d->offset[c], row, h, device, row, s.stream);
                }
            }
            NSS_CUDA_CHECK(cudaEventRecord(f->ready.event, s.stream));
        }
        window.mark_queued();
        for (CachedFrame* f : window.frames()) {
            if (!window.uploads(f)) NSS_CUDA_CHECK(cudaStreamWaitEvent(s.stream, f->ready.event, 0));
        }
        drain.armed = true;
        for (int c = 0; c < d->count; ++c) {
            const int plane = d->planes[c];
            const int w = nss::plane_width(d->vi, plane), h = nss::plane_height(d->vi, plane);
            const std::size_t row = static_cast<std::size_t>(w) * sizeof(float);
            PnlmPlane pl{};
            for (int t = 0; t < count; ++t)
                pl.frames[t] = window.frames()[static_cast<std::size_t>(t)]->planes.as<float>() + d->offset[c];
            pl.count = count;
            pl.center = center;
            pl.width = w;
            pl.height = h;
            pl.pitch = w;
            pl.scale = nss::pnlm_scale(d->sigma[plane], d->block);
            pl.a = d->a;
            pl.block = d->block;
            pl.group = d->group;
            float* out = s.out.as<float>() + d->offset[c];
            pnlm_run(pl, s.est.as<float>(), s.keys.get(), out, s.stream);
            if (direct_out[c]) {
                download_direct(out, row, row, h, vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), s.stream);
            } else {
                begin_download(out, row, row, h, out_bytes + d->offset[c] * sizeof(float), s.stream);
            }
        }
        // The stream goes to the next request once this one's work is queued.
        NSS_CUDA_CHECK(cudaEventRecord(stage->done.event, s.stream));
    }
    {
        NSS_CUDA_RANGE("pnlm.wait");
        drain.armed = false;
        NSS_CUDA_CHECK(cudaEventSynchronize(stage->done.event));
    }
    for (int c = 0; c < d->count; ++c) {
        if (direct_out[c]) continue;  // the DMA wrote it
        const int plane = d->planes[c];
        const std::size_t row = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
        finish_download(out_bytes + d->offset[c] * sizeof(float), row, nss::plane_height(d->vi, plane),
                        vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane));
    }
    for (const VSFrame* f : sources) frames.freeFrame(f);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<PnlmData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->stages.reset();
        d->cache.frames.clear();
        d->frame_memory.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<PnlmData>();
    PnlmData& d = *data;
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    static_cast<nss::PnlmParams&>(d) = nss::frontend::parse_pnlm(vsapi, in, d.vi, "nss_cuda");
    d.backend = parse_backend_args(vsapi, in, "PNLM");
    d.device = acquire_device(d.backend.device_id, "PNLM", core, vsapi);
    d.budget = nss::current_budget();
    d.frame_memory = std::make_unique<FrameMemory>(core, vsapi);
    std::size_t est = 0, keys = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        if (d.sigma[plane] == 0.f) continue;
        const int w = nss::plane_width(d.vi, plane), h = nss::plane_height(d.vi, plane);
        d.planes[d.count] = plane;
        d.offset[d.count + 1] = d.offset[d.count] + static_cast<std::size_t>(w) * h;
        ++d.count;
        est = std::max(est, pnlm_est_bytes(w, h, d.block));
        if (!pnlm_fused(d.a, d.group)) keys = std::max(keys, pnlm_keys_bytes(w, h, d.block, d.group));
    }
    d.frame_floats = d.offset[d.count];
    // The window of a request: at most 2d + 1 frames (fewer near the ends).
    const std::size_t window = std::min<std::size_t>(2 * static_cast<std::size_t>(d.d) + 1,
                                                     static_cast<std::size_t>(std::max(d.vi.numFrames, 1)));
    const std::size_t frame_bytes = std::max<std::size_t>(d.frame_floats, 1) * sizeof(float);
    // The output frame a stream's request holds (rows at most 64-byte aligned).
    std::size_t output_bytes = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const std::size_t row = (static_cast<std::size_t>(nss::plane_width(d.vi, plane)) * sizeof(float) + 63) & ~std::size_t{63};
        output_bytes += row * static_cast<std::size_t>(nss::plane_height(d.vi, plane));
    }
    // A stream: the output, the estimates and the keys. A stage: the pinned
    // output and the output frame it holds. A cached frame: device and pinned.
    const std::size_t stream_bytes = frame_bytes + std::max<std::size_t>(est, 1) + keys;
    const std::size_t stage_bytes = frame_bytes + output_bytes;
    const std::size_t cached_bytes = 2 * frame_bytes;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const auto plan_bytes = [&](std::size_t streams, std::size_t stages, std::size_t cached) {
        return streams * stream_bytes + stages * stage_bytes + cached * cached_bytes;
    };
    // Without num_streams: three (fewer where the limit does not hold them;
    // 1080p d = 1, R78: gray 452 / 513 / 536 fps with one, two, three
    // streams, YUV420 304 / 344 / 357, RGB 157 / 174 / 182).
    // kExtraStages more stages, and a cache of the window and a frame for
    // each stage (down to one stage a stream and the window under a limit).
    std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 3;
    while (!d.backend.streams_explicit && streams > 1 && plan_bytes(streams, streams, window) > limit) --streams;
    std::size_t stages = streams + kExtraStages, cached = window + stages;
    while (stages > streams && plan_bytes(streams, stages, cached) > limit) cached = window + --stages;
    while (cached > window && plan_bytes(streams, stages, cached) > limit) --cached;
    if (plan_bytes(streams, stages, cached) > limit) {
        throw std::invalid_argument("nss_cuda.PNLM: memory_limit_mb is too small for this clip: it needs at least " +
                                    std::to_string((plan_bytes(streams, stages, cached) >> 20) + 1) + " MiB");
    }
    d.backend.num_streams = static_cast<int>(streams);
    {
        DeviceGuard guard(d.device.index);
        for (std::size_t i = 0; i < cached; ++i) {
            auto f = std::make_unique<CachedFrame>();
            f->planes = DeviceBuffer(frame_bytes, d.budget);
            f->staging = PinnedBuffer(frame_bytes, d.budget);
            d.cache.frames.push_back(std::move(f));
        }
        std::vector<std::unique_ptr<Slot>> slots;
        for (std::size_t i = 0; i < streams; ++i) {
            auto s = std::make_unique<Slot>();
            s->out = DeviceBuffer(frame_bytes, d.budget);
            s->est = DeviceBuffer(std::max<std::size_t>(est, 1), d.budget);
            if (keys) s->keys = DeviceBuffer(keys, d.budget);
            slots.push_back(std::move(s));
        }
        d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
        std::vector<std::unique_ptr<Stage>> staged;
        for (std::size_t i = 0; i < stages; ++i) {
            staged.push_back(std::make_unique<Stage>());
            staged.back()->bytes = PinnedBuffer(frame_bytes, d.budget);
        }
        d.stages = std::make_unique<SlotPool<Stage>>(std::move(staged));
    }
    VSFilterDependency deps[1]{{d.node, d.d == 0 ? rpStrictSpatial : rpGeneral}};
    VSNode* node =
        vsapi->createVideoFilter2("PNLM", &d.vi, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps, 1, &d, core);
    if (!node) {
        freeFilter(data.release(), core, vsapi);
        throw std::runtime_error("nss_cuda.PNLM: failed to create filter");
    }
    data.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_pnlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kPnlmSignature);
    vspapi->registerFunction("PNLM", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr,
                             plugin);
}

}  // namespace nss_cuda
