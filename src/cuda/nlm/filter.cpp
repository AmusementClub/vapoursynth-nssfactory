// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NLM. Arguments and errors come from parse_nlm with the "nss_cuda"
// prefix (D14); the whole frame (distances, box sums, weights, accumulation,
// normalization) runs on the device (D3).
//
// Two paths. The tile path runs a frame as one launch: the window frames stay
// on the device in a cache shared by the streams, so that a frame is uploaded
// once however many windows it is part of, a thread stages its uploads before
// it takes a stream and copies its result out after it gave the stream back.
// The plane path (a window, a patch or a memory limit the tile path cannot
// take) keeps only the centre frame and the current backward/forward pair
// resident, so large d does not grow device memory.
#include "cuda/nlm/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/nlm_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

using nss::host_detail::temporal_last;
using nss::host_detail::temporal_slot_frame;

// Device copies of one frame's used planes (source and, with rclip, reference).
struct FramePlanes {
    std::array<DeviceBuffer, 3> src, ref;
};

struct Slot {
    Stream stream;
    FramePlanes center, bwd, fwd;
    DeviceBuffer hsum_bwd, hsum_fwd, weight, max_weight;
    std::array<DeviceBuffer, 3> wdst, out;
    PinnedBuffer staging;
};

struct Event {
    cudaEvent_t event = nullptr;
    Event() { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event) cudaEventDestroy(event);
    }
};

// Tile path: one frame of the clip on the device, with the pinned block its
// upload is staged in. The fields below the buffers are guarded by
// WindowCache::mu.
struct WindowFrame {
    FramePlanes planes;
    PinnedBuffer staging;  // regions_per_frame planes
    Event ready;           // behind the upload
    int n = -1;            // frame number, -1: none
    int pins = 0;          // requests reading it; it is not replaced while pinned
    bool queued = false;   // the upload is queued and ready is recorded
    bool failed = false;   // the request that was to upload it gave up
    std::uint64_t used = 0;
};

struct WindowCache {
    std::mutex mu;
    std::condition_variable changed;
    std::vector<std::unique_ptr<WindowFrame>> frames;
    std::uint64_t clock = 0;
};

struct TileSlot {
    Stream stream;
    std::array<DeviceBuffer, 3> out;
    // With d = 1 and room for them: the weight maps two frames saved for the
    // frames behind them, and which frames.
    std::array<DeviceBuffer, 2> weights;
    std::array<int, 2> weights_of{-1, -1};
};

// Where a frame in flight has its result downloaded to.
struct TileStage {
    PinnedBuffer bytes;  // nc planes
    Event done;          // behind the last download
};

struct NlmData : nss::NlmParams {
    nss::NodeRef node, rclip;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    DeviceInfo device;
    int width = 0, height = 0, nc = 1, first_plane = 0;
    NlmDistance distance = NlmDistance::Luma;
    std::size_t plane_bytes = 0;
    std::unique_ptr<SlotPool<Slot>> pool;  // plane path
    bool tiles = false;
    std::unique_ptr<SlotPool<TileSlot>> tile_pool;
    std::unique_ptr<SlotPool<TileStage>> tile_stages;
    WindowCache window;
};

// Staging regions (one plane each): centre, backward, forward uploads for
// src and ref, then the outputs.
int regions_per_frame(const NlmData& d) { return d.nc * (d.rclip ? 2 : 1); }
int staging_regions(const NlmData& d) { return regions_per_frame(d) * (d.d > 0 ? 3 : 1) + d.nc; }

std::size_t slot_bytes(const NlmData& d) {
    const int frames = d.d > 0 ? 3 : 1;
    const int device_planes = regions_per_frame(d) * frames + 4 + 2 * d.nc;
    return d.plane_bytes * (device_planes + staging_regions(d));
}

std::unique_ptr<Slot> make_slot(const NlmData& d) {
    auto s = std::make_unique<Slot>();
    auto frame = [&](FramePlanes& f) {
        for (int c = 0; c < d.nc; ++c) {
            f.src[c] = DeviceBuffer(d.plane_bytes, d.budget);
            if (d.rclip) f.ref[c] = DeviceBuffer(d.plane_bytes, d.budget);
        }
    };
    frame(s->center);
    if (d.d > 0) {
        frame(s->bwd);
        frame(s->fwd);
    }
    s->hsum_bwd = DeviceBuffer(d.plane_bytes, d.budget);
    s->hsum_fwd = DeviceBuffer(d.plane_bytes, d.budget);
    s->weight = DeviceBuffer(d.plane_bytes, d.budget);
    s->max_weight = DeviceBuffer(d.plane_bytes, d.budget);
    for (int c = 0; c < d.nc; ++c) {
        s->wdst[c] = DeviceBuffer(d.plane_bytes, d.budget);
        s->out[c] = DeviceBuffer(d.plane_bytes, d.budget);
    }
    s->staging = PinnedBuffer(d.plane_bytes * staging_regions(d), d.budget);
    return s;
}

NlmPlanes planes_of(const std::array<DeviceBuffer, 3>& buffers, int nc) {
    NlmPlanes p{};
    for (int c = 0; c < nc; ++c) p.p[c] = buffers[c].as<float>();
    return p;
}

float h2_inv_norm(const NlmData& d) {
    return (255.0f * 255.0f) /
           (3.0f * d.h * d.h * (static_cast<float>(2 * d.s + 1) * static_cast<float>(2 * d.s + 1)));
}

using Frames = std::vector<const VSFrame*>;

// Frames in flight beyond one per stream on the tile path: they stage their
// input and copy their result out while the streams work.
constexpr std::size_t kTileExtraStages = 2;
// Saved weight maps take at most 1 / kWeightShare of the device's memory.
constexpr std::size_t kWeightShare = 4;

void plane_frame(const NlmData& d, const Frames& srcf, const Frames& reff, VSFrame* dst, const VSAPI* vsapi) {
    auto slot = d.pool->acquire();
    Slot& s = *slot;
    const int w = d.width, h = d.height, nc = d.nc;
    const std::size_t row_bytes = static_cast<std::size_t>(w) * sizeof(float);
    auto* staging = s.staging.as<std::uint8_t>();
    const int per_frame = regions_per_frame(d);
    auto upload_frame = [&](FramePlanes& planes, int t, int region) {
        for (int c = 0; c < nc; ++c) {
            const int plane = d.first_plane + c;
            upload_plane(vsapi->getReadPtr(srcf[t], plane), vsapi->getStride(srcf[t], plane), row_bytes, h,
                         staging + d.plane_bytes * (region + c), planes.src[c].get(), row_bytes, s.stream);
            if (d.rclip) {
                upload_plane(vsapi->getReadPtr(reff[t], plane), vsapi->getStride(reff[t], plane), row_bytes, h,
                             staging + d.plane_bytes * (region + nc + c), planes.ref[c].get(), row_bytes, s.stream);
            }
        }
    };
    upload_frame(s.center, d.d, 0);
    float* wdst[3]{s.wdst[0].as<float>(), s.wdst[1].as<float>(), s.wdst[2].as<float>()};
    nlm_reset(s.weight.as<float>(), s.max_weight.as<float>(), wdst, nc, w, h, s.stream);

    const NlmPlanes center_src = planes_of(s.center.src, nc);
    const NlmPlanes center_ref = d.rclip ? planes_of(s.center.ref, nc) : center_src;
    const float h2 = h2_inv_norm(d);
    const int span = 2 * d.a + 1;
    for (int i = -d.d; i <= 0; ++i) {
        NlmPlanes bwd_src = center_src, fwd_src = center_src, bwd_ref = center_ref, fwd_ref = center_ref;
        if (i < 0) {
            // The pair's staging regions are reused: let the previous pair finish first.
            if (i > -d.d) s.stream.synchronize();
            upload_frame(s.bwd, d.d + i, per_frame);
            upload_frame(s.fwd, d.d - i, 2 * per_frame);
            bwd_src = planes_of(s.bwd.src, nc);
            fwd_src = planes_of(s.fwd.src, nc);
            bwd_ref = d.rclip ? planes_of(s.bwd.ref, nc) : bwd_src;
            fwd_ref = d.rclip ? planes_of(s.fwd.ref, nc) : fwd_src;
        }
        for (int oy = -d.a; oy <= d.a; ++oy) {
            for (int ox = -d.a; ox <= d.a; ++ox) {
                if (static_cast<std::int64_t>(i) * span * span + static_cast<std::int64_t>(oy) * span + ox >= 0) continue;
                nlm_distance_hsum(center_ref, bwd_ref, d.distance, ox, oy, d.s, w, h, s.hsum_bwd.as<float>(),
                                  s.stream);
                if (i < 0) {
                    nlm_distance_hsum(fwd_ref, center_ref, d.distance, ox, oy, d.s, w, h, s.hsum_fwd.as<float>(),
                                      s.stream);
                }
                NlmAccumArgs args{};
                args.hsum_bwd = s.hsum_bwd.as<float>();
                args.hsum_fwd = i < 0 ? s.hsum_fwd.as<float>() : s.hsum_bwd.as<float>();
                args.src_bwd = bwd_src;
                args.src_fwd = fwd_src;
                args.channels = nc;
                args.ox = ox;
                args.oy = oy;
                args.s = d.s;
                args.h2_inv_norm = h2;
                args.width = w;
                args.height = h;
                args.weight = s.weight.as<float>();
                args.max_weight = s.max_weight.as<float>();
                for (int c = 0; c < nc; ++c) args.wdst[c] = wdst[c];
                nlm_accumulate(args, s.stream);
            }
        }
    }
    const int out_region = per_frame * (d.d > 0 ? 3 : 1);
    for (int c = 0; c < nc; ++c) {
        nlm_finish(center_src.p[c], s.weight.as<float>(), s.max_weight.as<float>(), wdst[c], d.wref, w, h,
                   s.out[c].as<float>(), s.stream);
        begin_download(s.out[c].get(), row_bytes, row_bytes, h, staging + d.plane_bytes * (out_region + c), s.stream);
    }
    s.stream.synchronize();
    for (int c = 0; c < nc; ++c) {
        const int plane = d.first_plane + c;
        finish_download(staging + d.plane_bytes * (out_region + c), row_bytes, h, vsapi->getWritePtr(dst, plane),
                        vsapi->getStride(dst, plane));
    }
}

// The window of one request on the tile path: pins its frames in the cache
// and takes the cache's free frames for those that are not there. Unpins on
// destruction; a frame it did not get to upload is given up.
class Window {
public:
    Window(WindowCache& cache, int n, int radius, int clip_frames) : cache_(cache), slots_(2 * radius + 1) {
        std::vector<int> wanted;
        for (int t = 0; t < 2 * radius + 1; ++t) {
            const int fn = temporal_slot_frame(n, t, radius, clip_frames);
            if (std::find(wanted.begin(), wanted.end(), fn) == wanted.end()) wanted.push_back(fn);
        }
        std::unique_lock lock(cache_.mu);
        std::vector<WindowFrame*> found(wanted.size());
        // All of the window at once: a request never waits while it holds pins.
        cache_.changed.wait(lock, [&] {
            std::size_t missing = 0, free = 0;
            for (std::size_t i = 0; i < wanted.size(); ++i) {
                found[i] = nullptr;
                for (auto& f : cache_.frames) {
                    if (f->n == wanted[i]) found[i] = f.get();
                }
                missing += found[i] == nullptr;
            }
            for (auto& f : cache_.frames) {
                free += f->pins == 0 && std::find(found.begin(), found.end(), f.get()) == found.end();
            }
            return missing <= free;
        });
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            WindowFrame* f = found[i];
            if (!f) {
                // The free frame that was used longest ago.
                for (auto& other : cache_.frames) {
                    if (other->pins != 0 || std::find(found.begin(), found.end(), other.get()) != found.end()) continue;
                    if (!f || other->used < f->used) f = other.get();
                }
                f->n = wanted[i];
                f->queued = false;
                f->failed = false;
                found[i] = f;
                uploads_.push_back(f);
            }
            ++f->pins;
            f->used = ++cache_.clock;
            pinned_.push_back(f);
        }
        for (int t = 0; t < 2 * radius + 1; ++t) {
            const int fn = temporal_slot_frame(n, t, radius, clip_frames);
            slots_[t] = found[std::find(wanted.begin(), wanted.end(), fn) - wanted.begin()];
        }
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    ~Window() {
        {
            std::lock_guard lock(cache_.mu);
            for (WindowFrame* f : uploads_) {
                if (f->queued) continue;
                f->n = -1;
                f->failed = true;
            }
            for (WindowFrame* f : pinned_) --f->pins;
        }
        cache_.changed.notify_all();
    }

    WindowFrame& slot(int t) const { return *slots_[t]; }
    // The frames this request uploads.
    const std::vector<WindowFrame*>& uploads() const { return uploads_; }
    bool uploads(const WindowFrame* f) const { return std::find(uploads_.begin(), uploads_.end(), f) != uploads_.end(); }
    const std::vector<WindowFrame*>& pinned() const { return pinned_; }

    // Returns once the frames other requests upload are queued, so that their
    // events can be waited for on the device.
    void wait_queued() {
        std::unique_lock lock(cache_.mu);
        bool failed = false;
        cache_.changed.wait(lock, [&] {
            bool all = true;
            for (WindowFrame* f : pinned_) {
                failed = failed || f->failed;
                all = all && (f->queued || uploads(f));
            }
            return failed || all;
        });
        if (failed) throw std::runtime_error("nss_cuda.NLM: a frame of the window was not uploaded");
    }
    void mark_queued() {
        {
            std::lock_guard lock(cache_.mu);
            for (WindowFrame* f : uploads_) f->queued = true;
        }
        cache_.changed.notify_all();
    }

private:
    WindowCache& cache_;
    std::vector<WindowFrame*> slots_, pinned_, uploads_;
};

void tile_frame(NlmData& d, int n, const Frames& srcf, const Frames& reff, VSFrame* dst, const VSAPI* vsapi) {
    const int h = d.height, nc = d.nc;
    const std::size_t row_bytes = static_cast<std::size_t>(d.width) * sizeof(float);
    auto stage = d.tile_stages->acquire();
    Window window(d.window, n, d.d, d.vi.numFrames);
    {
        NSS_CUDA_RANGE("nlm.stage_in");
        for (WindowFrame* f : window.uploads()) {
            int t = 0;
            while (&window.slot(t) != f) ++t;
            auto* staging = f->staging.as<std::uint8_t>();
            for (int c = 0; c < nc; ++c) {
                const int plane = d.first_plane + c;
                stage_plane(vsapi->getReadPtr(srcf[t], plane), vsapi->getStride(srcf[t], plane), row_bytes, h,
                            staging + d.plane_bytes * c);
                if (d.rclip) {
                    stage_plane(vsapi->getReadPtr(reff[t], plane), vsapi->getStride(reff[t], plane), row_bytes, h,
                                staging + d.plane_bytes * (nc + c));
                }
            }
        }
    }
    window.wait_queued();
    auto* out = stage->bytes.as<std::uint8_t>();
    {
        auto slot = d.tile_pool->acquire();
        TileSlot& s = *slot;
        // On an error the queued copies must not outlive the pins of the
        // frames and the lease of the staging they read and write.
        struct Drain {
            cudaStream_t stream;
            bool armed = true;
            ~Drain() {
                if (armed) cudaStreamSynchronize(stream);
            }
        } drain{s.stream};
        for (WindowFrame* f : window.uploads()) {
            auto* staging = f->staging.as<std::uint8_t>();
            for (int c = 0; c < nc; ++c) {
                upload_staged(staging + d.plane_bytes * c, row_bytes, h, f->planes.src[c].get(), row_bytes, s.stream);
                if (d.rclip) {
                    upload_staged(staging + d.plane_bytes * (nc + c), row_bytes, h, f->planes.ref[c].get(), row_bytes,
                                  s.stream);
                }
            }
            NSS_CUDA_CHECK(cudaEventRecord(f->ready.event, s.stream));
        }
        window.mark_queued();
        for (WindowFrame* f : window.pinned()) {
            if (!window.uploads(f)) NSS_CUDA_CHECK(cudaStreamWaitEvent(s.stream, f->ready.event, 0));
        }
        NlmTileArgs args{};
        for (int t = 0; t < 2 * d.d + 1; ++t) {
            const FramePlanes& planes = window.slot(t).planes;
            args.src[t] = planes_of(planes.src, nc);
            args.ref[t] = d.rclip ? planes_of(planes.ref, nc) : args.src[t];
        }
        args.distance = d.distance;
        args.channels = nc;
        args.d = d.d;
        args.a = d.a;
        args.s = d.s;
        args.h2_inv_norm = h2_inv_norm(d);
        args.wref = d.wref;
        args.width = d.width;
        args.height = h;
        for (int c = 0; c < nc; ++c) args.out[c] = s.out[c].as<float>();
        int save = -1;
        if (s.weights[0].get()) {
            // Read what frame n - 1 saved on this stream, save into the other set.
            const int load = n > 0 && s.weights_of[0] == n - 1 ? 0 : n > 0 && s.weights_of[1] == n - 1 ? 1 : -1;
            save = load == 0 ? 1 : 0;
            args.weight_load = load < 0 ? nullptr : s.weights[load].as<float>();
            args.weight_save = s.weights[save].as<float>();
            s.weights_of[save] = -1;
        }
        nlm_tile(args, s.stream);
        if (save >= 0) s.weights_of[save] = n;
        for (int c = 0; c < nc; ++c) {
            begin_download(s.out[c].get(), row_bytes, row_bytes, h, out + d.plane_bytes * c, s.stream);
        }
        NSS_CUDA_CHECK(cudaEventRecord(stage->done.event, s.stream));
        drain.armed = false;
    }
    // The staging holds the result once the event has passed.
    {
        NSS_CUDA_RANGE("nlm.wait");
        NSS_CUDA_CHECK(cudaEventSynchronize(stage->done.event));
    }
    NSS_CUDA_RANGE("nlm.stage_out");
    for (int c = 0; c < nc; ++c) {
        const int plane = d.first_plane + c;
        finish_download(out + d.plane_bytes * c, row_bytes, h, vsapi->getWritePtr(dst, plane),
                        vsapi->getStride(dst, plane));
    }
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<NlmData*>(instance);
    if (activation == arInitial) {
        for (int i = std::max(0, n - d->d); i <= temporal_last(n, d->d, d->vi.numFrames); ++i) {
            vsapi->requestFrameFilter(i, d->node, ctx);
            if (d->rclip) vsapi->requestFrameFilter(i, d->rclip, ctx);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("nlm.frame");
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);

    const int ntemp = 2 * d->d + 1;
    std::vector<const VSFrame*> srcf(ntemp), reff(ntemp, nullptr);
    for (int t = 0; t < ntemp; ++t) {
        const int fn = temporal_slot_frame(n, t, d->d, d->vi.numFrames);
        srcf[t] = frames.getFrameFilter(fn, d->node, ctx);
        if (d->rclip) reff[t] = frames.getFrameFilter(fn, d->rclip, ctx);
    }
    const VSFrame* src_center = srcf[d->d];
    VSFrame* dst = nullptr;
    if (d->channels == nss::ChannelMode::Y && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{nullptr, src_center, src_center};
        const int pl[3]{0, 1, 2};
        dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else if (d->channels == nss::ChannelMode::UV && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{src_center, nullptr, nullptr};
        const int pl[3]{0, 1, 2};
        dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else {
        dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src_center, core);
    }

    if (d->tiles) {
        tile_frame(*d, n, srcf, reff, dst, vsapi);
    } else {
        plane_frame(*d, srcf, reff, dst, vsapi);
    }
    for (int t = 0; t < ntemp; ++t) {
        frames.freeFrame(srcf[t]);
        if (reff[t]) frames.freeFrame(reff[t]);
    }
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<NlmData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->tile_pool.reset();
        d->tile_stages.reset();
        d->window.frames.clear();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.NLM (D14).
    auto d = std::make_unique<NlmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::NlmParams&>(*d) = nss::frontend::parse_nlm(vsapi, in, d->vi, "nss_cuda");
    int err = 0;
    d->rclip = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        throw std::invalid_argument("nss_cuda.NLM: rclip must match clip");
    }
    d->backend = parse_backend_args(vsapi, in, "NLM");
    d->device = acquire_device(d->backend.device_id, "NLM", core, vsapi);
    d->budget = nss::current_budget();

    const bool uv = d->channels == nss::ChannelMode::UV;
    d->width = uv ? d->vi.width >> d->vi.format.subSamplingW : d->vi.width;
    d->height = uv ? d->vi.height >> d->vi.format.subSamplingH : d->vi.height;
    d->nc = d->channels == nss::ChannelMode::Y ? 1 : uv ? 2 : 3;
    d->first_plane = uv ? 1 : 0;
    d->distance = d->channels == nss::ChannelMode::Y     ? NlmDistance::Luma
                  : uv                                   ? NlmDistance::Chroma
                  : d->channels == nss::ChannelMode::YUV ? NlmDistance::Yuv
                                                         : NlmDistance::Rgb;
    d->plane_bytes = static_cast<std::size_t>(d->width) * d->height * sizeof(float);

    const std::size_t limit = d->budget ? d->budget->snapshot().limit : SIZE_MAX;
    // Tile path: the cache holds a window at least; by default every frame in
    // flight (two more than there are streams) can have its own new frame and
    // one is spare. A limit takes the frames in flight first, then the spare
    // cache frames; below a bare window the plane path remains.
    const std::size_t streams = static_cast<std::size_t>(d->backend.num_streams);
    const std::size_t window = static_cast<std::size_t>(2 * d->d + 1);
    std::size_t stages = streams + kTileExtraStages, cached = 0;
    // With d = 1 a frame's forward weight maps are the backward maps of the
    // frame after it, so every stream keeps two sets of (2a + 1)^2 maps: one
    // to read, one to write (in order 469 to 560 fps at 1080p; nothing for
    // other request orders). They are the first thing a limit takes, and
    // without a limit they may use a quarter of the device.
    const std::size_t span = static_cast<std::size_t>(2 * d->a + 1);
    std::size_t weight_sets = d->d == 1 ? 2 : 0;
    if (streams * weight_sets * span * span * d->plane_bytes > d->device.total_memory / kWeightShare) weight_sets = 0;
    auto tile_bytes = [&] {
        return d->plane_bytes * (cached * 2 * regions_per_frame(*d) + streams * d->nc +
                                 stages * (d->nc + d->vi.format.numPlanes) + streams * weight_sets * span * span);
    };
    d->tiles = nlm_tile_supported(d->d, d->a, d->s);
    if (d->tiles) {
        cached = window + stages;
        if (tile_bytes() > limit) weight_sets = 0;
        while (tile_bytes() > limit && stages > streams) cached = window + --stages;
        while (tile_bytes() > limit && cached > window) --cached;
        d->tiles = tile_bytes() <= limit;
    }
    if (!d->tiles) {
        // Per slot: device planes, pinned staging and the leased output frame.
        const std::size_t need = slot_bytes(*d) + d->plane_bytes * d->vi.format.numPlanes;
        if (!d->backend.streams_explicit) {
            d->backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / need, 1, kDefaultStreams));
        }
        if (limit / static_cast<std::size_t>(d->backend.num_streams) < need) {
            throw std::invalid_argument("nss_cuda.NLM: memory_limit_mb is too small for this clip: each of the " +
                                        std::to_string(d->backend.num_streams) + " stream(s) needs at least " +
                                        std::to_string((need >> 20) + 1) + " MiB");
        }
    }
    DeviceGuard guard(d->device.index);
    if (d->tiles) {
        const std::size_t frame_bytes = d->plane_bytes * regions_per_frame(*d);
        for (std::size_t i = 0; i < cached; ++i) {
            auto frame = std::make_unique<WindowFrame>();
            for (int c = 0; c < d->nc; ++c) {
                frame->planes.src[c] = DeviceBuffer(d->plane_bytes, d->budget);
                if (d->rclip) frame->planes.ref[c] = DeviceBuffer(d->plane_bytes, d->budget);
            }
            frame->staging = PinnedBuffer(frame_bytes, d->budget);
            d->window.frames.push_back(std::move(frame));
        }
        std::vector<std::unique_ptr<TileSlot>> slots;
        for (std::size_t i = 0; i < streams; ++i) {
            slots.push_back(std::make_unique<TileSlot>());
            for (int c = 0; c < d->nc; ++c) slots.back()->out[c] = DeviceBuffer(d->plane_bytes, d->budget);
            for (std::size_t w = 0; w < weight_sets; ++w) {
                slots.back()->weights[w] = DeviceBuffer(d->plane_bytes * span * span, d->budget);
            }
        }
        d->tile_pool = std::make_unique<SlotPool<TileSlot>>(std::move(slots));
        std::vector<std::unique_ptr<TileStage>> staged;
        for (std::size_t i = 0; i < stages; ++i) {
            staged.push_back(std::make_unique<TileStage>());
            staged.back()->bytes = PinnedBuffer(d->plane_bytes * d->nc, d->budget);
        }
        d->tile_stages = std::make_unique<SlotPool<TileStage>>(std::move(staged));
    } else {
        std::vector<std::unique_ptr<Slot>> slots;
        for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
        d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    }

    const auto pattern = d->d == 0 ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->rclip, pattern}};
    NlmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("NLM", &raw->vi, nss::checked_frame<getFrame>, freeFilter, fmParallel,
                                            deps, d->rclip ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error("nss_cuda.NLM: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNlmSignature);
    vspapi->registerFunction("NLM", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
