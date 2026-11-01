// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NLM. Arguments and errors come from parse_nlm with the "nss_cuda"
// prefix; the whole frame (distances, box sums, weights, accumulation,
// normalization) runs on the device.
//
// Two paths. The tile path runs a frame as one launch: the window frames stay
// on the device in a cache shared by the streams, so that a frame is uploaded
// once however many windows it is part of, a thread stages its uploads before
// it takes a stream and copies its result out after it gave the stream back.
// The plane path (a window, a patch or a memory limit the tile path cannot
// take) keeps only the centre frame and the current backward/forward pair
// resident, so large d does not grow device memory.
//
// Both copy page-locked VapourSynth frame planes (FrameMemory, under
// vapoursynth-classic) to and from the device directly; other planes go
// through pinned staging.
#include "cuda/nlm/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/frame_memory.hpp"
#include "cuda/runtime/gpu_frames.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "cuda/runtime/vs_api.hpp"
#include "frontend/nlm_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
    std::unique_ptr<GpuStream> gpu;
    FramePlanes center, bwd, fwd;
    DeviceBuffer hsum_bwd, hsum_fwd, weight, max_weight;
    std::array<DeviceBuffer, 3> wdst, out;
    PinnedBuffer staging;
};

struct Event {
    cudaEvent_t event = nullptr;
    explicit Event(unsigned flags = kStreamEvent) { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags)); }
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
    std::unique_ptr<GpuStream> gpu;
    std::array<DeviceBuffer, 3> out;
};

// d = 1 with room for them: weight maps a frame saved for the frame after it
// (NlmTileArgs::weight_save), shared by the streams so that the frame after
// it finds them whichever stream it gets. Guarded by WeightCache::mu.
struct WeightSet {
    DeviceBuffer maps;      // (2a + 1)^2 planes
    Event written, read;    // behind the last kernel that saved / loaded them
    int frame = -1;         // the frame that saved them, -1: none
    bool busy = false;      // a request is queueing a kernel that saves or loads them
    bool loaded = false;    // a kernel loaded them since they were saved
    std::uint64_t used = 0;
};

struct WeightCache {
    std::mutex mu;
    std::vector<std::unique_ptr<WeightSet>> sets;
    std::uint64_t clock = 0;
};

// The sets a frame's kernel loads and saves, taken under the cache's lock:
// the kernel waits on the device for the one that saved what it loads, and
// for the kernels that last saved or loaded the set it overwrites. Nothing
// waits on the host, so requests in any order cannot block one another; a
// frame whose predecessor has not queued its kernel yet computes in full.
class WeightLease {
public:
    WeightLease(WeightCache& cache, int n, cudaStream_t stream) : cache_(cache), stream_(stream) {
        if (cache.sets.empty()) return;
        std::lock_guard lock(cache.mu);
        for (auto& w : cache.sets) {
            if (n > 0 && w->frame == n - 1 && !w->busy && !w->loaded) load_ = w.get();
        }
        // A set that was loaded or holds nothing before the oldest that still waits for its frame.
        const auto rank = [](const WeightSet* w) { return std::pair(w->frame >= 0 && !w->loaded, w->used); };
        for (auto& w : cache.sets) {
            if (w.get() != load_ && !w->busy && (!save_ || rank(w.get()) < rank(save_))) save_ = w.get();
        }
        if (load_) {
            load_->busy = true;
            NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, load_->written.event, 0));
        }
        if (save_) {
            save_->busy = true;
            save_->frame = -1;
            NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, save_->written.event, 0));
            NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, save_->read.event, 0));
        }
    }
    WeightLease(const WeightLease&) = delete;
    WeightLease& operator=(const WeightLease&) = delete;
    // Without queued(), as after an error: the sets are given back holding
    // nothing, behind whatever was queued on the stream.
    ~WeightLease() {
        if (!load_ && !save_) return;
        std::lock_guard lock(cache_.mu);
        if (load_) {
            cudaEventRecord(load_->read.event, stream_);
            load_->busy = false;
        }
        if (save_) {
            cudaEventRecord(save_->written.event, stream_);
            save_->busy = false;
        }
        cudaGetLastError();
    }
    const float* load() const { return load_ ? load_->maps.as<float>() : nullptr; }
    float* save() const { return save_ ? save_->maps.as<float>() : nullptr; }
    // After the kernel of frame n is queued: the frame after it may load the
    // saved set from now on.
    void queued(int n) {
        if (!load_ && !save_) return;
        std::lock_guard lock(cache_.mu);
        if (load_) {
            NSS_CUDA_CHECK(cudaEventRecord(load_->read.event, stream_));
            load_->loaded = true;
            load_->busy = false;
        }
        if (save_) {
            NSS_CUDA_CHECK(cudaEventRecord(save_->written.event, stream_));
            save_->frame = n;
            save_->loaded = false;
            save_->used = ++cache_.clock;
            save_->busy = false;
        }
        load_ = save_ = nullptr;
    }

private:
    WeightCache& cache_;
    cudaStream_t stream_;
    WeightSet* load_ = nullptr;
    WeightSet* save_ = nullptr;
};

// Where a frame in flight has its result downloaded to.
struct TileStage {
    PinnedBuffer bytes;  // nc planes
    Event done{kHostWaitEvent};  // behind the last download; the request waits for it
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
    std::shared_ptr<GpuFrames> gpu;
    std::unique_ptr<SlotPool<Slot>> pool;  // plane path
    bool tiles = false;
    std::unique_ptr<SlotPool<TileSlot>> tile_pool;
    std::unique_ptr<SlotPool<TileStage>> tile_stages;
    WindowCache window;
    WeightCache weights;
    std::unique_ptr<FrameMemory> frame_memory;
};

// Staging regions (one plane each): centre, backward, forward uploads for
// src and ref, then the outputs.
int regions_per_frame(const NlmData& d) { return d.nc * (d.rclip ? 2 : 1); }
int staging_regions(const NlmData& d) { return regions_per_frame(d) * (d.d > 0 ? 3 : 1) + d.nc; }
int device_regions(const NlmData& d) { return regions_per_frame(d) * (d.d > 0 ? 3 : 1) + 4 + 2 * d.nc; }

std::size_t slot_bytes(const NlmData& d) {
    return d.plane_bytes * (device_regions(d) + staging_regions(d));
}

std::size_t slot_device_bytes(const NlmData& d) { return d.plane_bytes * device_regions(d); }

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
    if (d.gpu) s->gpu = std::make_unique<GpuStream>(d.gpu, s->stream);
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
// The output frame is made once the request holds the stage (tile path) or
// slot (plane path) that the memory plan counts it in, and that lease is held
// (by getFrame, through `Held`) until the frame is returned: made before, or
// held for less, more output frames than the plan counts were charged to
// memory_limit_mb at once.
using MakeOutput = std::function<VSFrame*()>;
struct Held {
    std::optional<SlotPool<TileStage>::Lease> stage;
    std::optional<SlotPool<Slot>::Lease> slot;
};

// Frames in flight beyond one per stream on the tile path: they stage their
// input and copy their result out while the streams work.
constexpr std::size_t kTileExtraStages = 2;
// Saved weight maps take at most 1 / kWeightShare of the device's memory.
constexpr std::size_t kWeightShare = 4;

void plane_frame(const NlmData& d, const Frames& srcf, const Frames& reff, const MakeOutput& make_output,
                 Held& held, const VSAPI* vsapi) {
    held.slot.emplace(d.pool->acquire());
    VSFrame* dst = make_output();
    Slot& s = **held.slot;
    if (s.gpu) s.gpu->begin();
    // On an error the queued copies must not outlive the frames (direct
    // DMA reads and writes them), the staging and the slot (as tile_frame).
    struct Drain {
        cudaStream_t stream;
        bool armed = true;
        ~Drain() {
            if (armed) cudaStreamSynchronize(stream);
        }
    } drain{s.stream};
    const int w = d.width, h = d.height, nc = d.nc;
    const std::size_t row_bytes = static_cast<std::size_t>(w) * sizeof(float);
    auto* staging = s.staging.as<std::uint8_t>();
    const int per_frame = regions_per_frame(d);
    // The frames are held until the stream has synchronized below.
    const auto upload = [&](const VSFrame* frame, int plane, int region, void* device) {
        if (s.gpu) {
            s.gpu->read(frame, plane, device, row_bytes, row_bytes, h);
        } else if (d.frame_memory->pinned(frame, plane)) {
            upload_direct(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, h, device,
                          row_bytes, s.stream);
        } else {
            upload_plane(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, h,
                         staging + d.plane_bytes * region, device, row_bytes, s.stream);
        }
    };
    auto upload_frame = [&](FramePlanes& planes, int t, int region) {
        for (int c = 0; c < nc; ++c) {
            const int plane = d.first_plane + c;
            upload(srcf[t], plane, region + c, planes.src[c].get());
            if (d.rclip) upload(reff[t], plane, region + nc + c, planes.ref[c].get());
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
                args.weight_mode = static_cast<NlmWeight>(d.weight);
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
    std::array<bool, 3> direct_out{};
    std::array<int, 3> output_planes{0, 1, 2};
    for (int c = 0; c < nc; ++c) {
        const int plane = d.first_plane + c;
        nlm_finish(center_src.p[c], s.weight.as<float>(), s.max_weight.as<float>(), wdst[c], d.wref, w, h,
                   s.out[c].as<float>(), s.stream);
        if (s.gpu) {
            s.gpu->write(s.out[c].get(), row_bytes, dst, plane, row_bytes, h);
            continue;
        }
        direct_out[c] = d.frame_memory->pinned(dst, plane);
        if (direct_out[c]) {
            download_direct(s.out[c].get(), row_bytes, row_bytes, h, vsapi->getWritePtr(dst, plane),
                            vsapi->getStride(dst, plane), s.stream);
        } else {
            begin_download(s.out[c].get(), row_bytes, row_bytes, h, staging + d.plane_bytes * (out_region + c),
                           s.stream);
        }
    }
    if (s.gpu) {
        for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
            if (plane >= d.first_plane && plane < d.first_plane + nc) continue;
            s.gpu->copy(srcf[d.d], plane, dst, plane,
                        static_cast<std::size_t>(nss::plane_width(d.vi, plane)) * sizeof(float),
                        nss::plane_height(d.vi, plane));
        }
        s.gpu->publish(dst, output_planes.data(), d.vi.format.numPlanes);
    }
    s.stream.synchronize();
    drain.armed = false;
    for (int c = 0; c < nc; ++c) {
        if (s.gpu || direct_out[c]) continue;
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

void tile_frame(NlmData& d, int n, const Frames& srcf, const Frames& reff, const MakeOutput& make_output,
                Held& held, const VSAPI* vsapi) {
    const int h = d.height, nc = d.nc;
    const std::size_t row_bytes = static_cast<std::size_t>(d.width) * sizeof(float);
    // Stages and streams go to the waiting requests in frame order: windows
    // then move through the cache as the clip does (in the order VapourSynth's
    // threads came, the cache dropped frames that waiting requests needed:
    // 1080p gray d = 1 on cached input uploaded 1.49 frames a frame), and a
    // frame mostly finds the weight maps of the frame before it.
    held.stage.emplace(d.tile_stages->acquire(n));
    auto& stage = *held.stage;
    VSFrame* dst = make_output();
    Window window(d.window, n, d.d, d.vi.numFrames);
    const std::vector<WindowFrame*>& uploads = window.uploads();
    // Per upload and staging region: the frame whose plane is uploaded
    // directly (page-locked; held until the stage's event has passed)
    // instead of from the staging.
    std::vector<std::array<const VSFrame*, 6>> direct_in(uploads.size());
    {
        NSS_CUDA_RANGE("nlm.stage_in");
        for (std::size_t i = 0; i < uploads.size(); ++i) {
            int t = 0;
            while (&window.slot(t) != uploads[i]) ++t;
            auto* staging = uploads[i]->staging.as<std::uint8_t>();
            const auto stage_in = [&](const VSFrame* frame, int plane, int region) {
                if (d.gpu || d.frame_memory->pinned(frame, plane)) {
                    direct_in[i][static_cast<std::size_t>(region)] = frame;
                    return;
                }
                stage_plane(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, h,
                            staging + d.plane_bytes * region);
            };
            for (int c = 0; c < nc; ++c) {
                stage_in(srcf[t], d.first_plane + c, c);
                if (d.rclip) stage_in(reff[t], d.first_plane + c, nc + c);
            }
        }
    }
    std::array<bool, 3> direct_out{};
    if (!d.gpu) {
        for (int c = 0; c < nc; ++c) direct_out[c] = d.frame_memory->pinned(dst, d.first_plane + c);
    }
    window.wait_queued();
    auto* out = stage->bytes.as<std::uint8_t>();
    {
        auto slot = d.tile_pool->acquire(n);
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
        if (s.gpu) s.gpu->begin();
        for (std::size_t i = 0; i < uploads.size(); ++i) {
            WindowFrame* f = uploads[i];
            auto* staging = f->staging.as<std::uint8_t>();
            const auto upload = [&](int plane, int region, void* device) {
                if (s.gpu) {
                    s.gpu->read(direct_in[i][static_cast<std::size_t>(region)], plane, device, row_bytes, row_bytes, h);
                } else if (const VSFrame* frame = direct_in[i][static_cast<std::size_t>(region)]) {
                    upload_direct(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, h,
                                  device, row_bytes, s.stream);
                } else {
                    upload_staged(staging + d.plane_bytes * region, row_bytes, h, device, row_bytes, s.stream);
                }
            };
            for (int c = 0; c < nc; ++c) {
                upload(d.first_plane + c, c, f->planes.src[c].get());
                if (d.rclip) upload(d.first_plane + c, nc + c, f->planes.ref[c].get());
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
        args.weight_mode = static_cast<NlmWeight>(d.weight);
        args.h2_inv_norm = h2_inv_norm(d);
        args.wref = d.wref;
        args.width = d.width;
        args.height = h;
        for (int c = 0; c < nc; ++c) args.out[c] = s.out[c].as<float>();
        WeightLease weights(d.weights, n, s.stream);
        args.weight_load = weights.load();
        args.weight_save = weights.save();
        nlm_tile(args, s.stream);
        weights.queued(n);
        for (int c = 0; c < nc; ++c) {
            const int plane = d.first_plane + c;
            if (s.gpu) {
                s.gpu->write(s.out[c].get(), row_bytes, dst, plane, row_bytes, h);
                s.gpu->publish(dst, &plane, 1);
            } else if (direct_out[c]) {
                download_direct(s.out[c].get(), row_bytes, row_bytes, h, vsapi->getWritePtr(dst, plane),
                                vsapi->getStride(dst, plane), s.stream);
            } else {
                begin_download(s.out[c].get(), row_bytes, row_bytes, h, out + d.plane_bytes * c, s.stream);
            }
        }
        if (s.gpu) {
            for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
                if (plane >= d.first_plane && plane < d.first_plane + nc) continue;
                const std::size_t row = static_cast<std::size_t>(nss::plane_width(d.vi, plane)) * sizeof(float);
                s.gpu->copy(srcf[d.d], plane, dst, plane, row, nss::plane_height(d.vi, plane));
                s.gpu->publish(dst, &plane, 1);
            }
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
        if (d.gpu || direct_out[c]) continue;
        const int plane = d.first_plane + c;
        // Three planes go out with non-temporal stores: RGB 1080p 236 -> 250 fps in three sessions, where
        // one plane lost 2% with them (docs/archive/cuda-head-r2-20261020.md).
        if (nc == 3) {
            copy_to_frame(out + d.plane_bytes * c, static_cast<std::ptrdiff_t>(row_bytes), vsapi->getWritePtr(dst, plane),
                          vsapi->getStride(dst, plane), row_bytes, h);
        } else {
            finish_download(out + d.plane_bytes * c, row_bytes, h, vsapi->getWritePtr(dst, plane),
                            vsapi->getStride(dst, plane));
        }
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
    // Exportable frames are consumed from availableFrames; temporal edge duplicates take references instead.
    const auto take = [&](VSNode* node, Frames& out, int t, int fn) {
        if (!d->gpu) return frames.getFrameFilter(fn, node, ctx);
        for (int u = 0; u < t; ++u) {
            if (out[u] && temporal_slot_frame(n, u, d->d, d->vi.numFrames) == fn) {
                return frames.adoptInput(vsapi->addFrameRef(out[u]));
            }
        }
        return d->gpu->get_frame(fn, node, ctx, frames);
    };
    for (int t = 0; t < ntemp; ++t) {
        const int fn = temporal_slot_frame(n, t, d->d, d->vi.numFrames);
        srcf[t] = take(d->node, srcf, t, fn);
        if (d->rclip) reff[t] = take(d->rclip, reff, t, fn);
    }
    const VSFrame* src_center = srcf[d->d];
    VSFrame* dst = nullptr;
    const MakeOutput make_output = [&]() -> VSFrame* {
        // Exportable input planes are read-only and cannot return to the core, so every output plane is fresh.
        if (d->gpu) {
            dst = d->gpu->new_frame(frames, &d->vi.format, d->vi.width, d->vi.height, src_center);
        } else if (d->channels == nss::ChannelMode::Y && d->vi.format.numPlanes > 1) {
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
        return dst;
    };
    Held held;  // released after keep() below
    if (d->tiles) {
        tile_frame(*d, n, srcf, reff, make_output, held, vsapi);
    } else {
        plane_frame(*d, srcf, reff, make_output, held, vsapi);
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
        d->weights.sets.clear();
        d->frame_memory.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.NLM.
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
    const VSVULKANAPI* gpu_api = take_clips("NLM", {{"clip", &d->node}, {"rclip", &d->rclip}}, core, vsapi);
    d->device = gpu_api ? acquire_core_device(d->backend, "NLM", core, vsapi, gpu_api)
                        : acquire_device(d->backend.device_id, "NLM", core, vsapi);
    d->budget = nss::current_budget();
    if (gpu_api) d->gpu = std::make_shared<GpuFrames>(core, gpu_api, vsapi);
    d->frame_memory = std::make_unique<FrameMemory>(core, vsapi);

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
    // The output frame a request holds: every plane of the format, as it is
    // charged (rows at most 64-byte aligned), whatever the channels filtered.
    std::size_t output_bytes = 0;
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const std::size_t row = (static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float) + 63) & ~std::size_t{63};
        output_bytes += row * static_cast<std::size_t>(nss::plane_height(d->vi, plane));
    }
    // Tile path: the cache holds a window at least; by default every frame in
    // flight (two more than there are streams) can have its own new frame and
    // one is spare. A limit takes the frames in flight first, then the spare
    // cache frames; below a bare window the plane path remains.
    const std::size_t window = static_cast<std::size_t>(2 * d->d + 1);
    std::size_t streams = 0, stages = 0, cached = 0, weight_sets = 0;
    // With d = 1 a frame's forward weight maps are the backward maps of the
    // frame after it: the streams share one set of (2a + 1)^2 maps more than
    // there are streams (WeightCache; a frame loads the set the frame before
    // it saved and saves into another), which saves work for frames rendered
    // in order (1080p gray, two streams: the kernel 1.33 to 1.07 ms). They are
    // the first thing a limit takes, and without a limit they may use a
    // quarter of the device.
    const std::size_t span = static_cast<std::size_t>(2 * d->a + 1);
    const auto plan_tiles = [&](std::size_t n) {
        streams = n;
        stages = streams + kTileExtraStages;
        cached = window + stages;
        weight_sets = d->d == 1 ? streams + 1 : 0;
        if (weight_sets * span * span * d->plane_bytes > d->device.total_memory / kWeightShare) weight_sets = 0;
    };
    auto tile_bytes = [&] {
        return d->plane_bytes * (cached * 2 * regions_per_frame(*d) + streams * d->nc +
                                 stages * d->nc + weight_sets * span * span) +
               stages * output_bytes;
    };
    d->tiles = nlm_tile_supported(d->d, d->a, d->s);
    if (d->tiles) {
        // Without num_streams: two (1080p YUV420 d = 1: 614 fps with one
        // stream, 758 with two or three), fewer where their weight maps or
        // the whole plan would not fit.
        plan_tiles(d->backend.streams_explicit ? static_cast<std::size_t>(d->backend.num_streams) : 2);
        while (!d->backend.streams_explicit && streams > 1 &&
               ((d->d == 1 && weight_sets == 0) || tile_bytes() > limit))
            plan_tiles(streams - 1);
        d->backend.num_streams = static_cast<int>(streams);
        if (tile_bytes() > limit) weight_sets = 0;
        while (tile_bytes() > limit && stages > streams) cached = window + --stages;
        while (tile_bytes() > limit && cached > window) --cached;
        d->tiles = tile_bytes() <= limit;
    }
    if (!d->tiles) {
        // Per slot: device planes, pinned staging and the leased output frame.
        const std::size_t need = slot_bytes(*d) + output_bytes;
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
            if (d->gpu) slots.back()->gpu = std::make_unique<GpuStream>(d->gpu, slots.back()->stream);
            for (int c = 0; c < d->nc; ++c) slots.back()->out[c] = DeviceBuffer(d->plane_bytes, d->budget);
        }
        d->tile_pool = std::make_unique<SlotPool<TileSlot>>(std::move(slots));
        // The maps only save work: a device without room for them (other
        // filters hold it) runs without, as before they existed.
        try {
            for (std::size_t w = 0; w < weight_sets; ++w) {
                d->weights.sets.push_back(std::make_unique<WeightSet>());
                d->weights.sets.back()->maps = DeviceBuffer(d->plane_bytes * span * span, d->budget);
            }
        } catch (const std::exception&) {
            d->weights.sets.clear();
        }
        std::vector<std::unique_ptr<TileStage>> staged;
        for (std::size_t i = 0; i < stages; ++i) {
            staged.push_back(std::make_unique<TileStage>());
            staged.back()->bytes = PinnedBuffer(d->plane_bytes * d->nc, d->budget);
        }
        d->tile_stages = std::make_unique<SlotPool<TileStage>>(std::move(staged));
    } else {
        if (d->gpu) d->gpu->reserve(slot_device_bytes(*d) * static_cast<std::size_t>(d->backend.num_streams));
        std::vector<std::unique_ptr<Slot>> slots;
        for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
        d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    }
    if (d->gpu && d->budget) d->gpu->reserve(d->budget->snapshot().bytes[static_cast<std::size_t>(nss::ResourceKind::Workspace)]);

    const auto pattern = d->d == 0 ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->rclip, pattern}};
    NlmData* raw = d.get();
    VSNode* node = vs_api::create_video_filter(vsapi, "NLM", &raw->vi, nss::checked_frame<getFrame>, freeFilter,
                                               fmParallel, raw->gpu != nullptr, deps, d->rclip ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error("nss_cuda.NLM: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi, bool gpu_frames) {
    const std::string args = residency_args(signature(nss::frontend::kNlmSignature), gpu_frames);
    vspapi->registerFunction("NLM", args.c_str(), clip_return(gpu_frames), nss::checked_create<create, kDefaultMemoryLimitMb>,
                             nullptr, plugin);
}

}  // namespace nss_cuda
