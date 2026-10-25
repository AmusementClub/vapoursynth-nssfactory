// SPDX-License-Identifier: GPL-2.0-only
// nss_vk.NLM (docs/vulkan-plan.md V4). Arguments and errors come from
// parse_nlm with the "nss_vk" prefix; the whole frame (distances, box sums,
// weights, accumulation, normalization) runs on the device, as nss_cuda.NLM.
//
// Two paths, as in nss_cuda. The tile path runs a frame as one dispatch
// (shaders/nlm/tile.comp): the window frames stay on the device in a cache
// shared by the slots, so that a frame is uploaded once however many windows
// it is part of; a slot that reads a frame another slot uploads waits for that
// upload on the device. The plane path (a window, a patch or a memory limit
// the tile path cannot take; shaders/nlm/plane.comp) keeps only the centre
// frame and the current backward/forward pair resident.
//
// On vapoursynth-classic (VSFrameMemory.h) the copy queues read the planes of
// imported frames and write the output into the frame itself (FrameMemory);
// other planes are written into BAR memory or staging by the host, and read
// back through the readback buffer, as before.
#include "frontend/nlm_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/frame_memory.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_vk {
namespace {

using nss::host_detail::temporal_last;
using nss::host_detail::temporal_slot_frame;

// The shaders' DISTANCE values (nss::ChannelMode Y / UV / YUV / RGB).
enum class Distance : std::uint32_t { Luma, Chroma, Yuv, Rgb };

// Tile path limits, as nss_cuda: the window frames a dispatch takes, the
// tile, the shared memory every D22 device has, the radii with unrolled sums.
constexpr int kTileFrames = 17;
constexpr int kTileWidth = 32, kTileHeight = 32, kTileThreads = kTileWidth * kTileHeight / 2;
constexpr std::size_t kTileSharedBytes = 48 * 1024;
constexpr int kTileFixedS = 8;
// Frames in flight beyond num_streams on the tile path (nss_cuda's stages).
// nss_cuda's two extra stages are host staging around its streams; here every
// frame in flight is a Slot of its own (command buffers, timeline, readback):
// the tile path has num_streams + 2 slots, the plane path num_streams.
constexpr std::size_t kTileExtraSlots = 2;

constexpr int map_pitch(int floats) { return (floats + 3) & ~3; }
// Shared floats of a tile: distance map, row sums, weights (nss_cuda tile_floats).
constexpr long long tile_floats(int a, int s) {
    const long long rw = kTileWidth + a, rh = kTileHeight + a, hp = map_pitch(kTileWidth + a),
                    dp = hp + map_pitch(2 * s);
    return dp * (rh + 2LL * s) + hp * (rh + 2LL * s) + rw * rh;
}
// And of the windows of three frames (the centre and one pair).
constexpr long long window_floats(int a, int s, int channels) {
    return 3LL * channels * (kTileWidth + 4LL * a + 2 * s) * (kTileHeight + 4LL * a + 2 * s);
}

struct TilePush {
    VkDeviceAddress frames, out;
    std::int32_t width, height, d, a, s;
    float h2_inv_norm, wref;
    std::uint32_t frame_floats, ref_offset;
    std::uint32_t slots[5];
    VkDeviceAddress weight_save, weight_load;
};
static_assert(sizeof(TilePush) == 88, "matches the push block of shaders/nlm/tile.comp");

struct HsumPush {
    VkDeviceAddress frames, hsum;
    std::uint32_t center, neighbor;
    std::int32_t ox, oy, s, width, height;
};
struct AccumulatePush {
    VkDeviceAddress frames, hsum_bwd, hsum_fwd, sums;
    std::uint32_t src_bwd, src_fwd;
    std::int32_t ox, oy, s, width, height;
    float h2_inv_norm;
};
struct ResetPush {
    VkDeviceAddress sums;
    std::uint32_t count;
};
struct FinishPush {
    VkDeviceAddress frames, sums, out;
    std::uint32_t src, count;
    float wref;
};

// Tile path: a frame of the clip in the cache (region `region` of the store).
// Guarded by WindowCache::mu.
struct WindowFrame {
    int region = 0;
    int n = -1;           // frame number, -1: none
    int pins = 0;         // requests reading it; it is not replaced while pinned
    bool queued = false;  // written, and its upload (if any) submitted
    bool failed = false;  // the request that was to upload it gave up
    std::uint64_t used = 0;
    // The upload stage that copies it to the device (null with BAR memory).
    VkSemaphore ready = VK_NULL_HANDLE;
    std::uint64_t ready_value = 0;
    // The stages that last used the region, one per slot timeline (the upload
    // that wrote it, the compute stages that read it): an upload stage that
    // copies another frame into the region waits for them on the device. The
    // host has waited for them already (a region is reused only once no
    // request pins it), but only a device-side dependency orders the copies
    // of different slots into one region (synchronization validation:
    // write-after-write on the store, with staging or imported frames).
    std::vector<std::pair<VkSemaphore, std::uint64_t>> uses;
};

struct WindowCache {
    std::mutex mu;
    std::condition_variable changed;
    std::vector<std::unique_ptr<WindowFrame>> frames;
    std::uint64_t clock = 0;
    PlaneStore store;  // the frames' planes
};

struct TileSlot {
    Slot slot;
    DownloadPlane out;  // nc planes stacked
    TileSlot(const std::shared_ptr<Device>& device, std::size_t row_bytes, int rows,
             const std::shared_ptr<nss::ResourceBudget>& budget)
        : slot(device), out(device, row_bytes, rows, budget) {}
    void settle() noexcept { slot.settle(); }
};

// d = 1 with room for them: weight maps a frame saved for the frame after it
// (tile.comp weight_save), shared by the slots so that the frame after it
// finds them whichever slot it gets (nss_cuda's WeightCache). Guarded by
// WeightCache::mu.
struct WeightSet {
    Buffer maps;  // (2a + 1)^2 planes
    // The compute stages of the last frames that saved and loaded them.
    std::pair<VkSemaphore, std::uint64_t> written{VK_NULL_HANDLE, 0}, read{VK_NULL_HANDLE, 0};
    int frame = -1;       // the frame that saved them, -1: none
    bool busy = false;    // a request is recording a dispatch that saves or loads them
    bool loaded = false;  // a dispatch loaded them since they were saved
    std::uint64_t used = 0;
};

struct WeightCache {
    std::mutex mu;
    std::vector<std::unique_ptr<WeightSet>> sets;
    std::uint64_t clock = 0;
};

// The sets a frame's dispatch loads and saves, taken under the cache's lock:
// its compute stage waits for the frame that saved what it loads, and for the
// frames that last saved or loaded the set it overwrites. Nothing waits on
// the host, so requests in any order cannot block one another; a frame whose
// predecessor has not submitted yet computes in full.
class WeightLease {
public:
    WeightLease(WeightCache& cache, int n, Slot& slot) : cache_(cache) {
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
        const auto wait = [&](const std::pair<VkSemaphore, std::uint64_t>& stage) {
            if (stage.first) slot.wait_for(Slot::Compute, stage.first, stage.second);
        };
        if (load_) {
            load_->busy = true;
            wait(load_->written);
        }
        if (save_) {
            save_->busy = true;
            save_->frame = -1;
            wait(save_->written);
            wait(save_->read);
        }
    }
    WeightLease(const WeightLease&) = delete;
    WeightLease& operator=(const WeightLease&) = delete;
    // Without submitted(), as after an error: the sets are given back holding
    // nothing (a frame that was not submitted wrote nothing).
    ~WeightLease() {
        if (!load_ && !save_) return;
        std::lock_guard lock(cache_.mu);
        if (load_) load_->busy = false;
        if (save_) save_->busy = false;
    }
    VkDeviceAddress load() const noexcept { return load_ ? load_->maps.address() : 0; }
    VkDeviceAddress save() const noexcept { return save_ ? save_->maps.address() : 0; }
    // After the slot submitted frame n: the frame after it may load the saved
    // set from now on.
    void submitted(int n, const Slot& slot) {
        if (!load_ && !save_) return;
        const auto compute = slot.done(Slot::Compute);
        std::lock_guard lock(cache_.mu);
        if (load_) {
            load_->read = compute;
            load_->loaded = true;
            load_->busy = false;
        }
        if (save_) {
            save_->written = compute;
            save_->frame = n;
            save_->loaded = false;
            save_->used = ++cache_.clock;
            save_->busy = false;
        }
        load_ = save_ = nullptr;
    }

private:
    WeightCache& cache_;
    WeightSet* load_ = nullptr;
    WeightSet* save_ = nullptr;
};

// Plane path: the slot's centre, backward and forward frames (regions 0-2),
// the row sums of the two pairs, the sums and the output.
struct PlaneSlot {
    Slot slot;
    PlaneStore frames;
    Buffer hsum_bwd, hsum_fwd, sums;
    DownloadPlane out;
    void settle() noexcept { slot.settle(); }
    PlaneSlot(const std::shared_ptr<Device>& device, std::size_t frame_bytes, int region_count,
              std::size_t plane_bytes, int nc, int rows, const std::shared_ptr<nss::ResourceBudget>& budget)
        : slot(device),
          frames(device, frame_bytes, region_count, budget),
          hsum_bwd(device, plane_bytes, Memory::Device, budget),
          hsum_fwd(device, plane_bytes, Memory::Device, budget),
          sums(device, plane_bytes * static_cast<std::size_t>(2 + nc), Memory::Device, budget),
          out(device, plane_bytes / static_cast<std::size_t>(rows), rows * nc, budget) {}
};

struct NlmData : nss::NlmParams {
    nss::NodeRef node, rclip;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    std::shared_ptr<Device> device;
    int width = 0, height = 0, nc = 1, first_plane = 0;
    Distance distance = Distance::Luma;
    std::size_t plane_bytes = 0;
    bool tiles = false;
    // Tile path.
    Pipeline tile;
    std::unique_ptr<SlotPool<TileSlot>> tile_pool;
    WindowCache window;
    WeightCache weights;
    // Plane path.
    Pipeline hsum, accumulate, reset, finish;
    std::unique_ptr<SlotPool<PlaneSlot>> pool;
    // Last: destroyed first, so that its imports go before the device can.
    std::unique_ptr<FrameMemory> frame_memory;
};

// Planes per frame: the averaged ones, then (with rclip) the reference ones.
int planes_per_frame(const NlmData& d) { return d.nc * (d.rclip ? 2 : 1); }
std::uint32_t frame_floats(const NlmData& d) {
    return static_cast<std::uint32_t>(d.plane_bytes / sizeof(float)) * static_cast<std::uint32_t>(planes_per_frame(d));
}
std::uint32_t ref_offset(const NlmData& d) {
    return d.rclip ? static_cast<std::uint32_t>(d.plane_bytes / sizeof(float)) * static_cast<std::uint32_t>(d.nc) : 0u;
}

float h2_inv_norm(const NlmData& d) {
    return (255.0f * 255.0f) /
           (3.0f * d.h * d.h * (static_cast<float>(2 * d.s + 1) * static_cast<float>(2 * d.s + 1)));
}

using Frames = std::vector<const VSFrame*>;

// A plane the upload stage copies into a store region: from an imported frame
// (`source`), else from the store's staging mirror.
struct PlaneUpload {
    int region = 0;
    std::size_t offset = 0;
    HostPlane source;
};

// Frame t of the window (averaged planes, then reference planes) into region
// `region` of `store`: the host writes the planes that are not imported, and
// `uploads` gets the copies the upload stage makes. The frames are held until
// the slot's wait.
void write_frame(const NlmData& d, PlaneStore& store, int region, const Frames& srcf, const Frames& reff, int t,
                 const VSAPI* vsapi, std::vector<PlaneUpload>& uploads) {
    const std::size_t row_bytes = static_cast<std::size_t>(d.width) * sizeof(float);
    const auto write = [&](const VSFrame* frame, int plane, int index) {
        const std::size_t offset = d.plane_bytes * static_cast<std::size_t>(index);
        const std::uint8_t* data = vsapi->getReadPtr(frame, plane);
        const std::ptrdiff_t stride = vsapi->getStride(frame, plane);
        if (const HostPlane source = d.frame_memory->pinned(frame, plane, data, stride, row_bytes)) {
            uploads.push_back({region, offset, source});
            return;
        }
        store.write(region, offset, data, stride, row_bytes, d.height, row_bytes);
        if (store.needs_copy()) uploads.push_back({region, offset, {}});
    };
    for (int c = 0; c < d.nc; ++c) {
        write(srcf[t], d.first_plane + c, c);
        if (d.rclip) write(reff[t], d.first_plane + c, d.nc + c);
    }
}

void record_uploads(const NlmData& d, const PlaneStore& store, const std::vector<PlaneUpload>& uploads,
                    VkCommandBuffer commands) {
    for (const PlaneUpload& u : uploads) {
        if (u.source) {
            store.record_upload(commands, u.region, u.offset, d.plane_bytes, u.source.buffer, u.source.offset);
        } else {
            store.record_upload(commands, u.region, u.offset, d.plane_bytes);
        }
    }
}

// The output planes the download stage copies into the frame itself (empty:
// through the readback buffer). Takes the write pointers.
using DirectOut = std::array<HostPlane, 3>;
DirectOut direct_out(const NlmData& d, VSFrame* dst, const VSAPI* vsapi) {
    const std::size_t row_bytes = static_cast<std::size_t>(d.width) * sizeof(float);
    DirectOut direct{};
    for (int c = 0; c < d.nc; ++c) {
        const int plane = d.first_plane + c;
        direct[c] = d.frame_memory->pinned(dst, plane, vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane),
                                           row_bytes);
    }
    return direct;
}

void record_result(const NlmData& d, const DownloadPlane& out, const DirectOut& direct, VkCommandBuffer commands) {
    for (int c = 0; c < d.nc; ++c) out.record_download(commands, c * d.height, d.height, direct[c].buffer, direct[c].offset);
    out.record_host_barrier(commands);
}

// After the slot's wait: the planes the copies did not write.
void read_result(const NlmData& d, const DownloadPlane& out, const DirectOut& direct, VSFrame* dst, const VSAPI* vsapi) {
    const std::size_t row_bytes = static_cast<std::size_t>(d.width) * sizeof(float);
    for (int c = 0; c < d.nc; ++c) {
        if (direct[c]) continue;
        const int plane = d.first_plane + c;
        out.read(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), row_bytes, c * d.height, d.height);
    }
}

// The output frame is made once the request holds its slot (make_output), so
// that frames in flight, and their charges, stay within the slots the memory
// plan counts however many VapourSynth threads ask.
using MakeOutput = std::function<VSFrame*()>;

void plane_frame(NlmData& d, const Frames& srcf, const Frames& reff, const MakeOutput& make_output,
                 const VSAPI* vsapi) {
    auto lease = d.pool->acquire();
    VSFrame* dst = make_output();
    const DirectOut direct = direct_out(d, dst, vsapi);
    PlaneSlot& p = *lease;
    Slot& slot = p.slot;
    const int w = d.width, h = d.height;
    const std::uint32_t count = static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h);
    const std::uint32_t ff = frame_floats(d), ro = ref_offset(d);
    const std::uint32_t center = 0, bwd = ff, fwd = 2 * ff;
    const float h2 = h2_inv_norm(d);
    const int span = 2 * d.a + 1;
    const VkDeviceAddress frames = p.frames.address();
    auto upload = [&](int region, int t) {
        std::vector<PlaneUpload> uploads;
        write_frame(d, p.frames, region, srcf, reff, t, vsapi, uploads);
        if (!uploads.empty()) record_uploads(d, p.frames, uploads, slot.commands(Slot::Upload));
    };
    upload(0, d.d);
    {
        const ResetPush push{p.sums.address(), count};
        d.reset.dispatch(slot.commands(Slot::Compute), &push, (count + 255) / 256);
        compute_barrier(slot.device(), slot.commands(Slot::Compute));
    }
    for (int i = -d.d; i <= 0; ++i) {
        // The pair's regions are reused: the previous pair has finished (wait below).
        if (i < 0) {
            upload(1, d.d + i);
            upload(2, d.d - i);
        }
        const std::uint32_t bwd_frame = i < 0 ? bwd : center, fwd_frame = i < 0 ? fwd : center;
        VkCommandBuffer commands = slot.commands(Slot::Compute);
        for (int oy = -d.a; oy <= d.a; ++oy) {
            for (int ox = -d.a; ox <= d.a; ++ox) {
                if (static_cast<std::int64_t>(i) * span * span + static_cast<std::int64_t>(oy) * span + ox >= 0) continue;
                const HsumPush first{frames, p.hsum_bwd.address(), center + ro, bwd_frame + ro, ox, oy, d.s, w, h};
                d.hsum.dispatch(commands, &first, (w + 31) / 32, (h + 7) / 8);
                if (i < 0) {
                    const HsumPush second{frames, p.hsum_fwd.address(), fwd_frame + ro, center + ro, ox, oy, d.s, w, h};
                    d.hsum.dispatch(commands, &second, (w + 31) / 32, (h + 7) / 8);
                }
                compute_barrier(slot.device(), commands);
                const AccumulatePush push{frames,    p.hsum_bwd.address(), i < 0 ? p.hsum_fwd.address() : p.hsum_bwd.address(),
                                          p.sums.address(), bwd_frame,     fwd_frame,
                                          ox,        oy,                   d.s,
                                          w,         h,                    h2};
                d.accumulate.dispatch(commands, &push, (w + 31) / 32, (h + 7) / 8);
                compute_barrier(slot.device(), commands);
            }
        }
        if (i == 0) {
            const FinishPush push{frames, p.sums.address(), p.out.address(), center, count, d.wref};
            d.finish.dispatch(commands, &push, (count + 255) / 256);
            record_result(d, p.out, direct, slot.commands(Slot::Download));
        }
        slot.submit();
        slot.wait();
    }
    read_result(d, p.out, direct, dst, vsapi);
}

// The window of one request on the tile path: pins its frames in the cache
// and takes the cache's free frames for those that are not there. Unpins on
// destruction; a frame it did not get to upload is given up (as nss_cuda).
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
                f->ready = VK_NULL_HANDLE;
                f->ready_value = 0;
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
    const std::vector<WindowFrame*>& uploads() const { return uploads_; }
    bool uploads(const WindowFrame* f) const { return std::find(uploads_.begin(), uploads_.end(), f) != uploads_.end(); }
    const std::vector<WindowFrame*>& pinned() const { return pinned_; }

    // Returns once the frames other requests upload are queued; their upload
    // stages are then known for the device-side waits.
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
        if (failed) throw std::runtime_error("nss_vk.NLM: a frame of the window was not uploaded");
    }
    void mark_queued(VkSemaphore ready, std::uint64_t value) {
        {
            std::lock_guard lock(cache_.mu);
            for (WindowFrame* f : uploads_) {
                f->queued = true;
                f->ready = ready;
                f->ready_value = value;
                // The upload waited for the region's earlier uses (or the
                // host wrote it after they were done).
                f->uses.clear();
                if (ready) f->uses.emplace_back(ready, value);
            }
        }
        cache_.changed.notify_all();
    }
    // Before the uploads are recorded: the stages that last used their regions.
    std::vector<std::pair<VkSemaphore, std::uint64_t>> previous_uses() const {
        std::lock_guard lock(cache_.mu);
        std::vector<std::pair<VkSemaphore, std::uint64_t>> uses;
        for (const WindowFrame* f : uploads_) uses.insert(uses.end(), f->uses.begin(), f->uses.end());
        return uses;
    }
    // After the submission: its compute stage read every frame of the window.
    void read_by(std::pair<VkSemaphore, std::uint64_t> compute) {
        if (!compute.first) return;
        std::lock_guard lock(cache_.mu);
        for (WindowFrame* f : pinned_) {
            auto use = std::find_if(f->uses.begin(), f->uses.end(), [&](const auto& u) { return u.first == compute.first; });
            if (use == f->uses.end()) {
                f->uses.push_back(compute);
            } else {
                use->second = std::max(use->second, compute.second);
            }
        }
    }

private:
    WindowCache& cache_;
    std::vector<WindowFrame*> slots_, pinned_, uploads_;
};

void tile_frame(NlmData& d, int n, const Frames& srcf, const Frames& reff, const MakeOutput& make_output,
                const VSAPI* vsapi) {
    // The pins outlive the lease: returning it settles the slot (waits for
    // submitted work), and only then may other requests reuse the cache
    // frames this frame reads.
    Window window(d.window, n, d.d, d.vi.numFrames);
    PlaneStore& store = d.window.store;
    // The host writes need no slot. Neither does waiting for the frames other
    // requests upload: a request holding a slot there could wait forever for
    // an uploader that waits for a slot (as nss_cuda, which waits before it
    // takes a stream).
    std::vector<PlaneUpload> uploads;
    for (WindowFrame* f : window.uploads()) {
        int t = 0;
        while (&window.slot(t) != f) ++t;
        write_frame(d, store, f->region, srcf, reff, t, vsapi, uploads);
    }
    // With BAR memory the frames the host wrote are on the device: the
    // requests that share them need not wait for this one's slot and
    // submission. Copies (staging, imported frames) are ready once this
    // request's upload stage is.
    const bool staged = !uploads.empty();
    if (!staged) window.mark_queued(VK_NULL_HANDLE, 0);
    window.wait_queued();
    auto lease = d.tile_pool->acquire();
    VSFrame* dst = make_output();
    const DirectOut direct = direct_out(d, dst, vsapi);
    Slot& slot = lease->slot;
    if (staged) {
        for (const auto& [semaphore, value] : window.previous_uses()) slot.wait_for(Slot::Upload, semaphore, value);
        record_uploads(d, store, uploads, slot.commands(Slot::Upload));
    }
    // Frames other requests upload: wait for their upload stages on the device.
    for (WindowFrame* f : window.pinned()) {
        if (!window.uploads(f)) slot.wait_for(Slot::Compute, f->ready, f->ready_value);
    }

    TilePush push{};
    push.frames = store.address();
    push.out = lease->out.address();
    push.width = d.width;
    push.height = d.height;
    push.d = d.d;
    push.a = d.a;
    push.s = d.s;
    push.h2_inv_norm = h2_inv_norm(d);
    push.wref = d.wref;
    push.frame_floats = frame_floats(d);
    push.ref_offset = ref_offset(d);
    for (int t = 0; t < 2 * d.d + 1; ++t) {
        push.slots[t / 4] |= static_cast<std::uint32_t>(window.slot(t).region) << (8 * (t % 4));
    }
    WeightLease weights(d.weights, n, slot);
    push.weight_save = weights.save();
    push.weight_load = weights.load();
    slot.begin_label(Slot::Compute, "nlm.tile");
    d.tile.dispatch(slot.commands(Slot::Compute), &push, static_cast<std::uint32_t>((d.width + kTileWidth - 1) / kTileWidth),
                    static_cast<std::uint32_t>((d.height + kTileHeight - 1) / kTileHeight));
    slot.end_label(Slot::Compute);
    record_result(d, lease->out, direct, slot.commands(Slot::Download));
    slot.submit();
    weights.submitted(n, slot);
    if (staged) {
        const auto uploaded = slot.done(Slot::Upload);
        window.mark_queued(uploaded.first, uploaded.second);
    }
    window.read_by(slot.done(Slot::Compute));
    slot.wait();
    read_result(d, lease->out, direct, dst, vsapi);
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
    const MakeOutput make_output = [&]() -> VSFrame* {
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
        return dst;
    };
    if (d->tiles) {
        tile_frame(*d, n, srcf, reff, make_output, vsapi);
    } else {
        plane_frame(*d, srcf, reff, make_output, vsapi);
    }
    for (int t = 0; t < ntemp; ++t) {
        frames.freeFrame(srcf[t]);
        if (reff[t]) frames.freeFrame(reff[t]);
    }
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    delete static_cast<NlmData*>(instance);
}

std::vector<std::uint32_t> constants(const NlmData& d) {
    return {static_cast<std::uint32_t>(d.distance), static_cast<std::uint32_t>(d.weight),
            static_cast<std::uint32_t>(d.nc)};
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.NLM.
    auto d = std::make_unique<NlmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::NlmParams&>(*d) = nss::frontend::parse_nlm(vsapi, in, d->vi, "nss_vk");
    int err = 0;
    d->rclip = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        throw std::invalid_argument("nss_vk.NLM: rclip must match clip");
    }
    d->backend = parse_backend_args(vsapi, in, "NLM");
    d->device = acquire_device(d->backend.device_id, "NLM", core, vsapi);
    d->budget = nss::current_budget();
    d->frame_memory = std::make_unique<FrameMemory>(d->device, core, vsapi);

    const bool uv = d->channels == nss::ChannelMode::UV;
    d->width = uv ? d->vi.width >> d->vi.format.subSamplingW : d->vi.width;
    d->height = uv ? d->vi.height >> d->vi.format.subSamplingH : d->vi.height;
    d->nc = d->channels == nss::ChannelMode::Y ? 1 : uv ? 2 : 3;
    d->first_plane = uv ? 1 : 0;
    d->distance = d->channels == nss::ChannelMode::Y     ? Distance::Luma
                  : uv                                   ? Distance::Chroma
                  : d->channels == nss::ChannelMode::YUV ? Distance::Yuv
                                                         : Distance::Rgb;
    d->plane_bytes = static_cast<std::size_t>(d->width) * d->height * sizeof(float);
    const std::size_t row_bytes = static_cast<std::size_t>(d->width) * sizeof(float);
    const std::size_t frame_bytes = d->plane_bytes * static_cast<std::size_t>(planes_per_frame(*d));
    // The plan assumes BAR uploads on a ReBAR device. Should BAR memory run
    // out (other users), PlaneStore falls back to staging, which doubles the
    // cache: under a limit that is reported as memory_limit_mb exceeded.
    const bool staged = !d->device->rebar();

    const std::size_t limit = d->budget ? d->budget->snapshot().limit : SIZE_MAX;
    // Tile path: the cache holds a window at least; by default every frame in
    // flight can have its own new frame and one is spare. A limit takes frames
    // in flight first, then the spare cache frames; below a bare window the
    // plane path remains (as nss_cuda).
    const std::size_t streams = static_cast<std::size_t>(d->backend.num_streams);
    const std::size_t window = static_cast<std::size_t>(2 * d->d + 1);
    std::size_t slots = streams + kTileExtraSlots, cached = 0;
    // With d = 1 a frame's forward weight maps are the backward maps of the
    // frame after it: the slots share one set of (2a + 1)^2 maps more than
    // there are streams (WeightCache; 1080p gray: the kernel 1.79 to 1.13 ms,
    // every frame rendered in order reusing, with one or two streams). They
    // are the first thing a limit takes, and without a limit they may use a
    // quarter of the device (nss_cuda).
    const std::size_t span = static_cast<std::size_t>(2 * d->a + 1);
    std::size_t weight_sets = d->d == 1 ? streams + 1 : 0;
    if (weight_sets * span * span * d->plane_bytes > d->device->caps().device_local_bytes / 4) weight_sets = 0;
    auto tile_bytes = [&] {
        return frame_bytes * cached * (staged ? 2 : 1) +
               d->plane_bytes * slots * (2 * static_cast<std::size_t>(d->nc) + d->vi.format.numPlanes) +
               d->plane_bytes * weight_sets * span * span;
    };
    const long long shared_maps = tile_floats(d->a, d->s);
    const long long shared_all = shared_maps + window_floats(d->a, d->s, d->nc);
    d->tiles = 2 * d->d + 1 <= kTileFrames &&
               shared_maps * static_cast<long long>(sizeof(float)) <= static_cast<long long>(kTileSharedBytes) &&
               d->device->caps().max_workgroup_invocations >= static_cast<std::uint32_t>(kTileThreads) &&
               // Float offsets into the cache are 32-bit (the plane path's into
               // its three frames are too: fine up to planes of 2^30 / 3 floats).
               frame_bytes / sizeof(float) * (window + slots + 1) < (1ull << 32);
    if (d->tiles) {
        cached = window + slots;
        if (tile_bytes() > limit) weight_sets = 0;
        while (tile_bytes() > limit && slots > streams) cached = window + --slots;
        while (tile_bytes() > limit && cached > window) --cached;
        d->tiles = tile_bytes() <= limit;
    }
    if (!d->tiles) {
        // Per slot: three frames (with staging), row sums, sums, output (with
        // readback) and the leased output frame.
        const int regions = d->d > 0 ? 3 : 1;
        const std::size_t need = frame_bytes * regions * (staged ? 2 : 1) +
                                 d->plane_bytes * (2 + (2 + d->nc) + 2 * d->nc + d->vi.format.numPlanes);
        if (!d->backend.streams_explicit) {
            d->backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / need, 1, kDefaultStreams));
        }
        if (limit / static_cast<std::size_t>(d->backend.num_streams) < need) {
            throw std::invalid_argument("nss_vk.NLM: memory_limit_mb is too small for this clip: each of the " +
                                        std::to_string(d->backend.num_streams) + " stream(s) needs at least " +
                                        std::to_string((need >> 20) + 1) + " MiB");
        }
    }

    if (d->tiles) {
        const bool windows = shared_all * static_cast<long long>(sizeof(float)) <= static_cast<long long>(kTileSharedBytes);
        const int fixed = d->s <= kTileFixedS ? d->s : -1;
        const std::string id = "nlm_tile_s" + (fixed < 0 ? std::string("x") : std::to_string(fixed)) + "_i" +
                               (windows ? "1" : "0");
        std::vector<std::uint32_t> spec = constants(*d);
        spec.push_back(static_cast<std::uint32_t>(windows ? shared_all : shared_maps));
        d->tile = Pipeline(d->device, {id, sizeof(TilePush), 0, spec});
        d->window.store = PlaneStore(d->device, frame_bytes, static_cast<int>(cached), d->budget);
        for (std::size_t i = 0; i < cached; ++i) {
            auto frame = std::make_unique<WindowFrame>();
            frame->region = static_cast<int>(i);
            d->window.frames.push_back(std::move(frame));
        }
        std::vector<std::unique_ptr<TileSlot>> leases;
        for (std::size_t i = 0; i < slots; ++i) {
            leases.push_back(std::make_unique<TileSlot>(d->device, row_bytes, d->height * d->nc, d->budget));
        }
        d->tile_pool = std::make_unique<SlotPool<TileSlot>>(std::move(leases));
        // The maps only save work: a device without room for them runs without.
        try {
            for (std::size_t w = 0; w < weight_sets; ++w) {
                d->weights.sets.push_back(std::make_unique<WeightSet>());
                d->weights.sets.back()->maps = Buffer(d->device, d->plane_bytes * span * span, Memory::Device, d->budget);
            }
        } catch (const std::exception&) {
            d->weights.sets.clear();
        }
    } else {
        const std::vector<std::uint32_t> spec = constants(*d);
        d->hsum = Pipeline(d->device, {"nlm_plane_hsum", sizeof(HsumPush), 0, spec});
        d->accumulate = Pipeline(d->device, {"nlm_plane_accumulate", sizeof(AccumulatePush), 0, spec});
        d->reset = Pipeline(d->device, {"nlm_plane_reset", sizeof(ResetPush), 0, spec});
        d->finish = Pipeline(d->device, {"nlm_plane_finish", sizeof(FinishPush), 0, spec});
        std::vector<std::unique_ptr<PlaneSlot>> leases;
        for (int i = 0; i < d->backend.num_streams; ++i) {
            leases.push_back(std::make_unique<PlaneSlot>(d->device, frame_bytes, d->d > 0 ? 3 : 1, d->plane_bytes,
                                                         d->nc, d->height, d->budget));
        }
        d->pool = std::make_unique<SlotPool<PlaneSlot>>(std::move(leases));
    }
    d->device->save_pipeline_cache();

    const auto pattern = d->d == 0 ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->rclip, pattern}};
    NlmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("NLM", &raw->vi, nss::checked_frame<getFrame>, freeFilter, fmParallel,
                                            deps, d->rclip ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error("nss_vk.NLM: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNlmSignature);
    vspapi->registerFunction("NLM", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_vk
