// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for patch-group filters (see group_driver.hpp).
#include "cuda/common/group_driver.hpp"
#include "cuda/common/aggregate.hpp"
#include "cuda/common/chunk_cache.hpp"
#include "cuda/common/match.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/frame_memory.hpp"
#include "cuda/runtime/gpu_frames.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/pointer_table.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "cuda/runtime/vs_api.hpp"
#include "frontend/contribution.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"
#include "nss/checked.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_cuda {
namespace {

using nss::host_detail::temporal_last;
using nss::host_detail::temporal_slot_frame;

// Upper bound on the per-batch group buffers; larger planes run in batches
// and the batch also shrinks so num_streams slots fit memory_limit_mb.
constexpr std::size_t kBatchBytes = 128u << 20;
// Frames (spatial, legacy) or chunks (rolling) staged at once (see plan_planes).
constexpr std::size_t kFramesInFlight = 3;
constexpr std::size_t kChunksPerStream = 3;

struct Event {
    cudaEvent_t event = nullptr;
    Event() { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event, kStreamEvent)); }
    Event(Event&& other) noexcept : event(std::exchange(other.event, nullptr)) {}
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event) cudaEventDestroy(event);
    }
};

// What a rolling slot keeps of one plane between chunks, so that the chunk
// after the one it ran last only does the new work: the window frames stay
// on the device and the sums that the last centers left for later frames are
// carried. A frame is finished as soon as its last center has run, so the
// sums are a ring of 2R+1 slices whatever the chunk: frame f is slice
// f % (2R+1), and window buffer f % (the source ring's size).
struct RollState {
    std::vector<DeviceBuffer> src_frames, ref_frames;
    std::vector<const float*> src_ptrs, ref_ptrs;
    DeviceBuffer acc;                   // ordered planes: 2R+1 num slices, then as many den slices
    DeviceBuffer fixed_num, fixed_den;  // fused planes: 2R+1 slices each
    // Two stages: the first stage's sums, laid out as the second's, and its
    // estimates in ref_frames. The source ring then has 4R+1 buffers (the
    // first stage runs 2R centers ahead), otherwise 2R+1.
    DeviceBuffer basic_acc, basic_num, basic_den;
    std::vector<int> resident;          // frame in each source buffer (the chunk at turn owns it)
    // Uploads run on the slot's upload stream, apart from the kernels: an
    // upload into a buffer waits for the last kernel that read the frame it
    // replaces (read[i], recorded behind every reader), and the kernels that
    // read the buffer wait for its upload (uploaded[i]). The guide clip's
    // buffers likewise (ref_read, ref_uploaded) when it has one.
    std::vector<Event> read, uploaded, ref_read, ref_uploaded;
    // Guarded by Driver::roll_mu. Chunks take a ticket when they are planned
    // and run in ticket order, so a plan can rely on the chunks before it.
    int plane = -1;
    int next_start = -1;          // the chunk that can carry on after the planned ones
    std::vector<int> promised;    // resident after the planned ones
    std::uint64_t issued = 0, serving = 0;
    std::uint64_t chain = 0;      // counts the chunks that started afresh
    std::uint64_t failed_chain = ~0ull;
};

struct Slot {
    Stream stream;
    std::unique_ptr<GpuStream> gpu;
    // Rolling: the source frames go up on `up`, a chunk's finished frames
    // come down together on `down` (out_free behind them, which the next
    // chunk waits for before it finishes a frame into `out`); `computed`
    // marks the end of a chunk's kernels for the download.
    Stream up, down;
    std::unique_ptr<GpuStream> gpu_up;
    Event out_free, computed;
    bool in_use = false;  // rolling: guarded by Driver::roll_mu
    std::vector<RollState> roll;
    // Window frames: 2R+1, 1 when spatial (rolling keeps its own in `roll`).
    std::vector<DeviceBuffer> src_frames, ref_frames, noisy_frames;
    std::vector<const float*> host_src_ptrs, host_ref_ptrs;
    DeviceBuffer src_ptrs, ref_ptrs;  // device arrays: window slot t -> plane pointer
    DeviceBuffer num, den;            // 2R+1 slices each
    DeviceBuffer fixed_num, fixed_den;  // spatial fused planes: fixed-point cells, one slice
    DeviceBuffer fixed_rows;          // fixed_finish scratch, one slice
    DeviceBuffer out;                 // finished plane
    DeviceBuffer matches, counts, values, scratch, patches;
    DeviceBuffer match_scratch;  // the tensor-core matcher's workspace (spatial planes that qualify)
    std::unique_ptr<OrderedAggregator> aggregator;
    // Rolling with several rounds: a center's estimates of its window frames
    // (2R+1 planes and their device pointer array) and the fixed-point sums
    // of its rounds before the last (2R+1 slices each).
    std::vector<DeviceBuffer> estimates;
    DeviceBuffer estimate_ptrs, round_num, round_den;
};

// Pinned staging for one frame (spatial, legacy) or one chunk (rolling) in
// flight. It is leased apart from the slots: a thread copies its input in
// before it takes a slot and copies its output out after it has given the
// slot back, so the stream works on another frame meanwhile. regions[r] is
// one tight plane.
struct Staging {
    PinnedBuffer bytes;
    std::vector<std::uint8_t*> regions;
    cudaEvent_t done = nullptr;  // recorded behind the frame's last download
    // Rolling: an event per plane, and where each plane's uploads are staged
    // (upload i at roll_base[plane] + i * roll_step[plane]).
    std::array<cudaEvent_t, 3> plane_done{};
    std::array<std::uint8_t*, 3> roll_base{};
    std::array<std::size_t, 3> roll_step{};
    Staging() = default;
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    ~Staging() {
        if (done) cudaEventDestroy(done);
        for (cudaEvent_t event : plane_done) {
            if (event) cudaEventDestroy(event);
        }
    }
};

struct Driver : GroupFilterConfig {
    std::size_t plane_floats = 0;  // largest plane (tight pitch)
    std::size_t match_scratch = 0;  // bytes per slot of the tensor-core matcher's workspace (0: lane kernels)
    int ntemp = 1;                 // 2R+1
    int regions = 0;               // staging regions per staging set
    int staging_sets = 0;          // frames or chunks in flight, at least num_streams
    bool channel_out[3]{true, true, true};  // joint: channel c is written (else copied)
    bool basic_out[3]{true, true, true};    // two_stage: the first stage filters channel c of a unit
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
    // Rolling: a permit per staging set, taken once a request holds its chunk
    // and kept until its frame is copied out, so that the output frames stay
    // within what the plan counts per staging set. The cache never drops a
    // chunk that a request holds as a hit (such a request then waits only for
    // a permit, and a permit holder only copies its frame out). A request that
    // computes its chunk holds no permit meanwhile and keeps its staging set
    // until its frame is out: the plan counts a chunk store per set, and the
    // cache may drop that chunk in the meantime. (A permit taken before the
    // chunk let the cache drop a chunk the requests for its other frames were
    // about to read: they computed it again, rolling ran at half speed.)
    struct OutputPermit {
        Stream stream;
        std::unique_ptr<GpuStream> gpu;
    };
    std::unique_ptr<SlotPool<OutputPermit>> outputs;
    // Spatial and legacy frames: direct DMA to and from page-locked VS frame
    // planes where the host allows it (vapoursynth-classic VSFrameMemory.h).
    std::unique_ptr<FrameMemory> frame_memory;
    std::shared_ptr<GpuFrames> gpu;
    // Rolling takes its slots by turn (see RollState) instead of from the pool.
    std::vector<Slot*> slots;
    std::vector<int> roll_state;  // plane -> index in Slot::roll
    int roll_states = 0;
    // The planes of a chunk are staged and queued together and then waited
    // for (each has its state and its staging); otherwise one after the other.
    bool roll_together = false;
    // A chunk's finished frames wait on the device and are downloaded
    // together; false (under a memory limit without room for them) downloads
    // each as it is finished.
    bool roll_batch_out = true;
    std::size_t staging_bytes = 0;
    std::mutex roll_mu;
    std::condition_variable roll_cv;

    // Rolling chunk cache: it holds cache_size chunks at first and grows to
    // cache_cap (both settled by plan_planes).
    std::size_t frame_bytes = 0;
    std::array<std::size_t, 3> plane_offset{};
    int cache_size = 1, cache_cap = 1;
    ChunkCache cache;
    PlanOrder order;
};

// A guide clip comes from the host; the device also holds guide frames when
// the first of two stages makes them.
bool guide_clip(const Driver& d) { return d.guide != nullptr; }
bool has_guide(const Driver& d) { return d.guide != nullptr || d.two_stage; }
// The active planes of every stage: they share the slot's per-batch buffers.
template <class D>
auto stage_planes(D& d) {
    std::vector<decltype(&d.planes[0])> planes;
    for (auto& p : d.planes) {
        if (p.active) planes.push_back(&p);
    }
    for (auto& p : d.basic) {
        if (d.two_stage && p.active) planes.push_back(&p);
    }
    return planes;
}
// Joint or multi-round filters keep their estimate in the window frames.
bool iterative(const Driver& d) { return (d.channels > 1 && !d.shared_match) || d.iters > 1; }
// Patch record arrays per batch: one, or one per channel of a shared_match unit.
std::size_t patch_sets(const Driver& d) { return d.shared_match ? static_cast<std::size_t>(d.channels) : 1; }
std::string prefix(const Driver& d) { return "nss_cuda." + d.name + ": "; }

const VSFrame* take_gpu_frame(Driver& d, GpuFrameCache& cache, VSNode* node, int fn, nss::FrameScope& frames,
                              VSFrameContext* ctx, const VSAPI*) {
    return cache.get(*d.gpu, fn, node, ctx, frames);
}

std::size_t scratch_floats(const Driver& d, const GroupPlane& p) {
    return d.scratch_floats ? d.scratch_floats(p, p.wiener) : 0;
}

bool any_fused(const Driver& d) {
    for (const GroupPlane* p : stage_planes(d)) {
        if (p->fused) return true;
    }
    return false;
}
// Some active plane goes through values, patch records and the ordered
// aggregator into the float num/den slices.
bool any_ordered(const Driver& d) {
    for (const GroupPlane* p : stage_planes(d)) {
        if (!p->fused) return true;
    }
    return false;
}

std::size_t per_ref_bytes(const Driver& d, const GroupPlane& p) {
    if (p.fused) {
        return scratch_floats(d, p) * sizeof(float) + sizeof(int) +
               static_cast<std::size_t>(p.group) * sizeof(DeviceMatch);
    }
    const std::size_t cube = static_cast<std::size_t>(d.channels) * p.group * p.block * p.block * sizeof(float);
    return cube + scratch_floats(d, p) * sizeof(float) + sizeof(int) +
           static_cast<std::size_t>(p.group) *
               (sizeof(DeviceMatch) + patch_sets(d) * sizeof(AggregatePatch) + OrderedAggregator::kBytesPerPatch);
}

// Bytes of the slot's per-batch buffers for the planes' current batches.
// make_slot sizes each buffer for the plane that needs it most, so with
// planes of different shapes (or fused beside ordered) this is more than any
// single plane's batch * per_ref_bytes.
std::size_t slot_batch_bytes(const Driver& d) {
    std::size_t matches = 0, counts = 0, scratch = 0, values = 0, patches = 0, sort = 0;
    for (const GroupPlane* plane : stage_planes(d)) {
        const GroupPlane& p = *plane;
        const std::size_t batch = static_cast<std::size_t>(p.batch);
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, p) * sizeof(float));
        if (p.fused) continue;
        values = std::max(values, batch * d.channels * p.group * p.block * p.block * sizeof(float));
        patches = std::max(patches, batch * p.group * patch_sets(d) * sizeof(AggregatePatch));
        sort = std::max(sort, batch * p.group * OrderedAggregator::kBytesPerPatch);
    }
    return matches + counts + scratch + values + patches + sort;
}

// Frames a rolling chunk reads on each side of itself, and the buffers of a
// state's source ring. The first of two stages reaches 2R further.
int roll_reach(const Driver& d) { return (d.two_stage ? 4 : 2) * d.radius; }
// The source buffers of a rolling state: the frames the centers read at once
// (with two stages the first runs 2R centers ahead) and one more, so that a
// frame's upload runs while the center before it computes (the buffer it
// replaces is read by no later center).
int src_ring(const Driver& d) { return (d.two_stage ? 4 * d.radius + 1 : d.ntemp) + 1; }

// Pinned bytes of one rolling staging set: every active plane's uploads at
// its own size, or one plane's at the largest size for planes that take turns.
std::size_t roll_staging_bytes(const Driver& d, bool together) {
    const std::size_t uploads =
        static_cast<std::size_t>(d.rolling.rolling_chunk + 2 * roll_reach(d)) * (d.guide ? 2 : 1) * d.channels;
    if (!together) return d.plane_floats * sizeof(float) * uploads;
    std::size_t bytes = 0;
    for (const GroupPlane& p : d.planes) {
        if (p.active) bytes += p.floats * sizeof(float) * uploads;
    }
    return bytes;
}

// The sums a rolling state keeps for plane `plane`, or for every plane (-1,
// the state the planes share).
struct RollNeeds {
    bool fused = false, ordered = false, basic_fused = false, basic_ordered = false;
};
RollNeeds roll_needs(const Driver& d, int plane) {
    RollNeeds needs;
    for (int i = 0; i < 3; ++i) {
        if (plane >= 0 && i != plane) continue;
        const GroupPlane& p = d.planes[i];
        const GroupPlane& b = d.basic[i];
        if (!p.active) continue;
        (p.fused ? needs.fused : needs.ordered) = true;
        if (d.two_stage && b.active) (b.fused ? needs.basic_fused : needs.basic_ordered) = true;
    }
    return needs;
}

int active_planes(const Driver& d) {
    int n = 0;
    for (const GroupPlane& p : d.planes) n += p.active;
    return n;
}

// Device bytes of the rolling states of one slot: one per active plane, or
// one that every plane uses in turn.
std::size_t roll_bytes(const Driver& d, bool per_plane) {
    const std::size_t nt = d.ntemp;
    const auto planes = [&](const RollNeeds& n) {
        const std::size_t frames = src_ring(d) + (has_guide(d) ? nt : 0);
        // Fixed-point sums take two float planes' worth per slice of num and of den.
        const std::size_t sums = ((n.fused ? 4 : 0) + (n.ordered ? 2 : 0) + (n.basic_fused ? 4 : 0) +
                                  (n.basic_ordered ? 2 : 0)) * nt;
        return (frames + sums) * static_cast<std::size_t>(d.channels);
    };
    if (!per_plane) return active_planes(d) ? d.plane_floats * sizeof(float) * planes(roll_needs(d, -1)) : 0;
    std::size_t bytes = 0;
    for (int plane = 0; plane < 3; ++plane) {
        if (d.planes[plane].active) bytes += d.planes[plane].floats * sizeof(float) * planes(roll_needs(d, plane));
    }
    return bytes;
}

// Staging regions: the uploads (src, ref), then the downloads.
// Spatial and legacy: one upload per window frame and channel; one download
// per channel (spatial) or 2 * ntemp fat rows per channel (legacy).
// Rolling: one upload per frame a chunk that starts afresh reads (the chunk
// and 2 * radius frames on each side); the results go straight into the
// chunk store.
int upload_regions(const Driver& d) { return d.ntemp * d.channels * (guide_clip(d) ? 2 : 1); }
// Window frames a spatial or legacy slot holds (the guide's too when the first stage makes them).
int device_frames(const Driver& d) { return d.ntemp * d.channels * (has_guide(d) ? 2 : 1); }
int chunk_window(const Driver& d) { return d.rolling.rolling_chunk + 2 * roll_reach(d); }
int chunk_uploads(const Driver& d) { return chunk_window(d) * (guide_clip(d) ? 2 : 1); }
int staging_regions(const Driver& d) {
    if (d.mode == GroupMode::Rolling) return chunk_uploads(d);
    return upload_regions(d) + (d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1) * d.channels;
}

// Each plane's size and reference grid (both stages), the frame layout of
// the chunk stores, and the staging regions of a set.
void size_planes(Driver& d) {
    std::size_t frame_bytes = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        GroupPlane& p = d.planes[plane];
        p.width = nss::plane_width(d.vi, plane);
        p.height = nss::plane_height(d.vi, plane);
        p.floats = static_cast<std::size_t>(p.width) * p.height;
        d.plane_floats = std::max(d.plane_floats, p.floats);
        d.plane_offset[plane] = frame_bytes;
        frame_bytes += p.floats * sizeof(float);
        p.wiener = has_guide(d);
        GroupPlane& b = d.basic[plane];
        b.width = p.width;
        b.height = p.height;
        b.floats = p.floats;
        for (GroupPlane* stage : {&p, &b}) {
            stage->group = std::min(stage->group, kMaxGroup);
            if (stage->active) stage->grid = make_raster_grid(p.width, p.height, stage->block, stage->step);
        }
    }
    d.frame_bytes = frame_bytes;
    d.regions = staging_regions(d);
}

// What a slot holds whatever its batch, before the rolling states.
struct SlotPlan {
    std::size_t fixed_base = 0;    // window frames, slices, result, staging (spatial, legacy), output, bins
    std::size_t output = 0;        // per staging set: the output frame(s) (rolling: and a chunk store)
    std::size_t chunk_bytes = 0;   // a chunk store (rolling)
    std::size_t shared_cache = 0;  // the rolling cache at its starting size, beside the slots
    std::size_t min_batch = 0;     // a minimum batch of groups (up to 1024 references)
    std::size_t batch_out = 0;     // a rolling chunk's frames waiting for their download (one is in fixed_base)
};

SlotPlan slot_plan(const Driver& d) {
    SlotPlan sp;
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const bool rolling = d.mode == GroupMode::Rolling;
    const int chunk = rolling ? d.rolling.rolling_chunk : 0;
    // Window frames (rolling keeps them in its states), float slices, the
    // result, and for fused planes the finish scratch and, when spatial, the
    // fixed-point num and den (two float planes' worth each). Several rounds
    // keep 2R+1 planes beside the window: the noisy frames (spatial, legacy)
    // or a rolling center's estimates, and then also the fixed-point sums of
    // its rounds before the last.
    const std::size_t device_planes = (rolling ? 0 : static_cast<std::size_t>(device_frames(d))) +
                                      static_cast<std::size_t>(d.channels) *
                                          ((any_ordered(d) ? 2 * d.ntemp : 0) + 1 + (d.iters > 1 ? d.ntemp : 0)) +
                                      (any_fused(d) ? 2 + 4 * static_cast<std::size_t>(d.channels) *
                                                              (rolling ? (d.iters > 1 ? d.ntemp : 0) : 1)
                                                    : 0);
    int max_w = 1, max_h = 1;
    for (const GroupPlane* p : stage_planes(d)) {
        if (!p->fused) {
            max_w = std::max(max_w, p->width);
            max_h = std::max(max_h, p->height);
        }
    }
    const std::size_t bins = any_ordered(d) ? OrderedAggregator::bin_bytes(max_w, max_h, d.ntemp, true) : 0;
    // Rolling also holds host chunk stores: one being built per slot, up to
    // cache_limit finished ones, and the output frames copied out of them.
    sp.chunk_bytes = d.frame_bytes * chunk;
    sp.output = rolling ? 2 * d.frame_bytes + sp.chunk_bytes
                        : d.frame_bytes * (d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1);
    sp.fixed_base = plane_bytes * device_planes + (rolling ? 0 : plane_bytes * d.regions) + sp.output + bins;
    // The cache starts at cache_chunks; what it may grow to is settled in size_staging.
    sp.shared_cache = rolling ? sp.chunk_bytes * d.rolling.cache_chunks : 0;
    for (const GroupPlane* p : stage_planes(d)) {
        sp.min_batch = std::max(sp.min_batch, std::min<std::size_t>(p->grid.count(), 1024) * per_ref_bytes(d, *p));
    }
    sp.batch_out = rolling ? plane_bytes * d.channels * (chunk - 1) : 0;
    return sp;
}

// Rolling: a state per plane lets every plane carry on from chunk to chunk
// and the planes of a chunk go together; under a limit that does not hold
// them the planes take turns on one state (and a plane then starts each
// chunk afresh, unless it is the only one). Likewise a chunk's finished
// frames wait on the device for one download only where they fit.
void choose_roll_layout(Driver& d, const SlotPlan& sp, std::size_t limit) {
    const bool rolling = d.mode == GroupMode::Rolling;
    bool per_plane = rolling && active_planes(d) > 1;
    if (rolling && limit != SIZE_MAX) {
        const std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 1;
        const auto fits = [&](bool planes, std::size_t extra) {
            return limit / streams >=
                   sp.fixed_base + extra + roll_bytes(d, planes) + roll_staging_bytes(d, planes) + sp.min_batch * 4 / 3;
        };
        per_plane = per_plane && fits(true, sp.batch_out);
        d.roll_batch_out = fits(per_plane, sp.batch_out);
    }
    d.roll_states = 0;
    d.roll_state.assign(3, 0);
    if (rolling) {
        for (int plane = 0; plane < 3; ++plane) {
            if (d.planes[plane].active && per_plane) d.roll_state[plane] = d.roll_states++;
        }
        if (!per_plane) d.roll_states = 1;
    }
    d.roll_together = per_plane;
    d.staging_bytes = rolling ? roll_staging_bytes(d, per_plane) : d.plane_floats * sizeof(float) * d.regions;
}

// With pinned staging the host copies bound most filters and one stream
// keeps up; direct DMA takes them out, and further streams then fill the
// gaps a frame's transfers leave between its kernels (1080p: BM3D YUV420
// 500 to 570 fps with two or three streams, MCWNNM RGB 184 to 237 with two
// and 269 with three; without direct DMA 0.99x for both).
std::size_t implicit_streams(const Driver& d) {
    if (d.mode == GroupMode::Rolling) return kDefaultStreams;
    const bool direct = d.frame_memory && d.frame_memory->available();
    return static_cast<std::size_t>(direct ? d.direct_streams : d.default_streams);
}

// The streams, unless given: the most that still run a whole plane as one
// batch (extra batches cost a matching/aggregation launch each), otherwise
// the most that fit at all. Each must have room for a minimum batch.
void choose_streams(Driver& d, const SlotPlan& sp, std::size_t fixed, std::size_t limit) {
    const std::size_t need = fixed + sp.min_batch * 4 / 3;
    if (!d.backend.streams_explicit) {
        std::size_t full_batch = 0;
        for (const GroupPlane* p : stage_planes(d)) {
            full_batch = std::max(full_batch, std::min<std::size_t>(kBatchBytes, p->grid.count() * per_ref_bytes(d, *p)));
        }
        const std::size_t whole = limit / (fixed + full_batch * 4 / 3 + 4096);
        d.backend.num_streams = static_cast<int>(
            std::clamp<std::size_t>(whole >= 1 ? whole : limit / need, 1, implicit_streams(d)));
    }
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < need) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB" +
                                    (sp.shared_cache ? ", plus " + std::to_string((sp.shared_cache >> 20) + 1) +
                                                           " MiB for the rolling cache"
                                                     : ""));
    }
}

// Each plane's batch of references: what a stream's share leaves beside the
// fixed buffers (at most kBatchBytes), shrunk until planes of different
// shapes fit the slot's shared buffers together.
void fit_batches(Driver& d, std::size_t fixed, std::size_t limit) {
    const std::size_t share =
        limit == SIZE_MAX ? kBatchBytes + fixed : limit / static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t batch_bytes = std::min(kBatchBytes, (share - fixed) / 4 * 3);
    for (GroupPlane* p : stage_planes(d)) {
        const std::size_t fit = std::max<std::size_t>(1, batch_bytes / per_ref_bytes(d, *p));
        p->batch = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(p->grid.count())));
    }
    for (int round = 0; round < 8 && slot_batch_bytes(d) > batch_bytes; ++round) {
        const double scale = static_cast<double>(batch_bytes) / static_cast<double>(slot_batch_bytes(d));
        bool changed = false;
        for (GroupPlane* p : stage_planes(d)) {
            const int next = std::max(1, static_cast<int>(p->batch * scale));
            changed = changed || next != p->batch;
            p->batch = next;
        }
        if (!changed) break;
    }
}

// A staging set and the output it feeds per frame (or rolling chunk) in
// flight. One per stream is in `fixed`; more keep a stream busy while other
// threads copy, as far as the limit allows. Three frames in flight saturate
// one stream (1080p BM3D on an RTX 5080: 454 fps with one set, 752 with two,
// 821 with three); beyond that the host copies only compete for memory
// bandwidth. A rolling chunk stages all its frames before its device work
// starts, so each stream needs chunks ahead of it (radius 1, one stream: 173
// fps with one set, 302 with two, 308 with four; three streams: 303, 304
// with four, 344 with nine). Under a limit the rolling cache grows only into
// what the plan leaves over.
void size_staging(Driver& d, const SlotPlan& sp, std::size_t fixed, std::size_t total, std::size_t limit) {
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t set_bytes = d.staging_bytes + sp.output;
    std::size_t extra = d.mode == GroupMode::Rolling ? (kChunksPerStream - 1) * streams
                                                     : std::max(streams, kFramesInFlight) - streams;
    if (total != SIZE_MAX) {
        const std::size_t used = streams * (fixed + slot_batch_bytes(d) / 3 * 4);
        extra = std::min(extra, limit > used ? (limit - used) / set_bytes : 0);
    }
    d.staging_sets = static_cast<int>(streams + extra);
    d.cache_size = d.rolling.cache_chunks;
    d.cache_cap = d.rolling.cache_limit;
    if (d.mode == GroupMode::Rolling && total != SIZE_MAX) {
        const std::size_t used = streams * (fixed + slot_batch_bytes(d) / 3 * 4) + extra * set_bytes;
        const std::size_t spare = limit > used ? (limit - used) / sp.chunk_bytes : 0;
        d.cache_cap = static_cast<int>(
            std::min<std::size_t>(d.rolling.cache_limit, d.rolling.cache_chunks + spare));
    }
}

// Workspace bytes of the tensor-core matcher for the spatial planes it takes
// (spatial_match_tensor), 0 when none does.
std::size_t tensor_scratch_bytes(const Driver& d) {
    if (d.radius > 0) return 0;
    int width = 0, height = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        MatchGeometry g{p->width, p->height, p->width, p->block, p->range, p->group};
        g.channels = d.shared_match ? 1 : d.channels;
        if (d.column_match && spatial_match_column_takes(g, p->grid)) continue;
        if (d.square_match && spatial_match_square_takes(g, p->grid) && g.lo() < kSquareTensorRadius) continue;
        if (spatial_match_tensor(g, p->grid)) width = std::max(width, p->width), height = std::max(height, p->height);
    }
    return width ? spatial_match_scratch_bytes(width, height) : 0;
}

// The memory plan: per slot the window frames, slices, result/accumulators,
// staging (sized for the largest plane), the leased frame's output VSFrame
// and at least a minimum batch of groups; then the streams, the batches, the
// staging sets and the rolling cache within memory_limit_mb.
void plan_planes(Driver& d) {
    size_planes(d);
    const SlotPlan sp = slot_plan(d);
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (total != SIZE_MAX && total <= sp.shared_cache) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - sp.shared_cache;
    choose_roll_layout(d, sp, limit);
    const bool rolling = d.mode == GroupMode::Rolling;
    std::size_t fixed = sp.fixed_base + (rolling ? roll_bytes(d, d.roll_together) + d.staging_bytes : 0) +
                        (d.roll_batch_out ? sp.batch_out : 0);
    // Spatial matching on tensor cores where a plane qualifies: one workspace per slot, for the largest such
    // plane, left out (the lane kernels give the same matches) when the limit has no room for it.
    d.match_scratch = tensor_scratch_bytes(d);
    if (d.match_scratch && limit != SIZE_MAX) {
        const std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 1;
        if (limit / streams < fixed + d.match_scratch + sp.min_batch * 4 / 3) d.match_scratch = 0;
    }
    fixed += d.match_scratch;
    choose_streams(d, sp, fixed, limit);
    fit_batches(d, fixed, limit);
    size_staging(d, sp, fixed, total, limit);
}

std::unique_ptr<Staging> make_staging(const Driver& d) {
    auto staging = std::make_unique<Staging>();
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    staging->bytes = PinnedBuffer(d.staging_bytes, d.budget);
    NSS_CUDA_CHECK(cudaEventCreateWithFlags(&staging->done, kHostWaitEvent));
    if (d.mode == GroupMode::Rolling) {
        std::uint8_t* at = staging->bytes.as<std::uint8_t>();
        for (int plane = 0; plane < 3; ++plane) {
            const GroupPlane& p = d.planes[plane];
            if (!p.active) continue;
            NSS_CUDA_CHECK(cudaEventCreateWithFlags(&staging->plane_done[plane], kHostWaitEvent));
            staging->roll_base[plane] = at;
            staging->roll_step[plane] = (d.roll_together ? p.floats : d.plane_floats) * sizeof(float);
            if (d.roll_together) at += staging->roll_step[plane] * chunk_uploads(d) * d.channels;
        }
        return staging;
    }
    for (int i = 0; i < d.regions; ++i) staging->regions.push_back(staging->bytes.as<std::uint8_t>() + plane_bytes * i);
    return staging;
}

std::unique_ptr<Slot> make_slot(const Driver& d) {
    auto slot = std::make_unique<Slot>();
    if (d.gpu) {
        slot->gpu = std::make_unique<GpuStream>(d.gpu, slot->stream);
        if (d.mode == GroupMode::Rolling) slot->gpu_up = std::make_unique<GpuStream>(d.gpu, slot->up);
    }
    const bool w = has_guide(d);
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t unit_bytes = plane_bytes * d.channels;
    std::size_t values = 0, matches = 0, counts = 0, patches = 0, max_patches = 0, scratch = 0;
    int max_w = 1, max_h = 1;
    for (const GroupPlane* plane : stage_planes(d)) {
        const GroupPlane& p = *plane;
        const std::size_t batch = static_cast<std::size_t>(p.batch);
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, p) * sizeof(float));
        if (p.fused) continue;
        max_patches = std::max(max_patches, batch * p.group);
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
        values = std::max(values, batch * d.channels * p.group * p.block * p.block * sizeof(float));
        patches = std::max(patches, batch * p.group * patch_sets(d) * sizeof(AggregatePatch));
    }
    const bool rolling = d.mode == GroupMode::Rolling;
    for (int t = 0; t < d.ntemp && !rolling; ++t) {
        slot->src_frames.emplace_back(unit_bytes, d.budget);
        slot->host_src_ptrs.push_back(slot->src_frames.back().as<float>());
        if (d.iters > 1) slot->noisy_frames.emplace_back(unit_bytes, d.budget);
        if (w) {
            slot->ref_frames.emplace_back(unit_bytes, d.budget);
            slot->host_ref_ptrs.push_back(slot->ref_frames.back().as<float>());
        }
    }
    if (rolling) {
        slot->roll.resize(d.roll_states);
        const bool shared = d.roll_states == 1;
        for (int plane = 0; plane < 3; ++plane) {
            const GroupPlane& p = d.planes[plane];
            if (!p.active) continue;
            RollState& r = slot->roll[d.roll_state[plane]];
            if (!r.src_frames.empty()) continue;  // the shared state, sized below for every plane
            // A unit of several channels keeps them side by side in every buffer.
            const std::size_t bytes = (shared ? d.plane_floats : p.floats) * sizeof(float) * d.channels;
            const RollNeeds needs = roll_needs(d, shared ? -1 : plane);
            for (int t = 0; t < src_ring(d); ++t) {
                r.src_frames.emplace_back(bytes, d.budget);
                r.src_ptrs.push_back(r.src_frames.back().as<float>());
                r.read.emplace_back();
                r.uploaded.emplace_back();
            }
            for (int t = 0; t < d.ntemp && w; ++t) {
                r.ref_frames.emplace_back(bytes, d.budget);
                r.ref_ptrs.push_back(r.ref_frames.back().as<float>());
                if (guide_clip(d)) {
                    r.ref_read.emplace_back();
                    r.ref_uploaded.emplace_back();
                }
            }
            if (needs.ordered) r.acc = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
            if (needs.fused) {
                r.fixed_num = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
                r.fixed_den = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
            }
            if (needs.basic_ordered) r.basic_acc = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
            if (needs.basic_fused) {
                r.basic_num = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
                r.basic_den = DeviceBuffer(bytes * 2 * d.ntemp, d.budget);
            }
            r.resident.assign(src_ring(d), -1);
            r.promised.assign(src_ring(d), -1);
        }
    }
    slot->src_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    if (w) slot->ref_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    std::vector<const float*> estimate_ptrs;
    if (rolling && d.iters > 1) {
        for (int t = 0; t < d.ntemp; ++t) {
            slot->estimates.emplace_back(unit_bytes, d.budget);
            estimate_ptrs.push_back(slot->estimates.back().as<float>());
        }
        const std::size_t round_cells = d.plane_floats * d.ntemp * d.channels;
        slot->estimate_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
        slot->round_num = DeviceBuffer(round_cells * sizeof(unsigned long long), d.budget);
        slot->round_den = DeviceBuffer(round_cells * sizeof(unsigned long long), d.budget);
    }
    if (any_ordered(d)) {
        slot->num = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
        slot->den = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
    }
    // The finished plane; rolling keeps a chunk's worth and downloads them together.
    slot->out = DeviceBuffer(unit_bytes * (rolling && d.roll_batch_out ? d.rolling.rolling_chunk : 1), d.budget);
    if (any_fused(d)) {
        if (!rolling) {
            slot->fixed_num = DeviceBuffer(d.plane_floats * d.channels * sizeof(unsigned long long), d.budget);
            slot->fixed_den = DeviceBuffer(d.plane_floats * d.channels * sizeof(unsigned long long), d.budget);
        }
        slot->fixed_rows = DeviceBuffer(d.plane_floats * sizeof(unsigned long long), d.budget);
    }
    slot->matches = DeviceBuffer(matches, d.budget);
    if (d.match_scratch) slot->match_scratch = DeviceBuffer(d.match_scratch, d.budget);
    slot->counts = DeviceBuffer(counts, d.budget);
    if (values) slot->values = DeviceBuffer(values, d.budget);
    if (scratch) slot->scratch = DeviceBuffer(scratch, d.budget);
    if (patches) slot->patches = DeviceBuffer(patches, d.budget);
    if (max_patches) {
        slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, d.ntemp, max_patches, d.budget, true);
    }
    // Spatial/legacy windows map slot t to buffer t for the slot's lifetime,
    // and so do a rolling center's estimates.
    if (!estimate_ptrs.empty()) {
        NSS_CUDA_CHECK(cudaMemcpy(slot->estimate_ptrs.get(), estimate_ptrs.data(), d.ntemp * sizeof(float*),
                                  cudaMemcpyHostToDevice));
    }
    if (rolling) {
        if (!estimate_ptrs.empty()) NSS_CUDA_CHECK(cudaDeviceSynchronize());  // see below
        return slot;
    }
    NSS_CUDA_CHECK(cudaMemcpy(slot->src_ptrs.get(), slot->host_src_ptrs.data(), d.ntemp * sizeof(float*),
                              cudaMemcpyHostToDevice));
    if (w) {
        NSS_CUDA_CHECK(cudaMemcpy(slot->ref_ptrs.get(), slot->host_ref_ptrs.data(), d.ntemp * sizeof(float*),
                                  cudaMemcpyHostToDevice));
    }
    // A small copy from pageable memory may still be in flight when
    // cudaMemcpy returns, and the slot's non-blocking stream does not wait
    // for it: without this, the first kernels can read the arrays unset.
    NSS_CUDA_CHECK(cudaDeviceSynchronize());
    return slot;
}

// All groups of one center into the slot's num/den slices (ntemp of them per
// channel, channel-major). The device pointer arrays map window slot t to the
// plane of frame temporal_slot_frame(center, t). Matching runs on the guide
// when match_guide is set, otherwise on the source/estimate frames. A fused
// plane adds to fixed-point slices instead: the slot's one spatial slice, or
// `rolling`, the chunk's (which the caller prepares). `frames`, a device
// pointer array like s.src_ptrs, replaces the window's source frames (a
// rolling center's estimates), for matching too unless match_guide.
void filter_center(const Driver& d, Slot& s, const GroupPlane& p, int center, bool match_guide,
                   const FixedTarget* rolling = nullptr, const float* const* frames = nullptr) {
    cudaStream_t stream = s.stream;
    const int radius = d.radius;
    // The first of two stages filters the channels its estimate is needed for.
    const bool first_stage = d.two_stage && &p >= d.basic && &p < d.basic + 3;
    const bool* active = first_stage ? d.basic_out : d.channel_out;
    MatchGeometry geometry{p.width, p.height, p.width, p.block, p.range, p.group};
    geometry.channels = d.shared_match ? 1 : d.channels;
    geometry.channel_step = static_cast<long long>(p.floats);
    TemporalWindow window{};
    if (radius > 0) {
        window.frames = match_guide ? s.ref_ptrs.as<const float*>() : frames ? frames : s.src_ptrs.as<const float*>();
        window.ntemp = d.ntemp;
        window.t0 = radius;
        window.radius = radius;
        window.valid_begin = std::max(0, radius - center);
        window.valid_end = radius + std::min(radius + 1, d.vi.numFrames - center);
        window.ps_num = p.ps_num;
        window.ps_range = p.ps_range;
    }
    FixedTarget fixed{s.fixed_num.as<unsigned long long>(), s.fixed_den.as<unsigned long long>(), p.width, p.floats};
    fixed.channel_step = p.floats;
    if (rolling) fixed = *rolling;
    if (p.fused && !rolling) fixed_clear(fixed, p.floats * d.channels, stream);
    const float* match_plane = radius > 0 ? nullptr : match_guide ? s.host_ref_ptrs[0] : s.host_src_ptrs[0];
    // The column or square matcher where it applies, except for the large
    // windows the tensor-core matcher takes (when this device and plan have it).
    const bool tensor_ok = radius == 0 && s.match_scratch.get() && spatial_match_tensor(geometry, p.grid);
    const bool column = radius == 0 && d.column_match && spatial_match_column_takes(geometry, p.grid);
    const bool square = radius == 0 && d.square_match && spatial_match_square_takes(geometry, p.grid) &&
                        !(tensor_ok && geometry.lo() >= kSquareTensorRadius);
    const bool tensor = !column && !square && tensor_ok;
    if (tensor) spatial_match_prepare(match_plane, geometry, s.match_scratch.get(), stream);
    for (int begin = 0; begin < p.grid.count(); begin += p.batch) {
        const int count = std::min(p.batch, p.grid.count() - begin);
        {
            NSS_CUDA_RANGE("group.match");
            if (radius > 0) {
                // The centre frame by the square matcher where it takes it.
                if (!(d.square_match &&
                      predictive_match_square(geometry, window, p.grid, begin, count, s.matches.as<DeviceMatch>(),
                                              s.counts.as<int>(), stream))) {
                    predictive_match(geometry, window, p.grid, begin, count, s.matches.as<DeviceMatch>(),
                                     s.counts.as<int>(), stream);
                }
            } else if (column) {
                if (!spatial_match_column(match_plane, geometry, p.grid, begin, count, s.matches.as<DeviceMatch>(),
                                          s.counts.as<int>(), stream)) {
                    spatial_match(match_plane, geometry, p.grid, begin, count, s.matches.as<DeviceMatch>(),
                                  s.counts.as<int>(), stream);
                }
            } else if (!square || !spatial_match_square(match_plane, geometry, p.grid, begin, count,
                                                        s.matches.as<DeviceMatch>(), s.counts.as<int>(), stream)) {
                spatial_match(match_plane, geometry, p.grid, begin, count, s.matches.as<DeviceMatch>(),
                              s.counts.as<int>(), stream, tensor ? s.match_scratch.get() : nullptr);
            }
        }
        GroupLaunch launch{};
        launch.src = frames ? frames : s.src_ptrs.as<const float*>();
        launch.guide = p.wiener ? s.ref_ptrs.as<const float*>() : nullptr;
        launch.pitch = p.width;
        launch.matches = s.matches.as<DeviceMatch>();
        launch.counts = s.counts.as<int>();
        launch.batch = count;
        launch.plane = &p;
        launch.channels = d.channels;
        launch.channel_step = geometry.channel_step;
        launch.values = p.fused ? nullptr : s.values.as<float>();
        launch.scratch = s.scratch.as<float>();
        launch.patches = p.fused ? nullptr : s.patches.as<AggregatePatch>();
        launch.channel_active = active;
        if (p.fused) launch.fused = fixed;
        launch.stream = stream;
        {
            NSS_CUDA_RANGE("group.filter");
            d.launch(launch);
        }
        if (p.fused) continue;
        NSS_CUDA_RANGE("group.aggregate");
        const std::size_t channel_values = static_cast<std::size_t>(count) * p.group * p.block * p.block;
        for (int c = 0; c < d.channels; ++c) {
            const AggregateTarget target{s.num.as<float>() + c * d.ntemp * p.floats,
                                         s.den.as<float>() + c * d.ntemp * p.floats,
                                         p.width, p.height, p.width, d.ntemp, p.floats};
            if (d.shared_match && !active[c]) continue;
            const AggregatePatch* patches =
                s.patches.as<AggregatePatch>() + (d.shared_match ? static_cast<std::size_t>(c) * count * p.group : 0);
            s.aggregator->run(s.values.as<float>() + c * channel_values, patches, count * p.group, p.block, target, stream,
                              begin > 0);
        }
    }
}

// A plane copied unfiltered into dst: as is, or as identity slices (legacy).
void copy_identity(const Driver& d, const VSAPI* vsapi, VSFrame* dst, const VSFrame* src, int plane) {
    const GroupPlane& p = d.planes[plane];
    auto* op = vsapi->getWritePtr(dst, plane);
    const std::ptrdiff_t ds = vsapi->getStride(dst, plane);
    const auto* sp = vsapi->getReadPtr(src, plane);
    const std::ptrdiff_t ss = vsapi->getStride(src, plane);
    if (d.radius > 0) {
        nss::host_detail::temporal_identity(reinterpret_cast<float*>(op), static_cast<int>(ds / sizeof(float)),
                                            reinterpret_cast<const float*>(sp), static_cast<int>(ss / sizeof(float)),
                                            p.width, p.height, d.radius);
    } else {
        vsh::bitblt(op, ds, sp, ss, static_cast<std::size_t>(p.width) * sizeof(float), p.height);
    }
}

// Two stages: the first stage's estimate of every channel into the guide
// frame, where the second stage matches and takes its Wiener reference. A
// channel the first stage leaves alone is the source.
void run_first_stage(const Driver& d, Slot& s, int unit, int n) {
    const GroupPlane& p = d.planes[unit];
    const GroupPlane& b = d.basic[unit];
    if (b.active) filter_center(d, s, b, n, false);
    for (int c = 0; c < d.channels; ++c) {
        const std::size_t channel = static_cast<std::size_t>(c) * p.floats;
        const float* source = s.host_src_ptrs[0] + channel;
        float* estimate = s.ref_frames[0].as<float>() + channel;
        if (!b.active || !d.basic_out[c]) {
            NSS_CUDA_CHECK(cudaMemcpyAsync(estimate, source, p.floats * sizeof(float), cudaMemcpyDeviceToDevice,
                                           s.stream));
        } else if (b.fused) {
            const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + channel,
                                    s.fixed_den.as<unsigned long long>() + channel, p.width, p.floats};
            fixed_finish(fixed, b.block, source, p.width, p.height, s.fixed_rows.as<unsigned long long>(), estimate,
                         s.stream);
        } else {
            aggregate_finish(s.num.as<float>() + channel, s.den.as<float>() + channel, source, p.width, p.height,
                             p.width, estimate, s.stream);
        }
    }
}

// The unit's rounds over the window frames. Between rounds the estimate of
// every window frame becomes its finished slice (relaxed toward the noisy
// frame at the start of the next round); the last round of the legacy
// output leaves its slices for the download.
void run_rounds(const Driver& d, Slot& s, const GroupPlane& p, int n) {
    const std::size_t unit_floats = p.floats * d.channels;
    for (int iter = 0; iter < d.iters; ++iter) {
        for (int t = 0; t < d.ntemp && iter > 0; ++t) {
            iter_regularize(s.src_frames[t].as<float>(), s.noisy_frames[t].as<float>(), unit_floats, d.delta, s.stream);
        }
        filter_center(d, s, p, n, (guide_clip(d) || d.two_stage) && iter == 0);
        if (!iterative(d) || (d.radius > 0 && iter + 1 == d.iters)) break;
        for (int t = 0; t < d.ntemp; ++t) {
            for (int c = 0; c < d.channels; ++c) {
                float* est = s.src_frames[t].as<float>() + c * p.floats;
                const std::size_t slice = (static_cast<std::size_t>(c) * d.ntemp + t) * p.floats;
                if (p.fused) {  // spatial: one slice per channel
                    const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + slice,
                                            s.fixed_den.as<unsigned long long>() + slice, p.width, p.floats};
                    fixed_finish(fixed, p.block, est, p.width, p.height, s.fixed_rows.as<unsigned long long>(), est,
                                 s.stream);
                    continue;
                }
                aggregate_finish(s.num.as<float>() + slice, s.den.as<float>() + slice, est, p.width, p.height, p.width,
                                 est, s.stream);
            }
        }
    }
}

// Queues the download of each written channel of the unit: download(c, k,
// device rows) for the finished plane (spatial, k = 0) or the num and den
// slices (legacy: slice sl -> num rows at 2*sl*h, den rows at (2*sl+1)*h).
template <class Download>
void queue_downloads(const Driver& d, Slot& s, const GroupPlane& p, Download download) {
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    for (int c = 0; c < d.channels; ++c) {
        if (!d.channel_out[c]) continue;
        if (d.radius == 0) {
            const float* result = s.src_frames[0].as<float>() + c * p.floats;
            // s.out is reused for every channel: its download is
            // stream-ordered before the next finish overwrites it.
            const std::size_t channel = static_cast<std::size_t>(c) * p.floats;
            // An iterative filter's last finish left the result in the
            // source frame.
            if (p.fused && !iterative(d)) {
                const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + channel,
                                        s.fixed_den.as<unsigned long long>() + channel, p.width, p.floats};
                fixed_finish(fixed, p.block, s.host_src_ptrs[0] + channel, p.width, p.height,
                             s.fixed_rows.as<unsigned long long>(), s.out.as<float>(), s.stream);
                result = s.out.as<float>();
            } else if (!iterative(d)) {
                aggregate_finish(s.num.as<float>() + channel, s.den.as<float>() + channel, s.host_src_ptrs[0] + channel,
                                 p.width, p.height, p.width, s.out.as<float>(), s.stream);
                result = s.out.as<float>();
            }
            download(c, 0, result);
            continue;
        }
        for (int sl = 0; sl < d.ntemp; ++sl) {
            const std::size_t slice = (static_cast<std::size_t>(c) * d.ntemp + sl) * p.floats;
            download(c, 2 * sl, s.num.as<float>() + slice);
            download(c, 2 * sl + 1, s.den.as<float>() + slice);
        }
    }
}

// Spatial and legacy: one output frame per call.
const VSFrame* frame_output(Driver* d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* vsapi) {
    // Lease the staging set before allocating the output so at most
    // staging_sets output frames are charged to memory_limit_mb at any time.
    auto staging = [&] {
        NSS_CUDA_RANGE("group.staging");
        return d->staging->acquire();
    }();
    Staging& g = *staging;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const int radius = d->radius;
    const bool w = guide_clip(*d);
    std::vector<const VSFrame*> srcf(d->ntemp), reff(d->ntemp, nullptr);
    for (int t = 0; t < d->ntemp; ++t) {
        const int fn = temporal_slot_frame(n, t, radius, d->vi.numFrames);
        srcf[t] = frames.getFrameFilter(fn, d->node, ctx);
        if (w) reff[t] = frames.getFrameFilter(fn, d->guide, ctx);
    }
    const VSFrame* src0 = srcf[radius];
    VSFrame* dst = frames.newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src0, core);
    nss::stamp_contribution(dst, radius, n, d->model, vsapi);

    const int downloads = upload_regions(*d);
    const int channels = d->channels;
    const auto identity = [&](int plane) { copy_identity(*d, vsapi, dst, src0, plane); };
    // A unit is one plane, or all three planes of a joint filter.
    for (int unit = 0; unit < (channels > 1 ? 1 : d->vi.format.numPlanes); ++unit) {
        const GroupPlane& p = d->planes[unit];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (!p.active) {
            for (int c = 0; c < channels; ++c) identity(unit + c);
            continue;
        }
        // An input plane page-locked for direct DMA skips its staging copy
        // (direct_in[region]); likewise an output plane (direct_out[c]).
        std::vector<const VSFrame*> direct_in(static_cast<std::size_t>(downloads), nullptr);
        std::array<bool, 3> direct_out{};
        const auto stage = [&](int region, const VSFrame* frame, int plane) {
            if (d->frame_memory->pinned(frame, plane)) {
                direct_in[region] = frame;
                return;
            }
            stage_plane(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, p.height,
                        g.regions[region]);
        };
        int region = 0;
        {
            NSS_CUDA_RANGE("group.stage_in");
            for (int t = 0; t < d->ntemp; ++t) {
                for (int c = 0; c < channels; ++c) {
                    stage(region++, srcf[t], unit + c);
                    if (w) stage(region++, reff[t], unit + c);
                }
            }
        }
        // Download region of row k of channel c.
        const int rows = radius == 0 ? 1 : 2 * d->ntemp;
        const auto slot_region = [&](int c, int k) { return downloads + c * rows + k; };
        for (int c = 0; c < channels; ++c) {
            direct_out[c] = d->channel_out[c] && d->frame_memory->pinned(dst, unit + c);
        }
        // The slot is held only while this unit's work is queued: everything
        // the next frame queues on the stream runs after it, and the staging
        // regions belong to this frame until its event has passed.
        {
            NSS_CUDA_RANGE("group.queue");
            auto slot = d->pool->acquire();
            Slot& s = *slot;
            // On an error the queued copies must not outlive the leases of
            // the staging they read and write.
            struct Drain {
                cudaStream_t stream;
                bool armed = true;
                ~Drain() {
                    if (armed) cudaStreamSynchronize(stream);
                }
            } drain{s.stream};
            region = 0;
            const auto upload = [&](const VSFrame* frame, int plane, float* device) {
                if (direct_in[region]) {
                    upload_direct(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, p.height,
                                  device, row_bytes, s.stream);
                } else {
                    upload_staged(g.regions[region], row_bytes, p.height, device, row_bytes, s.stream);
                }
                ++region;
            };
            for (int t = 0; t < d->ntemp; ++t) {
                for (int c = 0; c < channels; ++c) {
                    upload(srcf[t], unit + c, s.src_frames[t].as<float>() + c * p.floats);
                    if (w) upload(reff[t], unit + c, s.ref_frames[t].as<float>() + c * p.floats);
                }
            }
            const std::size_t unit_floats = p.floats * channels;
            for (int t = 0; t < d->ntemp && d->iters > 1; ++t) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.noisy_frames[t].get(), s.src_frames[t].get(), unit_floats * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s.stream));
            }
            if (d->two_stage) run_first_stage(*d, s, unit, n);
            run_rounds(*d, s, p, n);
            queue_downloads(*d, s, p, [&](int c, int k, const float* device) {
                if (direct_out[c]) {
                    auto* op = static_cast<std::uint8_t*>(static_cast<void*>(vsapi->getWritePtr(dst, unit + c)));
                    const std::ptrdiff_t ds = vsapi->getStride(dst, unit + c);
                    download_direct(device, row_bytes, row_bytes, p.height, op + static_cast<std::ptrdiff_t>(k) * p.height * ds,
                                    ds, s.stream);
                } else {
                    begin_download(device, row_bytes, row_bytes, p.height, g.regions[slot_region(c, k)], s.stream);
                }
            });
            NSS_CUDA_CHECK(cudaEventRecord(g.done, s.stream));
            drain.armed = false;
        }
        // The staging regions hold the result once the event has passed.
        {
            NSS_CUDA_RANGE("group.wait");
            NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
        }
        NSS_CUDA_RANGE("group.stage_out");
        for (int c = 0; c < channels; ++c) {
            if (!d->channel_out[c]) {
                identity(unit + c);
                continue;
            }
            if (direct_out[c]) continue;  // the DMA wrote it
            auto* op = static_cast<std::uint8_t*>(static_cast<void*>(vsapi->getWritePtr(dst, unit + c)));
            const std::ptrdiff_t ds = vsapi->getStride(dst, unit + c);
            for (int k = 0; k < rows; ++k) {
                finish_download(g.regions[slot_region(c, k)], row_bytes, p.height,
                                op + static_cast<std::ptrdiff_t>(k) * p.height * ds, ds);
            }
        }
    }
    for (int t = 0; t < d->ntemp; ++t) {
        frames.freeFrame(srcf[t]);
        if (reff[t]) frames.freeFrame(reff[t]);
    }
    return frames.keep(dst);
}

const VSFrame* frame_output_gpu(Driver* d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* vsapi) {
    auto staging = d->staging->acquire();
    Staging& g = *staging;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const int radius = d->radius;
    const bool w = guide_clip(*d);
    std::vector<const VSFrame*> srcf(d->ntemp), reff(d->ntemp, nullptr);
    const auto take = [&](VSNode* node, std::vector<const VSFrame*>& out, int t, int fn) {
        for (int u = 0; u < t; ++u) {
            if (out[u] && temporal_slot_frame(n, u, radius, d->vi.numFrames) == fn) {
                return frames.adoptInput(vsapi->addFrameRef(out[u]));
            }
        }
        return d->gpu->get_frame(fn, node, ctx, frames);
    };
    for (int t = 0; t < d->ntemp; ++t) {
        const int fn = temporal_slot_frame(n, t, radius, d->vi.numFrames);
        srcf[t] = take(d->node, srcf, t, fn);
        if (w) reff[t] = take(d->guide, reff, t, fn);
    }
    const VSFrame* src0 = srcf[radius];
    VSFrame* dst = d->gpu->new_frame(frames, &d->vi_out.format, d->vi_out.width, d->vi_out.height, src0);
    nss::stamp_contribution(dst, radius, n, d->model, vsapi);

    const int channels = d->channels;
    for (int unit = 0; unit < (channels > 1 ? 1 : d->vi.format.numPlanes); ++unit) {
        const GroupPlane& p = d->planes[unit];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        auto slot = d->pool->acquire();
        Slot& s = *slot;
        s.gpu->begin();
        struct Drain {
            cudaStream_t stream;
            bool armed = true;
            ~Drain() {
                if (armed) cudaStreamSynchronize(stream);
            }
        } drain{s.stream};
        const auto identity = [&](int plane) {
            if (!radius) {
                s.gpu->copy(src0, plane, dst, plane, row_bytes, p.height);
                return;
            }
            for (int sl = 0; sl < d->ntemp; ++sl) {
                if (sl == radius) {
                    s.gpu->copy_rows(src0, plane, 0, dst, plane, 2 * sl * p.height, row_bytes, p.height);
                    s.gpu->fill(dst, plane, (2 * sl + 1) * p.height, p.height, 0x3f800000);
                } else {
                    s.gpu->fill(dst, plane, 2 * sl * p.height, p.height, 0);
                    s.gpu->fill(dst, plane, (2 * sl + 1) * p.height, p.height, 0);
                }
            }
        };
        if (p.active) {
            for (int t = 0; t < d->ntemp; ++t) {
                for (int c = 0; c < channels; ++c) {
                    s.gpu->read(srcf[t], unit + c, s.src_frames[t].as<float>() + c * p.floats, row_bytes, row_bytes,
                                p.height);
                    if (w) {
                        s.gpu->read(reff[t], unit + c, s.ref_frames[t].as<float>() + c * p.floats, row_bytes, row_bytes,
                                    p.height);
                    }
                }
            }
            const std::size_t unit_floats = p.floats * channels;
            for (int t = 0; t < d->ntemp && d->iters > 1; ++t) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.noisy_frames[t].get(), s.src_frames[t].get(), unit_floats * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s.stream));
            }
            if (d->two_stage) run_first_stage(*d, s, unit, n);
            run_rounds(*d, s, p, n);
            queue_downloads(*d, s, p, [&](int c, int k, const float* device) {
                s.gpu->write_rows(device, row_bytes, dst, unit + c, row_bytes, p.height, k * p.height);
            });
        }
        for (int c = 0; c < channels; ++c) {
            const int plane = unit + c;
            if (!p.active || !d->channel_out[c]) identity(plane);
            s.gpu->publish(dst, &plane, 1);
        }
        NSS_CUDA_CHECK(cudaEventRecord(g.done, s.stream));
        NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
        drain.armed = false;
    }
    for (int t = 0; t < d->ntemp; ++t) {
        frames.freeFrame(srcf[t]);
        if (reff[t]) frames.freeFrame(reff[t]);
    }
    return frames.keep(dst);
}

// The work of a chunk, in the order the stream runs it: its centers, and
// every chunk frame as soon as its last center has run (the source ring
// still holds it then, and its slice of the sums is free for a later frame).
// With two stages the first is brought far enough before each center f of
// the second: its estimates up to frame f + R, each of which needs its
// centers up to R further (and those their source frames another R on).
// v.center(center, next): a center of the (second) stage; frames before
// `next` are finished or not this chunk's, so what it adds to them is dropped.
// v.output(frame). v.basic_center(center, next) and v.basic_frame(frame):
// the first stage's centers and finished estimates.
// A chunk that carries on resumes where the one before it stopped, which
// follows from its start alone.
template <class Visitor>
void walk_chunk(const Driver& d, int start, int count, bool carry, Visitor&& v) {
    const int radius = d.radius, last = d.vi.numFrames - 1;
    const int last_center = std::min(start + count - 1 + radius, last);
    const int first_center = carry ? std::min(start + radius, last_center + 1) : std::max(0, start - radius);
    int frame = carry ? std::min(start - 1 + 2 * radius, last) + 1 : std::max(0, first_center - radius);
    int center = carry ? std::min(start - 1 + 3 * radius, last) + 1 : std::max(0, frame - radius);
    int out = start;
    for (int f = first_center; f <= last_center; ++f) {
        for (; d.two_stage && frame <= std::min(f + radius, last); ++frame) {
            for (; center <= std::min(frame + radius, last); ++center) v.basic_center(center, frame);
            v.basic_frame(frame);
        }
        v.center(f, out);
        for (; out < start + count && std::min(out + radius, last) <= f; ++out) v.output(out);
    }
    for (; out < start + count; ++out) v.output(out);
}

// One plane of one chunk on a rolling slot: planned (and given its turn) at
// construction, run between enter() and leave(). A chunk that follows the
// last one planned for a state carries on from it; any other starts afresh.
// If a chunk fails, those that were planned to carry on from it fail too.
constexpr std::uint64_t kCarryBacklog = 3;  // chunks waiting on a slot before another slot is preferred

struct RollTurn {
    Driver& d;
    Slot* slot = nullptr;
    RollState* state = nullptr;
    bool carry = false;
    int first_center = 0, last_center = -1;
    std::vector<int> uploads;  // frames to stage, in upload order
    std::uint64_t ticket = 0, chain = 0;
    bool entered = false, left = false;

    // The caller holds d.roll_mu: the planes of a chunk take their tickets in one step.
    RollTurn(Driver& driver, int plane, int start, int count, const std::lock_guard<std::mutex>&) : d(driver) {
        const int radius = d.radius, nframes = d.vi.numFrames;
        const int index = d.roll_state[plane];
        const auto backlog = [&](const Slot* s) {
            std::uint64_t n = 0;
            for (const RollState& r : s->roll) n += r.issued - r.serving;
            return n;
        };
        Slot* idle = d.slots.front();
        for (Slot* s : d.slots) {
            const RollState& r = s->roll[index];
            if (r.plane == plane && r.next_start == start && r.failed_chain != r.chain && !slot) slot = s;
            if (backlog(s) < backlog(idle)) idle = s;
        }
        // Carrying on saves more than a second stream adds, unless the slot is far behind.
        carry = slot && (backlog(slot) < kCarryBacklog || backlog(idle) >= backlog(slot));
        if (!carry) slot = idle;
        state = &slot->roll[index];
        // Everything that can throw comes before the state changes: a ticket
        // that nobody holds would stop the chunks behind it.
        const int ring = src_ring(d);
        std::vector<int> promised = carry ? state->promised : std::vector<int>(ring, -1);
        last_center = temporal_last(start + count - 1, radius, nframes);
        first_center = carry ? std::min(start + radius, last_center + 1) : std::max(0, start - radius);
        uploads.reserve(static_cast<std::size_t>(chunk_window(d)));
        // The source frames come with the centers that read them first: the
        // first stage's with two stages.
        const auto window = [&](int center) {
            for (int t = 0; t < d.ntemp; ++t) {
                const int fn = temporal_slot_frame(center, t, radius, nframes);
                if (promised[fn % ring] != fn) {
                    uploads.push_back(fn);
                    promised[fn % ring] = fn;
                }
            }
        };
        struct Plan {
            decltype(window)& stage;
            bool two_stage;
            void basic_center(int center, int) { stage(center); }
            void basic_frame(int) {}
            void center(int center, int) {
                if (!two_stage) stage(center);
            }
            void output(int) {}
        };
        walk_chunk(d, start, count, carry, Plan{window, d.two_stage});
        state->promised.swap(promised);
        if (!carry) ++state->chain;
        ticket = state->issued++;
        chain = state->chain;
        state->plane = plane;
        state->next_start = start + count;
    }
    RollTurn(const RollTurn&) = delete;
    RollTurn& operator=(const RollTurn&) = delete;

    void enter() {
        std::unique_lock lock(d.roll_mu);
        d.roll_cv.wait(lock, [&] { return !slot->in_use && state->serving == ticket; });
        slot->in_use = true;
        entered = true;
        if (state->failed_chain == chain) throw std::runtime_error(prefix(d) + "an earlier chunk of this run failed");
    }
    void leave() { finish(false); }
    ~RollTurn() {
        if (left) return;
        if (!entered) {
            std::unique_lock lock(d.roll_mu);
            d.roll_cv.wait(lock, [&] { return state->serving == ticket; });
        }
        finish(true);
    }

private:
    void finish(bool failed) {
        {
            std::lock_guard lock(d.roll_mu);
            if (failed) {
                state->failed_chain = chain;
                if (state->chain == chain) state->next_start = -1;
            }
            if (entered) slot->in_use = false;
            ++state->serving;
            left = true;
        }
        d.roll_cv.notify_all();
    }
};

// One plane of a rolling chunk on its slot, as walk_chunk visits it: the
// state's sums (a unit's channels one after the other, 2R+1 ring cells each,
// the first stage's too), the frames staged at base + i * step, and the
// device pointer arrays of the current center's window.
using RollDirect = std::array<const VSFrame*, 6>;

struct PlaneRun {
    Driver& d;
    Slot& s;
    RollState& r;
    const RollTurn& turn;
    const GroupPlane& p;
    const GroupPlane& b;
    ChunkStore& store;
    const std::uint8_t* base;
    std::size_t step, row_bytes, ring_floats, basic_floats;
    int plane, start, radius, nframes, nt, channels, ring, clips;
    float *acc_num, *acc_den, *basic_acc_num, *basic_acc_den;
    FixedTarget fixed, basic;
    std::size_t staged = 0;
    std::vector<const float*> sp, rp;
    // Per upload: the frames whose planes are uploaded directly (page-locked;
    // index clip * channels + c), null where staged.
    const std::vector<RollDirect>& direct;
    const VSAPI* vsapi;
    bool out_free = false;  // the downloads of the chunk before have released s.out

    // A slice is zero before its frame's first center: every one when the
    // chunk starts afresh, and again once its frame is finished.
    void clear_basic(int cell_index) {
        for (int c = 0; c < channels && d.two_stage && b.active; ++c) {
            if (!d.basic_out[c]) continue;
            const std::size_t cell = c * basic_floats + static_cast<std::size_t>(cell_index) * p.floats;
            if (b.fused) {
                fixed_clear(FixedTarget{basic.num + cell, basic.den + cell}, p.floats, s.stream);
            } else {
                NSS_CUDA_CHECK(cudaMemsetAsync(basic_acc_num + cell, 0, p.floats * sizeof(float), s.stream));
                NSS_CUDA_CHECK(cudaMemsetAsync(basic_acc_den + cell, 0, p.floats * sizeof(float), s.stream));
            }
        }
    }
    // A joint unit's kernel adds to every channel, written or not.
    void clear_sums(int cell_index) {
        for (int c = 0; c < channels; ++c) {
            if (d.shared_match && !d.channel_out[c]) continue;
            const std::size_t cell = c * ring_floats + static_cast<std::size_t>(cell_index) * p.floats;
            if (p.fused) {
                fixed_clear(FixedTarget{fixed.num + cell, fixed.den + cell}, p.floats, s.stream);
            } else {
                NSS_CUDA_CHECK(cudaMemsetAsync(acc_num + cell, 0, p.floats * sizeof(float), s.stream));
                NSS_CUDA_CHECK(cudaMemsetAsync(acc_den + cell, 0, p.floats * sizeof(float), s.stream));
            }
        }
    }

    // The device's window of `center`: source frames, and the estimates for the second stage.
    void point(int center, bool estimates) {
        for (int t = 0; t < nt; ++t) {
            const int fn = temporal_slot_frame(center, t, radius, nframes);
            sp[t] = r.src_ptrs[fn % ring];
            if (estimates) rp[t] = r.ref_ptrs[fn % nt];
            // The window's uploads (ones that ran long ago cost nothing to wait for).
            NSS_CUDA_CHECK(cudaStreamWaitEvent(s.stream, r.uploaded[fn % ring].event, 0));
            if (clips == 2) NSS_CUDA_CHECK(cudaStreamWaitEvent(s.stream, r.ref_uploaded[fn % nt].event, 0));
        }
        // Stream-ordered after the previous center's kernels; the values
        // travel as kernel arguments (runtime/pointer_table.hpp).
        const auto table = [&](const DeviceBuffer& target, const std::vector<const float*>& values) {
            std::vector<const void*> v(values.begin(), values.begin() + nt);
            write_pointer_table(target.get(), v.data(), nt, s.stream);
        };
        table(s.src_ptrs, sp);
        if (estimates) table(s.ref_ptrs, rp);
    }
    // Behind the kernels of `center`: its window's buffers were last read here.
    void mark_read(int center) {
        for (int t = 0; t < nt; ++t) {
            const int fn = temporal_slot_frame(center, t, radius, nframes);
            NSS_CUDA_CHECK(cudaEventRecord(r.read[fn % ring].event, s.stream));
            if (clips == 2) NSS_CUDA_CHECK(cudaEventRecord(r.ref_read[fn % nt].event, s.stream));
        }
    }
    // The window frames of `center` the device does not hold, in the plan's order.
    void upload(int center) {
        for (int t = 0; t < nt; ++t) {
            const int fn = temporal_slot_frame(center, t, radius, nframes);
            if (r.resident[fn % ring] == fn) continue;
            if (staged >= turn.uploads.size() || turn.uploads[staged] != fn) {
                throw std::logic_error(prefix(d) + "rolling plan and device state disagree");
            }
            const auto put = [&](int clip, int c, float* device) {
                if (const VSFrame* f = direct[staged][static_cast<std::size_t>(clip * channels + c)]) {
                    if (d.gpu) {
                        s.gpu_up->read(f, plane + c, device, row_bytes, row_bytes, p.height);
                    } else {
                        upload_direct(vsapi->getReadPtr(f, plane + c), vsapi->getStride(f, plane + c), row_bytes, p.height,
                                      device, row_bytes, s.up);
                    }
                } else {
                    upload_staged(base + ((staged * clips + clip) * channels + c) * step, row_bytes, p.height, device,
                                  row_bytes, s.up);
                }
            };
            // On the upload stream, once the frames these buffers held are read for the last time.
            NSS_CUDA_CHECK(cudaStreamWaitEvent(s.up, r.read[fn % ring].event, 0));
            if (clips == 2) NSS_CUDA_CHECK(cudaStreamWaitEvent(s.up, r.ref_read[fn % nt].event, 0));
            for (int c = 0; c < channels; ++c) {
                put(0, c, r.src_frames[fn % ring].as<float>() + c * p.floats);
                if (clips == 2) put(1, c, r.ref_frames[fn % nt].as<float>() + c * p.floats);
            }
            NSS_CUDA_CHECK(cudaEventRecord(r.uploaded[fn % ring].event, s.up));
            if (clips == 2) NSS_CUDA_CHECK(cudaEventRecord(r.ref_uploaded[fn % nt].event, s.up));
            ++staged;
            r.resident[fn % ring] = fn;
        }
    }
    void basic_center(int center, int next) {
        upload(center);
        if (!b.active) return;
        point(center, false);
        // Logical slice i is frame next + i, in ring cell (next + i) % (2R+1).
        basic.slices = std::min(nt, nframes - next);
        basic.slice_first = next % nt;
        basic.slice_base = center - radius - next;
        filter_center(d, s, b, center, false, &basic);
        for (int target = std::max(next, center - radius);
             !b.fused && target <= temporal_last(center, radius, nframes); ++target) {
            for (int c = 0; c < channels; ++c) {
                if (!d.basic_out[c]) continue;
                const std::size_t slice = (static_cast<std::size_t>(c) * nt + (target - center + radius)) * p.floats;
                const std::size_t cell = c * basic_floats + static_cast<std::size_t>(target % nt) * p.floats;
                accumulate_slice(basic_acc_num + cell, basic_acc_den + cell, s.num.as<float>() + slice,
                                 s.den.as<float>() + slice, p.floats, s.stream);
            }
        }
        mark_read(center);
    }
    void basic_frame(int frame) {
        for (int c = 0; c < channels; ++c) {
            const float* source = r.src_frames[frame % ring].as<float>() + c * p.floats;
            float* estimate = r.ref_frames[frame % nt].as<float>() + c * p.floats;
            const std::size_t cell = c * basic_floats + static_cast<std::size_t>(frame % nt) * p.floats;
            if (!b.active || !d.basic_out[c]) {
                // A channel the first stage leaves alone: its estimate is the source.
                NSS_CUDA_CHECK(cudaMemcpyAsync(estimate, source, p.floats * sizeof(float), cudaMemcpyDeviceToDevice,
                                               s.stream));
            } else if (b.fused) {
                fixed_finish(FixedTarget{basic.num + cell, basic.den + cell, p.width, p.floats}, b.block, source,
                             p.width, p.height, s.fixed_rows.as<unsigned long long>(), estimate, s.stream);
            } else {
                aggregate_finish(basic_acc_num + cell, basic_acc_den + cell, source, p.width, p.height, p.width,
                                 estimate, s.stream);
            }
        }
        clear_basic(frame % nt);  // the cell is the next frame's from here on
        NSS_CUDA_CHECK(cudaEventRecord(r.read[frame % ring].event, s.stream));
    }
    void center(int center, int next) {
        if (!d.two_stage) upload(center);
        point(center, p.wiener);
        const float* const* estimates = d.iters > 1 ? rounds(center) : nullptr;
        // Logical slice i is frame next + i, in ring cell (next + i) % (2R+1).
        fixed.slices = std::min(nt, nframes - next);
        fixed.slice_first = next % nt;
        fixed.slice_base = center - radius - next;
        filter_center(d, s, p, center, p.wiener && d.iters == 1, &fixed, estimates);
        // Ordered planes: each frame takes its centers in ascending
        // order, within a chunk and from one chunk to the next.
        for (int target = std::max(next, center - radius);
             !p.fused && target <= temporal_last(center, radius, nframes); ++target) {
            for (int c = 0; c < channels; ++c) {
                if (!d.channel_out[c]) continue;
                const std::size_t slice = (static_cast<std::size_t>(c) * nt + (target - center + radius)) * p.floats;
                const std::size_t cell = c * ring_floats + static_cast<std::size_t>(target % nt) * p.floats;
                accumulate_slice(acc_num + cell, acc_den + cell, s.num.as<float>() + slice, s.den.as<float>() + slice,
                                 p.floats, s.stream);
            }
        }
        mark_read(center);
    }
    // Every round of a center but the last, as the legacy output does it
    // for its center alone: the round's sums cover the window's frames
    // (slot t in slice t; a joint unit's channels each have nt slices),
    // and each window frame's estimate is finished from them (unchanged
    // where nothing covers it), then relaxed toward its noisy frame for
    // the next round, which matches and filters the estimates. Every
    // channel of a unit is finished: the next round matches on all of
    // them. Returns the device pointer array of the estimates.
    const float* const* rounds(int center) {
        FixedTarget sums{s.round_num.as<unsigned long long>(), s.round_den.as<unsigned long long>(), p.width,
                         p.floats};
        sums.slices = sums.slice_ring = nt;
        sums.channel_step = ring_floats;
        const std::size_t unit_floats = p.floats * channels;
        const auto noisy = [&](int t) {
            return r.src_frames[temporal_slot_frame(center, t, radius, nframes) % ring].as<float>();
        };
        const float* const* estimates = s.estimate_ptrs.as<const float*>();
        for (int iter = 0; iter + 1 < d.iters; ++iter) {
            fixed_clear(sums, ring_floats * channels, s.stream);
            for (int t = 0; t < nt && iter > 0; ++t) {
                iter_regularize(s.estimates[t].as<float>(), noisy(t), unit_floats, d.delta, s.stream);
            }
            filter_center(d, s, p, center, p.wiener && iter == 0, &sums, iter > 0 ? estimates : nullptr);
            for (int t = 0; t < nt; ++t) {
                for (int c = 0; c < channels; ++c) {
                    const std::size_t slice = c * ring_floats + static_cast<std::size_t>(t) * p.floats;
                    float* const estimate = s.estimates[t].as<float>() + c * p.floats;
                    fixed_finish(FixedTarget{sums.num + slice, sums.den + slice, p.width, p.floats}, p.block,
                                 iter > 0 ? estimate : noisy(t) + c * p.floats, p.width, p.height,
                                 s.fixed_rows.as<unsigned long long>(), estimate, s.stream);
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            iter_regularize(s.estimates[t].as<float>(), noisy(t), unit_floats, d.delta, s.stream);
        }
        return estimates;
    }
    void output(int frame) {
        if (d.roll_batch_out && !out_free) {
            // The chunk before on this slot downloads s.out on the download
            // stream while this chunk's first centers run.
            NSS_CUDA_CHECK(cudaStreamWaitEvent(s.stream, s.out_free.event, 0));
            out_free = true;
        }
        for (int c = 0; c < channels; ++c) {
            if (!d.channel_out[c]) continue;
            // The chunk's frames wait in s.out and are downloaded together at
            // the end: downloads between the centers measured 30% longer each.
            // Without room for that, s.out is one plane, downloaded before
            // the next finish overwrites it (stream order).
            float* result = s.out.as<float>() +
                            (d.roll_batch_out ? (static_cast<std::size_t>(frame - start) * channels + c) * p.floats : 0);
            const std::size_t cell = c * ring_floats + static_cast<std::size_t>(frame % nt) * p.floats;
            const float* source = r.src_frames[frame % ring].as<float>() + c * p.floats;
            if (p.fused) {
                fixed_finish(FixedTarget{fixed.num + cell, fixed.den + cell, p.width, p.floats}, p.block, source,
                             p.width, p.height, s.fixed_rows.as<unsigned long long>(), result, s.stream);
            } else {
                aggregate_finish(acc_num + cell, acc_den + cell, source, p.width, p.height, p.width, result, s.stream);
            }
            if (!d.roll_batch_out) {
                if (d.gpu) {
                    NSS_CUDA_CHECK(cudaMemcpyAsync(store.plane(frame - start, plane + c), result, row_bytes * p.height,
                                                   cudaMemcpyDeviceToDevice, s.stream));
                } else {
                    begin_download(result, row_bytes, row_bytes, p.height, store.plane(frame - start, plane + c), s.stream);
                }
            }
        }
        clear_sums(frame % nt);  // the slice is a later frame's from here on
        NSS_CUDA_CHECK(cudaEventRecord(r.read[frame % ring].event, s.stream));
    }
};


// Queues one plane of the chunk on its slot, in walk_chunk's order: the
// uploads staged at base + i * step, the centers, and each chunk frame's
// finish and download into the store. `event` is recorded behind the last
// download.
void queue_plane(Driver& d, RollTurn& turn, int plane, int start, int count, ChunkStore& store,
                 const std::uint8_t* base, std::size_t step, cudaEvent_t event, const std::vector<RollDirect>& direct,
                 const VSAPI* vsapi) {
    const GroupPlane& p = d.planes[plane];
    const GroupPlane& b = d.basic[plane];
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    const int radius = d.radius, nframes = d.vi.numFrames, nt = d.ntemp;
    const int channels = d.channels, ring = src_ring(d), clips = guide_clip(d) ? 2 : 1;
    NSS_CUDA_RANGE("group.queue");
    turn.enter();
    Slot& s = *turn.slot;
    if (s.gpu_up) s.gpu_up->begin();
    RollState& r = *turn.state;
    struct Drain {
        const Slot& s;
        bool armed = true;
        ~Drain() {
            if (!armed) return;
            for (cudaStream_t stream : {s.up.get(), s.stream.get(), s.down.get()}) cudaStreamSynchronize(stream);
        }
    } drain{s};
    // Every buffer of the state holds the unit's channels one after the
    // other: a frame's planes, and 2R+1 ring cells per channel of the sums
    // (the first stage's too).
    const std::size_t ring_floats = static_cast<std::size_t>(nt) * p.floats;
    const std::size_t basic_floats = ring_floats;
    float* acc_num = r.acc.as<float>();
    float* acc_den = p.fused ? nullptr : acc_num + ring_floats * channels;
    float* basic_acc_num = r.basic_acc.as<float>();
    float* basic_acc_den = !b.active || b.fused ? nullptr : basic_acc_num + basic_floats * channels;
    FixedTarget fixed{r.fixed_num.as<unsigned long long>(), r.fixed_den.as<unsigned long long>(), p.width, p.floats};
    fixed.slice_ring = nt;
    fixed.channel_step = ring_floats;
    FixedTarget basic{r.basic_num.as<unsigned long long>(), r.basic_den.as<unsigned long long>(), p.width, p.floats};
    basic.slice_ring = nt;
    basic.channel_step = basic_floats;
    PlaneRun run{d, s, r, turn, p, b, store, base, step, row_bytes, ring_floats, basic_floats,
                 plane, start, radius, nframes, nt, channels, ring, clips,
                 acc_num, acc_den, basic_acc_num, basic_acc_den, fixed, basic, 0,
                 std::vector<const float*>(nt), std::vector<const float*>(nt), direct, vsapi};
    if (!turn.carry) {
        std::fill(r.resident.begin(), r.resident.end(), -1);
        for (int cell = 0; cell < nt; ++cell) {
            run.clear_basic(cell);
            run.clear_sums(cell);
        }
    }
    walk_chunk(d, start, count, turn.carry, run);
    if (d.roll_batch_out) {
        // The finished frames come down while the next chunk's centers run.
        NSS_CUDA_CHECK(cudaEventRecord(s.computed.event, s.stream));
        NSS_CUDA_CHECK(cudaStreamWaitEvent(s.down, s.computed.event, 0));
        for (int i = 0; i < count; ++i) {
            for (int c = 0; c < channels; ++c) {
                if (!d.channel_out[c]) continue;
                const float* result = s.out.as<float>() + (static_cast<std::size_t>(i) * channels + c) * p.floats;
                if (d.gpu) {
                    NSS_CUDA_CHECK(cudaMemcpyAsync(store.plane(i, plane + c), result, row_bytes * p.height,
                                                   cudaMemcpyDeviceToDevice, s.down));
                } else {
                    begin_download(result, row_bytes, row_bytes, p.height, store.plane(i, plane + c), s.down);
                }
            }
        }
        NSS_CUDA_CHECK(cudaEventRecord(s.out_free.event, s.down));
        NSS_CUDA_CHECK(cudaEventRecord(event, s.down));
    } else {
        NSS_CUDA_CHECK(cudaEventRecord(event, s.stream));
    }
    drain.armed = false;
    turn.leave();
}

// Rolling: compute the chunk [start, start + count). The frames it reads are
// staged before a slot is taken and the results are copied out after it has
// been given back, as for a spatial frame.
// `staging` takes the staging set, which the caller keeps until its frame is
// copied out (see Driver::outputs).
std::shared_ptr<ChunkStore> compute_chunk(Driver* d, std::optional<SlotPool<Staging>::Lease>& staging, int start,
                                          int count, VSFrameContext* ctx, const VSAPI* vsapi,
                                          nss::FrameScope& frames, GpuFrameCache& cache) {
    d->order.wait_for_previous(start, d->rolling.rolling_chunk, d->cache);
    {
        NSS_CUDA_RANGE("group.staging");
        staging.emplace(d->staging->acquire());
    }
    Staging& g = **staging;
    DeviceGuard guard(d->device.index);
    const bool w = guide_clip(*d);
    auto store = d->cache.make_store(start, count);
    const auto get = [&](VSNode* node, int fn) {
        return d->gpu ? take_gpu_frame(*d, cache, node, fn, frames, ctx, vsapi) : frames.getFrameFilter(fn, node, ctx);
    };

    const int clips = w ? 2 : 1;
    // A unit of several channels is planes[0]; its channels that are not
    // written are copied like the planes of an inactive unit.
    const bool unit = d->channels > 1;
    std::vector<int> active;
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        if (p.active && !(unit && plane > 0)) active.push_back(plane);
        if (unit ? d->planes[0].active && d->channel_out[plane] : p.active) continue;
        if (d->gpu) continue;
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        for (int i = 0; i < count; ++i) {
            const VSFrame* f = get(d->node, start + i);
            vsh::bitblt(store->plane(i, plane), row_bytes, vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane),
                        row_bytes, p.height);
            frames.freeFrame(f);
        }
    }
    // Planes with staging and state of their own go together: all staged, all
    // queued, then waited for, so the device has the next plane while one
    // runs. Otherwise each plane is staged, queued and waited for in turn.
    const std::size_t step = d->roll_together ? active.size() : 1;
    for (std::size_t first = 0; first < active.size(); first += step) {
        // Queued planes whose copies may still run if a later one fails.
        struct Pending {
            std::vector<cudaStream_t> streams;
            ~Pending() {
                for (cudaStream_t stream : streams) cudaStreamSynchronize(stream);
            }
        } pending;
        // The plans say which frames the device will not hold: only those are staged.
        std::vector<std::unique_ptr<RollTurn>> turns;
        turns.reserve(step);
        {
            std::lock_guard lock(d->roll_mu);
            for (std::size_t i = 0; i < step; ++i) {
                turns.push_back(std::make_unique<RollTurn>(*d, active[first + i], start, count, lock));
            }
        }
        if (first == 0) d->order.planned(start, d->rolling.rolling_chunk);
        // Planes page-locked for direct DMA are uploaded from their frames,
        // which are held until the planes' events have passed.
        std::vector<std::vector<RollDirect>> direct(step);
        std::vector<const VSFrame*> held;
        {
            NSS_CUDA_RANGE("group.stage_in");
            for (std::size_t j = 0; j < step; ++j) {
                const int plane = active[first + j];
                const GroupPlane& p = d->planes[plane];
                const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
                std::uint8_t* base = g.roll_base[plane];
                const std::size_t region = g.roll_step[plane];
                const RollTurn& turn = *turns[j];
                direct[j].assign(turn.uploads.size(), RollDirect{});
                const std::size_t channels = static_cast<std::size_t>(d->channels);
                const auto stage = [&](std::size_t i, int clip, const VSFrame* f) {
                    bool used = false;
                    for (std::size_t c = 0; c < channels; ++c) {
                        if (d->gpu || d->frame_memory->pinned(f, plane + static_cast<int>(c))) {
                            direct[j][i][clip * channels + c] = f;
                            used = true;
                        } else {
                            stage_plane(vsapi->getReadPtr(f, plane + c), vsapi->getStride(f, plane + c), row_bytes,
                                        p.height, base + ((i * clips + clip) * channels + c) * region);
                        }
                    }
                    if (used) {
                        held.push_back(f);
                    } else {
                        frames.freeFrame(f);
                    }
                };
                for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
                    const int fn = turn.uploads[i];
                    stage(i, 0, get(d->node, fn));
                    if (w) stage(i, 1, get(d->guide, fn));
                }
            }
        }
        // A slot is held only while a plane's work is queued (see frame_output).
        pending.streams.reserve(3 * step);
        for (std::size_t j = 0; j < step; ++j) {
            const int plane = active[first + j];
            queue_plane(*d, *turns[j], plane, start, count, *store, g.roll_base[plane], g.roll_step[plane],
                        g.plane_done[plane], direct[j], vsapi);
            for (cudaStream_t stream : {turns[j]->slot->up.get(), turns[j]->slot->stream.get(), turns[j]->slot->down.get()})
                pending.streams.push_back(stream);
        }
        // The store holds these planes once their events have passed; the
        // staging is free for the next planes then.
        NSS_CUDA_RANGE("group.wait");
        for (std::size_t j = 0; j < step; ++j) NSS_CUDA_CHECK(cudaEventSynchronize(g.plane_done[active[first + j]]));
        pending.streams.clear();
        for (const VSFrame* f : held) frames.freeFrame(f);
    }
    return store;
}

ChunkCache::Chunk rolling_chunk(Driver* d, std::optional<SlotPool<Staging>::Lease>& staging, int start, int count,
                               VSFrameContext* ctx, const VSAPI* vsapi, nss::FrameScope& frames, GpuFrameCache& cache) {
    return d->cache.get(start, [&] { return compute_chunk(d, staging, start, count, ctx, vsapi, frames, cache); });
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Driver*>(instance);
    const int nframes = d->vi.numFrames;
    if (activation == arInitial) {
        int first = n, last = n;
        if (d->mode == GroupMode::Legacy) {
            first = std::max(0, n - d->radius);
            last = temporal_last(n, d->radius, nframes);
        } else if (d->mode == GroupMode::Rolling) {
            // Always the full chunk dependency window: a cache hit seen now may
            // be evicted before arAllFramesReady.
            const int start = n / d->rolling.rolling_chunk * d->rolling.rolling_chunk;
            const int count = std::min(d->rolling.rolling_chunk, nframes - start);
            first = std::max(0, start - roll_reach(*d));
            last = temporal_last(start + count - 1, roll_reach(*d), nframes);
        }
        for (int i = first; i <= last; ++i) {
            vsapi->requestFrameFilter(i, d->node, ctx);
            if (d->guide) vsapi->requestFrameFilter(i, d->guide, ctx);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("group.frame");
    if (d->mode != GroupMode::Rolling) return d->gpu ? frame_output_gpu(d, n, ctx, core, vsapi) : frame_output(d, n, ctx, core, vsapi);

    const int start = n / d->rolling.rolling_chunk * d->rolling.rolling_chunk;
    const int count = std::min(d->rolling.rolling_chunk, nframes - start);
    // The chunk, then the permit for the output frame (see Driver::outputs); a chunk this request computes
    // keeps its staging set until the frame is out.
    std::optional<SlotPool<Staging>::Lease> staging;
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    GpuFrameCache cache(vsapi);
    auto chunk = rolling_chunk(d, staging, start, count, ctx, vsapi, frames, cache);
    auto permit = d->outputs->acquire();
    const VSFrame* src = d->gpu ? take_gpu_frame(*d, cache, d->node, n, frames, ctx, vsapi)
                                : frames.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = d->gpu ? d->gpu->new_frame(frames, &d->vi.format, d->vi.width, d->vi.height, src)
                          : frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    nss::stamp_contribution(dst, 0, n, d->model, vsapi);
    if (permit->gpu) permit->gpu->begin();
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (permit->gpu) {
            const bool written = d->channels > 1 ? d->planes[0].active && d->channel_out[plane] : p.active;
            if (written) {
                permit->gpu->write(chunk->plane(n - chunk->start, plane), row_bytes, dst, plane, row_bytes, p.height);
            } else {
                permit->gpu->copy(src, plane, dst, plane, row_bytes, p.height);
            }
            permit->gpu->publish(dst, &plane, 1);
        } else {
            copy_to_frame(chunk->plane(n - chunk->start, plane), static_cast<std::ptrdiff_t>(row_bytes),
                          vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), row_bytes, p.height);
        }
    }
    if (permit->gpu) permit->stream.synchronize();
    frames.freeFrame(src);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Driver*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->staging.reset();
        d->cache.clear();
        d->frame_memory.reset();
    }
    delete d;
}

}  // namespace

void group_filter_install(GroupFilterConfig&& config, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<Driver>();
    static_cast<GroupFilterConfig&>(*d) = std::move(config);
    d->mode = d->radius == 0 ? GroupMode::Spatial : d->mode;
    d->ntemp = 2 * d->radius + 1;
    d->vi_out = d->vi;
    if (d->mode == GroupMode::Legacy) d->vi_out.height = nss::checked_fat_height(d->vi.height, d->radius);
    d->budget = nss::current_budget();
    // A rolling center's rounds before the last add into fixed-point sums.
    for (const GroupPlane& p : d->planes) {
        if (d->mode == GroupMode::Rolling && d->iters > 1 && p.active && !p.fused) {
            throw std::logic_error(prefix(*d) + "rolling rounds need a fused plane");
        }
    }
    // Fused planes finish straight from their accumulators; the legacy output
    // is the float slices of each center.
    if (d->mode == GroupMode::Legacy) {
        for (GroupPlane& p : d->planes) p.fused = false;
    }
    if (d->channels > 1) {
        bool any = false;
        for (int c = 0; c < d->channels; ++c) {
            d->channel_out[c] = d->planes[c].active;
            any = any || d->planes[c].active;
            if (c > 0) d->planes[c].active = false;  // planes[0] carries the unit
        }
        d->planes[0].active = any;
    }
    if (d->two_stage) {
        if (d->radius > 0 && d->mode != GroupMode::Rolling) {
            throw std::logic_error(prefix(*d) + "two stages need spatial or rolling output");
        }
        if (iterative(*d) || d->guide) throw std::logic_error(prefix(*d) + "two stages take one round and no guide clip");
        for (int c = 0; c < 3; ++c) {
            GroupPlane& b = d->basic[c];
            if (d->channels > 1) {
                // A unit: the estimate of channel c is needed where the second stage
                // filters it, and of channel 0 for the matching. basic[0] carries the unit.
                d->basic_out[c] = b.active && (d->channel_out[c] || c == 0);
            } else {
                b.active = b.active && d->planes[c].active;
            }
        }
        if (d->channels > 1) {
            d->basic[0].active = d->planes[0].active && (d->basic_out[0] || d->basic_out[1] || d->basic_out[2]);
            d->basic[1].active = d->basic[2].active = false;
        }
    }
    if (d->gpu_api) d->gpu = std::make_shared<GpuFrames>(core, d->gpu_api, vsapi);
    DeviceGuard guard(d->device.index);
    d->frame_memory = std::make_unique<FrameMemory>(core, vsapi);
    plan_planes(*d);
    d->cache.configure(d->rolling.rolling_chunk, d->frame_bytes, d->plane_offset, d->cache_size, d->cache_cap,
                       chunk_blocks(d->budget, d->gpu != nullptr));
    if (d->gpu && d->budget) {
        d->gpu->reserve(d->budget->snapshot().bytes[static_cast<std::size_t>(nss::ResourceKind::Workspace)] +
                        d->frame_bytes * static_cast<std::size_t>(d->cache_cap));
    }
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    for (auto& slot : slots) d->slots.push_back(slot.get());
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    if (d->staging_sets) {
        std::vector<std::unique_ptr<Staging>> sets;
        for (int i = 0; i < d->staging_sets; ++i) sets.push_back(make_staging(*d));
        d->staging = std::make_unique<SlotPool<Staging>>(std::move(sets));
    }
    if (d->mode == GroupMode::Rolling) {
        std::vector<std::unique_ptr<Driver::OutputPermit>> permits;
        for (int i = 0; i < std::max(d->staging_sets, 1); ++i) {
            auto permit = std::make_unique<Driver::OutputPermit>();
            if (d->gpu) permit->gpu = std::make_unique<GpuStream>(d->gpu, permit->stream);
            permits.push_back(std::move(permit));
        }
        d->outputs = std::make_unique<SlotPool<Driver::OutputPermit>>(std::move(permits));
    }

    const auto pattern = d->mode == GroupMode::Spatial ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->guide, pattern}};
    Driver* raw = d.get();
    VSNode* node = vs_api::create_video_filter(vsapi, raw->name.c_str(), &raw->vi_out, nss::checked_frame<getFrame>,
                                               freeFilter, fmParallel, raw->gpu != nullptr, deps, d->guide ? 2 : 1, raw,
                                               core);
    if (!node) throw std::runtime_error(prefix(*d) + "failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace nss_cuda
