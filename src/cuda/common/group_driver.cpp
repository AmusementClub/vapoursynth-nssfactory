// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for patch-group filters (see group_driver.hpp).
#include "cuda/common/group_driver.hpp"
#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
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
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
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
    std::unique_ptr<OrderedAggregator> aggregator;
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

// Pinned blocks for chunk stores, reused so that a chunk neither allocates
// nor faults in fresh pages. A block goes back when its chunk is dropped.
struct StorePool {
    std::mutex mu;
    std::vector<PinnedBuffer> free;
};

// The finished frames of one chunk. The device downloads straight into it.
struct ChunkStore {
    int start = 0, count = 0;
    PinnedBuffer bytes;  // rolling_chunk frames, each its planes back to back (tight rows)
    std::size_t frame_bytes = 0;
    std::array<std::size_t, 3> plane_offset{};
    std::shared_ptr<StorePool> pool;
    // Final plane of frame start + i.
    float* plane(int i, int p) const {
        return reinterpret_cast<float*>(bytes.as<std::uint8_t>() + i * frame_bytes + plane_offset[p]);
    }
    ChunkStore() = default;
    ChunkStore(const ChunkStore&) = delete;
    ChunkStore& operator=(const ChunkStore&) = delete;
    ~ChunkStore() {
        if (!pool || !bytes.get()) return;
        std::lock_guard lock(pool->mu);
        pool->free.push_back(std::move(bytes));
    }
};

struct Driver : GroupFilterConfig {
    std::size_t plane_floats = 0;  // largest plane (tight pitch)
    int ntemp = 1;                 // 2R+1
    int regions = 0;               // staging regions per staging set
    int staging_sets = 0;          // frames or chunks in flight, at least num_streams
    bool channel_out[3]{true, true, true};  // joint: channel c is written (else copied)
    bool basic_out[3]{true, true, true};    // two_stage: the first stage filters channel c of a unit
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
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

    // Rolling chunk cache (LRU) and chunks being computed. The cache holds
    // cache_size chunks and grows to cache_cap when chunks it dropped are
    // asked for again (two such misses per step).
    std::shared_ptr<StorePool> stores;
    std::size_t frame_bytes = 0;
    std::array<std::size_t, 3> plane_offset{};
    int cache_size = 1, cache_cap = 1;
    std::deque<int> dropped;  // starts of recently dropped chunks
    int dropped_hits = 0;
    std::mutex cache_mu;
    std::condition_variable cache_cv;
    std::list<std::shared_ptr<const ChunkStore>> cache;
    std::set<int> computing;
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
int src_ring(const Driver& d) { return d.two_stage ? 4 * d.radius + 1 : d.ntemp; }

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

void plan_planes(Driver& d) {
    const bool w = has_guide(d);
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
    d.regions = staging_regions(d);
    // Per slot: window frames, slices, result/accumulators, staging (sized for
    // the largest plane), the leased frame's output VSFrame, and at least a
    // minimum batch of groups.
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const bool rolling = d.mode == GroupMode::Rolling;
    const int chunk = rolling ? d.rolling.rolling_chunk : 0;
    // Window frames (rolling keeps them in its states), float slices, the
    // result, and for fused planes the finish scratch and, when spatial, the
    // fixed-point num and den (two float planes' worth each).
    const std::size_t device_planes = (rolling ? 0 : static_cast<std::size_t>(device_frames(d))) +
                                      static_cast<std::size_t>(d.channels) *
                                          ((any_ordered(d) ? 2 * d.ntemp : 0) + 1 + (d.iters > 1 ? d.ntemp : 0)) +
                                      (any_fused(d) ? 2 + (rolling ? 0 : 4 * static_cast<std::size_t>(d.channels)) : 0);
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
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
    const std::size_t chunk_bytes = frame_bytes * chunk;
    const std::size_t output = d.mode == GroupMode::Rolling ? 2 * frame_bytes + chunk_bytes
                                                       : frame_bytes * (d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1);
    const std::size_t fixed_base =
        plane_bytes * device_planes + (rolling ? 0 : plane_bytes * d.regions) + output + bins;
    d.frame_bytes = frame_bytes;
    // The cache starts at cache_chunks; what it may grow to is settled below.
    const std::size_t shared_cache = d.mode == GroupMode::Rolling ? chunk_bytes * d.rolling.cache_chunks : 0;
    std::size_t min_batch = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        min_batch = std::max(min_batch, std::min<std::size_t>(p->grid.count(), 1024) * per_ref_bytes(d, *p));
    }
    if (total != SIZE_MAX && total <= shared_cache) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - shared_cache;
    // A state per plane lets every plane carry on from chunk to chunk and the
    // planes of a chunk go together; under a limit that does not hold them
    // the planes take turns on one state (and a plane then starts each chunk
    // afresh, unless it is the only one).
    bool per_plane = rolling && active_planes(d) > 1;
    // The frames of a chunk that wait for their download (one is in fixed_base).
    const std::size_t batch_out = rolling ? plane_bytes * d.channels * (chunk - 1) : 0;
    if (rolling && limit != SIZE_MAX) {
        const std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 1;
        const auto fits = [&](bool planes, std::size_t extra) {
            return limit / streams >=
                   fixed_base + extra + roll_bytes(d, planes) + roll_staging_bytes(d, planes) + min_batch * 4 / 3;
        };
        per_plane = per_plane && fits(true, batch_out);
        d.roll_batch_out = fits(per_plane, batch_out);
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
    d.staging_bytes = rolling ? roll_staging_bytes(d, per_plane) : plane_bytes * d.regions;
    const std::size_t fixed = fixed_base + (rolling ? roll_bytes(d, per_plane) + d.staging_bytes : 0) +
                              (d.roll_batch_out ? batch_out : 0);
    const std::size_t need = fixed + min_batch * 4 / 3;
    if (!d.backend.streams_explicit) {
        // Prefer the most streams that still run a whole plane as one batch
        // (extra batches cost a matching/aggregation launch each); otherwise
        // the most streams that fit at all.
        std::size_t full_batch = 0;
        for (const GroupPlane* p : stage_planes(d)) {
            full_batch = std::max(full_batch, std::min<std::size_t>(kBatchBytes, p->grid.count() * per_ref_bytes(d, *p)));
        }
        const std::size_t whole = limit / (fixed + full_batch * 4 / 3 + 4096);
        d.backend.num_streams = static_cast<int>(
            std::clamp<std::size_t>(whole >= 1 ? whole : limit / need, 1, kDefaultStreams));
    }
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < need) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB" +
                                    (shared_cache ? ", plus " + std::to_string((shared_cache >> 20) + 1) +
                                                        " MiB for the rolling cache"
                                                  : ""));
    }
    const std::size_t share =
        limit == SIZE_MAX ? kBatchBytes + fixed : limit / static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t batch_bytes = std::min(kBatchBytes, (share - fixed) / 4 * 3);
    for (GroupPlane* p : stage_planes(d)) {
        const std::size_t fit = std::max<std::size_t>(1, batch_bytes / per_ref_bytes(d, *p));
        p->batch = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(p->grid.count())));
    }
    // Planes of different shapes share the slot's buffers; shrink the batches
    // until the buffers together fit what one plane alone was given.
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
    // A staging set and the output it feeds per frame (or rolling chunk) in
    // flight. One per stream is in `fixed`; more keep a stream busy while
    // other threads copy, as far as the limit allows. Three frames in flight
    // saturate one stream (1080p BM3D on an RTX 5080: 454 fps with one set,
    // 752 with two, 821 with three); beyond that the host copies only compete
    // for memory bandwidth. A rolling chunk stages all its frames before its
    // device work starts, so each stream needs chunks ahead of it (radius 1,
    // one stream: 173 fps with one set, 302 with two, 308 with four; three
    // streams: 303, 304 with four, 344 with nine).
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t set_bytes = d.staging_bytes + output;
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
        // Under a limit the cache grows only into what the plan leaves over.
        const std::size_t used = streams * (fixed + slot_batch_bytes(d) / 3 * 4) + extra * set_bytes;
        const std::size_t spare = limit > used ? (limit - used) / chunk_bytes : 0;
        d.cache_cap = static_cast<int>(
            std::min<std::size_t>(d.rolling.cache_limit, d.rolling.cache_chunks + spare));
    }
}

std::unique_ptr<Staging> make_staging(const Driver& d) {
    auto staging = std::make_unique<Staging>();
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    staging->bytes = PinnedBuffer(d.staging_bytes, d.budget);
    NSS_CUDA_CHECK(cudaEventCreateWithFlags(&staging->done, cudaEventDisableTiming));
    if (d.mode == GroupMode::Rolling) {
        std::uint8_t* at = staging->bytes.as<std::uint8_t>();
        for (int plane = 0; plane < 3; ++plane) {
            const GroupPlane& p = d.planes[plane];
            if (!p.active) continue;
            NSS_CUDA_CHECK(cudaEventCreateWithFlags(&staging->plane_done[plane], cudaEventDisableTiming));
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
            }
            for (int t = 0; t < d.ntemp && w; ++t) {
                r.ref_frames.emplace_back(bytes, d.budget);
                r.ref_ptrs.push_back(r.ref_frames.back().as<float>());
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
    slot->counts = DeviceBuffer(counts, d.budget);
    if (values) slot->values = DeviceBuffer(values, d.budget);
    if (scratch) slot->scratch = DeviceBuffer(scratch, d.budget);
    if (patches) slot->patches = DeviceBuffer(patches, d.budget);
    if (max_patches) {
        slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, d.ntemp, max_patches, d.budget, true);
    }
    // Spatial/legacy windows map slot t to buffer t for the slot's lifetime.
    if (rolling) return slot;
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
// `rolling`, the chunk's (which the caller prepares).
void filter_center(const Driver& d, Slot& s, const GroupPlane& p, int center, bool match_guide,
                   const FixedTarget* rolling = nullptr) {
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
        window.frames = (match_guide ? s.ref_ptrs : s.src_ptrs).as<const float*>();
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
    for (int begin = 0; begin < p.grid.count(); begin += p.batch) {
        const int count = std::min(p.batch, p.grid.count() - begin);
        {
            NSS_CUDA_RANGE("group.match");
            if (radius > 0) {
                predictive_match(geometry, window, p.grid, begin, count, s.matches.as<DeviceMatch>(), s.counts.as<int>(),
                                 stream);
            } else {
                const float* plane = match_guide ? s.host_ref_ptrs[0] : s.host_src_ptrs[0];
                spatial_match(plane, geometry, p.grid, begin, count, s.matches.as<DeviceMatch>(), s.counts.as<int>(),
                              stream);
            }
        }
        GroupLaunch launch{};
        launch.src = s.src_ptrs.as<const float*>();
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
    const auto identity = [&](int plane) {
        const GroupPlane& p = d->planes[plane];
        auto* op = vsapi->getWritePtr(dst, plane);
        const std::ptrdiff_t ds = vsapi->getStride(dst, plane);
        const auto* sp = vsapi->getReadPtr(src0, plane);
        const std::ptrdiff_t ss = vsapi->getStride(src0, plane);
        if (radius > 0) {
            nss::host_detail::temporal_identity(reinterpret_cast<float*>(op), static_cast<int>(ds / sizeof(float)),
                                                reinterpret_cast<const float*>(sp),
                                                static_cast<int>(ss / sizeof(float)), p.width, p.height, radius);
        } else {
            vsh::bitblt(op, ds, sp, ss, static_cast<std::size_t>(p.width) * sizeof(float), p.height);
        }
    };
    // A unit is one plane, or all three planes of a joint filter.
    for (int unit = 0; unit < (channels > 1 ? 1 : d->vi.format.numPlanes); ++unit) {
        const GroupPlane& p = d->planes[unit];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (!p.active) {
            for (int c = 0; c < channels; ++c) identity(unit + c);
            continue;
        }
        const auto stage = [&](int region, const VSFrame* frame, int plane) {
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
            for (int t = 0; t < d->ntemp; ++t) {
                for (int c = 0; c < channels; ++c) {
                    upload_staged(g.regions[region++], row_bytes, p.height, s.src_frames[t].as<float>() + c * p.floats,
                                  row_bytes, s.stream);
                    if (w) {
                        upload_staged(g.regions[region++], row_bytes, p.height, s.ref_frames[t].as<float>() + c * p.floats,
                                      row_bytes, s.stream);
                    }
                }
            }
            const std::size_t unit_floats = p.floats * channels;
            for (int t = 0; t < d->ntemp && d->iters > 1; ++t) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.noisy_frames[t].get(), s.src_frames[t].get(), unit_floats * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s.stream));
            }
            if (d->two_stage) {
                // First stage: the estimate of every channel into the guide
                // frame, where the second stage matches and takes its Wiener
                // reference. A channel the first stage leaves alone is the source.
                const GroupPlane& b = d->basic[unit];
                if (b.active) filter_center(*d, s, b, n, false);
                for (int c = 0; c < channels; ++c) {
                    const std::size_t channel = static_cast<std::size_t>(c) * p.floats;
                    const float* source = s.host_src_ptrs[0] + channel;
                    float* estimate = s.ref_frames[0].as<float>() + channel;
                    if (!b.active || !d->basic_out[c]) {
                        NSS_CUDA_CHECK(cudaMemcpyAsync(estimate, source, p.floats * sizeof(float),
                                                       cudaMemcpyDeviceToDevice, s.stream));
                    } else if (b.fused) {
                        const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + channel,
                                                s.fixed_den.as<unsigned long long>() + channel, p.width, p.floats};
                        fixed_finish(fixed, b.block, source, p.width, p.height, s.fixed_rows.as<unsigned long long>(),
                                     estimate, s.stream);
                    } else {
                        aggregate_finish(s.num.as<float>() + channel, s.den.as<float>() + channel, source, p.width,
                                         p.height, p.width, estimate, s.stream);
                    }
                }
            }
            for (int iter = 0; iter < d->iters; ++iter) {
                for (int t = 0; t < d->ntemp && iter > 0; ++t) {
                    iter_regularize(s.src_frames[t].as<float>(), s.noisy_frames[t].as<float>(), unit_floats, d->delta, s.stream);
                }
                filter_center(*d, s, p, n, (w || d->two_stage) && iter == 0);
                if (!iterative(*d) || (radius > 0 && iter + 1 == d->iters)) break;
                // The estimate of every window frame becomes the finished slice.
                for (int t = 0; t < d->ntemp; ++t) {
                    for (int c = 0; c < channels; ++c) {
                        float* est = s.src_frames[t].as<float>() + c * p.floats;
                        const std::size_t slice = (static_cast<std::size_t>(c) * d->ntemp + t) * p.floats;
                        if (p.fused) {  // spatial: one slice per channel
                            const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + slice,
                                                    s.fixed_den.as<unsigned long long>() + slice, p.width, p.floats};
                            fixed_finish(fixed, p.block, est, p.width, p.height, s.fixed_rows.as<unsigned long long>(),
                                         est, s.stream);
                            continue;
                        }
                        aggregate_finish(s.num.as<float>() + slice, s.den.as<float>() + slice, est, p.width, p.height, p.width,
                                         est, s.stream);
                    }
                }
            }
            for (int c = 0; c < channels; ++c) {
                if (!d->channel_out[c]) continue;
                if (radius == 0) {
                    const float* result = s.src_frames[0].as<float>() + c * p.floats;
                    // s.out is reused for every channel: its download is
                    // stream-ordered before the next finish overwrites it.
                    const std::size_t channel = static_cast<std::size_t>(c) * p.floats;
                    // An iterative filter's last finish left the result in the
                    // source frame.
                    if (p.fused && !iterative(*d)) {
                        const FixedTarget fixed{s.fixed_num.as<unsigned long long>() + channel,
                                                s.fixed_den.as<unsigned long long>() + channel, p.width, p.floats};
                        fixed_finish(fixed, p.block, s.host_src_ptrs[0] + channel, p.width, p.height,
                                     s.fixed_rows.as<unsigned long long>(), s.out.as<float>(), s.stream);
                        result = s.out.as<float>();
                    } else if (!iterative(*d)) {
                        aggregate_finish(s.num.as<float>() + channel, s.den.as<float>() + channel,
                                         s.host_src_ptrs[0] + channel, p.width, p.height, p.width, s.out.as<float>(),
                                         s.stream);
                        result = s.out.as<float>();
                    }
                    begin_download(result, row_bytes, row_bytes, p.height, g.regions[slot_region(c, 0)], s.stream);
                } else {
                    // Fat layout: slice sl -> num rows at 2*sl*h, den rows at (2*sl+1)*h.
                    for (int sl = 0; sl < d->ntemp; ++sl) {
                        const std::size_t slice = (static_cast<std::size_t>(c) * d->ntemp + sl) * p.floats;
                        begin_download(s.num.as<float>() + slice, row_bytes, row_bytes, p.height,
                                       g.regions[slot_region(c, 2 * sl)], s.stream);
                        begin_download(s.den.as<float>() + slice, row_bytes, row_bytes, p.height,
                                       g.regions[slot_region(c, 2 * sl + 1)], s.stream);
                    }
                }
            }
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

// Queues one plane of the chunk on its slot, in walk_chunk's order: the
// uploads staged at base + i * step, the centers, and each chunk frame's
// finish and download into the store. `event` is recorded behind the last
// download.
void queue_plane(Driver& d, RollTurn& turn, int plane, int start, int count, ChunkStore& store,
                 const std::uint8_t* base, std::size_t step, cudaEvent_t event) {
    const GroupPlane& p = d.planes[plane];
    const GroupPlane& b = d.basic[plane];
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    const int radius = d.radius, nframes = d.vi.numFrames, nt = d.ntemp;
    const int channels = d.channels, ring = src_ring(d), clips = guide_clip(d) ? 2 : 1;
    NSS_CUDA_RANGE("group.queue");
    turn.enter();
    Slot& s = *turn.slot;
    RollState& r = *turn.state;
    struct Drain {
        cudaStream_t stream;
        bool armed = true;
        ~Drain() {
            if (armed) cudaStreamSynchronize(stream);
        }
    } drain{s.stream};
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
    // A slice is zero before its frame's first center: every one when the
    // chunk starts afresh, and again once its frame is finished.
    const auto clear_basic = [&](int cell_index) {
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
    };
    const auto clear_sums = [&](int cell_index) {
        for (int c = 0; c < channels; ++c) {
            if (!d.channel_out[c]) continue;
            const std::size_t cell = c * ring_floats + static_cast<std::size_t>(cell_index) * p.floats;
            if (p.fused) {
                fixed_clear(FixedTarget{fixed.num + cell, fixed.den + cell}, p.floats, s.stream);
            } else {
                NSS_CUDA_CHECK(cudaMemsetAsync(acc_num + cell, 0, p.floats * sizeof(float), s.stream));
                NSS_CUDA_CHECK(cudaMemsetAsync(acc_den + cell, 0, p.floats * sizeof(float), s.stream));
            }
        }
    };
    if (!turn.carry) {
        std::fill(r.resident.begin(), r.resident.end(), -1);
        for (int cell = 0; cell < nt; ++cell) {
            clear_basic(cell);
            clear_sums(cell);
        }
    }
    struct Run {
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
        decltype(clear_basic)& clear;
        decltype(clear_sums)& clear_frame;
        std::size_t staged = 0;
        std::vector<const float*> sp, rp;

        // The device's window of `center`: source frames, and the estimates for the second stage.
        void point(int center, bool estimates) {
            for (int t = 0; t < nt; ++t) {
                const int fn = temporal_slot_frame(center, t, radius, nframes);
                sp[t] = r.src_ptrs[fn % ring];
                if (estimates) rp[t] = r.ref_ptrs[fn % nt];
            }
            // Pageable sources: staged before cudaMemcpyAsync returns, and
            // stream-ordered after the previous center's kernels.
            NSS_CUDA_CHECK(cudaMemcpyAsync(s.src_ptrs.get(), sp.data(), nt * sizeof(float*), cudaMemcpyHostToDevice,
                                           s.stream));
            if (estimates) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.ref_ptrs.get(), rp.data(), nt * sizeof(float*), cudaMemcpyHostToDevice,
                                               s.stream));
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
                for (int c = 0; c < channels; ++c) {
                    upload_staged(base + (staged * clips * channels + c) * step, row_bytes, p.height,
                                  r.src_frames[fn % ring].as<float>() + c * p.floats, row_bytes, s.stream);
                    if (clips == 2) {
                        upload_staged(base + ((staged * clips + 1) * channels + c) * step, row_bytes, p.height,
                                      r.ref_frames[fn % nt].as<float>() + c * p.floats, row_bytes, s.stream);
                    }
                }
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
            clear(frame % nt);  // the cell is the next frame's from here on
        }
        void center(int center, int next) {
            if (!d.two_stage) upload(center);
            point(center, p.wiener);
            // Logical slice i is frame next + i, in ring cell (next + i) % (2R+1).
            fixed.slices = std::min(nt, nframes - next);
            fixed.slice_first = next % nt;
            fixed.slice_base = center - radius - next;
            filter_center(d, s, p, center, p.wiener, &fixed);
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
        }
        void output(int frame) {
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
                    begin_download(result, row_bytes, row_bytes, p.height, store.plane(frame - start, plane + c), s.stream);
                }
            }
            clear_frame(frame % nt);  // the slice is a later frame's from here on
        }
    };
    Run run{d, s, r, turn, p, b, store, base, step, row_bytes, ring_floats, basic_floats, plane, start, radius, nframes, nt,
            channels, ring, clips, acc_num, acc_den, basic_acc_num, basic_acc_den, fixed, basic, clear_basic, clear_sums, 0,
            std::vector<const float*>(nt), std::vector<const float*>(nt)};
    walk_chunk(d, start, count, turn.carry, run);
    for (int i = 0; i < count && d.roll_batch_out; ++i) {
        for (int c = 0; c < channels; ++c) {
            if (!d.channel_out[c]) continue;
            begin_download(s.out.as<float>() + (static_cast<std::size_t>(i) * channels + c) * p.floats, row_bytes, row_bytes,
                           p.height, store.plane(i, plane + c), s.stream);
        }
    }
    NSS_CUDA_CHECK(cudaEventRecord(event, s.stream));
    drain.armed = false;
    turn.leave();
}

// Rolling: compute the chunk [start, start + count). The frames it reads are
// staged before a slot is taken and the results are copied out after it has
// been given back, as for a spatial frame.
std::shared_ptr<ChunkStore> compute_chunk(Driver* d, int start, int count, VSFrameContext* ctx, const VSAPI* vsapi) {
    auto staging = [&] {
        NSS_CUDA_RANGE("group.staging");
        return d->staging->acquire();
    }();
    Staging& g = *staging;
    DeviceGuard guard(d->device.index);
    const int radius = d->radius;
    const int nframes = d->vi.numFrames;
    const bool w = guide_clip(*d);
    nss::FrameScope frames(vsapi);
    auto store = std::make_shared<ChunkStore>();
    store->start = start;
    store->count = count;
    store->frame_bytes = d->frame_bytes;
    store->plane_offset = d->plane_offset;
    store->pool = d->stores;
    {
        std::lock_guard lock(d->stores->mu);
        if (!d->stores->free.empty()) {
            store->bytes = std::move(d->stores->free.back());
            d->stores->free.pop_back();
        }
    }
    if (!store->bytes.get()) store->bytes = PinnedBuffer(d->frame_bytes * d->rolling.rolling_chunk, d->budget);

    const int clips = w ? 2 : 1;
    // A unit of several channels is planes[0]; its channels that are not
    // written are copied like the planes of an inactive unit.
    const bool unit = d->channels > 1;
    std::vector<int> active;
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        if (p.active && !(unit && plane > 0)) active.push_back(plane);
        if (unit ? d->planes[0].active && d->channel_out[plane] : p.active) continue;
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        for (int i = 0; i < count; ++i) {
            const VSFrame* f = frames.getFrameFilter(start + i, d->node, ctx);
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
        {
            NSS_CUDA_RANGE("group.stage_in");
            for (std::size_t j = 0; j < step; ++j) {
                const int plane = active[first + j];
                const GroupPlane& p = d->planes[plane];
                const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
                std::uint8_t* base = g.roll_base[plane];
                const std::size_t region = g.roll_step[plane];
                const RollTurn& turn = *turns[j];
                for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
                    const int fn = turn.uploads[i];
                    const std::size_t channels = static_cast<std::size_t>(d->channels);
                    const VSFrame* f = frames.getFrameFilter(fn, d->node, ctx);
                    for (std::size_t c = 0; c < channels; ++c) {
                        stage_plane(vsapi->getReadPtr(f, plane + c), vsapi->getStride(f, plane + c), row_bytes, p.height,
                                    base + (i * clips * channels + c) * region);
                    }
                    frames.freeFrame(f);
                    if (w) {
                        const VSFrame* rf = frames.getFrameFilter(fn, d->guide, ctx);
                        for (std::size_t c = 0; c < channels; ++c) {
                            stage_plane(vsapi->getReadPtr(rf, plane + c), vsapi->getStride(rf, plane + c), row_bytes,
                                        p.height, base + ((i * clips + 1) * channels + c) * region);
                        }
                        frames.freeFrame(rf);
                    }
                }
            }
        }
        // A slot is held only while a plane's work is queued (see frame_output).
        pending.streams.reserve(step);
        for (std::size_t j = 0; j < step; ++j) {
            const int plane = active[first + j];
            queue_plane(*d, *turns[j], plane, start, count, *store, g.roll_base[plane], g.roll_step[plane],
                        g.plane_done[plane]);
            pending.streams.push_back(turns[j]->slot->stream);
        }
        // The store holds these planes once their events have passed; the
        // staging is free for the next planes then.
        NSS_CUDA_RANGE("group.wait");
        for (std::size_t j = 0; j < step; ++j) NSS_CUDA_CHECK(cudaEventSynchronize(g.plane_done[active[first + j]]));
        pending.streams.clear();
    }
    return store;
}

std::shared_ptr<const ChunkStore> rolling_chunk(Driver* d, int start, int count, VSFrameContext* ctx,
                                                const VSAPI* vsapi) {
    std::unique_lock lock(d->cache_mu);
    for (;;) {
        for (auto it = d->cache.begin(); it != d->cache.end(); ++it) {
            if ((*it)->start == start) {
                auto hit = *it;
                d->cache.splice(d->cache.begin(), d->cache, it);
                return hit;
            }
        }
        if (!d->computing.count(start)) break;
        d->cache_cv.wait(lock);  // another thread is computing this chunk
    }
    // A miss on a chunk the cache dropped recently: after two of them the
    // cache keeps one more chunk, up to cache_cap.
    if (d->cache_size < d->cache_cap) {
        const auto again = std::find(d->dropped.begin(), d->dropped.end(), start);
        if (again != d->dropped.end()) {
            d->dropped.erase(again);
            if (++d->dropped_hits >= 2) {
                d->dropped_hits = 0;
                ++d->cache_size;
            }
        }
    }
    d->computing.insert(start);
    lock.unlock();
    std::shared_ptr<ChunkStore> computed;
    try {
        computed = compute_chunk(d, start, count, ctx, vsapi);
    } catch (...) {
        lock.lock();
        d->computing.erase(start);
        d->cache_cv.notify_all();
        throw;
    }
    lock.lock();
    d->computing.erase(start);
    d->cache.push_front(computed);
    while (static_cast<int>(d->cache.size()) > d->cache_size) {
        d->dropped.push_back(d->cache.back()->start);
        d->cache.pop_back();
    }
    while (d->dropped.size() > static_cast<std::size_t>(std::max(8, 2 * d->cache_cap))) d->dropped.pop_front();
    d->cache_cv.notify_all();
    return computed;
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
    if (d->mode != GroupMode::Rolling) return frame_output(d, n, ctx, core, vsapi);

    const int start = n / d->rolling.rolling_chunk * d->rolling.rolling_chunk;
    const int count = std::min(d->rolling.rolling_chunk, nframes - start);
    auto chunk = rolling_chunk(d, start, count, ctx, vsapi);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    nss::stamp_contribution(dst, 0, n, d->model, vsapi);
    frames.freeFrame(src);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane),
                    chunk->plane(n - chunk->start, plane), row_bytes, row_bytes, p.height);
    }
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Driver*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->staging.reset();
        d->cache.clear();
        d->stores.reset();
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
    if (d->mode == GroupMode::Rolling && iterative(*d)) {
        throw std::logic_error(prefix(*d) + "rolling mode does not support joint or multi-round filters");
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
    DeviceGuard guard(d->device.index);
    plan_planes(*d);
    d->stores = std::make_shared<StorePool>();
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    for (auto& slot : slots) d->slots.push_back(slot.get());
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    if (d->staging_sets) {
        std::vector<std::unique_ptr<Staging>> sets;
        for (int i = 0; i < d->staging_sets; ++i) sets.push_back(make_staging(*d));
        d->staging = std::make_unique<SlotPool<Staging>>(std::move(sets));
    }

    const auto pattern = d->mode == GroupMode::Spatial ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->guide, pattern}};
    Driver* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2(raw->name.c_str(), &raw->vi_out, nss::checked_frame<getFrame>, freeFilter,
                                            fmParallel, deps, d->guide ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error(prefix(*d) + "failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace nss_cuda
