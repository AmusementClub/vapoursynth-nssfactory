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
// carried. Rings: frame f is window buffer f % (2R+1) and slice f % K of the
// sums and of chunk_src, K = rolling_chunk + 2R.
struct RollState {
    std::vector<DeviceBuffer> src_frames, ref_frames;
    std::vector<const float*> src_ptrs, ref_ptrs;
    DeviceBuffer chunk_src;             // source planes, for pixels no patch covers
    DeviceBuffer acc;                   // ordered planes: K num slices, then K den slices
    DeviceBuffer fixed_num, fixed_den;  // fused planes: K slices each
    std::vector<int> resident;          // frame in each window buffer (the chunk at turn owns it)
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
    Staging() = default;
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    ~Staging() {
        if (done) cudaEventDestroy(done);
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
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
    // Rolling takes its slots by turn (see RollState) instead of from the pool.
    std::vector<Slot*> slots;
    std::vector<int> roll_state;  // plane -> index in Slot::roll
    int roll_states = 0;
    bool roll_carry = true;  // false under a memory limit that has no room for the carried slices
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

bool has_guide(const Driver& d) { return d.guide != nullptr; }
// Joint or multi-round filters keep their estimate in the window frames.
bool iterative(const Driver& d) { return d.channels > 1 || d.iters > 1; }
std::string prefix(const Driver& d) { return "nss_cuda." + d.name + ": "; }

std::size_t scratch_floats(const Driver& d, const GroupPlane& p) {
    return d.scratch_floats ? d.scratch_floats(p, has_guide(d)) : 0;
}

bool any_fused(const Driver& d) {
    for (const GroupPlane& p : d.planes) {
        if (p.active && p.fused) return true;
    }
    return false;
}
// Some active plane goes through values, patch records and the ordered
// aggregator into the float num/den slices.
bool any_ordered(const Driver& d) {
    for (const GroupPlane& p : d.planes) {
        if (p.active && !p.fused) return true;
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
               (sizeof(DeviceMatch) + sizeof(AggregatePatch) + OrderedAggregator::kBytesPerPatch);
}

// Bytes of the slot's per-batch buffers for the planes' current batches.
// make_slot sizes each buffer for the plane that needs it most, so with
// planes of different shapes (or fused beside ordered) this is more than any
// single plane's batch * per_ref_bytes.
std::size_t slot_batch_bytes(const Driver& d) {
    std::size_t matches = 0, counts = 0, scratch = 0, values = 0, patches = 0, sort = 0;
    for (const GroupPlane& p : d.planes) {
        if (!p.active) continue;
        const std::size_t batch = static_cast<std::size_t>(p.batch);
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, p) * sizeof(float));
        if (p.fused) continue;
        values = std::max(values, batch * d.channels * p.group * p.block * p.block * sizeof(float));
        patches = std::max(patches, batch * p.group * sizeof(AggregatePatch));
        sort = std::max(sort, batch * p.group * OrderedAggregator::kBytesPerPatch);
    }
    return matches + counts + scratch + values + patches + sort;
}

// Staging regions: the uploads (src, ref), then the downloads.
// Spatial and legacy: one upload per window frame and channel; one download
// per channel (spatial) or 2 * ntemp fat rows per channel (legacy).
// Rolling: one upload per frame a chunk reads (the chunk and 2 * radius
// frames on each side); the results go straight into the chunk store.
// Slices of a rolling plane's sums: the chunk and, to carry on from chunk to
// chunk, the 2R frames behind it that its last centers already add to.
int roll_slices(const Driver& d) { return d.rolling.rolling_chunk + (d.roll_carry ? 2 * d.radius : 0); }

// Device bytes of the rolling states of one slot: one per active plane, or
// one that every plane uses in turn (then only a single-plane clip carries).
std::size_t roll_bytes(const Driver& d, bool per_plane) {
    const std::size_t k = roll_slices(d), clips = has_guide(d) ? 2 : 1;
    const auto planes = [&](bool fused, bool ordered) {
        return d.ntemp * clips + k + (fused ? 4 * k : 0) + (ordered ? 2 * k : 0);
    };
    if (!per_plane) return d.plane_floats * sizeof(float) * planes(any_fused(d), any_ordered(d));
    std::size_t bytes = 0;
    for (const GroupPlane& p : d.planes) {
        if (p.active) bytes += p.floats * sizeof(float) * planes(p.fused, !p.fused);
    }
    return bytes;
}
int active_planes(const Driver& d) {
    int n = 0;
    for (const GroupPlane& p : d.planes) n += p.active;
    return n;
}

int upload_regions(const Driver& d) { return d.ntemp * d.channels * (has_guide(d) ? 2 : 1); }
int chunk_window(const Driver& d) { return d.rolling.rolling_chunk + 4 * d.radius; }
int chunk_uploads(const Driver& d) { return chunk_window(d) * (has_guide(d) ? 2 : 1); }
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
        if (!p.active) continue;
        p.group = std::min(p.group, kMaxGroup);
        p.grid = make_raster_grid(p.width, p.height, p.block, p.step);
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
    const std::size_t device_planes = (rolling ? 0 : static_cast<std::size_t>(upload_regions(d))) +
                                      static_cast<std::size_t>(d.channels) *
                                          ((any_ordered(d) ? 2 * d.ntemp : 0) + 1 + (d.iters > 1 ? d.ntemp : 0)) +
                                      (any_fused(d) ? (rolling ? 2 : 6) : 0);
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    int max_w = 1, max_h = 1;
    for (const GroupPlane& p : d.planes) {
        if (p.active && !p.fused) {
            max_w = std::max(max_w, p.width);
            max_h = std::max(max_h, p.height);
        }
    }
    const std::size_t bins = any_ordered(d) ? OrderedAggregator::bin_bytes(max_w, max_h, d.ntemp, true) : 0;
    // Rolling also holds host chunk stores: one being built per slot, up to
    // cache_limit finished ones, and the output frames copied out of them.
    const std::size_t chunk_bytes = frame_bytes * chunk;
    const std::size_t output = d.mode == GroupMode::Rolling ? 2 * frame_bytes + chunk_bytes
                                                       : frame_bytes * (d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1);
    const std::size_t fixed_base = plane_bytes * device_planes + plane_bytes * d.regions + output + bins;
    d.frame_bytes = frame_bytes;
    // The cache starts at cache_chunks; what it may grow to is settled below.
    const std::size_t shared_cache = d.mode == GroupMode::Rolling ? chunk_bytes * d.rolling.cache_chunks : 0;
    std::size_t min_batch = 0;
    for (const GroupPlane& p : d.planes) {
        if (p.active) min_batch = std::max(min_batch, std::min<std::size_t>(p.grid.count(), 1024) * per_ref_bytes(d, p));
    }
    if (total != SIZE_MAX && total <= shared_cache) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - shared_cache;
    // A state per plane lets every plane carry from chunk to chunk; under a
    // limit that does not hold them the planes share one, and under one that
    // does not hold the carried slices either every chunk starts afresh.
    bool per_plane = rolling && active_planes(d) > 1;
    if (rolling && limit != SIZE_MAX) {
        const std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 1;
        const auto fits = [&](bool planes) {
            return limit / streams >= fixed_base + roll_bytes(d, planes) + min_batch * 4 / 3;
        };
        per_plane = per_plane && fits(true);
        if (!per_plane && !fits(false)) d.roll_carry = false;
    }
    d.roll_states = 0;
    d.roll_state.assign(3, 0);
    if (rolling) {
        for (int plane = 0; plane < 3; ++plane) {
            if (d.planes[plane].active && per_plane) d.roll_state[plane] = d.roll_states++;
        }
        if (!per_plane) d.roll_states = 1;
    }
    const std::size_t fixed = fixed_base + (rolling ? roll_bytes(d, per_plane) : 0);
    const std::size_t need = fixed + min_batch * 4 / 3;
    if (!d.backend.streams_explicit) {
        // Prefer the most streams that still run a whole plane as one batch
        // (extra batches cost a matching/aggregation launch each); otherwise
        // the most streams that fit at all.
        std::size_t full_batch = 0;
        for (const GroupPlane& p : d.planes) {
            if (p.active) {
                full_batch = std::max(full_batch, std::min<std::size_t>(kBatchBytes, p.grid.count() * per_ref_bytes(d, p)));
            }
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
    for (GroupPlane& p : d.planes) {
        if (!p.active) continue;
        const std::size_t fit = std::max<std::size_t>(1, batch_bytes / per_ref_bytes(d, p));
        p.batch = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(p.grid.count())));
    }
    // Planes of different shapes share the slot's buffers; shrink the batches
    // until the buffers together fit what one plane alone was given.
    for (int round = 0; round < 8 && slot_batch_bytes(d) > batch_bytes; ++round) {
        const double scale = static_cast<double>(batch_bytes) / static_cast<double>(slot_batch_bytes(d));
        bool changed = false;
        for (GroupPlane& p : d.planes) {
            if (!p.active) continue;
            const int next = std::max(1, static_cast<int>(p.batch * scale));
            changed = changed || next != p.batch;
            p.batch = next;
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
    const std::size_t set_bytes = plane_bytes * d.regions + output;
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
    staging->bytes = PinnedBuffer(plane_bytes * d.regions, d.budget);
    for (int i = 0; i < d.regions; ++i) staging->regions.push_back(staging->bytes.as<std::uint8_t>() + plane_bytes * i);
    NSS_CUDA_CHECK(cudaEventCreateWithFlags(&staging->done, cudaEventDisableTiming));
    return staging;
}

std::unique_ptr<Slot> make_slot(const Driver& d) {
    auto slot = std::make_unique<Slot>();
    const bool w = has_guide(d);
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t unit_bytes = plane_bytes * d.channels;
    std::size_t values = 0, matches = 0, counts = 0, patches = 0, max_patches = 0, scratch = 0;
    int max_w = 1, max_h = 1;
    for (const GroupPlane& p : d.planes) {
        if (!p.active) continue;
        const std::size_t batch = static_cast<std::size_t>(p.batch);
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, p) * sizeof(float));
        if (p.fused) continue;
        max_patches = std::max(max_patches, batch * p.group);
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
        values = std::max(values, batch * d.channels * p.group * p.block * p.block * sizeof(float));
        patches = std::max(patches, batch * p.group * sizeof(AggregatePatch));
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
        const std::size_t k = roll_slices(d);
        slot->roll.resize(d.roll_states);
        const bool shared = d.roll_states == 1;
        for (int plane = 0; plane < 3; ++plane) {
            const GroupPlane& p = d.planes[plane];
            if (!p.active) continue;
            RollState& r = slot->roll[d.roll_state[plane]];
            if (!r.src_frames.empty()) continue;  // the shared state, sized below for every plane
            const std::size_t bytes = (shared ? d.plane_floats : p.floats) * sizeof(float);
            for (int t = 0; t < d.ntemp; ++t) {
                r.src_frames.emplace_back(bytes, d.budget);
                r.src_ptrs.push_back(r.src_frames.back().as<float>());
                if (w) {
                    r.ref_frames.emplace_back(bytes, d.budget);
                    r.ref_ptrs.push_back(r.ref_frames.back().as<float>());
                }
            }
            r.chunk_src = DeviceBuffer(bytes * k, d.budget);
            if (shared ? any_ordered(d) : !p.fused) r.acc = DeviceBuffer(bytes * 2 * k, d.budget);
            if (shared ? any_fused(d) : p.fused) {
                r.fixed_num = DeviceBuffer(bytes * 2 * k, d.budget);
                r.fixed_den = DeviceBuffer(bytes * 2 * k, d.budget);
            }
            r.resident.assign(d.ntemp, -1);
            r.promised.assign(d.ntemp, -1);
        }
    }
    slot->src_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    if (w) slot->ref_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    if (any_ordered(d)) {
        slot->num = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
        slot->den = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
    }
    slot->out = DeviceBuffer(unit_bytes, d.budget);
    if (any_fused(d)) {
        if (!rolling) {
            slot->fixed_num = DeviceBuffer(d.plane_floats * sizeof(unsigned long long), d.budget);
            slot->fixed_den = DeviceBuffer(d.plane_floats * sizeof(unsigned long long), d.budget);
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
    const bool w = has_guide(d);
    const int radius = d.radius;
    MatchGeometry geometry{p.width, p.height, p.width, p.block, p.range, p.group};
    geometry.channels = d.channels;
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
    const FixedTarget fixed = rolling ? *rolling
                                      : FixedTarget{s.fixed_num.as<unsigned long long>(),
                                                    s.fixed_den.as<unsigned long long>(), p.width, p.floats};
    if (p.fused && !rolling) fixed_clear(fixed, p.floats, stream);
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
        launch.guide = w ? s.ref_ptrs.as<const float*>() : nullptr;
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
            s.aggregator->run(s.values.as<float>() + c * channel_values, s.patches.as<AggregatePatch>(), count * p.group,
                              p.block, target, stream, begin > 0);
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
    const bool w = has_guide(*d);
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
            for (int iter = 0; iter < d->iters; ++iter) {
                for (int t = 0; t < d->ntemp && iter > 0; ++t) {
                    iter_regularize(s.src_frames[t].as<float>(), s.noisy_frames[t].as<float>(), unit_floats, d->delta, s.stream);
                }
                filter_center(*d, s, p, n, w && iter == 0);
                if (!iterative(*d) || (radius > 0 && iter + 1 == d->iters)) break;
                // The estimate of every window frame becomes the finished slice.
                for (int t = 0; t < d->ntemp; ++t) {
                    for (int c = 0; c < channels; ++c) {
                        float* est = s.src_frames[t].as<float>() + c * p.floats;
                        const std::size_t slice = (static_cast<std::size_t>(c) * d->ntemp + t) * p.floats;
                        aggregate_finish(s.num.as<float>() + slice, s.den.as<float>() + slice, est, p.width, p.height, p.width,
                                         est, s.stream);
                    }
                }
            }
            for (int c = 0; c < channels; ++c) {
                if (!d->channel_out[c]) continue;
                if (radius == 0) {
                    const float* result = s.src_frames[0].as<float>() + c * p.floats;
                    if (p.fused) {
                        const FixedTarget fixed{s.fixed_num.as<unsigned long long>(), s.fixed_den.as<unsigned long long>(),
                                                p.width, p.floats};
                        fixed_finish(fixed, p.block, s.host_src_ptrs[0], p.width, p.height,
                                     s.fixed_rows.as<unsigned long long>(), s.out.as<float>(), s.stream);
                        result = s.out.as<float>();
                    } else if (!iterative(*d)) {
                        aggregate_finish(s.num.as<float>(), s.den.as<float>(), s.host_src_ptrs[0], p.width, p.height, p.width,
                                         s.out.as<float>(), s.stream);
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

    RollTurn(Driver& driver, int plane, int start, int count) : d(driver) {
        const int radius = d.radius, nframes = d.vi.numFrames;
        const int index = d.roll_state[plane];
        std::lock_guard lock(d.roll_mu);
        const auto backlog = [&](const Slot* s) {
            std::uint64_t n = 0;
            for (const RollState& r : s->roll) n += r.issued - r.serving;
            return n;
        };
        Slot* idle = d.slots.front();
        for (Slot* s : d.slots) {
            const RollState& r = s->roll[index];
            if (d.roll_carry && r.plane == plane && r.next_start == start && r.failed_chain != r.chain && !slot) slot = s;
            if (backlog(s) < backlog(idle)) idle = s;
        }
        // Carrying on saves more than a second stream adds, unless the slot is far behind.
        carry = slot && (backlog(slot) < kCarryBacklog || backlog(idle) >= backlog(slot));
        if (!carry) slot = idle;
        state = &slot->roll[index];
        if (!carry) {
            ++state->chain;
            std::fill(state->promised.begin(), state->promised.end(), -1);
        }
        ticket = state->issued++;
        chain = state->chain;
        last_center = temporal_last(start + count - 1, radius, nframes);
        first_center = carry ? std::min(start + radius, last_center + 1) : std::max(0, start - radius);
        for (int center = first_center; center <= last_center; ++center) {
            for (int t = 0; t < d.ntemp; ++t) {
                const int fn = temporal_slot_frame(center, t, radius, nframes);
                if (state->promised[fn % d.ntemp] != fn) {
                    uploads.push_back(fn);
                    state->promised[fn % d.ntemp] = fn;
                }
            }
        }
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
                state->next_start = -1;
            }
            if (entered) slot->in_use = false;
            ++state->serving;
            left = true;
        }
        d.roll_cv.notify_all();
    }
};

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
    const bool w = has_guide(*d);
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
    const int k = roll_slices(*d);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (!p.active) {
            for (int i = 0; i < count; ++i) {
                const VSFrame* f = frames.getFrameFilter(start + i, d->node, ctx);
                vsh::bitblt(store->plane(i, plane), row_bytes, vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane),
                            row_bytes, p.height);
                frames.freeFrame(f);
            }
            continue;
        }
        // The plan says which frames the device will not hold: only those are staged.
        RollTurn turn(*d, plane, start, count);
        {
            NSS_CUDA_RANGE("group.stage_in");
            for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
                const int fn = turn.uploads[i];
                const VSFrame* f = frames.getFrameFilter(fn, d->node, ctx);
                stage_plane(vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane), row_bytes, p.height,
                            g.regions[i * clips]);
                frames.freeFrame(f);
                if (w) {
                    const VSFrame* rf = frames.getFrameFilter(fn, d->guide, ctx);
                    stage_plane(vsapi->getReadPtr(rf, plane), vsapi->getStride(rf, plane), row_bytes, p.height,
                                g.regions[i * clips + 1]);
                    frames.freeFrame(rf);
                }
            }
        }
        // The slot is held only while this plane's work is queued (see frame_output).
        {
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
            float* acc_num = r.acc.as<float>();
            float* acc_den = p.fused ? nullptr : acc_num + p.floats * k;
            // Logical slice i is frame start + i, in ring cell (start + i) % k.
            FixedTarget fixed{r.fixed_num.as<unsigned long long>(), r.fixed_den.as<unsigned long long>(), p.width, p.floats};
            fixed.slices = std::min(k, nframes - start);
            fixed.slice_first = start % k;
            fixed.slice_ring = k;
            if (!turn.carry) std::fill(r.resident.begin(), r.resident.end(), -1);
            // The carried frames keep their sums; the others start from zero.
            for (int f = turn.carry ? start + 2 * radius : start; f < start + k; ++f) {
                const std::size_t cell = static_cast<std::size_t>(f % k) * p.floats;
                if (p.fused) {
                    fixed_clear(FixedTarget{fixed.num + cell, fixed.den + cell}, p.floats, s.stream);
                } else {
                    NSS_CUDA_CHECK(cudaMemsetAsync(acc_num + cell, 0, p.floats * sizeof(float), s.stream));
                    NSS_CUDA_CHECK(cudaMemsetAsync(acc_den + cell, 0, p.floats * sizeof(float), s.stream));
                }
            }
            std::size_t staged = 0;
            std::vector<const float*> sp(d->ntemp), rp(d->ntemp);
            for (int center = turn.first_center; center <= turn.last_center; ++center) {
                for (int t = 0; t < d->ntemp; ++t) {
                    const int fn = temporal_slot_frame(center, t, radius, nframes);
                    const int ring = fn % d->ntemp;
                    if (r.resident[ring] != fn) {
                        if (staged >= turn.uploads.size() || turn.uploads[staged] != fn) {
                            throw std::logic_error(prefix(*d) + "rolling plan and device state disagree");
                        }
                        upload_staged(g.regions[staged * clips], row_bytes, p.height, r.src_frames[ring].get(), row_bytes,
                                      s.stream);
                        if (w) {
                            upload_staged(g.regions[staged * clips + 1], row_bytes, p.height, r.ref_frames[ring].get(),
                                          row_bytes, s.stream);
                        }
                        ++staged;
                        if (fn >= start && fn < start + k) {
                            NSS_CUDA_CHECK(cudaMemcpyAsync(r.chunk_src.as<float>() + static_cast<std::size_t>(fn % k) * p.floats,
                                                           r.src_frames[ring].get(), p.floats * sizeof(float),
                                                           cudaMemcpyDeviceToDevice, s.stream));
                        }
                        r.resident[ring] = fn;
                    }
                    sp[t] = r.src_ptrs[ring];
                    if (w) rp[t] = r.ref_ptrs[ring];
                }
                // Pageable sources: staged before cudaMemcpyAsync returns, and
                // stream-ordered after the previous center's kernels.
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.src_ptrs.get(), sp.data(), d->ntemp * sizeof(float*),
                                               cudaMemcpyHostToDevice, s.stream));
                if (w) {
                    NSS_CUDA_CHECK(cudaMemcpyAsync(s.ref_ptrs.get(), rp.data(), d->ntemp * sizeof(float*),
                                                   cudaMemcpyHostToDevice, s.stream));
                }
                // Window slot t of this center is frame center - radius + t.
                fixed.slice_base = center - radius - start;
                filter_center(*d, s, p, center, w, &fixed);
                // Ordered planes: each frame takes its centers in ascending
                // order, within a chunk and from one chunk to the next.
                for (int target = std::max(start, center - radius);
                     !p.fused && target <= std::min(start + k - 1, temporal_last(center, radius, nframes)); ++target) {
                    const int slice = target - center + radius;
                    const std::size_t cell = static_cast<std::size_t>(target % k) * p.floats;
                    accumulate_slice(acc_num + cell, acc_den + cell, s.num.as<float>() + slice * p.floats,
                                     s.den.as<float>() + slice * p.floats, p.floats, s.stream);
                }
            }
            for (int i = 0; i < count; ++i) {
                // s.out is reused for every frame: the download is stream-ordered
                // before the next finish overwrites it.
                const std::size_t cell = static_cast<std::size_t>((start + i) % k) * p.floats;
                const float* source = r.chunk_src.as<float>() + cell;
                if (p.fused) {
                    fixed_finish(FixedTarget{fixed.num + cell, fixed.den + cell, p.width, p.floats}, p.block, source,
                                 p.width, p.height, s.fixed_rows.as<unsigned long long>(), s.out.as<float>(), s.stream);
                } else {
                    aggregate_finish(acc_num + cell, acc_den + cell, source, p.width, p.height, p.width,
                                     s.out.as<float>(), s.stream);
                }
                begin_download(s.out.get(), row_bytes, row_bytes, p.height, store->plane(i, plane), s.stream);
            }
            NSS_CUDA_CHECK(cudaEventRecord(g.done, s.stream));
            drain.armed = false;
            turn.leave();
        }
        // The store holds this plane once the event has passed; the staging
        // regions are free for the next plane then.
        NSS_CUDA_RANGE("group.wait");
        NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
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
            first = std::max(0, start - 2 * d->radius);
            last = temporal_last(start + count - 1, 2 * d->radius, nframes);
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
    if (any_fused(*d) && iterative(*d)) {
        throw std::logic_error(prefix(*d) + "fused aggregation does not support joint or multi-round filters");
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
