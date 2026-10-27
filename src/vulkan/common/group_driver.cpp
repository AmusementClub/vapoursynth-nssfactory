// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for the patch-group filters (see group_driver.hpp).
#include "vulkan/common/group_driver.hpp"

#include "frontend/contribution.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"
#include "gpuhost/chunk_cache.hpp"
#include "nss/checked.hpp"
#include "vulkan/runtime/frame_memory.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_vk {
namespace {

using nss::host_detail::temporal_last;
using nss::host_detail::temporal_slot_frame;
// Rolling: finished chunks in host-cached memory the copies write.
using ChunkCache = nss::gpuhost::ChunkCache<Buffer>;
using ChunkStore = nss::gpuhost::ChunkStore<Buffer>;
using nss::gpuhost::PlanOrder;

// The matchers' group limit (TWSC's; the BM3D family stays within 64).
constexpr int kMaxGroup = 256;
// Upper bound on a slot's per-batch group buffers; larger planes run in
// batches, and the batch also shrinks so that the slots fit memory_limit_mb.
constexpr std::size_t kBatchBytes = 128u << 20;
// Frames in flight beyond num_streams: each is a slot of its own, so that the
// host copies of one frame run while the device works on another. One paid
// (1080p BM3D, RTX 5080, both cores: +32-43%), a second did not (+0.4-5% for
// +83-209 MiB, 2026-10-26). A memory limit takes it first.
constexpr std::size_t kExtraSlots = 1;
// Output frames beyond the slots: a request holds an output permit until its
// frame is returned and gives its slot back before that (as NLM, #103), so
// one request can finish while another uses its slot. A memory limit takes it
// before the extra slots.
constexpr std::size_t kExtraOutputs = 1;
// Rolling: staging sets per stream (nss_cuda's kChunksPerStream: a chunk
// stages all its frames before its device work starts, so a stream needs
// chunks staged ahead of it).
constexpr std::size_t kChunksPerStream = 3;

// A unit: one plane, or the three planes of a shared_match filter.
struct Unit {
    int first = 0;
    int channels = 1;
};

// A spatial or legacy frame in flight. The planes have the frame's layout
// (plane p at Driver::plane_offset[p] floats); the sums hold one unit at a
// time.
struct GroupSlot {
    explicit GroupSlot(std::shared_ptr<Device> device) : slot(std::move(device)) {}
    Slot slot;
    // Region t * clips + k: window slot t of the source (k = 0) and of the
    // guide clip (k = 1); a spatial window has one frame.
    PlaneStore inputs;
    Buffer estimate;  // two stages: the first stage's estimate (the second's guide)
    // Per unit 2 * ntemp device addresses: its source window, then its guide's.
    Buffer tables;
    // Fixed-point sums of the unit being filtered: channel c's ntemp slices
    // from c * ntemp * plane floats.
    Buffer num, den;
    Buffer rows;      // fixed_finish scratch, one plane
    Buffer out;       // the finished planes (spatial) or fat planes (legacy): fat_slices x the frame layout
    Buffer readback;  // what the copies did not write into the frame
    Buffer matches, counts, scratch;
    void settle() noexcept { slot.settle(); }
};

// What a rolling slot keeps of one unit between chunks, so that the chunk
// after the one it ran last only does the new work (as nss_cuda's RollState):
// the window frames stay on the device and the sums the last centers left for
// later frames are carried. A frame is finished as soon as its last center
// has run, so the sums are a ring of 2R+1 slices whatever the chunk: frame f
// is slice f % (2R+1), and source buffer f % src_ring.
struct RollState {
    std::vector<Buffer> src;  // src_ring frames of the unit (channels side by side)
    std::vector<Buffer> ref;  // ref_ring: the guide clip's frames, or the first stage's estimates
    Buffer num, den;          // 2R+1 ring cells per channel (channel c from c * (2R+1) * plane floats)
    Buffer basic_num, basic_den;  // two stages: the first stage's, laid out alike
    std::vector<int> resident;    // frame in each source buffer
    // Guarded by Driver::roll_mu. Chunks take a ticket when they are planned
    // and run in ticket order, so a plan can rely on the chunks before it.
    int plane = -1;
    int next_start = -1;        // the chunk that can carry on after the planned ones
    std::vector<int> promised;  // resident after the planned ones
    std::uint64_t issued = 0, serving = 0;
    std::uint64_t chain = 0;  // counts the chunks that started afresh
    std::uint64_t failed_chain = ~0ull;
    bool fresh_last = true;  // the last planned chunk started afresh
};

// One submission in flight on a rolling slot: a chunk's uploads, kernels and
// downloads.
struct RollLane {
    explicit RollLane(std::shared_ptr<Device> device) : slot(std::move(device)) {}
    Slot slot;
    bool in_use = false;  // guarded by Driver::roll_mu: from entering until its wait is done
    Buffer tables;        // the centers' window tables (written by the upload stage)
    Buffer out;           // roll_batch_out: the chunk's finished planes at the store's layout
};

// A rolling "stream": its states and two lanes that take its chunks in
// turn, so that a chunk's uploads and the download of the chunk before it
// run on the copy queues beside the kernels of the other (as nss_cuda's
// up / down streams, #96). The kernels of a slot run one chunk after the
// other: each lane's compute stage waits for the last one submitted.
struct RollSlot {
    std::vector<RollState> roll;
    std::array<std::unique_ptr<RollLane>, 2> lanes;
    // Guarded by Driver::roll_mu:
    int next_lane = 0;
    bool recording = false;  // a chunk is recorded and submitted (it sets `computed`)
    std::pair<VkSemaphore, std::uint64_t> computed{};  // the kernels of the last chunk submitted
    Buffer rows;
    Buffer out;  // without roll_batch_out: one unit, copied into the store after each finish
    Buffer matches, counts, scratch;
};

// The host side of one rolling chunk in flight: the uploads that are not
// direct, written before the chunk takes its slot (as nss_cuda's staging
// sets: the host copies of one chunk run while the device works on another).
// Upload i of a unit at base[first plane] + i * step[first plane].
struct RollStaging {
    Buffer bytes;
    std::array<std::size_t, 3> base{}, step{};
    void settle() noexcept {}
};

struct Driver : GroupFilterConfig {
    VSVideoInfo vi_out{};
    int ntemp = 1;  // 2R+1
    std::vector<Unit> units;
    std::array<std::size_t, 3> plane_offset{};  // floats
    std::size_t frame_floats = 0;
    std::size_t plane_floats = 0;  // largest plane
    int unit_channels = 1;         // largest unit
    std::size_t output_bytes = 0;  // an output VSFrame
    bool channel_out[3]{true, true, true};  // shared_match: channel c is written (else copied)
    bool basic_out[3]{true, true, true};    // two_stage: the first stage filters channel c of a unit
    std::size_t slots = 0, permits = 0;
    std::unique_ptr<Matcher> matcher;
    std::unique_ptr<Aggregator> aggregator;
    std::unique_ptr<SlotPool<GroupSlot>> pool;
    struct OutputPermit {
        void settle() noexcept {}
    };
    std::unique_ptr<SlotPool<OutputPermit>> outputs;

    // Rolling. The slots are taken by turn (RollTurn), not from a pool. A
    // state per active unit lets every plane carry on from chunk to chunk and
    // the units of a chunk go in one submission (roll_together); under a
    // limit that does not hold them the units take turns on one state.
    std::vector<std::unique_ptr<RollSlot>> roll_slots;
    std::vector<int> roll_state;  // unit's first plane -> index in RollSlot::roll
    int roll_states = 0;
    bool roll_together = false;
    // A chunk's finished frames wait on the device and are downloaded
    // together; false (under a limit without room for them) copies each into
    // the store on the compute queue as it is finished.
    bool roll_batch_out = true;
    std::size_t staging_bytes = 0, roll_table_words = 0;
    int staging_sets = 0;
    std::unique_ptr<SlotPool<RollStaging>> staging;
    std::mutex roll_mu;
    std::condition_variable roll_cv;
    int cache_size = 1, cache_cap = 1;
    ChunkCache cache;
    PlanOrder order;

    // Last: destroyed first, so that its imports go before the device can.
    std::unique_ptr<FrameMemory> frame_memory;
};

std::string prefix(const Driver& d) { return "nss_vk." + d.name + ": "; }
bool guide_clip(const Driver& d) { return d.guide != nullptr; }
bool has_guide(const Driver& d) { return d.guide != nullptr || d.two_stage; }
int clips(const Driver& d) { return guide_clip(d) ? 2 : 1; }
bool rolling(const Driver& d) { return d.mode == GroupMode::Rolling; }
// Slices of a plane in a slot's output: a num and a den per window slot (legacy).
int fat_slices(const Driver& d) { return d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1; }
std::size_t frame_bytes(const Driver& d) { return d.frame_floats * sizeof(float); }

// Frames a rolling chunk reads on each side of itself; the first of two
// stages reaches 2R further.
int roll_reach(const Driver& d) { return (d.two_stage ? 4 : 2) * d.radius; }
int chunk_window(const Driver& d) { return d.rolling.rolling_chunk + 2 * roll_reach(d); }
// The source buffers of a rolling state (and the guide clip's). A chunk's
// uploads run on the upload queue before its kernels, beside the kernels of
// the chunk before it on the slot's other lane (see RollSlot), so the ring
// holds what both read: a chunk that carries on reads its frames and `reach`
// more, the next one `chunk` further; one that starts afresh reads
// chunk_window frames (and the chunk after it waits; RollTurn::wait_upload).
int src_ring(const Driver& d) {
    return std::max(2 * d.rolling.rolling_chunk + roll_reach(d), chunk_window(d));
}
// Frames of the second clip a state keeps: the guide clip's (a ring like the
// source's), or the first stage's estimates (2R+1, made on the device).
int ref_ring(const Driver& d) { return guide_clip(d) ? src_ring(d) : d.two_stage ? d.ntemp : 0; }
std::size_t chunk_bytes(const Driver& d) { return frame_bytes(d) * static_cast<std::size_t>(d.rolling.rolling_chunk); }

// The active planes of every stage: they share the slot's per-batch buffers.
std::vector<GroupPlane*> stage_planes(Driver& d) {
    std::vector<GroupPlane*> planes;
    for (const Unit& u : d.units) {
        if (d.planes[u.first].active) planes.push_back(&d.planes[u.first]);
        if (d.two_stage && d.basic[u.first].active) planes.push_back(&d.basic[u.first]);
    }
    return planes;
}

std::vector<const Unit*> active_units(const Driver& d) {
    std::vector<const Unit*> units;
    for (const Unit& u : d.units) {
        if (d.planes[u.first].active) units.push_back(&u);
    }
    return units;
}

bool basic_active(const Driver& d, const Unit& u) { return d.two_stage && d.basic[u.first].active; }

std::size_t scratch_floats(const Driver& d, const GroupPlane& p) {
    return d.scratch_floats ? d.scratch_floats(p, p.wiener) : 0;
}

std::size_t per_ref_bytes(const Driver& d, const GroupPlane& p) {
    return scratch_floats(d, p) * sizeof(float) + sizeof(int) + static_cast<std::size_t>(p.group) * sizeof(DeviceMatch);
}

// Bytes of the per-batch buffers for the planes' current batches: each is
// sized for the plane that needs it most.
struct BatchBytes {
    std::size_t matches = 0, counts = 0, scratch = 0;
    std::size_t total() const { return matches + counts + scratch; }
};
BatchBytes batch_bytes(Driver& d) {
    BatchBytes b;
    for (const GroupPlane* p : stage_planes(d)) {
        const std::size_t batch = static_cast<std::size_t>(p->batch);
        b.matches = std::max(b.matches, batch * p->group * sizeof(DeviceMatch));
        b.counts = std::max(b.counts, batch * sizeof(int));
        b.scratch = std::max(b.scratch, batch * scratch_floats(d, *p) * sizeof(float));
    }
    return b;
}

void size_planes(Driver& d) {
    std::size_t floats = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const int width = nss::plane_width(d.vi, plane), height = nss::plane_height(d.vi, plane);
        d.plane_offset[plane] = floats;
        floats += static_cast<std::size_t>(width) * height;
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(width) * height);
        const std::size_t row = (static_cast<std::size_t>(width) * sizeof(float) + 63) & ~std::size_t{63};
        d.output_bytes += row * static_cast<std::size_t>(height) * fat_slices(d);
        for (GroupPlane* stage : {&d.planes[plane], &d.basic[plane]}) {
            stage->width = width;
            stage->height = height;
            stage->floats = static_cast<std::size_t>(width) * height;
            stage->group = std::min(stage->group, kMaxGroup);
            stage->wiener = stage == &d.planes[plane] && has_guide(d);
            if (stage->active) stage->grid = make_raster_grid(width, height, stage->block, stage->step);
        }
    }
    d.frame_floats = floats;
}

// The smallest batch a slot must hold (up to 1024 references) and the
// largest that runs a whole plane at once.
std::pair<std::size_t, std::size_t> batch_bounds(Driver& d) {
    std::size_t min_batch = 0, full_batch = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        const std::size_t refs = static_cast<std::size_t>(p->grid.count());
        min_batch = std::max(min_batch, std::min(kBatchBytes, std::min<std::size_t>(refs, 1024) * per_ref_bytes(d, *p)));
        full_batch = std::max(full_batch, std::min(kBatchBytes, refs * per_ref_bytes(d, *p)));
    }
    return {min_batch, full_batch};
}

// Each plane's batch: at most `budget` bytes of per-batch buffers, shrunk
// until planes of different shapes fit the shared buffers together.
void fit_batches(Driver& d, std::size_t budget) {
    budget = std::min(kBatchBytes, budget);
    // NSS_VK_GROUP_BATCH (tests): at most that many references a batch.
    const char* cap = std::getenv("NSS_VK_GROUP_BATCH");
    const std::size_t most = cap && std::atoi(cap) > 0 ? static_cast<std::size_t>(std::atoi(cap)) : SIZE_MAX;
    for (GroupPlane* p : stage_planes(d)) {
        const std::size_t fit = std::max<std::size_t>(1, budget / per_ref_bytes(d, *p));
        p->batch = static_cast<int>(std::min({fit, most, static_cast<std::size_t>(p->grid.count())}));
    }
    for (int round = 0; round < 8 && batch_bytes(d).total() > budget; ++round) {
        const double scale = static_cast<double>(budget) / static_cast<double>(batch_bytes(d).total());
        bool changed = false;
        for (GroupPlane* p : stage_planes(d)) {
            const int next = std::max(1, static_cast<int>(p->batch * scale));
            changed = changed || next != p->batch;
            p->batch = next;
        }
        if (!changed) break;
    }
}

[[noreturn]] void too_small(const Driver& d, std::size_t streams, std::size_t need, std::size_t cache) {
    throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for this clip: each of the " +
                                std::to_string(streams) + " stream(s) needs at least " +
                                std::to_string((need >> 20) + 1) + " MiB" +
                                (cache ? ", plus " + std::to_string((cache >> 20) + 1) + " MiB for the rolling cache"
                                       : ""));
}

// What a spatial or legacy slot holds whatever its batch.
std::size_t slot_fixed_bytes(const Driver& d) {
    const std::size_t nt = static_cast<std::size_t>(d.ntemp);
    // The plan assumes BAR uploads on a ReBAR device; without one the
    // inputs have a staging mirror.
    const std::size_t inputs = frame_bytes(d) * nt * clips(d) * (d.device->rebar() ? 1 : 2);
    const std::size_t cells = d.plane_floats * static_cast<std::size_t>(d.unit_channels) * nt;
    const std::size_t out = frame_bytes(d) * fat_slices(d);
    const std::size_t tables = d.units.size() * 2 * nt * sizeof(VkDeviceAddress);
    return inputs + (d.two_stage ? frame_bytes(d) : 0) + cells * 8 * 2 + d.plane_floats * 8 + out * 2 + tables + 4096;
}

// The memory plan of spatial and legacy output: num_streams (unless given),
// the slots and output permits in flight, and each plane's batch of
// references, within memory_limit_mb.
void plan_frames(Driver& d) {
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const std::size_t fixed = slot_fixed_bytes(d);
    const auto [min_batch, full_batch] = batch_bounds(d);
    // A slot and the output frame of its request.
    const std::size_t need = fixed + min_batch + d.output_bytes;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams =
            static_cast<int>(std::clamp<std::size_t>(limit / (fixed + full_batch + d.output_bytes), 1, kDefaultStreams));
    }
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    if (limit / streams < need) too_small(d, streams, need, 0);
    // Extra output permits, then extra slots, while the whole batches fit.
    d.slots = streams + kExtraSlots;
    std::size_t extra_outputs = kExtraOutputs;
    const auto total = [&](std::size_t batch) {
        return d.slots * (fixed + batch + d.output_bytes) + extra_outputs * d.output_bytes;
    };
    if (total(min_batch) > limit) extra_outputs = 0;
    while (d.slots > streams && total(min_batch) > limit) --d.slots;
    d.permits = d.slots + extra_outputs;
    // Each plane's batch: what a slot's share leaves beside its fixed buffers.
    fit_batches(d, limit == SIZE_MAX ? SIZE_MAX
                                     : (limit - extra_outputs * d.output_bytes) / d.slots - fixed - d.output_bytes);
}

// Rolling: the device bytes of a state (frames and sums of a unit of planes
// of `floats`), and of a slot's states: one per active unit, or one that the
// units use in turn (sized for the largest plane).
std::size_t roll_state_bytes(const Driver& d, std::size_t floats, bool basic) {
    const std::size_t nt = static_cast<std::size_t>(d.ntemp), ch = static_cast<std::size_t>(d.unit_channels);
    const std::size_t frames = static_cast<std::size_t>(src_ring(d) + ref_ring(d));
    return floats * ch * (frames * sizeof(float) + nt * 8 * 2 * (basic ? 2 : 1));
}
std::size_t roll_bytes(const Driver& d, bool per_unit) {
    std::size_t bytes = 0;
    bool basic = false;
    for (const Unit* u : active_units(d)) {
        if (per_unit) bytes += roll_state_bytes(d, d.planes[u->first].floats, basic_active(d, *u));
        basic = basic || basic_active(d, *u);
    }
    return per_unit || active_units(d).empty() ? bytes : roll_state_bytes(d, d.plane_floats, basic);
}
// Words of a slot's window tables: two of 2R+1 addresses a center, for the
// centers of both stages of every unit of a chunk (at most chunk + 6R each).
std::size_t roll_table_words(const Driver& d) {
    const std::size_t centers = static_cast<std::size_t>(d.rolling.rolling_chunk + 2 * roll_reach(d) + 2);
    return d.units.size() * 2 * centers * 2 * static_cast<std::size_t>(d.ntemp);
}
// A slot: its states, the finish scratch, and per lane the tables and the
// chunk's planes (or one unit the lanes share).
std::size_t roll_slot_bytes(const Driver& d, bool per_unit, bool batch_out) {
    const std::size_t out = batch_out ? 2 * chunk_bytes(d) : d.plane_floats * d.unit_channels * sizeof(float);
    return roll_bytes(d, per_unit) + d.plane_floats * 8 + out + 2 * roll_table_words(d) * 8 + 4096;
}
// Host bytes of a staging set's uploads: every active unit's at its own size,
// or one unit's at the largest size for units that take turns.
std::size_t roll_staging_bytes(const Driver& d, bool together) {
    const std::size_t uploads = static_cast<std::size_t>(chunk_window(d)) * clips(d) * d.unit_channels;
    if (!together) return d.plane_floats * sizeof(float) * uploads;
    std::size_t bytes = 0;
    for (const Unit* u : active_units(d)) bytes += d.planes[u->first].floats * sizeof(float) * uploads;
    return bytes;
}
// A staging set and what its request holds until the frame is out: two
// output frames and a chunk store (as nss_cuda's SlotPlan::output).
std::size_t roll_set_bytes(const Driver& d, bool together) {
    return roll_staging_bytes(d, together) + 2 * d.output_bytes + chunk_bytes(d);
}

// The memory plan of rolling output (as nss_cuda's plan_planes): the rolling
// cache at its starting size beside the slots; the state layout; the
// streams; the batches; the staging sets; and how far the cache may grow.
void plan_rolling(Driver& d) {
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const std::size_t cache = chunk_bytes(d) * static_cast<std::size_t>(d.rolling.cache_chunks);
    if (total != SIZE_MAX && total <= cache) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - cache;
    const auto [min_batch, full_batch] = batch_bounds(d);
    bool per_unit = active_units(d).size() > 1;
    bool batch_out = true;
    if (limit != SIZE_MAX) {
        const std::size_t streams = d.backend.streams_explicit ? static_cast<std::size_t>(d.backend.num_streams) : 1;
        const auto fits = [&](bool units, bool out) {
            return limit / streams >= roll_slot_bytes(d, units, out) + roll_set_bytes(d, units) + min_batch;
        };
        per_unit = per_unit && fits(true, true);
        batch_out = fits(per_unit, true);
    }
    d.roll_together = per_unit;
    d.roll_batch_out = batch_out;
    d.roll_states = 0;
    d.roll_state.assign(3, 0);
    for (const Unit* u : active_units(d)) {
        if (per_unit) d.roll_state[static_cast<std::size_t>(u->first)] = d.roll_states++;
    }
    if (!per_unit) d.roll_states = 1;
    d.staging_bytes = roll_staging_bytes(d, per_unit);
    d.roll_table_words = roll_table_words(d);
    const std::size_t fixed = roll_slot_bytes(d, per_unit, batch_out);
    const std::size_t set = roll_set_bytes(d, per_unit);
    // Each stream: its slot, a staging set and at least a minimum batch.
    const std::size_t need = fixed + set + min_batch;
    if (!d.backend.streams_explicit) {
        const std::size_t whole = limit / (fixed + set + full_batch);
        d.backend.num_streams =
            static_cast<int>(std::clamp<std::size_t>(whole >= 1 ? whole : limit / need, 1, kDefaultStreams));
    }
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    if (limit / streams < need) too_small(d, streams, need, total == SIZE_MAX ? 0 : cache);
    fit_batches(d, limit == SIZE_MAX ? SIZE_MAX : limit / streams - fixed - set);
    // More staging sets keep a stream busy while other threads copy, as far
    // as the limit allows; the cache grows only into what is left over.
    std::size_t extra = (kChunksPerStream - 1) * streams;
    d.cache_size = d.rolling.cache_chunks;
    d.cache_cap = d.rolling.cache_limit;
    if (limit != SIZE_MAX) {
        const std::size_t used = streams * (fixed + batch_bytes(d).total() + set);
        extra = std::min(extra, limit > used ? (limit - used) / set : 0);
        const std::size_t left = used + extra * set;
        const std::size_t spare = limit > left ? (limit - left) / chunk_bytes(d) : 0;
        d.cache_cap = static_cast<int>(
            std::min<std::size_t>(d.rolling.cache_limit, static_cast<std::size_t>(d.rolling.cache_chunks) + spare));
    }
    d.staging_sets = static_cast<int>(streams + extra);
    d.slots = streams;
    d.permits = static_cast<std::size_t>(std::max(d.staging_sets, 1));
}

// Writes `bytes` to a device buffer once, at filter creation.
void upload_now(const std::shared_ptr<Device>& device, const Buffer& target, const void* data, std::size_t bytes) {
    Buffer staging(device, bytes, Memory::Staging);
    std::memcpy(staging.mapped(), data, bytes);
    staging.flush();
    Slot slot(device);
    const VkBufferCopy copy{0, 0, bytes};
    device->vk().vkCmdCopyBuffer(slot.commands(Slot::Upload), staging.get(), target.get(), 1, &copy);
    slot.submit();
    slot.wait();
}

struct Batches {
    Buffer matches, counts, scratch;
};
Batches make_batches(Driver& d) {
    const BatchBytes b = batch_bytes(d);
    Batches out;
    out.matches = Buffer(d.device, std::max<std::size_t>(b.matches, 16), Memory::Device, d.budget);
    out.counts = Buffer(d.device, std::max<std::size_t>(b.counts, 4), Memory::Device, d.budget);
    if (b.scratch) out.scratch = Buffer(d.device, b.scratch, Memory::Device, d.budget);
    return out;
}

std::unique_ptr<GroupSlot> make_slot(Driver& d) {
    auto s = std::make_unique<GroupSlot>(d.device);
    const std::size_t nt = static_cast<std::size_t>(d.ntemp);
    s->inputs = PlaneStore(d.device, frame_bytes(d), d.ntemp * clips(d), d.budget);
    if (d.two_stage) s->estimate = Buffer(d.device, frame_bytes(d), Memory::Device, d.budget);
    const std::size_t cells = d.plane_floats * static_cast<std::size_t>(d.unit_channels) * nt;
    s->num = Buffer(d.device, cells * 8, Memory::Device, d.budget);
    s->den = Buffer(d.device, cells * 8, Memory::Device, d.budget);
    s->rows = Buffer(d.device, d.plane_floats * 8, Memory::Device, d.budget);
    s->out = Buffer(d.device, frame_bytes(d) * fat_slices(d), Memory::Device, d.budget);
    s->readback = Buffer(d.device, frame_bytes(d) * fat_slices(d), Memory::Readback, d.budget);
    Batches b = make_batches(d);
    s->matches = std::move(b.matches);
    s->counts = std::move(b.counts);
    s->scratch = std::move(b.scratch);
    // The tables never change: a unit's source window and its guide's.
    std::vector<VkDeviceAddress> tables;
    for (const Unit& u : d.units) {
        const VkDeviceAddress offset = d.plane_offset[static_cast<std::size_t>(u.first)] * sizeof(float);
        const auto region = [&](std::size_t t, int k) {
            return s->inputs.address() + (t * clips(d) + k) * frame_bytes(d) + offset;
        };
        for (std::size_t t = 0; t < nt; ++t) tables.push_back(region(t, 0));
        for (std::size_t t = 0; t < nt; ++t) {
            tables.push_back(guide_clip(d)  ? region(t, 1)
                             : d.two_stage ? s->estimate.address() + offset
                                           : 0);
        }
    }
    s->tables = Buffer(d.device, std::max<std::size_t>(tables.size(), 1) * sizeof(VkDeviceAddress), Memory::Device, d.budget);
    upload_now(d.device, s->tables, tables.data(), tables.size() * sizeof(VkDeviceAddress));
    return s;
}

std::unique_ptr<RollSlot> make_roll_slot(Driver& d) {
    auto s = std::make_unique<RollSlot>();
    const std::size_t nt = static_cast<std::size_t>(d.ntemp);
    s->roll.resize(static_cast<std::size_t>(d.roll_states));
    const bool shared = !d.roll_together;
    bool basic = false;
    for (const Unit* u : active_units(d)) basic = basic || basic_active(d, *u);
    for (const Unit* u : active_units(d)) {
        RollState& r = s->roll[static_cast<std::size_t>(d.roll_state[static_cast<std::size_t>(u->first)])];
        if (!r.src.empty()) continue;  // the shared state, sized for every unit
        const std::size_t floats = (shared ? d.plane_floats : d.planes[u->first].floats) * d.unit_channels;
        for (int t = 0; t < src_ring(d); ++t) r.src.emplace_back(d.device, floats * sizeof(float), Memory::Device, d.budget);
        for (int t = 0; t < ref_ring(d); ++t) r.ref.emplace_back(d.device, floats * sizeof(float), Memory::Device, d.budget);
        r.num = Buffer(d.device, floats * nt * 8, Memory::Device, d.budget);
        r.den = Buffer(d.device, floats * nt * 8, Memory::Device, d.budget);
        if (shared ? basic : basic_active(d, *u)) {
            r.basic_num = Buffer(d.device, floats * nt * 8, Memory::Device, d.budget);
            r.basic_den = Buffer(d.device, floats * nt * 8, Memory::Device, d.budget);
        }
        r.resident.assign(static_cast<std::size_t>(src_ring(d)), -1);
        r.promised.assign(static_cast<std::size_t>(src_ring(d)), -1);
    }
    for (auto& lane : s->lanes) {
        lane = std::make_unique<RollLane>(d.device);
        lane->tables = Buffer(d.device, d.roll_table_words * 8, Memory::Device, d.budget);
        if (d.roll_batch_out) lane->out = Buffer(d.device, chunk_bytes(d), Memory::Device, d.budget);
    }
    s->rows = Buffer(d.device, d.plane_floats * 8, Memory::Device, d.budget);
    if (!d.roll_batch_out) {
        s->out = Buffer(d.device, d.plane_floats * d.unit_channels * sizeof(float), Memory::Device, d.budget);
    }
    Batches b = make_batches(d);
    s->matches = std::move(b.matches);
    s->counts = std::move(b.counts);
    s->scratch = std::move(b.scratch);
    return s;
}

std::unique_ptr<RollStaging> make_staging(Driver& d) {
    auto g = std::make_unique<RollStaging>();
    g->bytes = Buffer(d.device, std::max<std::size_t>(d.staging_bytes, 4), Memory::Staging, d.budget);
    std::size_t at = 0;
    const std::size_t uploads = static_cast<std::size_t>(chunk_window(d)) * clips(d) * d.unit_channels;
    for (const Unit* u : active_units(d)) {
        const std::size_t plane = static_cast<std::size_t>(u->first);
        g->base[plane] = d.roll_together ? at : 0;
        g->step[plane] = (d.roll_together ? d.planes[u->first].floats : d.plane_floats) * sizeof(float);
        if (d.roll_together) at += g->step[plane] * uploads;
    }
    return g;
}

void memory_barrier(const Device& device, VkCommandBuffer commands, VkPipelineStageFlags2 src_stage,
                    VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    device.vk().vkCmdPipelineBarrier2(commands, &dependency);
}

// Copies on the compute queue between kernels: after the kernels before
// them (which may read or write what they overwrite), before the kernels
// after them.
constexpr VkAccessFlags2 kShaderAccess = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
constexpr VkAccessFlags2 kCopyAccess = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
void before_copies(const Device& device, VkCommandBuffer commands) {
    memory_barrier(device, commands, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kShaderAccess, VK_PIPELINE_STAGE_2_COPY_BIT,
                   kCopyAccess);
}
void after_copies(const Device& device, VkCommandBuffer commands) {
    memory_barrier(device, commands, VK_PIPELINE_STAGE_2_COPY_BIT, kCopyAccess, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   kShaderAccess);
}
void copy_buffer(const Device& device, VkCommandBuffer commands, VkBuffer from, VkDeviceSize from_offset, VkBuffer to,
                 VkDeviceSize to_offset, VkDeviceSize bytes) {
    const VkBufferCopy region{from_offset, to_offset, bytes};
    device.vk().vkCmdCopyBuffer(commands, from, to, 1, &region);
}

// Where one stage of a center's groups reads.
struct StageIO {
    VkDeviceAddress src = 0;    // table: window slot t -> source unit
    VkDeviceAddress guide = 0;  // the Wiener reference's table, 0 without one
    VkDeviceAddress match = 0;  // the search's: a plane (spatial) or a table (temporal)
};

struct BatchBuffers {
    const Buffer& matches;
    const Buffer& counts;
    const Buffer& scratch;
};

// The groups of one stage of a unit at `center` matched (on io.match,
// channel 0) and filtered into `fixed`, batch after batch, recorded into
// `commands` (compute). `write` names the channels filtered.
void filter_center(Driver& d, VkCommandBuffer commands, const BatchBuffers& buffers, const Unit& u, const GroupPlane& p,
                   int center, const StageIO& io, const FixedTarget& fixed, const bool* write) {
    MatchGeometry geometry{p.width, p.height, p.width, p.block, p.range, p.group};
    geometry.channels = d.shared_match ? 1 : u.channels;
    geometry.channel_step = static_cast<long long>(p.floats);
    TemporalWindow window{};
    if (d.radius > 0) {
        window.frames = io.match;
        window.ntemp = d.ntemp;
        window.t0 = d.radius;
        window.radius = d.radius;
        window.valid_begin = std::max(0, d.radius - center);
        window.valid_end = d.radius + std::min(d.radius + 1, d.vi.numFrames - center);
        window.ps_num = p.ps_num;
        window.ps_range = p.ps_range;
    }
    for (int begin = 0; begin < p.grid.count(); begin += p.batch) {
        const int count = std::min(p.batch, p.grid.count() - begin);
        if (d.radius > 0) {
            d.matcher->predictive(commands, geometry, window, p.grid, begin, count, buffers.matches.address(),
                                  buffers.counts.address());
        } else {
            d.matcher->spatial(commands, io.match, geometry, p.grid, begin, count, buffers.matches.address(),
                               buffers.counts.address());
        }
        compute_barrier(*d.device, commands);
        GroupLaunch launch{};
        launch.commands = commands;
        launch.src = io.src;
        launch.guide = p.wiener ? io.guide : 0;
        launch.pitch = p.width;
        launch.matches = buffers.matches.address();
        launch.counts = buffers.counts.address();
        launch.batch = count;
        launch.plane = &p;
        launch.channels = u.channels;
        launch.channel_step = geometry.channel_step;
        launch.scratch = buffers.scratch ? buffers.scratch.address() : 0;
        launch.channel_active = write;
        launch.fused = fixed;
        d.launch(launch);
        // The next batch's matches and scratch reuse the buffers.
        compute_barrier(*d.device, commands);
    }
}

// The slot's sums as a target: ntemp slices per channel, window slot t in slice t.
FixedTarget window_target(const Driver& d, const Buffer& num, const Buffer& den, const GroupPlane& p) {
    FixedTarget fixed{};
    fixed.num = num.address();
    fixed.den = den.address();
    fixed.pitch = p.width;
    fixed.slice_step = static_cast<std::int64_t>(p.floats);
    fixed.channel_step = static_cast<std::int64_t>(p.floats) * d.ntemp;
    fixed.slices = fixed.slice_ring = d.ntemp;
    return fixed;
}

// One stage of a unit of a spatial or legacy frame: its groups into the
// slot's sums, which then are finished (spatial) or turned into the fat
// frame's slices (legacy) at `target` for the channels `write` names.
void run_stage(Driver& d, GroupSlot& s, const Unit& u, const GroupPlane& p, int center, const StageIO& io,
               VkDeviceAddress source, VkDeviceAddress target, const bool* write) {
    VkCommandBuffer commands = s.slot.commands(Slot::Compute);
    // The sums hold one unit; the clear waits for the previous unit's finish.
    const std::size_t cells = p.floats * static_cast<std::size_t>(u.channels) * d.ntemp;
    fixed_clear(*d.device, commands, s.num.get(), 0, cells * 8);
    fixed_clear(*d.device, commands, s.den.get(), 0, cells * 8);
    const FixedTarget fixed = window_target(d, s.num, s.den, p);
    filter_center(d, commands, BatchBuffers{s.matches, s.counts, s.scratch}, u, p, center, io, fixed, write);
    // A unit's channels lie p.floats * fat_slices floats apart in `target`.
    for (int c = 0; c < u.channels; ++c) {
        if (!write[c]) continue;
        const VkDeviceAddress channel = static_cast<VkDeviceAddress>(c) * p.floats;
        const VkDeviceAddress out = target + channel * fat_slices(d) * sizeof(float);
        if (d.mode != GroupMode::Legacy) {
            d.aggregator->fixed_finish(commands, fixed.num + channel * 8, fixed.den + channel * 8, p.width, p.block,
                                       source + channel * sizeof(float), p.width, p.height, s.rows.address(), out);
            compute_barrier(*d.device, commands);  // the rows are reused, and the target read next
            continue;
        }
        for (int t = 0; t < d.ntemp; ++t) {
            const VkDeviceAddress slice = (channel * d.ntemp + static_cast<VkDeviceAddress>(t) * p.floats) * 8;
            d.aggregator->fixed_slices(commands, fixed.num + slice, fixed.den + slice, p.width, p.block, p.width,
                                       p.height, s.rows.address(),
                                       out + static_cast<VkDeviceAddress>(2 * t) * p.floats * sizeof(float));
            compute_barrier(*d.device, commands);
        }
    }
}

// Every unit of a spatial or legacy frame, on the slot's compute queue.
void run_units(Driver& d, GroupSlot& s, int n) {
    VkCommandBuffer commands = s.slot.commands(Slot::Compute);
    const std::size_t nt = static_cast<std::size_t>(d.ntemp);
    for (std::size_t i = 0; i < d.units.size(); ++i) {
        const Unit& u = d.units[i];
        const GroupPlane& p = d.planes[u.first];
        if (!p.active) continue;
        const VkDeviceAddress offset = d.plane_offset[static_cast<std::size_t>(u.first)] * sizeof(float);
        // The center's source frame: window slot R.
        const VkDeviceAddress source =
            s.inputs.address() + static_cast<VkDeviceAddress>(d.radius) * clips(d) * frame_bytes(d) + offset;
        StageIO io;
        io.src = s.tables.address() + i * 2 * nt * sizeof(VkDeviceAddress);
        io.guide = io.src + nt * sizeof(VkDeviceAddress);
        // Spatial: the plane the groups are matched on; temporal: its window.
        io.match = d.radius > 0 ? (guide_clip(d) ? io.guide : io.src)
                                : (guide_clip(d) ? s.inputs.address() + frame_bytes(d) + offset : source);
        const VkDeviceAddress target = s.out.address() + offset * fat_slices(d);
        if (d.two_stage) {
            // The first stage's estimate of every channel: filtered, or the source.
            const GroupPlane& b = d.basic[u.first];
            const VkDeviceAddress estimate = s.estimate.address() + offset;
            bool copy = false;
            for (int c = 0; c < u.channels; ++c) {
                if (b.active && d.basic_out[c]) continue;
                const VkDeviceSize at = offset + static_cast<VkDeviceAddress>(c) * p.floats * sizeof(float);
                copy_buffer(*d.device, commands, s.inputs.buffer(), at, s.estimate.get(), at, p.floats * sizeof(float));
                copy = true;
            }
            if (copy) after_copies(*d.device, commands);
            if (b.active) {
                StageIO first;
                first.src = io.src;
                first.match = source;
                run_stage(d, s, u, b, n, first, source, estimate, d.basic_out);
            }
            io.match = estimate;
        }
        run_stage(d, s, u, p, n, io, source, target, d.channel_out);
    }
}

// Planes the device filters (the planes of active units).
std::vector<int> device_planes(const Driver& d) {
    std::vector<int> planes;
    for (const Unit& u : d.units) {
        if (!d.planes[u.first].active) continue;
        for (int c = 0; c < u.channels; ++c) planes.push_back(u.first + c);
    }
    return planes;
}

// Planes the frame takes from the device (the others are the source's).
bool written(const Driver& d, int plane) {
    for (const Unit& u : d.units) {
        if (plane >= u.first && plane < u.first + u.channels) return d.planes[u.first].active && d.channel_out[plane - u.first];
    }
    return false;
}

std::size_t row_bytes(const Driver& d, int plane) {
    return static_cast<std::size_t>(nss::plane_width(d.vi, plane)) * sizeof(float);
}

// Spatial and legacy output: one frame per call.
const VSFrame* frame_output(Driver* d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* vsapi) {
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const int nt = d->ntemp, radius = d->radius;
    std::vector<const VSFrame*> srcf(static_cast<std::size_t>(nt)), reff(static_cast<std::size_t>(nt), nullptr);
    for (int t = 0; t < nt; ++t) {
        const int fn = temporal_slot_frame(n, t, radius, d->vi.numFrames);
        srcf[static_cast<std::size_t>(t)] = frames.getFrameFilter(fn, d->node, ctx);
        if (guide_clip(*d)) reff[static_cast<std::size_t>(t)] = frames.getFrameFilter(fn, d->guide, ctx);
    }
    const VSFrame* src = srcf[static_cast<std::size_t>(radius)];
    const std::vector<int> planes = device_planes(*d);
    const bool legacy = d->mode == GroupMode::Legacy;
    const int slices = fat_slices(*d);
    // The slot, then the output permit, then the output frame (as NLM): the
    // frames in flight stay within what the plan counts.
    std::optional<SlotPool<Driver::OutputPermit>::Lease> permit;
    VSFrame* dst = nullptr;
    {
        auto lease = d->pool->acquire();
        permit.emplace(d->outputs->acquire());
        if (legacy) {
            dst = frames.newVideoFrame(&d->vi_out.format, d->vi_out.width, d->vi_out.height, src, core);
        } else {
            const VSFrame* from[3]{};
            const int from_plane[3]{0, 1, 2};
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) from[plane] = written(*d, plane) ? nullptr : src;
            dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, from, from_plane, src, core);
        }
        nss::stamp_contribution(dst, radius, n, d->model, vsapi);
        GroupSlot& s = *lease;
        Slot& slot = s.slot;
        // Uploads: a plane of an imported frame is copied from it, the others
        // are written into BAR memory or staging by the host.
        const auto put = [&](const VSFrame* frame, int plane, int region) {
            const std::size_t bytes = row_bytes(*d, plane);
            const int height = nss::plane_height(d->vi, plane);
            const std::size_t offset = d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float);
            const std::uint8_t* data = vsapi->getReadPtr(frame, plane);
            const std::ptrdiff_t stride = vsapi->getStride(frame, plane);
            // The upload stage only where there are copies (BAR writes need none).
            if (const HostPlane imported = d->frame_memory->pinned(frame, plane, data, stride, bytes)) {
                slot.retain(frame, vsapi);
                s.inputs.record_upload(slot.commands(Slot::Upload), region, offset, bytes * height, imported.buffer,
                                       imported.offset);
                return;
            }
            s.inputs.write(region, offset, data, stride, bytes, height, bytes);
            if (s.inputs.needs_copy()) s.inputs.record_upload(slot.commands(Slot::Upload), region, offset, bytes * height);
        };
        for (int t = 0; t < nt; ++t) {
            for (const int plane : planes) {
                put(srcf[static_cast<std::size_t>(t)], plane, t * clips(*d));
                if (guide_clip(*d)) put(reff[static_cast<std::size_t>(t)], plane, t * clips(*d) + 1);
            }
        }
        run_units(*d, s, n);
        // Downloads: into the frame itself where it is imported, else into
        // the readback buffer. A legacy plane is its 2R+1 num / den slice
        // pairs one after the other, as in the fat frame.
        VkCommandBuffer download = slot.commands(Slot::Download);
        std::array<bool, 3> direct{};
        bool retained = false;
        for (const int plane : planes) {
            if (!written(*d, plane)) continue;
            const std::size_t bytes = row_bytes(*d, plane) * nss::plane_height(d->vi, plane) * slices;
            const std::size_t offset = d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float) * slices;
            const HostPlane imported = d->frame_memory->pinned(dst, plane, vsapi->getWritePtr(dst, plane),
                                                               vsapi->getStride(dst, plane), row_bytes(*d, plane));
            if (imported) {
                direct[static_cast<std::size_t>(plane)] = true;
                if (!retained) slot.retain(dst, vsapi);
                retained = true;
            }
            copy_buffer(*d->device, download, s.out.get(), offset, imported ? imported.buffer : s.readback.get(),
                        imported ? imported.offset : offset, bytes);
        }
        memory_barrier(*d->device, download, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        slot.submit();
        slot.wait();
        s.readback.invalidate();
        for (const int plane : planes) {
            if (!written(*d, plane) || direct[static_cast<std::size_t>(plane)]) continue;
            const std::size_t bytes = row_bytes(*d, plane);
            copy_rows(s.readback.as<std::uint8_t>() + d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float) * slices,
                      static_cast<std::ptrdiff_t>(bytes), vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane),
                      bytes, nss::plane_height(d->vi, plane) * slices);
        }
    }
    // Legacy planes the device leaves alone: identity slices.
    for (int plane = 0; legacy && plane < d->vi.format.numPlanes; ++plane) {
        if (written(*d, plane)) continue;
        nss::host_detail::temporal_identity(
            reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane)),
            static_cast<int>(vsapi->getStride(dst, plane) / static_cast<std::ptrdiff_t>(sizeof(float))),
            reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane)),
            static_cast<int>(vsapi->getStride(src, plane) / static_cast<std::ptrdiff_t>(sizeof(float))),
            nss::plane_width(d->vi, plane), nss::plane_height(d->vi, plane), radius);
    }
    for (int t = 0; t < nt; ++t) {
        frames.freeFrame(srcf[static_cast<std::size_t>(t)]);
        if (reff[static_cast<std::size_t>(t)]) frames.freeFrame(reff[static_cast<std::size_t>(t)]);
    }
    return frames.keep(dst);
}

// ---------------------------------------------------------------------------
// Rolling output (as nss_cuda's group driver).

// The work of a chunk, in the order the device runs it: its centers, and
// every chunk frame as soon as its last center has run (the source ring still
// holds it then, and its slice of the sums is free for a later frame). With
// two stages the first is brought far enough before each center f of the
// second: its estimates up to frame f + R, each of which needs its centers up
// to R further (and those their source frames another R on).
// v.center(center, next): a center of the (second) stage; frames before
// `next` are finished or not this chunk's, so what it adds to them is
// dropped. v.output(frame). v.basic_center(center, next) and
// v.basic_frame(frame): the first stage's centers and finished estimates. A
// chunk that carries on resumes where the one before it stopped, which
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

// One unit of one chunk on a rolling slot: planned (and given its turn) at
// construction, run between entering and leaving. A chunk that follows the
// last one planned for a state carries on from it; any other starts afresh.
// If a chunk fails, those that were planned to carry on from it fail too.
constexpr std::uint64_t kCarryBacklog = 3;  // chunks waiting on a slot before another slot is preferred

struct RollTurn {
    Driver& d;
    RollSlot* slot = nullptr;
    RollState* state = nullptr;
    bool carry = false;
    // The uploads wait for the kernels before them: a chunk that starts
    // afresh rewrites the ring, and one that follows such a chunk may evict
    // frames it reads (src_ring holds two chunks that carry on).
    bool wait_upload = false;
    std::vector<int> uploads;  // frames to stage, in upload order
    std::uint64_t ticket = 0, chain = 0;
    bool entered = false, left = false;

    // The caller holds d.roll_mu: the units of a chunk take their tickets in one step.
    RollTurn(Driver& driver, int plane, int start, int count, const std::lock_guard<std::mutex>&) : d(driver) {
        const int radius = d.radius, nframes = d.vi.numFrames;
        const std::size_t index = static_cast<std::size_t>(d.roll_state[static_cast<std::size_t>(plane)]);
        const auto backlog = [&](const RollSlot* s) {
            std::uint64_t n = 0;
            for (const RollState& r : s->roll) n += r.issued - r.serving;
            return n;
        };
        RollSlot* idle = d.roll_slots.front().get();
        for (const auto& owned : d.roll_slots) {
            RollSlot* s = owned.get();
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
        std::vector<int> promised = carry ? state->promised : std::vector<int>(static_cast<std::size_t>(ring), -1);
        uploads.reserve(static_cast<std::size_t>(chunk_window(d)));
        // The source frames come with the centers that read them first: the
        // first stage's with two stages.
        const auto window = [&](int center) {
            for (int t = 0; t < d.ntemp; ++t) {
                const int fn = temporal_slot_frame(center, t, radius, nframes);
                if (promised[static_cast<std::size_t>(fn % ring)] != fn) {
                    uploads.push_back(fn);
                    promised[static_cast<std::size_t>(fn % ring)] = fn;
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
        wait_upload = !carry || state->fresh_last;
        state->fresh_last = !carry;
        if (!carry) ++state->chain;
        ticket = state->issued++;
        chain = state->chain;
        state->plane = plane;
        state->next_start = start + count;
    }
    RollTurn(const RollTurn&) = delete;
    RollTurn& operator=(const RollTurn&) = delete;

    // Enters the turns of one slot together (their units go in one
    // submission): once each is its state's next, no other chunk is being
    // recorded on the slot, and its next lane is free. Returns that lane
    // (in use until release()) and the kernels it must follow.
    static RollLane& enter(const std::vector<RollTurn*>& turns, std::pair<VkSemaphore, std::uint64_t>& computed) {
        Driver& d = turns.front()->d;
        RollSlot* slot = turns.front()->slot;
        std::unique_lock lock(d.roll_mu);
        d.roll_cv.wait(lock, [&] {
            if (slot->recording || slot->lanes[static_cast<std::size_t>(slot->next_lane)]->in_use) return false;
            for (const RollTurn* turn : turns) {
                if (turn->state->serving != turn->ticket) return false;
            }
            return true;
        });
        RollLane& lane = *slot->lanes[static_cast<std::size_t>(slot->next_lane)];
        slot->next_lane ^= 1;
        slot->recording = true;
        lane.in_use = true;
        computed = slot->computed;
        for (RollTurn* turn : turns) turn->entered = true;
        for (const RollTurn* turn : turns) {
            if (turn->state->failed_chain == turn->chain) {
                slot->recording = false;
                lane.in_use = false;
                lock.unlock();
                d.roll_cv.notify_all();
                throw std::runtime_error(prefix(d) + "an earlier chunk of this run failed");
            }
        }
        return lane;
    }
    // After the submission (`submitted`: its kernels) or a failure (null):
    // the slot takes its next chunk; the lane stays in use until its wait.
    static void release(Driver& d, RollSlot& slot, RollLane& lane, const Slot* submitted) {
        {
            std::lock_guard lock(d.roll_mu);
            if (slot.recording) {
                slot.recording = false;
                if (submitted) slot.computed = submitted->done(Slot::Compute);
            }
            if (!submitted) lane.in_use = false;
        }
        d.roll_cv.notify_all();
    }
    static void free_lane(Driver& d, RollLane& lane) {
        {
            std::lock_guard lock(d.roll_mu);
            lane.in_use = false;
        }
        d.roll_cv.notify_all();
    }
    void leave() { finish(false); }
    // A wait that failed after the turn was left: the chunks planned to carry
    // on from this one fail too.
    void fail_after_leave() {
        {
            std::lock_guard lock(d.roll_mu);
            state->failed_chain = chain;
            if (state->chain == chain) state->next_start = -1;
        }
        d.roll_cv.notify_all();
    }
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
            ++state->serving;
            left = true;
        }
        d.roll_cv.notify_all();
    }
};

// Per upload of a unit: the imported planes (index clip * channels + c),
// empty where staged.
using RollDirect = std::array<HostPlane, 6>;

// The uploads of one unit of a chunk, recorded into `commands` (the lane's
// upload stage): every frame its plan stages, from the staging set (base + i
// * step) or the imported frame, into the ring buffers (the guide clip's
// too).
void record_uploads(Driver& d, VkCommandBuffer commands, RollState& r, const RollTurn& turn, const Unit& u,
                    const RollStaging& staging, const std::vector<RollDirect>& direct) {
    const int ring = src_ring(d), channels = u.channels, nclips = clips(d);
    const std::size_t first = static_cast<std::size_t>(u.first);
    const std::size_t plane_bytes = d.planes[u.first].floats * sizeof(float);
    if (!turn.carry) std::fill(r.resident.begin(), r.resident.end(), -1);
    for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
        const int fn = turn.uploads[i];
        const auto put = [&](int clip, int c, const Buffer& device) {
            const VkDeviceSize to = static_cast<VkDeviceSize>(c) * plane_bytes;
            if (const HostPlane& h = direct[i][static_cast<std::size_t>(clip * channels + c)]) {
                copy_buffer(*d.device, commands, h.buffer, h.offset, device.get(), to, plane_bytes);
                return;
            }
            const std::size_t at = staging.base[first] + ((i * nclips + clip) * channels + c) * staging.step[first];
            copy_buffer(*d.device, commands, staging.bytes.get(), at, device.get(), to, plane_bytes);
        };
        for (int c = 0; c < channels; ++c) {
            put(0, c, r.src[static_cast<std::size_t>(fn % ring)]);
            if (nclips == 2) put(1, c, r.ref[static_cast<std::size_t>(fn % ring)]);
        }
        r.resident[static_cast<std::size_t>(fn % ring)] = fn;
    }
}

// One unit of a rolling chunk on its slot, as walk_chunk visits it, recorded
// into the lane's compute commands: the state's sums (a unit's channels one
// after the other, 2R+1 ring cells each, the first stage's too), the window
// tables (words for the upload stage) and the chunk's finished planes.
struct RollRun {
    Driver& d;
    RollSlot& s;
    RollLane& lane;
    RollState& r;
    const Unit& u;
    const GroupPlane& p;
    const GroupPlane& b;
    const ChunkStore& store;
    VkCommandBuffer commands;
    std::vector<std::uint64_t>& tables;
    int start;

    int nt() const { return d.ntemp; }
    int ring() const { return src_ring(d); }
    // The second clip's buffer of frame fn: the guide clip's ring, or the estimates'.
    const Buffer& ref(int fn) const {
        return r.ref[static_cast<std::size_t>(fn % (guide_clip(d) ? ring() : nt()))];
    }
    std::size_t ring_floats() const { return static_cast<std::size_t>(d.ntemp) * p.floats; }
    std::size_t plane_bytes() const { return p.floats * sizeof(float); }
    FixedTarget target(const Buffer& num, const Buffer& den) const {
        FixedTarget fixed{};
        fixed.num = num.address();
        fixed.den = den.address();
        fixed.pitch = p.width;
        fixed.slice_step = static_cast<std::int64_t>(p.floats);
        fixed.channel_step = static_cast<std::int64_t>(ring_floats());
        fixed.slice_ring = d.ntemp;
        return fixed;
    }
    // Logical slice i is frame next + i, in ring cell (next + i) % (2R+1).
    FixedTarget slices(FixedTarget fixed, int center, int next) const {
        fixed.slices = std::min(d.ntemp, d.vi.numFrames - next);
        fixed.slice_first = next % d.ntemp;
        fixed.slice_base = center - d.radius - next;
        return fixed;
    }
    VkDeviceSize cell(int c, int index) const {
        return (static_cast<std::size_t>(c) * ring_floats() + static_cast<std::size_t>(index) * p.floats) * 8;
    }

    // A slice is zero before its frame's first center: every one when the
    // chunk starts afresh, and again once its frame is finished.
    void clear_basic(int index) {
        for (int c = 0; c < u.channels && basic_active(d, u); ++c) {
            if (!d.basic_out[c]) continue;
            fixed_clear(*d.device, commands, r.basic_num.get(), cell(c, index), p.floats * 8);
            fixed_clear(*d.device, commands, r.basic_den.get(), cell(c, index), p.floats * 8);
        }
    }
    void clear_sums(int index) {
        for (int c = 0; c < u.channels; ++c) {
            if (d.shared_match && !d.channel_out[c]) continue;
            fixed_clear(*d.device, commands, r.num.get(), cell(c, index), p.floats * 8);
            fixed_clear(*d.device, commands, r.den.get(), cell(c, index), p.floats * 8);
        }
    }

    // The window tables of `center` (its source frames, and the guide's or
    // the estimates' with `guide`), written by the upload stage.
    StageIO point(int center, bool guide) {
        if (tables.size() + 2 * static_cast<std::size_t>(nt()) > d.roll_table_words) {
            throw std::logic_error(prefix(d) + "rolling window tables overflow");
        }
        StageIO io;
        io.src = lane.tables.address() + tables.size() * sizeof(std::uint64_t);
        for (int t = 0; t < nt(); ++t) {
            const int fn = temporal_slot_frame(center, t, d.radius, d.vi.numFrames);
            tables.push_back(r.src[static_cast<std::size_t>(fn % ring())].address());
        }
        if (guide) {
            io.guide = lane.tables.address() + tables.size() * sizeof(std::uint64_t);
            for (int t = 0; t < nt(); ++t) {
                tables.push_back(ref(temporal_slot_frame(center, t, d.radius, d.vi.numFrames)).address());
            }
        }
        return io;
    }
    // The window frames of `center` are on the device (record_uploads put
    // them there, or a chunk before this one did).
    void check(int center) const {
        for (int t = 0; t < nt(); ++t) {
            const int fn = temporal_slot_frame(center, t, d.radius, d.vi.numFrames);
            if (r.resident[static_cast<std::size_t>(fn % ring())] != fn) {
                throw std::logic_error(prefix(d) + "rolling plan and device state disagree");
            }
        }
    }
    BatchBuffers batches() const { return BatchBuffers{s.matches, s.counts, s.scratch}; }

    void basic_center(int center, int next) {
        check(center);
        if (!b.active) return;
        StageIO io = point(center, false);
        io.match = io.src;
        filter_center(d, commands, batches(), u, b, center, io, slices(target(r.basic_num, r.basic_den), center, next),
                      d.basic_out);
    }
    void basic_frame(int frame) {
        bool copied = false;
        const Buffer& source = r.src[static_cast<std::size_t>(frame % ring())];
        const Buffer& estimate = ref(frame);
        for (int c = 0; c < u.channels; ++c) {
            const VkDeviceSize channel = static_cast<VkDeviceSize>(c) * plane_bytes();
            if (!b.active || !d.basic_out[c]) {
                // A channel the first stage leaves alone: its estimate is the source.
                if (!copied) before_copies(*d.device, commands);
                copied = true;
                copy_buffer(*d.device, commands, source.get(), channel, estimate.get(), channel, plane_bytes());
                continue;
            }
            const VkDeviceSize at = cell(c, frame % nt());
            d.aggregator->fixed_finish(commands, r.basic_num.address() + at, r.basic_den.address() + at, p.width, b.block,
                                       source.address() + channel, p.width, p.height, s.rows.address(),
                                       estimate.address() + channel);
            compute_barrier(*d.device, commands);
        }
        if (copied) after_copies(*d.device, commands);
        clear_basic(frame % nt());  // the cell is the next frame's from here on
    }
    void center(int center, int next) {
        if (!d.two_stage) check(center);
        StageIO io = point(center, p.wiener);
        io.match = p.wiener ? io.guide : io.src;
        filter_center(d, commands, batches(), u, p, center, io, slices(target(r.num, r.den), center, next),
                      d.channel_out);
    }
    void output(int frame) {
        const Buffer& source = r.src[static_cast<std::size_t>(frame % ring())];
        for (int c = 0; c < u.channels; ++c) {
            if (!d.channel_out[c]) continue;
            const VkDeviceSize channel = static_cast<VkDeviceSize>(c) * plane_bytes();
            // The chunk's planes wait in s.out at the store's layout and are
            // downloaded together after the kernels; without room for that,
            // s.out is one unit, copied into the store after each finish.
            const std::size_t plane = static_cast<std::size_t>(u.first + c);
            const VkDeviceSize in_store = static_cast<VkDeviceSize>(frame - start) * frame_bytes(d) +
                                          d.plane_offset[plane] * sizeof(float);
            const Buffer& out = d.roll_batch_out ? lane.out : s.out;
            const VkDeviceSize result = d.roll_batch_out ? in_store : channel;
            const VkDeviceSize at = cell(c, frame % nt());
            d.aggregator->fixed_finish(commands, r.num.address() + at, r.den.address() + at, p.width, p.block,
                                       source.address() + channel, p.width, p.height, s.rows.address(),
                                       out.address() + result);
            compute_barrier(*d.device, commands);
            if (!d.roll_batch_out) {
                before_copies(*d.device, commands);
                copy_buffer(*d.device, commands, s.out.get(), result, store.bytes.get(), in_store, plane_bytes());
                after_copies(*d.device, commands);  // the next finish overwrites s.out
            }
        }
        clear_sums(frame % nt());  // the slice is a later frame's from here on
    }
};

// The units of a chunk that run on one slot, in one submission of a lane:
// entered together; the uploads (upload stage), the units' centers and
// finishes in walk_chunk's order (compute stage, after the slot's last
// kernels) and the chunk's planes into the store (download stage); the turns
// are left once it is submitted, so that the next chunk records on the other
// lane while this one runs, and the lane is free again after its wait.
void queue_units(Driver& d, const std::vector<RollTurn*>& turns, const std::vector<const Unit*>& units,
                 const ChunkStore& store, const RollStaging& staging,
                 const std::vector<const std::vector<RollDirect>*>& direct, const std::vector<const VSFrame*>& held,
                 int start, int count, const VSAPI* vsapi) {
    std::pair<VkSemaphore, std::uint64_t> computed{};
    RollLane& lane = RollTurn::enter(turns, computed);
    RollSlot& s = *turns.front()->slot;
    // On an error before the submission, the lane back to idle (anything
    // submitted is waited for) and the slot to its next chunk.
    struct Settle {
        Driver& d;
        RollSlot& s;
        RollLane& lane;
        bool armed = true;
        ~Settle() {
            if (!armed) return;
            lane.slot.settle();
            RollTurn::release(d, s, lane, nullptr);
        }
    } settle{d, s, lane};
    bool wait_upload = false;
    for (const RollTurn* turn : turns) wait_upload = wait_upload || turn->wait_upload;
    if (wait_upload) lane.slot.wait_for(Slot::Upload, computed.first, computed.second);
    for (std::size_t j = 0; j < turns.size(); ++j) {
        if (turns[j]->uploads.empty()) continue;
        record_uploads(d, lane.slot.commands(Slot::Upload), *turns[j]->state, *turns[j], *units[j], staging, *direct[j]);
    }
    lane.slot.wait_for(Slot::Compute, computed.first, computed.second);
    VkCommandBuffer commands = lane.slot.commands(Slot::Compute);
    std::vector<std::uint64_t> tables;
    for (std::size_t j = 0; j < turns.size(); ++j) {
        const Unit& u = *units[j];
        RollState& r = *turns[j]->state;
        RollRun run{d, s, lane, r, u, d.planes[u.first], d.basic[u.first], store, commands, tables, start};
        if (!turns[j]->carry) {
            for (int index = 0; index < d.ntemp; ++index) {
                run.clear_basic(index);
                run.clear_sums(index);
            }
        }
        walk_chunk(d, start, count, turns[j]->carry, run);
    }
    if (!tables.empty()) {
        // The window tables, before the kernels (the stages chain).
        VkCommandBuffer upload = lane.slot.commands(Slot::Upload);
        const std::size_t bytes = tables.size() * sizeof(std::uint64_t);
        for (std::size_t at = 0; at < bytes; at += 65536) {
            d.device->vk().vkCmdUpdateBuffer(upload, lane.tables.get(), at, std::min<std::size_t>(65536, bytes - at),
                                             reinterpret_cast<const std::uint8_t*>(tables.data()) + at);
        }
    }
    if (d.roll_batch_out) {
        VkCommandBuffer download = lane.slot.commands(Slot::Download);
        for (const Unit* u : units) {
            for (int i = 0; i < count; ++i) {
                for (int c = 0; c < u->channels; ++c) {
                    if (!d.channel_out[c]) continue;
                    const std::size_t plane = static_cast<std::size_t>(u->first + c);
                    const VkDeviceSize at =
                        static_cast<VkDeviceSize>(i) * frame_bytes(d) + d.plane_offset[plane] * sizeof(float);
                    copy_buffer(*d.device, download, lane.out.get(), at, store.bytes.get(), at,
                                d.planes[u->first].floats * sizeof(float));
                }
            }
        }
        memory_barrier(*d.device, download, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    } else {
        memory_barrier(*d.device, commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    }
    // The imported frames the copies read stay referenced until the work is done.
    for (const VSFrame* f : held) lane.slot.retain(f, vsapi);
    lane.slot.submit();
    settle.armed = false;
    RollTurn::release(d, s, lane, &lane.slot);
    for (RollTurn* turn : turns) turn->leave();
    try {
        lane.slot.wait();
    } catch (...) {
        for (RollTurn* turn : turns) turn->fail_after_leave();
        RollTurn::free_lane(d, lane);
        throw;
    }
    RollTurn::free_lane(d, lane);
}

// Rolling: compute the chunk [start, start + count). The frames it reads are
// staged before a slot is taken, as for nss_cuda. `staging` takes the
// staging set, which the caller keeps until its frame is copied out (see
// Driver::outputs).
std::shared_ptr<ChunkStore> compute_chunk(Driver* d, std::optional<SlotPool<RollStaging>::Lease>& staging, int start,
                                          int count, VSFrameContext* ctx, const VSAPI* vsapi) {
    d->order.wait_for_previous(start, d->rolling.rolling_chunk, d->cache);
    staging.emplace(d->staging->acquire());
    RollStaging& g = **staging;
    nss::FrameScope frames(vsapi);
    auto store = d->cache.make_store(start, count);
    // Planes the device does not write are the source's.
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        if (written(*d, plane)) continue;
        for (int i = 0; i < count; ++i) {
            const VSFrame* f = frames.getFrameFilter(start + i, d->node, ctx);
            copy_rows(vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane), store->plane(i, plane),
                      static_cast<std::ptrdiff_t>(row_bytes(*d, plane)), row_bytes(*d, plane),
                      nss::plane_height(d->vi, plane));
            frames.freeFrame(f);
        }
    }
    const std::vector<const Unit*> active = active_units(*d);
    // Units with staging and state of their own go together: all staged, then
    // queued (one submission per slot they land on). Otherwise each unit is
    // staged and queued in turn.
    const std::size_t step = d->roll_together ? active.size() : 1;
    const int nclips = clips(*d);
    for (std::size_t first = 0; first < active.size(); first += step) {
        // The plans say which frames the device will not hold: only those are staged.
        std::vector<std::unique_ptr<RollTurn>> turns;
        turns.reserve(step);
        {
            std::lock_guard lock(d->roll_mu);
            for (std::size_t i = 0; i < step; ++i) {
                turns.push_back(std::make_unique<RollTurn>(*d, active[first + i]->first, start, count, lock));
            }
        }
        if (first == 0) d->order.planned(start, d->rolling.rolling_chunk);
        // Planes imported for direct I/O are copied from their frames, which
        // are held until the work is done.
        std::vector<std::vector<RollDirect>> direct(step);
        std::vector<const VSFrame*> held;
        struct Held {
            nss::FrameScope& frames;
            std::vector<const VSFrame*>& held;
            ~Held() {
                for (const VSFrame* f : held) frames.freeFrame(f);
            }
        } release{frames, held};
        for (std::size_t j = 0; j < step; ++j) {
            const Unit& u = *active[first + j];
            const std::size_t bytes = row_bytes(*d, u.first);
            const int height = nss::plane_height(d->vi, u.first);
            const std::size_t base = g.base[static_cast<std::size_t>(u.first)];
            const std::size_t region = g.step[static_cast<std::size_t>(u.first)];
            const RollTurn& turn = *turns[j];
            direct[j].assign(turn.uploads.size(), RollDirect{});
            const std::size_t channels = static_cast<std::size_t>(u.channels);
            const auto stage = [&](std::size_t i, int clip, const VSFrame* f) {
                bool used = false;
                for (std::size_t c = 0; c < channels; ++c) {
                    const int plane = u.first + static_cast<int>(c);
                    const std::uint8_t* data = vsapi->getReadPtr(f, plane);
                    const std::ptrdiff_t stride = vsapi->getStride(f, plane);
                    if (const HostPlane h = d->frame_memory->pinned(f, plane, data, stride, bytes)) {
                        direct[j][i][static_cast<std::size_t>(clip) * channels + c] = h;
                        used = true;
                        continue;
                    }
                    const std::size_t at = base + ((i * nclips + clip) * channels + c) * region;
                    copy_rows(data, stride, g.bytes.as<std::uint8_t>() + at, static_cast<std::ptrdiff_t>(bytes), bytes,
                              height);
                    g.bytes.flush(at, bytes * height);
                }
                if (used) {
                    held.push_back(f);
                } else {
                    frames.freeFrame(f);
                }
            };
            for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
                const int fn = turn.uploads[i];
                stage(i, 0, frames.getFrameFilter(fn, d->node, ctx));
                if (guide_clip(*d)) stage(i, 1, frames.getFrameFilter(fn, d->guide, ctx));
            }
        }
        // One submission per slot the units landed on, in their order.
        std::vector<bool> queued(step, false);
        for (std::size_t j = 0; j < step; ++j) {
            if (queued[j]) continue;
            std::vector<RollTurn*> group;
            std::vector<const Unit*> units;
            std::vector<const std::vector<RollDirect>*> planes;
            for (std::size_t k = j; k < step; ++k) {
                if (queued[k] || turns[k]->slot != turns[j]->slot) continue;
                queued[k] = true;
                group.push_back(turns[k].get());
                units.push_back(active[first + k]);
                planes.push_back(&direct[k]);
            }
            queue_units(*d, group, units, *store, g, planes, held, start, count, vsapi);
        }
    }
    store->bytes.invalidate();
    return store;
}

const VSFrame* rolling_output(Driver* d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* vsapi) {
    const int start = n / d->rolling.rolling_chunk * d->rolling.rolling_chunk;
    const int count = std::min(d->rolling.rolling_chunk, d->vi.numFrames - start);
    // The chunk, then the permit for the output frame (see Driver::outputs);
    // a chunk this request computes keeps its staging set until the frame is out.
    std::optional<SlotPool<RollStaging>::Lease> staging;
    auto chunk = d->cache.get(start, [&] { return compute_chunk(d, staging, start, count, ctx, vsapi); });
    auto permit = d->outputs->acquire();
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    nss::stamp_contribution(dst, 0, n, d->model, vsapi);
    frames.freeFrame(src);
    // Out of the chunk store with non-temporal stores (the frame is not in cache).
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const std::size_t bytes = row_bytes(*d, plane);
        stream_rows(chunk->plane(n - chunk->start, plane), static_cast<std::ptrdiff_t>(bytes),
                    vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), bytes, nss::plane_height(d->vi, plane));
    }
    return frames.keep(dst);
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
        } else if (rolling(*d)) {
            // Always the full chunk dependency window: a cache hit seen now
            // may be evicted before arAllFramesReady.
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
    return rolling(*d) ? rolling_output(d, n, ctx, core, vsapi) : frame_output(d, n, ctx, core, vsapi);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    delete static_cast<Driver*>(instance);
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
    if (d->channels > 1) {
        if (!d->shared_match) throw std::logic_error(prefix(*d) + "joint units are not ported yet (V7)");
        bool any = false;
        for (int c = 0; c < d->channels; ++c) {
            d->channel_out[c] = d->planes[c].active;
            any = any || d->planes[c].active;
            if (c > 0) d->planes[c].active = false;  // planes[0] carries the unit
        }
        d->planes[0].active = any;
        d->units.push_back(Unit{0, d->channels});
        d->unit_channels = d->channels;
    } else {
        for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) d->units.push_back(Unit{plane, 1});
    }
    if (d->two_stage) {
        if (d->mode == GroupMode::Legacy) throw std::logic_error(prefix(*d) + "two stages need spatial or rolling output");
        if (d->guide) throw std::logic_error(prefix(*d) + "two stages take no guide clip");
        for (int c = 0; c < 3; ++c) {
            GroupPlane& b = d->basic[c];
            if (d->channels > 1) {
                // The estimate of channel c is needed where the second stage
                // filters it, and of channel 0 for the matching.
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
    d->frame_memory = std::make_unique<FrameMemory>(d->device, core, vsapi);
    size_planes(*d);
    if (rolling(*d)) {
        plan_rolling(*d);
    } else {
        plan_frames(*d);
    }
    d->matcher = std::make_unique<Matcher>(d->device);
    d->aggregator = std::make_unique<Aggregator>(d->device);
    for (const GroupPlane* p : stage_planes(*d)) {
        MatchGeometry g{p->width, p->height, p->width, p->block, p->range, p->group};
        g.channels = d->shared_match ? 1 : d->channels;
        g.channel_step = static_cast<long long>(p->floats);
        const int ps_num = d->radius > 0 ? p->ps_num : 0;
        if (!d->matcher->supports(g, ps_num)) throw std::logic_error(prefix(*d) + "no matcher for this search");
        d->matcher->prepare(g, d->radius > 0, ps_num);
        if (d->prepare) d->prepare(*p, p->wiener);
    }
    if (rolling(*d)) {
        for (std::size_t i = 0; i < d->slots; ++i) d->roll_slots.push_back(make_roll_slot(*d));
        std::vector<std::unique_ptr<RollStaging>> sets;
        for (int i = 0; i < d->staging_sets; ++i) sets.push_back(make_staging(*d));
        d->staging = std::make_unique<SlotPool<RollStaging>>(std::move(sets));
        std::array<std::size_t, 3> offsets{};
        for (int plane = 0; plane < 3; ++plane) offsets[plane] = d->plane_offset[plane] * sizeof(float);
        d->cache.configure(d->rolling.rolling_chunk, frame_bytes(*d), offsets, d->cache_size, d->cache_cap,
                           [device = d->device, budget = d->budget](std::size_t bytes) {
                               return Buffer(device, bytes, Memory::Readback, budget);
                           });
    } else {
        std::vector<std::unique_ptr<GroupSlot>> slots;
        for (std::size_t i = 0; i < d->slots; ++i) slots.push_back(make_slot(*d));
        d->pool = std::make_unique<SlotPool<GroupSlot>>(std::move(slots));
    }
    std::vector<std::unique_ptr<Driver::OutputPermit>> permits;
    for (std::size_t i = 0; i < d->permits; ++i) permits.push_back(std::make_unique<Driver::OutputPermit>());
    d->outputs = std::make_unique<SlotPool<Driver::OutputPermit>>(std::move(permits));
    d->device->save_pipeline_cache();

    const auto pattern = d->mode == GroupMode::Spatial ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->guide, pattern}};
    Driver* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2(raw->name.c_str(), &raw->vi_out, nss::checked_frame<getFrame>, freeFilter,
                                            fmParallel, deps, d->guide ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error(prefix(*d) + "failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace nss_vk
