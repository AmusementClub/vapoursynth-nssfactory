// SPDX-License-Identifier: GPL-2.0-only
// Shared host driver for the patch-group filters (see group_driver.hpp).
#include "vulkan/common/group_driver.hpp"

#include "frontend/contribution.hpp"
#include "frontend/validate.hpp"
#include "nss/checked.hpp"
#include "vulkan/runtime/frame_memory.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_vk {
namespace {

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

// A unit: one plane, or the three planes of a shared_match filter.
struct Unit {
    int first = 0;
    int channels = 1;
};

// A frame in flight. The planes have the frame's layout (plane p at
// Driver::plane_offset[p] floats); the sums hold one unit at a time.
struct GroupSlot {
    explicit GroupSlot(std::shared_ptr<Device> device) : slot(std::move(device)) {}
    Slot slot;
    // Region 0: the source frame; region 1: the guide clip's frame.
    PlaneStore inputs;
    Buffer estimate;  // two stages: the first stage's estimate (the second's guide)
    // Per unit two device addresses: its source and its guide (window slot 0;
    // a spatial window has one frame).
    Buffer tables;
    Buffer num, den;  // fixed-point sums of the unit being filtered (channel c at c * plane floats)
    Buffer rows;      // fixed_finish scratch, one plane
    Buffer out;       // the finished planes
    Buffer readback;  // what the copies did not write into the frame
    Buffer matches, counts, scratch;
    void settle() noexcept { slot.settle(); }
};

struct Driver : GroupFilterConfig {
    VSVideoInfo vi_out{};
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
    // Last: destroyed first, so that its imports go before the device can.
    std::unique_ptr<FrameMemory> frame_memory;
};

std::string prefix(const Driver& d) { return "nss_vk." + d.name + ": "; }
bool guide_clip(const Driver& d) { return d.guide != nullptr; }
bool has_guide(const Driver& d) { return d.guide != nullptr || d.two_stage; }

// The active planes of every stage: they share the slot's per-batch buffers.
std::vector<GroupPlane*> stage_planes(Driver& d) {
    std::vector<GroupPlane*> planes;
    for (const Unit& u : d.units) {
        if (d.planes[u.first].active) planes.push_back(&d.planes[u.first]);
        if (d.two_stage && d.basic[u.first].active) planes.push_back(&d.basic[u.first]);
    }
    return planes;
}

std::size_t scratch_floats(const Driver& d, const GroupPlane& p) {
    return d.scratch_floats ? d.scratch_floats(p, p.wiener) : 0;
}

std::size_t per_ref_bytes(const Driver& d, const GroupPlane& p) {
    return scratch_floats(d, p) * sizeof(float) + sizeof(int) + static_cast<std::size_t>(p.group) * sizeof(DeviceMatch);
}

// Bytes of the per-batch buffers for the planes' current batches: each is
// sized for the plane that needs it most.
std::size_t batch_bytes(Driver& d) {
    std::size_t matches = 0, counts = 0, scratch = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        const std::size_t batch = static_cast<std::size_t>(p->batch);
        matches = std::max(matches, batch * p->group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, *p) * sizeof(float));
    }
    return matches + counts + scratch;
}

void size_planes(Driver& d) {
    std::size_t floats = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const int width = nss::plane_width(d.vi, plane), height = nss::plane_height(d.vi, plane);
        d.plane_offset[plane] = floats;
        floats += static_cast<std::size_t>(width) * height;
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(width) * height);
        const std::size_t row = (static_cast<std::size_t>(width) * sizeof(float) + 63) & ~std::size_t{63};
        d.output_bytes += row * static_cast<std::size_t>(height);
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

// What a slot holds whatever its batch.
std::size_t slot_fixed_bytes(const Driver& d) {
    const std::size_t frame_bytes = d.frame_floats * sizeof(float);
    // The plan assumes BAR uploads on a ReBAR device; without one the
    // inputs have a staging mirror.
    const std::size_t inputs = frame_bytes * (guide_clip(d) ? 2 : 1) * (d.device->rebar() ? 1 : 2);
    const std::size_t cells = d.plane_floats * static_cast<std::size_t>(d.unit_channels);
    return inputs + (d.two_stage ? frame_bytes : 0) + cells * 8 * 2 + d.plane_floats * 8 + frame_bytes * 2 + 4096;
}

// The memory plan: num_streams (unless given), the slots and output permits
// in flight, and each plane's batch of references, within memory_limit_mb.
void plan(Driver& d) {
    size_planes(d);
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const std::size_t fixed = slot_fixed_bytes(d);
    std::size_t min_batch = 0, full_batch = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        const std::size_t refs = static_cast<std::size_t>(p->grid.count());
        min_batch = std::max(min_batch, std::min(kBatchBytes, std::min<std::size_t>(refs, 1024) * per_ref_bytes(d, *p)));
        full_batch = std::max(full_batch, std::min(kBatchBytes, refs * per_ref_bytes(d, *p)));
    }
    // A slot and the output frame of its request.
    const std::size_t need = fixed + min_batch + d.output_bytes;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams =
            static_cast<int>(std::clamp<std::size_t>(limit / (fixed + full_batch + d.output_bytes), 1, kDefaultStreams));
    }
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    if (limit / streams < need) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(streams) + " stream(s) needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB");
    }
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
    const std::size_t share =
        limit == SIZE_MAX ? SIZE_MAX : (limit - extra_outputs * d.output_bytes) / d.slots - fixed - d.output_bytes;
    const std::size_t budget = std::min(kBatchBytes, share);
    // NSS_VK_GROUP_BATCH (tests): at most that many references a batch.
    const char* cap = std::getenv("NSS_VK_GROUP_BATCH");
    const std::size_t most = cap && std::atoi(cap) > 0 ? static_cast<std::size_t>(std::atoi(cap)) : SIZE_MAX;
    for (GroupPlane* p : stage_planes(d)) {
        const std::size_t fit = std::max<std::size_t>(1, budget / per_ref_bytes(d, *p));
        p->batch = static_cast<int>(std::min({fit, most, static_cast<std::size_t>(p->grid.count())}));
    }
    for (int round = 0; round < 8 && batch_bytes(d) > budget; ++round) {
        const double scale = static_cast<double>(budget) / static_cast<double>(batch_bytes(d));
        bool changed = false;
        for (GroupPlane* p : stage_planes(d)) {
            const int next = std::max(1, static_cast<int>(p->batch * scale));
            changed = changed || next != p->batch;
            p->batch = next;
        }
        if (!changed) break;
    }
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

std::unique_ptr<GroupSlot> make_slot(Driver& d) {
    auto s = std::make_unique<GroupSlot>(d.device);
    const std::size_t frame_bytes = d.frame_floats * sizeof(float);
    s->inputs = PlaneStore(d.device, frame_bytes, guide_clip(d) ? 2 : 1, d.budget);
    if (d.two_stage) s->estimate = Buffer(d.device, frame_bytes, Memory::Device, d.budget);
    const std::size_t cells = d.plane_floats * static_cast<std::size_t>(d.unit_channels);
    s->num = Buffer(d.device, cells * 8, Memory::Device, d.budget);
    s->den = Buffer(d.device, cells * 8, Memory::Device, d.budget);
    s->rows = Buffer(d.device, d.plane_floats * 8, Memory::Device, d.budget);
    s->out = Buffer(d.device, frame_bytes, Memory::Device, d.budget);
    s->readback = Buffer(d.device, frame_bytes, Memory::Readback, d.budget);
    std::size_t matches = 0, counts = 0, scratch = 0;
    for (const GroupPlane* p : stage_planes(d)) {
        const std::size_t batch = static_cast<std::size_t>(p->batch);
        matches = std::max(matches, batch * p->group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        scratch = std::max(scratch, batch * scratch_floats(d, *p) * sizeof(float));
    }
    s->matches = Buffer(d.device, std::max<std::size_t>(matches, 16), Memory::Device, d.budget);
    s->counts = Buffer(d.device, std::max<std::size_t>(counts, 4), Memory::Device, d.budget);
    if (scratch) s->scratch = Buffer(d.device, scratch, Memory::Device, d.budget);
    // The tables never change: a unit's source and guide.
    std::vector<VkDeviceAddress> tables;
    for (const Unit& u : d.units) {
        const VkDeviceAddress offset = d.plane_offset[static_cast<std::size_t>(u.first)] * sizeof(float);
        tables.push_back(s->inputs.address() + offset);
        tables.push_back(guide_clip(d)  ? s->inputs.address() + frame_bytes + offset
                         : d.two_stage ? s->estimate.address() + offset
                                       : 0);
    }
    s->tables = Buffer(d.device, std::max<std::size_t>(tables.size(), 1) * sizeof(VkDeviceAddress), Memory::Device, d.budget);
    upload_now(d.device, s->tables, tables.data(), tables.size() * sizeof(VkDeviceAddress));
    return s;
}

void transfer_barrier(const Device& device, VkCommandBuffer commands, VkPipelineStageFlags2 src_stage,
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

// One stage of a unit: its groups matched (on `match_unit`, channel 0) and
// filtered into the slot's sums, which then are finished into `target` for
// the channels `write` names (the others are left alone).
void run_stage(Driver& d, GroupSlot& s, int unit_index, const GroupPlane& p, VkDeviceAddress match_unit,
               VkDeviceAddress guide_table, VkDeviceAddress source_unit, VkDeviceAddress target_unit,
               const bool* write) {
    const Unit& u = d.units[static_cast<std::size_t>(unit_index)];
    VkCommandBuffer commands = s.slot.commands(Slot::Compute);
    // The sums hold one unit; the clear waits for the previous unit's finish.
    const std::size_t cells = p.floats * static_cast<std::size_t>(u.channels);
    fixed_clear(*d.device, commands, s.num.get(), 0, cells * 8);
    fixed_clear(*d.device, commands, s.den.get(), 0, cells * 8);
    MatchGeometry geometry{p.width, p.height, p.width, p.block, p.range, p.group};
    geometry.channels = d.shared_match ? 1 : u.channels;
    geometry.channel_step = static_cast<long long>(p.floats);
    FixedTarget fixed{};
    fixed.num = s.num.address();
    fixed.den = s.den.address();
    fixed.pitch = p.width;
    fixed.slice_step = static_cast<std::int64_t>(p.floats);
    fixed.channel_step = static_cast<std::int64_t>(p.floats);
    for (int begin = 0; begin < p.grid.count(); begin += p.batch) {
        const int count = std::min(p.batch, p.grid.count() - begin);
        d.matcher->spatial(commands, match_unit, geometry, p.grid, begin, count, s.matches.address(), s.counts.address());
        compute_barrier(*d.device, commands);
        GroupLaunch launch{};
        launch.commands = commands;
        launch.src = s.tables.address() + static_cast<VkDeviceAddress>(unit_index) * 2 * sizeof(VkDeviceAddress);
        launch.guide = p.wiener ? guide_table : 0;
        launch.pitch = p.width;
        launch.matches = s.matches.address();
        launch.counts = s.counts.address();
        launch.batch = count;
        launch.plane = &p;
        launch.channels = u.channels;
        launch.channel_step = geometry.channel_step;
        launch.scratch = s.scratch ? s.scratch.address() : 0;
        launch.channel_active = write;
        launch.fused = fixed;
        d.launch(launch);
        // The next batch's matches and scratch reuse the buffers.
        compute_barrier(*d.device, commands);
    }
    for (int c = 0; c < u.channels; ++c) {
        if (!write[c]) continue;
        const VkDeviceAddress channel = static_cast<VkDeviceAddress>(c) * p.floats;
        d.aggregator->fixed_finish(commands, fixed.num + channel * 8, fixed.den + channel * 8, p.width, p.block,
                                   source_unit + channel * sizeof(float), p.width, p.height, s.rows.address(),
                                   target_unit + channel * sizeof(float));
        compute_barrier(*d.device, commands);  // the rows are reused, and the target read next
    }
}

// Every unit of a frame, on the slot's compute queue.
void run_units(Driver& d, GroupSlot& s) {
    VkCommandBuffer commands = s.slot.commands(Slot::Compute);
    const std::size_t frame_bytes = d.frame_floats * sizeof(float);
    for (std::size_t i = 0; i < d.units.size(); ++i) {
        const Unit& u = d.units[i];
        const GroupPlane& p = d.planes[u.first];
        if (!p.active) continue;
        const VkDeviceAddress offset = d.plane_offset[static_cast<std::size_t>(u.first)] * sizeof(float);
        const VkDeviceAddress source = s.inputs.address() + offset;
        const VkDeviceAddress guide_table =
            s.tables.address() + (static_cast<VkDeviceAddress>(i) * 2 + 1) * sizeof(VkDeviceAddress);
        VkDeviceAddress match = source;
        if (guide_clip(d)) match = s.inputs.address() + frame_bytes + offset;
        if (d.two_stage) {
            // The first stage's estimate of every channel: filtered, or the source.
            const GroupPlane& b = d.basic[u.first];
            const VkDeviceAddress estimate = s.estimate.address() + offset;
            bool copy = false;
            for (int c = 0; c < u.channels; ++c) {
                if (b.active && d.basic_out[c]) continue;
                const VkBufferCopy region{offset + static_cast<VkDeviceAddress>(c) * p.floats * sizeof(float),
                                          offset + static_cast<VkDeviceAddress>(c) * p.floats * sizeof(float),
                                          p.floats * sizeof(float)};
                d.device->vk().vkCmdCopyBuffer(commands, s.inputs.buffer(), s.estimate.get(), 1, &region);
                copy = true;
            }
            if (copy) {
                transfer_barrier(*d.device, commands, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            }
            if (b.active) run_stage(d, s, static_cast<int>(i), b, source, 0, source, estimate, d.basic_out);
            match = estimate;
        }
        run_stage(d, s, static_cast<int>(i), p, match, guide_table, source, s.out.address() + offset, d.channel_out);
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

const VSFrame* frame_output(Driver* d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* vsapi) {
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    const VSFrame* ref = guide_clip(*d) ? frames.getFrameFilter(n, d->guide, ctx) : nullptr;
    const std::vector<int> planes = device_planes(*d);
    // The slot, then the output permit, then the output frame (as NLM): the
    // frames in flight stay within what the plan counts.
    std::optional<SlotPool<Driver::OutputPermit>::Lease> permit;
    VSFrame* dst = nullptr;
    {
        auto lease = d->pool->acquire();
        permit.emplace(d->outputs->acquire());
        const VSFrame* from[3]{};
        const int from_plane[3]{0, 1, 2};
        for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) from[plane] = written(*d, plane) ? nullptr : src;
        dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, from, from_plane, src, core);
        nss::stamp_contribution(dst, 0, n, d->model, vsapi);
        GroupSlot& s = *lease;
        Slot& slot = s.slot;
        // Uploads: a plane of an imported frame is copied from it, the others
        // are written into BAR memory or staging by the host.
        const auto put = [&](const VSFrame* frame, int plane, int region) {
            const std::size_t row_bytes = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
            const int height = nss::plane_height(d->vi, plane);
            const std::size_t offset = d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float);
            const std::uint8_t* data = vsapi->getReadPtr(frame, plane);
            const std::ptrdiff_t stride = vsapi->getStride(frame, plane);
            // The upload stage only where there are copies (BAR writes need none).
            if (const HostPlane imported = d->frame_memory->pinned(frame, plane, data, stride, row_bytes)) {
                slot.retain(frame, vsapi);
                s.inputs.record_upload(slot.commands(Slot::Upload), region, offset, row_bytes * height, imported.buffer,
                                       imported.offset);
                return;
            }
            s.inputs.write(region, offset, data, stride, row_bytes, height, row_bytes);
            if (s.inputs.needs_copy()) s.inputs.record_upload(slot.commands(Slot::Upload), region, offset, row_bytes * height);
        };
        for (const int plane : planes) {
            put(src, plane, 0);
            if (ref) put(ref, plane, 1);
        }
        run_units(*d, s);
        // Downloads: into the frame itself where it is imported, else into
        // the readback buffer.
        VkCommandBuffer download = slot.commands(Slot::Download);
        std::array<bool, 3> direct{};
        bool retained = false;
        for (const int plane : planes) {
            if (!written(*d, plane)) continue;
            const std::size_t row_bytes = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
            const int height = nss::plane_height(d->vi, plane);
            const std::size_t offset = d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float);
            const HostPlane imported = d->frame_memory->pinned(dst, plane, vsapi->getWritePtr(dst, plane),
                                                               vsapi->getStride(dst, plane), row_bytes);
            const VkBufferCopy region{offset, imported ? imported.offset : offset, row_bytes * height};
            if (imported) {
                direct[static_cast<std::size_t>(plane)] = true;
                if (!retained) slot.retain(dst, vsapi);
                retained = true;
            }
            d->device->vk().vkCmdCopyBuffer(download, s.out.get(), imported ? imported.buffer : s.readback.get(), 1,
                                            &region);
        }
        transfer_barrier(*d->device, download, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        slot.submit();
        slot.wait();
        s.readback.invalidate();
        for (const int plane : planes) {
            if (!written(*d, plane) || direct[static_cast<std::size_t>(plane)]) continue;
            const std::size_t row_bytes = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
            copy_rows(s.readback.as<std::uint8_t>() + d->plane_offset[static_cast<std::size_t>(plane)] * sizeof(float),
                      static_cast<std::ptrdiff_t>(row_bytes), vsapi->getWritePtr(dst, plane),
                      vsapi->getStride(dst, plane), row_bytes, nss::plane_height(d->vi, plane));
        }
    }
    frames.freeFrame(src);
    if (ref) frames.freeFrame(ref);
    return frames.keep(dst);
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Driver*>(instance);
    if (activation == arInitial) {
        vsapi->requestFrameFilter(n, d->node, ctx);
        if (d->guide) vsapi->requestFrameFilter(n, d->guide, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    return frame_output(d, n, ctx, core, vsapi);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    delete static_cast<Driver*>(instance);
}

}  // namespace

void group_filter_install(GroupFilterConfig&& config, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<Driver>();
    static_cast<GroupFilterConfig&>(*d) = std::move(config);
    d->vi_out = d->vi;
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
    plan(*d);
    d->matcher = std::make_unique<Matcher>(d->device);
    d->aggregator = std::make_unique<Aggregator>(d->device);
    for (const GroupPlane* p : stage_planes(*d)) {
        MatchGeometry g{p->width, p->height, p->width, p->block, p->range, p->group};
        g.channels = d->shared_match ? 1 : d->channels;
        g.channel_step = static_cast<long long>(p->floats);
        if (!d->matcher->supports(g, 0)) throw std::logic_error(prefix(*d) + "no matcher for this search");
        d->matcher->prepare(g, false, 0);
        if (d->prepare) d->prepare(*p, p->wiener);
    }
    std::vector<std::unique_ptr<GroupSlot>> slots;
    for (std::size_t i = 0; i < d->slots; ++i) slots.push_back(make_slot(*d));
    d->pool = std::make_unique<SlotPool<GroupSlot>>(std::move(slots));
    std::vector<std::unique_ptr<Driver::OutputPermit>> permits;
    for (std::size_t i = 0; i < d->permits; ++i) permits.push_back(std::make_unique<Driver::OutputPermit>());
    d->outputs = std::make_unique<SlotPool<Driver::OutputPermit>>(std::move(permits));
    d->device->save_pipeline_cache();

    VSFilterDependency deps[2]{{d->node, rpStrictSpatial}, {d->guide, rpStrictSpatial}};
    Driver* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2(raw->name.c_str(), &raw->vi_out, nss::checked_frame<getFrame>, freeFilter,
                                            fmParallel, deps, d->guide ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error(prefix(*d) + "failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace nss_vk
