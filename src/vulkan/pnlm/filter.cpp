// SPDX-License-Identifier: GPL-2.0-only
// nss_vk.PNLM (docs/vulkan-plan.md V15): parse_pnlm with the "nss_vk" prefix
// around the kernels of shaders/pnlm (model in include/nss/params/pnlm.hpp,
// the CPU kernel's and nss_cuda's arithmetic). Each plane with a sigma is
// filtered on its own; a plane with sigma 0 is the source's (shared, not
// copied).
//
// The frames a request searches (the distinct frames n - d .. n + d) stay on
// the device in a cache shared by the slots (common/window_cache.hpp, as
// NLM's tile path): a frame is uploaded once however many windows it is part
// of. A frame's filtered planes lie side by side in its region. Per plane the
// compute stage writes the plane's parameters (the window's frame addresses,
// the grid, the matcher's lattice passes), matches (shaders/pnlm/match.comp:
// keys, or the weighted estimates for small windows), weights the keys
// (weight.comp) and gathers the output (gather.comp). CPU frames are uploaded
// through BAR memory or staging and read back, or copied directly under
// vapoursynth-classic (FrameMemory); GPU frames (VapourSynth R80+, D29) are
// copied on VapourSynth's device (GpuFrames).
#include "frontend/ownership.hpp"
#include "frontend/pnlm_args.hpp"
#include "frontend/validate.hpp"
#include "vulkan/common/window_cache.hpp"
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/frame_memory.hpp"
#include "vulkan/runtime/gpu_frames.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/pipeline.hpp"
#include "vulkan/runtime/slot.hpp"
#include "vulkan/runtime/timing.hpp"
#include "vulkan/runtime/vs_api.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_vk {
namespace {

// Frames slots and requests hold beyond num_streams (as NLM's tile path): a
// request writes its uploads before it takes a slot, and holds an output
// permit until its frame is returned.
constexpr std::size_t kExtraSlots = 2;
constexpr std::size_t kExtraOutputs = 1;
constexpr int kSide = 16, kRefs = kSide - 1;
constexpr int kMaxFrames = 64;  // shaders/pnlm/common.glsl Params::frames (2 kPnlmMaxD + 1 = 63)
static_assert(2 * nss::kPnlmMaxD + 1 <= kMaxFrames);

// shaders/pnlm/common.glsl Params (std430: no padding in this order).
struct ParamsBlock {
    std::uint64_t frames[kMaxFrames];
    std::int32_t count, center, width, height, pitch, a, group, nx, ny, passes;
    std::int32_t pass[16];
    float inv_area, offset, inv_h2, inv_s2;
};
static_assert(sizeof(ParamsBlock) == 632, "matches Params in shaders/pnlm/common.glsl");
constexpr std::size_t kParamsStride = 640;  // a plane's block in the slot's params buffer

struct MatchPush {
    VkDeviceAddress params, keys, counts, est;
};
struct WeightPush {
    VkDeviceAddress params, keys, counts, est;
    std::int32_t refs, pad;
};
struct GatherPush {
    VkDeviceAddress est, out;
    std::int32_t nx, ny, width, height, pitch, pad;
};

struct PnlmSlot {
    Slot slot;
    Buffer params;    // kParamsStride a filtered plane
    Buffer est;       // the estimates of a plane's references
    Buffer keys;      // the key path: keys, then counts (empty when fused)
    Buffer out;       // the filtered planes at the frame layout
    Buffer readback;  // CPU frames: what the copies did not write into the frame
    explicit PnlmSlot(std::shared_ptr<Device> device) : slot(std::move(device)) {}
    void settle() noexcept { slot.settle(); }
};

struct PnlmData : nss::PnlmParams {
    nss::NodeRef node;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    std::shared_ptr<Device> device;
    int planes[3] = {};     // the filtered planes
    int count = 0;
    std::size_t offset[4] = {};  // their offsets in a frame, floats
    std::size_t frame_floats = 0;
    bool fused = false;
    std::size_t keys_bytes = 0;  // of the largest plane (0: fused)
    Pipeline match, weight, gather;
    WindowCache window;
    std::unique_ptr<SlotPool<PnlmSlot>> pool;
    struct OutputPermit {
        void settle() noexcept {}
    };
    std::unique_ptr<SlotPool<OutputPermit>> outputs;
    TimingStats timing;
    std::unique_ptr<GpuFrames> gpu;
    // Last: destroyed first, so that its imports go before the device can.
    std::unique_ptr<FrameMemory> frame_memory;
};

int width_of(const PnlmData& d, int c) { return nss::plane_width(d.vi, d.planes[c]); }
int height_of(const PnlmData& d, int c) { return nss::plane_height(d.vi, d.planes[c]); }
std::size_t row_bytes(const PnlmData& d, int c) { return static_cast<std::size_t>(width_of(d, c)) * sizeof(float); }
std::size_t frame_bytes(const PnlmData& d) { return std::max<std::size_t>(d.frame_floats, 1) * sizeof(float); }

// The square matcher's lattice passes for a plane (nss_cuda's launch_match):
// the lattice from 0, and lattices shifted so that a clamped last column or
// row falls on them. Returns the workgroups.
int lattice_passes(ParamsBlock& b, int q) {
    const int p = 2 * q, max_x = b.width - p, max_y = b.height - p;
    const bool clamp_x = max_x % q != 0, clamp_y = max_y % q != 0;
    const int lattice_cols = clamp_x ? b.nx - 1 : b.nx, lattice_rows = clamp_y ? b.ny - 1 : b.ny;
    const int blocks_x = (lattice_cols + kRefs - 1) / kRefs, bands = (lattice_rows + kRefs - 1) / kRefs;
    const int shift_x = max_x - q * (kRefs - 1), shift_y = max_y - q * (kRefs - 1);
    int blocks = 0;
    b.passes = 0;
    const auto add = [&](int ox, int oy, int bx, int nbands) {
        if (bx <= 0 || nbands <= 0) return;
        int* pass = b.pass + 4 * b.passes++;
        pass[0] = ox;
        pass[1] = oy;
        pass[2] = bx;
        pass[3] = blocks;
        blocks += bx * nbands;
    };
    if (blocks_x > 0) add(0, 0, blocks_x, bands);
    if (clamp_x) add(shift_x, 0, 1, bands);
    if (clamp_y) {
        if (blocks_x > 0) add(0, shift_y, blocks_x, 1);
        if (clamp_x) add(shift_x, shift_y, 1, 1);
    }
    return blocks;
}

// A plane the upload stage copies into a store region: from an imported frame
// (`source`), a GPU frame (`gpu`), else from the store's staging mirror.
struct PlaneUpload {
    int region = 0;
    int c = 0;  // filtered plane
    HostPlane source;
    const VSFrame* frame = nullptr;
    bool gpu = false;
};

void write_frame(PnlmData& d, const WindowFrame& f, const VSFrame* frame, const VSAPI* vsapi,
                 std::vector<PlaneUpload>& uploads) {
    for (int c = 0; c < d.count; ++c) {
        const int plane = d.planes[c];
        if (d.gpu) {
            uploads.push_back({f.region, c, {}, frame, true});
            continue;
        }
        const std::uint8_t* data = vsapi->getReadPtr(frame, plane);
        const std::ptrdiff_t stride = vsapi->getStride(frame, plane);
        if (const HostPlane source = d.frame_memory->pinned(frame, plane, data, stride, row_bytes(d, c))) {
            uploads.push_back({f.region, c, source, frame, false});
            continue;
        }
        d.window.store.write(f.region, d.offset[c] * sizeof(float), data, stride, row_bytes(d, c), height_of(d, c),
                             row_bytes(d, c));
        if (d.window.store.needs_copy()) uploads.push_back({f.region, c, {}, nullptr, false});
    }
}

void record_uploads(const PnlmData& d, const std::vector<PlaneUpload>& uploads, Slot& slot, const VSAPI* vsapi) {
    const PlaneStore& store = d.window.store;
    VkCommandBuffer commands = slot.commands(Slot::Upload);
    for (const PlaneUpload& u : uploads) {
        const std::size_t offset = d.offset[u.c] * sizeof(float);
        const std::size_t bytes = row_bytes(d, u.c) * static_cast<std::size_t>(height_of(d, u.c));
        if (u.gpu) {
            d.gpu->read(slot, Slot::Upload, u.frame, d.planes[u.c], store.buffer(), store.address(),
                        store.region_bytes() * static_cast<std::size_t>(u.region) + offset, row_bytes(d, u.c),
                        height_of(d, u.c));
        } else if (u.source) {
            slot.retain(u.frame, vsapi);
            store.record_upload(commands, u.region, offset, bytes, u.source.buffer, u.source.offset);
        } else {
            store.record_upload(commands, u.region, offset, bytes);
        }
    }
}

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
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    std::vector<const VSFrame*> sources;
    for (int i = first; i <= last; ++i) sources.push_back(frames.getFrameFilter(i, d->node, ctx));
    const int count = last - first + 1, center = n - first;
    const VSFrame* src = sources[static_cast<std::size_t>(center)];

    // The pins outlive the lease: returning it settles the slot, and only
    // then may other requests reuse the cache frames this frame reads.
    Window window(d->window, n, d->d, d->vi.numFrames);
    // Window slot t holds frame n - d + t (frames clamped to the clip repeat).
    const auto frame_of = [&](int i) -> WindowFrame& { return window.slot(i - n + d->d); };
    // The host writes need no slot; nor does waiting for the frames other
    // requests upload (see NLM's tile_frame).
    std::vector<PlaneUpload> uploads;
    for (WindowFrame* f : window.uploads()) write_frame(*d, *f, sources[static_cast<std::size_t>(f->n - first)], vsapi, uploads);
    const bool staged = !uploads.empty();
    if (!staged) window.mark_queued(VK_NULL_HANDLE, 0);
    window.wait_queued();
    auto lease = d->pool->acquire();
    std::optional<SlotPool<PnlmData::OutputPermit>::Lease> permit;
    permit.emplace(d->outputs->acquire());
    PnlmSlot& s = *lease;
    Slot& slot = s.slot;
    FrameTimer timer(slot);

    // Planes with sigma 0 are the source's.
    VSFrame* dst = nullptr;
    {
        const VSFrame* from[3]{};
        const int from_plane[3]{0, 1, 2};
        bool shared = false;
        for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
            from[plane] = d->sigma[plane] == 0.f ? src : nullptr;
            shared = shared || from[plane];
        }
        if (shared) {
            dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, from, from_plane, src, core);
        } else if (d->gpu) {
            dst = d->gpu->new_frame(frames, &d->vi.format, d->vi.width, d->vi.height, src, core);
        } else {
            dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
        }
    }
    if (staged) {
        for (const auto& [semaphore, value] : window.previous_uses()) slot.wait_for(Slot::Upload, semaphore, value);
        record_uploads(*d, uploads, slot, vsapi);
    }
    for (WindowFrame* f : window.pinned()) {
        if (!window.uploads(f)) slot.wait_for(Slot::Compute, f->ready, f->ready_value);
    }

    // Every plane's parameters, then its kernels.
    VkCommandBuffer commands = slot.commands(Slot::Compute);
    const int q = d->block / 2;
    std::array<int, 3> blocks{};
    std::array<ParamsBlock, 3> params{};
    for (int c = 0; c < d->count; ++c) {
        ParamsBlock& b = params[static_cast<std::size_t>(c)];
        for (int i = first; i <= last; ++i) {
            b.frames[i - first] = d->window.store.address() +
                                  d->window.store.region_bytes() * static_cast<std::size_t>(frame_of(i).region) +
                                  d->offset[c] * sizeof(float);
        }
        b.count = count;
        b.center = center;
        b.width = width_of(*d, c);
        b.height = height_of(*d, c);
        b.pitch = b.width;
        b.a = d->a;
        b.group = d->group;
        b.nx = nss::pnlm_grid_count(b.width, d->block);
        b.ny = nss::pnlm_grid_count(b.height, d->block);
        const nss::PnlmScale scale = nss::pnlm_scale(d->sigma[d->planes[c]], d->block);
        b.inv_area = scale.inv_area;
        b.offset = scale.offset;
        b.inv_h2 = scale.inv_h2;
        b.inv_s2 = scale.inv_s2;
        blocks[static_cast<std::size_t>(c)] = lattice_passes(b, q);
        d->device->vk().vkCmdUpdateBuffer(commands, s.params.get(), kParamsStride * static_cast<std::size_t>(c),
                                          sizeof(ParamsBlock), &b);
    }
    {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        d->device->vk().vkCmdPipelineBarrier2(commands, &dependency);
    }
    for (int c = 0; c < d->count; ++c) {
        const ParamsBlock& b = params[static_cast<std::size_t>(c)];
        const VkDeviceAddress block = s.params.address() + kParamsStride * static_cast<std::size_t>(c);
        const VkDeviceAddress keys = s.keys ? s.keys.address() : 0;
        const std::size_t refs = static_cast<std::size_t>(b.nx) * static_cast<std::size_t>(b.ny);
        const VkDeviceAddress counts = s.keys ? keys + refs * static_cast<std::size_t>(d->group) * sizeof(std::uint64_t) : 0;
        if (blocks[static_cast<std::size_t>(c)] > 0) {
            const MatchPush match{block, keys, counts, s.est.address()};
            d->match.dispatch(commands, &match, static_cast<std::uint32_t>(blocks[static_cast<std::size_t>(c)]));
            compute_barrier(*d->device, commands);
            if (d->count == 1) timer.match_end();
            if (!d->fused) {
                const int per = 256 / nss::pnlm_lanes(d->block * d->block);
                const WeightPush weight{block, keys, counts, s.est.address(), static_cast<std::int32_t>(refs), 0};
                d->weight.dispatch(commands, &weight, static_cast<std::uint32_t>((refs + per - 1) / per));
                compute_barrier(*d->device, commands);
            }
        }
        const GatherPush gather{s.est.address(), s.out.address() + d->offset[c] * sizeof(float), b.nx, b.ny,
                                b.width, b.height, b.pitch, 0};
        d->gather.dispatch(commands, &gather, static_cast<std::uint32_t>((b.width + 31) / 32),
                           static_cast<std::uint32_t>((b.height + 7) / 8));
        // The next plane's matcher overwrites the estimates.
        compute_barrier(*d->device, commands);
        if (d->count == 1) timer.filter_end();
    }

    // Results: into the GPU frame or the imported frame, else the readback buffer.
    timer.download_begin();
    std::array<bool, 3> direct{};
    if (d->gpu) {
        for (int c = 0; c < d->count; ++c) {
            d->gpu->write(slot, Slot::Download, s.out.get(), s.out.address(), d->offset[c] * sizeof(float), dst,
                          d->planes[c], row_bytes(*d, c), height_of(*d, c));
        }
    } else {
        VkCommandBuffer download = slot.commands(Slot::Download);
        bool retained = false;
        for (int c = 0; c < d->count; ++c) {
            const int plane = d->planes[c];
            const std::size_t at = d->offset[c] * sizeof(float);
            const HostPlane imported = d->frame_memory->pinned(dst, plane, vsapi->getWritePtr(dst, plane),
                                                               vsapi->getStride(dst, plane), row_bytes(*d, c));
            if (imported && !retained) {
                slot.retain(dst, vsapi);
                retained = true;
            }
            direct[static_cast<std::size_t>(c)] = static_cast<bool>(imported);
            const VkBufferCopy region{at, imported ? imported.offset : at,
                                      row_bytes(*d, c) * static_cast<std::size_t>(height_of(*d, c))};
            d->device->vk().vkCmdCopyBuffer(download, s.out.get(), imported ? imported.buffer : s.readback.get(), 1,
                                            &region);
        }
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        d->device->vk().vkCmdPipelineBarrier2(download, &dependency);
    }
    timer.download_end();
    timer.upload_end();
    slot.submit();
    if (d->gpu) {
        for (int c = 0; c < d->count; ++c) d->gpu->publish(slot, dst, d->planes[c]);
    }
    if (staged) {
        const auto uploaded = slot.done(Slot::Upload);
        window.mark_queued(uploaded.first, uploaded.second);
    }
    window.read_by(slot.done(Slot::Compute));
    timer.wait();
    d->timing.add(timer.values());
    if (!d->gpu) {
        s.readback.invalidate();
        for (int c = 0; c < d->count; ++c) {
            if (direct[static_cast<std::size_t>(c)]) continue;
            const int plane = d->planes[c];
            copy_rows(s.readback.as<std::uint8_t>() + d->offset[c] * sizeof(float),
                      static_cast<std::ptrdiff_t>(row_bytes(*d, c)), vsapi->getWritePtr(dst, plane),
                      vsapi->getStride(dst, plane), row_bytes(*d, c), height_of(*d, c));
        }
    }
    for (const VSFrame* f : sources) frames.freeFrame(f);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<PnlmData*>(instance);
    if (timing_enabled()) d->timing.print("nss_vk.PNLM");
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.PNLM.
    auto d = std::make_unique<PnlmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::PnlmParams&>(*d) = nss::frontend::parse_pnlm(vsapi, in, d->vi, "nss_vk");
    d->backend = parse_backend_args(vsapi, in, "PNLM");
    const VSVULKANAPI* gpu_api = take_clips("PNLM", {{"clip", &d->node}}, core, vsapi);
    d->device = gpu_api ? acquire_core_device(d->backend, "PNLM", core, vsapi, gpu_api)
                        : acquire_device(d->backend.device_id, "PNLM", core, vsapi);
    d->budget = nss::current_budget();
    if (gpu_api) d->gpu = std::make_unique<GpuFrames>(d->device, vsapi);
    d->frame_memory = std::make_unique<FrameMemory>(d->device, core, vsapi);
    d->window.filter = "nss_vk.PNLM";

    std::size_t est = 0;
    d->fused = d->a <= 4 && d->group <= 16;  // nss_cuda's pnlm_fused
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        if (d->sigma[plane] == 0.f) continue;
        const int w = nss::plane_width(d->vi, plane), h = nss::plane_height(d->vi, plane);
        d->planes[d->count] = plane;
        d->offset[d->count + 1] = d->offset[d->count] + static_cast<std::size_t>(w) * h;
        ++d->count;
        const std::size_t refs =
            static_cast<std::size_t>(nss::pnlm_grid_count(w, d->block)) * nss::pnlm_grid_count(h, d->block);
        est = std::max(est, refs * static_cast<std::size_t>(d->block * d->block) * sizeof(float));
        if (!d->fused) {
            d->keys_bytes = std::max(d->keys_bytes, refs * static_cast<std::size_t>(d->group) * sizeof(std::uint64_t) +
                                                        refs * sizeof(std::int32_t));
        }
    }
    d->frame_floats = d->offset[d->count];

    // The plan (as NLM's tile path): a cache of the window and a frame for
    // each slot, num_streams slots and kExtraSlots more, kExtraOutputs output
    // frames beyond them; under a limit the extras go first, then cache frames.
    const bool staged = !d->gpu && !d->device->rebar();
    std::size_t output_bytes = 0;
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        if (d->sigma[plane] == 0.f) continue;  // shared with the source
        const std::size_t row = (static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float) + 63) & ~std::size_t{63};
        output_bytes += row * static_cast<std::size_t>(nss::plane_height(d->vi, plane));
    }
    const std::size_t window = std::min<std::size_t>(2 * static_cast<std::size_t>(d->d) + 1,
                                                     static_cast<std::size_t>(std::max(d->vi.numFrames, 1)));
    const std::size_t slot_bytes = kParamsStride * 3 + std::max<std::size_t>(est, 4) + d->keys_bytes +
                                   frame_bytes(*d) * (d->gpu ? 1 : 2);
    const std::size_t limit = d->budget ? d->budget->snapshot().limit : SIZE_MAX;
    // Without num_streams: three for d <= 1, one beyond (1080p gray, R78,
    // fps with one / two / three: defaults 690 / 736 / 809, YUV420 439 / 510 /
    // 527, RGB 228 / 247 / 267; d = 2, a = 2: 841 / 779 / 700).
    if (!d->backend.streams_explicit) d->backend.num_streams = d->d <= 1 ? 3 : 1;
    const std::size_t streams = static_cast<std::size_t>(d->backend.num_streams);
    std::size_t slots = streams + kExtraSlots, extra_outputs = kExtraOutputs, cached = window + slots;
    const auto plan_bytes = [&] {
        return frame_bytes(*d) * cached * (staged ? 2 : 1) + slot_bytes * slots + (slots + extra_outputs) * output_bytes;
    };
    if (plan_bytes() > limit) extra_outputs = 0;
    while (plan_bytes() > limit && slots > streams) cached = window + --slots;
    while (plan_bytes() > limit && cached > window) --cached;
    if (plan_bytes() > limit) {
        throw std::invalid_argument("nss_vk.PNLM: memory_limit_mb is too small for this clip: it needs at least " +
                                    std::to_string((plan_bytes() >> 20) + 1) + " MiB");
    }

    const int q = d->block / 2;
    const std::size_t lanes = static_cast<std::size_t>(nss::pnlm_lanes(d->block * d->block));
    // NSS_VK_FORCE_EMULATED_WARP=1 (diagnostics, tests): the emulated warp
    // layer on any device.
    const char* force = std::getenv("NSS_VK_FORCE_EMULATED_WARP");
    const bool emulate = d->device->support().native_lanes < lanes || (force && force[0] == '1');
    const std::uint32_t width = emulate ? 0
                                        : std::clamp<std::uint32_t>(32, d->device->caps().min_subgroup_size,
                                                                    d->device->caps().max_subgroup_size);
    const std::string layer = emulate ? "_e" : "_n";
    const std::string qs = "_q" + std::to_string(q);
    const int k = d->group <= 8 ? 7 : d->group <= 16 ? 15 : 31;
    if (d->fused) {
        d->match = Pipeline(d->device, {"pnlm_match" + qs + "_k" + std::to_string(k) + "_m2" + layer,
                                        sizeof(MatchPush), width, {}});
    } else {
        d->match = Pipeline(d->device, {"pnlm_match" + qs + "_k" + std::to_string(k) + "_m1", sizeof(MatchPush), 0, {}});
        d->weight = Pipeline(d->device, {"pnlm_weight" + qs + layer, sizeof(WeightPush), width, {}});
    }
    d->gather = Pipeline(d->device, {"pnlm_gather" + qs, sizeof(GatherPush), 0, {}});
    d->window.store = PlaneStore(d->device, frame_bytes(*d), static_cast<int>(cached), d->budget,
                                 d->gpu ? UploadPlane::Mode::device : UploadPlane::Mode::automatic);
    for (std::size_t i = 0; i < cached; ++i) {
        auto frame = std::make_unique<WindowFrame>();
        frame->region = static_cast<int>(i);
        d->window.frames.push_back(std::move(frame));
    }
    std::vector<std::unique_ptr<PnlmSlot>> leases;
    for (std::size_t i = 0; i < slots; ++i) {
        auto s = std::make_unique<PnlmSlot>(d->device);
        s->params = Buffer(d->device, kParamsStride * 3, Memory::Device, d->budget);
        s->est = Buffer(d->device, std::max<std::size_t>(est, 4), Memory::Device, d->budget);
        if (d->keys_bytes) s->keys = Buffer(d->device, d->keys_bytes, Memory::Device, d->budget);
        s->out = Buffer(d->device, frame_bytes(*d), Memory::Device, d->budget);
        if (!d->gpu) s->readback = Buffer(d->device, frame_bytes(*d), Memory::Readback, d->budget);
        leases.push_back(std::move(s));
    }
    d->pool = std::make_unique<SlotPool<PnlmSlot>>(std::move(leases));
    std::vector<std::unique_ptr<PnlmData::OutputPermit>> permits;
    for (std::size_t i = 0; i < slots + extra_outputs; ++i) permits.push_back(std::make_unique<PnlmData::OutputPermit>());
    d->outputs = std::make_unique<SlotPool<PnlmData::OutputPermit>>(std::move(permits));
    d->device->save_pipeline_cache();

    VSFilterDependency deps[1]{{d->node, d->d == 0 ? rpStrictSpatial : rpGeneral}};
    PnlmData* raw = d.get();
    VSNode* node = vs_api::create_video_filter(vsapi, "PNLM", &raw->vi, nss::checked_frame<getFrame>, freeFilter,
                                               fmParallel, raw->gpu != nullptr, deps, 1, raw, core);
    if (!node) throw std::runtime_error("nss_vk.PNLM: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_pnlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi, bool gpu_frames) {
    const std::string args = residency_args(signature(nss::frontend::kPnlmSignature), gpu_frames);
    vspapi->registerFunction("PNLM", args.c_str(), clip_return(gpu_frames), nss::checked_create<create, kDefaultMemoryLimitMb>,
                             nullptr, plugin);
}

}  // namespace nss_vk
