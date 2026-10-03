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

// A staging region plus an event marking when the device has finished with
// it, so host code never overwrites bytes that are still in flight.
struct StagingRegion {
    std::uint8_t* bytes = nullptr;
    cudaEvent_t done = nullptr;
};

struct Slot {
    Stream stream;
    // Window frames: 2R+1 (rolling uses them as a ring indexed by frame), 1 when spatial.
    std::vector<DeviceBuffer> src_frames, ref_frames, noisy_frames;
    std::vector<const float*> host_src_ptrs, host_ref_ptrs;
    DeviceBuffer src_ptrs, ref_ptrs;  // device arrays: window slot t -> plane pointer
    DeviceBuffer num, den;            // 2R+1 slices each
    DeviceBuffer out;                 // finished plane
    DeviceBuffer acc, chunk_src;      // rolling: num/den per chunk frame, chunk source planes
    DeviceBuffer matches, counts, values, scratch, patches;
    PinnedBuffer staging;
    std::vector<StagingRegion> regions;
    std::unique_ptr<OrderedAggregator> aggregator;
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() {
        for (auto& r : regions)
            if (r.done) cudaEventDestroy(r.done);
    }
};

struct ChunkStore {
    int start = 0, count = 0;
    // frames[i][plane]: final plane of frame start + i (tight rows).
    std::vector<std::array<nss::ResourceVector<float>, 3>> frames;
};

struct Driver : GroupFilterConfig {
    std::size_t plane_floats = 0;  // largest plane (tight pitch)
    int ntemp = 1;                 // 2R+1
    int regions = 0;               // staging regions per slot
    bool channel_out[3]{true, true, true};  // joint: channel c is written (else copied)
    std::unique_ptr<SlotPool<Slot>> pool;

    // Rolling chunk cache (LRU) and chunks being computed.
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

std::size_t per_ref_bytes(const Driver& d, const GroupPlane& p) {
    const std::size_t cube = static_cast<std::size_t>(d.channels) * p.group * p.block * p.block * sizeof(float);
    return cube + scratch_floats(d, p) * sizeof(float) + sizeof(int) +
           static_cast<std::size_t>(p.group) *
               (sizeof(DeviceMatch) + sizeof(AggregatePatch) + OrderedAggregator::kBytesPerPatch);
}

// Staging regions: one upload per window frame and channel (src, ref), then
// downloads (legacy: 2 * ntemp fat rows per channel; spatial: one per channel;
// rolling: one per chunk frame).
int upload_regions(const Driver& d) { return d.ntemp * d.channels * (has_guide(d) ? 2 : 1); }
int staging_regions(const Driver& d) {
    const int downloads = d.mode == GroupMode::Legacy    ? 2 * d.ntemp * d.channels
                          : d.mode == GroupMode::Rolling ? d.rolling.rolling_chunk
                                                         : d.channels;
    return upload_regions(d) + downloads;
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
    const int chunk = d.mode == GroupMode::Rolling ? d.rolling.rolling_chunk : 0;
    const std::size_t device_planes = static_cast<std::size_t>(upload_regions(d)) +
                                      static_cast<std::size_t>(d.channels) * (2 * d.ntemp + 1 + (d.iters > 1 ? d.ntemp : 0)) +
                                      3 * chunk;
    // Rolling also holds host chunk stores: one being built per slot, up to
    // cache_limit finished ones, and the output frames copied out of them.
    const std::size_t chunk_bytes = frame_bytes * chunk;
    const std::size_t output = d.mode == GroupMode::Rolling ? 2 * frame_bytes + chunk_bytes
                                                       : frame_bytes * (d.mode == GroupMode::Legacy ? 2 * d.ntemp : 1);
    const std::size_t fixed = plane_bytes * device_planes + plane_bytes * d.regions + output;
    const std::size_t shared_cache = d.mode == GroupMode::Rolling ? chunk_bytes * d.rolling.cache_limit : 0;
    std::size_t min_batch = 0;
    for (const GroupPlane& p : d.planes) {
        if (p.active) min_batch = std::max(min_batch, std::min<std::size_t>(p.grid.count(), 1024) * per_ref_bytes(d, p));
    }
    const std::size_t need = fixed + min_batch * 4 / 3;
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (total != SIZE_MAX && total <= shared_cache) {
        throw std::invalid_argument(prefix(d) + "memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - shared_cache;
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
                                    std::to_string((need >> 20) + 1) + " MiB");
    }
    const std::size_t share =
        limit == SIZE_MAX ? kBatchBytes + fixed : limit / static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t batch_bytes = std::min(kBatchBytes, (share - fixed) / 4 * 3);
    for (GroupPlane& p : d.planes) {
        if (!p.active) continue;
        const std::size_t fit = std::max<std::size_t>(1, batch_bytes / per_ref_bytes(d, p));
        p.batch = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(p.grid.count())));
    }
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
        max_patches = std::max(max_patches, batch * p.group);
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
        values = std::max(values, batch * d.channels * p.group * p.block * p.block * sizeof(float));
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        patches = std::max(patches, batch * p.group * sizeof(AggregatePatch));
        scratch = std::max(scratch, batch * scratch_floats(d, p) * sizeof(float));
    }
    for (int t = 0; t < d.ntemp; ++t) {
        slot->src_frames.emplace_back(unit_bytes, d.budget);
        slot->host_src_ptrs.push_back(slot->src_frames.back().as<float>());
        if (d.iters > 1) slot->noisy_frames.emplace_back(unit_bytes, d.budget);
        if (w) {
            slot->ref_frames.emplace_back(unit_bytes, d.budget);
            slot->host_ref_ptrs.push_back(slot->ref_frames.back().as<float>());
        }
    }
    slot->src_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    if (w) slot->ref_ptrs = DeviceBuffer(d.ntemp * sizeof(float*), d.budget);
    slot->num = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
    slot->den = DeviceBuffer(unit_bytes * d.ntemp, d.budget);
    slot->out = DeviceBuffer(unit_bytes, d.budget);
    if (d.mode == GroupMode::Rolling) {
        slot->acc = DeviceBuffer(plane_bytes * 2 * d.rolling.rolling_chunk, d.budget);
        slot->chunk_src = DeviceBuffer(plane_bytes * d.rolling.rolling_chunk, d.budget);
    }
    slot->matches = DeviceBuffer(matches, d.budget);
    slot->counts = DeviceBuffer(counts, d.budget);
    slot->values = DeviceBuffer(values, d.budget);
    if (scratch) slot->scratch = DeviceBuffer(scratch, d.budget);
    slot->patches = DeviceBuffer(patches, d.budget);
    slot->staging = PinnedBuffer(plane_bytes * d.regions, d.budget);
    for (int i = 0; i < d.regions; ++i) {
        StagingRegion region{slot->staging.as<std::uint8_t>() + plane_bytes * i, nullptr};
        NSS_CUDA_CHECK(cudaEventCreateWithFlags(&region.done, cudaEventDisableTiming));
        slot->regions.push_back(region);
    }
    slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, d.ntemp, max_patches, d.budget);
    // Spatial/legacy windows map slot t to buffer t for the slot's lifetime.
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

// Upload one VS plane into a device plane through staging region r.
void upload(Slot& s, int r, const VSFrame* frame, int plane, const GroupPlane& p, void* device, const VSAPI* vsapi) {
    StagingRegion& region = s.regions[r];
    NSS_CUDA_CHECK(cudaEventSynchronize(region.done));
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    upload_plane(vsapi->getReadPtr(frame, plane), vsapi->getStride(frame, plane), row_bytes, p.height, region.bytes,
                 device, row_bytes, s.stream);
    NSS_CUDA_CHECK(cudaEventRecord(region.done, s.stream));
}

// Queue a device plane into staging region r (complete after a stream sync).
void download(Slot& s, int r, const void* device, const GroupPlane& p) {
    StagingRegion& region = s.regions[r];
    NSS_CUDA_CHECK(cudaEventSynchronize(region.done));
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    begin_download(device, row_bytes, row_bytes, p.height, region.bytes, s.stream);
    NSS_CUDA_CHECK(cudaEventRecord(region.done, s.stream));
}

// All groups of one center into the slot's num/den slices (ntemp of them per
// channel, channel-major). The device pointer arrays map window slot t to the
// plane of frame temporal_slot_frame(center, t). Matching runs on the guide
// when match_guide is set, otherwise on the source/estimate frames.
void filter_center(const Driver& d, Slot& s, const GroupPlane& p, int center, bool match_guide) {
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
        launch.values = s.values.as<float>();
        launch.scratch = s.scratch.as<float>();
        launch.patches = s.patches.as<AggregatePatch>();
        launch.stream = stream;
        {
            NSS_CUDA_RANGE("group.filter");
            d.launch(launch);
        }
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
    // Lease the slot before allocating the output so at most num_streams
    // output frames are charged to memory_limit_mb at any time.
    auto slot = d->pool->acquire();
    Slot& s = *slot;
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
        int region = 0;
        for (int t = 0; t < d->ntemp; ++t) {
            for (int c = 0; c < channels; ++c) {
                upload(s, region++, srcf[t], unit + c, p, s.src_frames[t].as<float>() + c * p.floats, vsapi);
                if (w) upload(s, region++, reff[t], unit + c, p, s.ref_frames[t].as<float>() + c * p.floats, vsapi);
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
        // Download region of row k of channel c.
        const int rows = radius == 0 ? 1 : 2 * d->ntemp;
        const auto slot_region = [&](int c, int k) { return downloads + c * rows + k; };
        for (int c = 0; c < channels; ++c) {
            if (!d->channel_out[c]) continue;
            if (radius == 0) {
                const float* result = s.src_frames[0].as<float>() + c * p.floats;
                if (!iterative(*d)) {
                    aggregate_finish(s.num.as<float>(), s.den.as<float>(), s.host_src_ptrs[0], p.width, p.height, p.width,
                                     s.out.as<float>(), s.stream);
                    result = s.out.as<float>();
                }
                download(s, slot_region(c, 0), result, p);
            } else {
                // Fat layout: slice sl -> num rows at 2*sl*h, den rows at (2*sl+1)*h.
                for (int sl = 0; sl < d->ntemp; ++sl) {
                    const std::size_t slice = (static_cast<std::size_t>(c) * d->ntemp + sl) * p.floats;
                    download(s, slot_region(c, 2 * sl), s.num.as<float>() + slice, p);
                    download(s, slot_region(c, 2 * sl + 1), s.den.as<float>() + slice, p);
                }
            }
        }
        // The next plane reuses these regions; finish this plane's copies first.
        s.stream.synchronize();
        for (int c = 0; c < channels; ++c) {
            if (!d->channel_out[c]) {
                identity(unit + c);
                continue;
            }
            auto* op = static_cast<std::uint8_t*>(static_cast<void*>(vsapi->getWritePtr(dst, unit + c)));
            const std::ptrdiff_t ds = vsapi->getStride(dst, unit + c);
            for (int k = 0; k < rows; ++k) {
                finish_download(s.regions[slot_region(c, k)].bytes, row_bytes, p.height,
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

// Rolling: compute the chunk [start, start + count) on one slot.
std::shared_ptr<ChunkStore> compute_chunk(Driver* d, int start, int count, VSFrameContext* ctx, const VSAPI* vsapi) {
    auto slot = d->pool->acquire();
    Slot& s = *slot;
    DeviceGuard guard(d->device.index);
    const int radius = d->radius;
    const int nframes = d->vi.numFrames;
    const bool w = has_guide(*d);
    nss::FrameScope frames(vsapi);
    auto store = std::make_shared<ChunkStore>();
    store->start = start;
    store->count = count;
    store->frames.resize(count);

    const int first_download = upload_regions(*d);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (!p.active) {
            for (int i = 0; i < count; ++i) {
                const VSFrame* f = frames.getFrameFilter(start + i, d->node, ctx);
                auto& out = store->frames[i][plane];
                out.resize(p.floats);
                vsh::bitblt(out.data(), row_bytes, vsapi->getReadPtr(f, plane), vsapi->getStride(f, plane), row_bytes,
                            p.height);
                frames.freeFrame(f);
            }
            continue;
        }
        float* acc_num = s.acc.as<float>();
        float* acc_den = acc_num + p.floats * count;
        NSS_CUDA_CHECK(cudaMemsetAsync(acc_num, 0, p.floats * 2 * count * sizeof(float), s.stream));
        // Device ring: frame f lives in window buffer f % ntemp; the distinct
        // frames of one window never collide. resident[i] is the frame held.
        std::vector<int> resident(d->ntemp, -1);
        const int first_center = std::max(0, start - radius);
        const int last_center = temporal_last(start + count - 1, radius, nframes);
        std::vector<const float*> sp(d->ntemp), rp(d->ntemp);
        for (int center = first_center; center <= last_center; ++center) {
            for (int t = 0; t < d->ntemp; ++t) {
                const int fn = temporal_slot_frame(center, t, radius, nframes);
                const int ring = fn % d->ntemp;
                if (resident[ring] != fn) {
                    const VSFrame* f = frames.getFrameFilter(fn, d->node, ctx);
                    upload(s, ring * (w ? 2 : 1), f, plane, p, s.src_frames[ring].get(), vsapi);
                    frames.freeFrame(f);
                    if (w) {
                        const VSFrame* rf = frames.getFrameFilter(fn, d->guide, ctx);
                        upload(s, ring * 2 + 1, rf, plane, p, s.ref_frames[ring].get(), vsapi);
                        frames.freeFrame(rf);
                    }
                    if (fn >= start && fn < start + count) {
                        NSS_CUDA_CHECK(cudaMemcpyAsync(s.chunk_src.as<float>() + (fn - start) * p.floats,
                                                       s.src_frames[ring].get(), p.floats * sizeof(float),
                                                       cudaMemcpyDeviceToDevice, s.stream));
                    }
                    resident[ring] = fn;
                }
                sp[t] = s.host_src_ptrs[ring];
                if (w) rp[t] = s.host_ref_ptrs[ring];
            }
            // Pageable sources: staged before cudaMemcpyAsync returns, and
            // stream-ordered after the previous center's kernels.
            NSS_CUDA_CHECK(cudaMemcpyAsync(s.src_ptrs.get(), sp.data(), d->ntemp * sizeof(float*),
                                           cudaMemcpyHostToDevice, s.stream));
            if (w) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(s.ref_ptrs.get(), rp.data(), d->ntemp * sizeof(float*),
                                               cudaMemcpyHostToDevice, s.stream));
            }
            filter_center(*d, s, p, center, w);
            for (int target = std::max(start, center - radius);
                 target <= std::min(start + count - 1, temporal_last(center, radius, nframes)); ++target) {
                const int slice = target - center + radius;
                accumulate_slice(acc_num + (target - start) * p.floats, acc_den + (target - start) * p.floats,
                                 s.num.as<float>() + slice * p.floats, s.den.as<float>() + slice * p.floats, p.floats,
                                 s.stream);
            }
        }
        for (int i = 0; i < count; ++i) {
            // s.out is reused for every frame: the download is stream-ordered
            // before the next finish overwrites it.
            aggregate_finish(acc_num + i * p.floats, acc_den + i * p.floats, s.chunk_src.as<float>() + i * p.floats,
                        p.width, p.height, p.width, s.out.as<float>(), s.stream);
            download(s, first_download + i, s.out.get(), p);
        }
        s.stream.synchronize();
        for (int i = 0; i < count; ++i) {
            auto& out = store->frames[i][plane];
            out.resize(p.floats);
            finish_download(s.regions[first_download + i].bytes, row_bytes, p.height, out.data(), row_bytes);
        }
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
    d->computing.insert(start);
    lock.unlock();
    std::shared_ptr<ChunkStore> computed;
    try {
        nss::ResourceScope chunk_scope(d->budget, nss::make_resource_account(nss::ResourceKind::Cached));
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
    while (static_cast<int>(d->cache.size()) > d->rolling.cache_limit) d->cache.pop_back();
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
    const auto& local = chunk->frames[n - chunk->start];
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const GroupPlane& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), local[plane].data(), row_bytes,
                    row_bytes, p.height);
    }
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Driver*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
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
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));

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
