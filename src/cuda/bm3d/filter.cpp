// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM3D. Arguments, defaults and errors come from the shared
// frontend (parse_bm3d with the "nss_cuda" prefix, D14); every frame is
// matched, filtered and aggregated on one device (D3) and only the final
// plane is copied back. Temporal radius > 0 lands in plan phase C5.
#include "cuda/bm3d/kernels.hpp"
#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/bm3d_args.hpp"
#include "frontend/contribution.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal_args.hpp"
#include "frontend/validate.hpp"
#include "nss/contracts.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

// Upper bound on the per-batch group buffers (values, ref cube, matches,
// patches, aggregation scratch); larger planes are processed in batches. The
// batch also shrinks so num_streams slots fit memory_limit_mb (D14).
constexpr std::size_t kBatchBytes = 128u << 20;

struct PlanePlan {
    bool active = false;
    int width = 0, height = 0;
    int block = 0, group = 0, step = 0, range = 0;
    float sigma = 0.f;
    RasterGrid grid{};
    int batch = 0;
};

struct Slot {
    Stream stream;
    DeviceBuffer src, ref, num, den, out, matches, counts, values, ref_cube, patches;
    PinnedBuffer staging;
    std::unique_ptr<OrderedAggregator> aggregator;  // shared by all planes
};

struct Bm3dData : nss::Bm3dParams {
    nss::NodeRef node, ref;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    DeviceInfo device;
    std::array<PlanePlan, 3> planes{};
    std::size_t plane_floats = 0;  // largest plane (tight pitch)
    std::array<std::size_t, 3> staging_offset{};  // per plane: src, (ref,) output regions
    std::size_t staging_bytes = 0;
    std::unique_ptr<SlotPool<Slot>> pool;
};

std::size_t per_ref_bytes(const PlanePlan& p, bool wiener) {
    const std::size_t cube = static_cast<std::size_t>(p.group) * p.block * p.block * sizeof(float);
    return cube * (wiener ? 2 : 1) + sizeof(int) +
           static_cast<std::size_t>(p.group) *
               (sizeof(DeviceMatch) + sizeof(AggregatePatch) + OrderedAggregator::kBytesPerPatch);
}

void plan_planes(Bm3dData& d) {
    const bool wiener = d.ref != nullptr;
    const int regions = wiener ? 3 : 2;  // staging: src (+ref) upload, output download
    std::size_t frame_bytes = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        PlanePlan& p = d.planes[plane];
        p.width = nss::plane_width(d.vi, plane);
        p.height = nss::plane_height(d.vi, plane);
        const std::size_t bytes = static_cast<std::size_t>(p.width) * p.height * sizeof(float);
        d.plane_floats = std::max(d.plane_floats, bytes / sizeof(float));
        d.staging_offset[plane] = d.staging_bytes;
        d.staging_bytes += bytes * regions;
        frame_bytes += bytes;
        p.sigma = d.sigma[plane];
        p.active = p.sigma != 0.f;
        if (!p.active) continue;
        p.block = d.block_size[plane];
        p.group = std::min(d.group_size[plane], kMaxGroup);
        p.step = d.block_step[plane];
        p.range = d.bm_range[plane];
        p.grid = make_raster_grid(p.width, p.height, p.block, p.step);
    }
    // Per slot: device planes, pinned staging, the leased frame's output
    // VSFrame, plus at least a minimum batch of groups.
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t fixed = plane_bytes * (wiener ? 5 : 4) + d.staging_bytes + frame_bytes;
    std::size_t min_batch = 0;
    for (const PlanePlan& p : d.planes) {
        if (p.active) min_batch = std::max(min_batch, std::min<std::size_t>(p.grid.count(), 1024) * per_ref_bytes(p, wiener));
    }
    const std::size_t need = fixed + min_batch * 4 / 3;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / need, 1, kDefaultStreams));
    }
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < need) {
        throw std::invalid_argument("nss_cuda.BM3D: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB");
    }
    const std::size_t share = limit == SIZE_MAX ? kBatchBytes + fixed : limit / static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t batch_bytes = std::min(kBatchBytes, (share - fixed) / 4 * 3);
    for (PlanePlan& p : d.planes) {
        if (!p.active) continue;
        const std::size_t fit = std::max<std::size_t>(1, batch_bytes / per_ref_bytes(p, wiener));
        p.batch = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(p.grid.count())));
    }
}

std::unique_ptr<Slot> make_slot(const Bm3dData& d) {
    auto slot = std::make_unique<Slot>();
    const bool wiener = d.ref != nullptr;
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    std::size_t values = 0, matches = 0, counts = 0, patches = 0, max_patches = 0;
    int max_w = 1, max_h = 1;
    for (const PlanePlan& p : d.planes) {
        if (!p.active) continue;
        const std::size_t batch = static_cast<std::size_t>(p.batch);
        max_patches = std::max(max_patches, batch * p.group);
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
        values = std::max(values, batch * p.group * p.block * p.block * sizeof(float));
        matches = std::max(matches, batch * p.group * sizeof(DeviceMatch));
        counts = std::max(counts, batch * sizeof(int));
        patches = std::max(patches, batch * p.group * sizeof(AggregatePatch));
    }
    slot->src = DeviceBuffer(plane_bytes, d.budget);
    if (wiener) slot->ref = DeviceBuffer(plane_bytes, d.budget);
    slot->num = DeviceBuffer(plane_bytes, d.budget);
    slot->den = DeviceBuffer(plane_bytes, d.budget);
    slot->out = DeviceBuffer(plane_bytes, d.budget);
    slot->matches = DeviceBuffer(matches, d.budget);
    slot->counts = DeviceBuffer(counts, d.budget);
    slot->values = DeviceBuffer(values, d.budget);
    if (wiener) slot->ref_cube = DeviceBuffer(values, d.budget);
    slot->patches = DeviceBuffer(patches, d.budget);
    slot->staging = PinnedBuffer(d.staging_bytes, d.budget);
    slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, 1, max_patches, d.budget);
    return slot;
}

void process_plane(const Bm3dData& d, Slot& s, int plane, const VSFrame* src, const VSFrame* ref, VSFrame* dst,
                   const VSAPI* vsapi, std::uint8_t* staging) {
    const PlanePlan& p = d.planes[plane];
    const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
    const std::size_t plane_bytes = row_bytes * p.height;
    cudaStream_t stream = s.stream;
    upload_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes, p.height, staging,
                 s.src.get(), row_bytes, stream);
    if (ref) {
        upload_plane(vsapi->getReadPtr(ref, plane), vsapi->getStride(ref, plane), row_bytes, p.height,
                     staging + plane_bytes, s.ref.get(), row_bytes, stream);
    }
    const float* match_plane = ref ? s.ref.as<float>() : s.src.as<float>();
    const MatchGeometry geometry{p.width, p.height, p.width, p.block, p.range, p.group};
    const AggregateTarget target{s.num.as<float>(), s.den.as<float>(), p.width, p.height, p.width, 1, 0};
    for (int begin = 0; begin < p.grid.count(); begin += p.batch) {
        const int count = std::min(p.batch, p.grid.count() - begin);
        {
            NSS_CUDA_RANGE("bm3d.match");
            spatial_match(match_plane, geometry, p.grid, begin, count, s.matches.as<DeviceMatch>(), s.counts.as<int>(),
                          stream);
        }
        Bm3dGroupArgs args{};
        args.src = s.src.as<float>();
        args.ref = ref ? s.ref.as<float>() : nullptr;
        args.pitch = p.width;
        args.matches = s.matches.as<DeviceMatch>();
        args.counts = s.counts.as<int>();
        args.batch = count;
        args.block = p.block;
        args.group = p.group;
        args.sigma = p.sigma;
        args.values = s.values.as<float>();
        args.ref_cube = ref ? s.ref_cube.as<float>() : nullptr;
        args.patches = s.patches.as<AggregatePatch>();
        args.slice = 0;
        {
            NSS_CUDA_RANGE("bm3d.filter");
            bm3d_filter_groups(args, stream);
        }
        NSS_CUDA_RANGE("bm3d.aggregate");
        s.aggregator->run(s.values.as<float>(), s.patches.as<AggregatePatch>(), count * p.group, p.block, target, stream,
                          begin > 0);
    }
    bm3d_finish(s.num.as<float>(), s.den.as<float>(), s.src.as<float>(), p.width, p.height, p.width, s.out.as<float>(),
                stream);
    begin_download(s.out.get(), row_bytes, row_bytes, p.height, staging + plane_bytes * (ref ? 2 : 1), stream);
    (void)dst;
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Bm3dData*>(instance);
    if (activation == arInitial) {
        vsapi->requestFrameFilter(n, d->node, ctx);
        if (d->ref) vsapi->requestFrameFilter(n, d->ref, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;

    NSS_CUDA_RANGE("bm3d.frame");
    // Lease the slot before allocating the output so at most num_streams
    // output frames are charged to memory_limit_mb at any time.
    auto slot = d->pool->acquire();
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    const VSFrame* ref = d->ref ? frames.getFrameFilter(n, d->ref, ctx) : nullptr;
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    nss::stamp_contribution(dst, 0, n, nss::Model::BM3D, vsapi);
    auto* staging = slot->staging.as<std::uint8_t>();
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const PlanePlan& p = d->planes[plane];
        if (!p.active) {
            const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
            vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                        vsapi->getStride(src, plane), row_bytes, p.height);
            continue;
        }
        process_plane(*d, *slot, plane, src, ref, dst, vsapi, staging + d->staging_offset[plane]);
    }
    slot->stream.synchronize();
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const PlanePlan& p = d->planes[plane];
        if (!p.active) continue;
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        const std::size_t used = row_bytes * p.height;
        finish_download(staging + d->staging_offset[plane] + used * (d->ref ? 2 : 1), row_bytes, p.height,
                        vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane));
    }
    frames.freeFrame(src);
    if (ref) frames.freeFrame(ref);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Bm3dData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order as nss.BM3D so shared errors match (D14).
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    nss::frontend::parse_temporal_mode(vsapi, in, radius, "BM3D", "nss_cuda");
    auto d = std::make_unique<Bm3dData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    int err = 0;
    d->ref = nss::get_node(vsapi, in, "ref", 0, &err);
    if (err) d->ref = nullptr;
    const VSVideoInfo* ref_vi = d->ref ? vsapi->getVideoInfo(d->ref) : nullptr;
    static_cast<nss::Bm3dParams&>(*d) = nss::frontend::parse_bm3d(vsapi, in, d->vi, ref_vi, "nss_cuda");
    if (d->radius > 0) {
        throw std::invalid_argument("nss_cuda.BM3D: radius > 0 is not implemented yet (CUDA plan phase C5)");
    }
    d->backend = parse_backend_args(vsapi, in, "BM3D");
    d->device = acquire_device(d->backend.device_id, "BM3D", core, vsapi);
    d->budget = nss::current_budget();
    DeviceGuard guard(d->device.index);
    bm3d_init_tables(d->device.index);
    plan_planes(*d);
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));

    VSFilterDependency deps[2]{{d->node, rpStrictSpatial}, {d->ref, rpStrictSpatial}};
    Bm3dData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("BM3D", &raw->vi, nss::checked_frame<getFrame>, freeFilter, fmParallel,
                                            deps, d->ref ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error("nss_cuda.BM3D: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_bm3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kBm3dSignature);
    vspapi->registerFunction("BM3D", args.c_str(), "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}

}  // namespace nss_cuda
