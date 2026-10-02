// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.VAggregate: reduces versioned fat intermediates (from any
// backend's legacy temporal output) to final frames on the device (D8). For
// frame n it uploads slice n - c + R of every contributing center c in
// [n - R, n + R], sums them in ascending center order and normalizes with the
// CPU's identity rule (den == 1 -> num, den > 1e-12 -> num / den, else src).
#include "cuda/bm3d/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/contribution.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/vaggregate_args.hpp"
#include "frontend/validate.hpp"
#include "nss/params.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

using nss::host_detail::temporal_last;

struct Slot {
    Stream stream;
    std::vector<DeviceBuffer> nums, dens;  // one plane per contributing center
    DeviceBuffer num_ptrs, den_ptrs, src, out;
    std::vector<const float*> host_num_ptrs, host_den_ptrs;
    PinnedBuffer staging;  // (2 * centers + 2) plane regions: num/den uploads, src, out
};

struct VAggData : nss::VAggregateParams {
    nss::NodeRef clip, src;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    DeviceInfo device;
    std::size_t plane_floats = 0;
    int centers = 1;  // 2R+1
    std::unique_ptr<SlotPool<Slot>> pool;
};

std::unique_ptr<Slot> make_slot(const VAggData& d) {
    auto s = std::make_unique<Slot>();
    const std::size_t bytes = d.plane_floats * sizeof(float);
    for (int c = 0; c < d.centers; ++c) {
        s->nums.emplace_back(bytes, d.budget);
        s->dens.emplace_back(bytes, d.budget);
        s->host_num_ptrs.push_back(s->nums.back().as<float>());
        s->host_den_ptrs.push_back(s->dens.back().as<float>());
    }
    s->num_ptrs = DeviceBuffer(d.centers * sizeof(float*), d.budget);
    s->den_ptrs = DeviceBuffer(d.centers * sizeof(float*), d.budget);
    NSS_CUDA_CHECK(cudaMemcpy(s->num_ptrs.get(), s->host_num_ptrs.data(), d.centers * sizeof(float*),
                              cudaMemcpyHostToDevice));
    NSS_CUDA_CHECK(cudaMemcpy(s->den_ptrs.get(), s->host_den_ptrs.data(), d.centers * sizeof(float*),
                              cudaMemcpyHostToDevice));
    s->src = DeviceBuffer(bytes, d.budget);
    s->out = DeviceBuffer(bytes, d.budget);
    s->staging = PinnedBuffer(bytes * (2 * d.centers + 2), d.budget);
    return s;
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<VAggData*>(instance);
    const int first = std::max(0, n - d->radius);
    const int last = temporal_last(n, d->radius, d->vi.numFrames);
    if (activation == arInitial) {
        for (int c = first; c <= last; ++c) vsapi->requestFrameFilter(c, d->clip, ctx);
        vsapi->requestFrameFilter(n, d->src, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("vaggregate.frame");
    auto slot = d->pool->acquire();
    Slot& s = *slot;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    std::vector<const VSFrame*> fats;
    int model = -1;
    for (int c = first; c <= last; ++c) {
        const VSFrame* frame = frames.getFrameFilter(c, d->clip, ctx);
        const auto identity = nss::validate_contribution(frame, d->radius, c, d->allow_legacy, vsapi, "nss_cuda");
        if (model >= 0 && model != identity.model)
            throw std::invalid_argument("nss_cuda.VAggregate: mixed contribution models");
        model = identity.model;
        fats.push_back(frame);
    }
    const VSFrame* src = frames.getFrameFilter(n, d->src, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    const int count = last - first + 1;
    const std::size_t region = d->plane_floats * sizeof(float);
    auto* staging = s.staging.as<std::uint8_t>();
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const std::size_t row_bytes = static_cast<std::size_t>(pw) * sizeof(float);
        if (!d->planes[plane]) {
            vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                        vsapi->getStride(src, plane), row_bytes, ph);
            continue;
        }
        for (int i = 0; i < count; ++i) {
            const VSFrame* fat = fats[i];
            const std::ptrdiff_t stride = vsapi->getStride(fat, plane);
            const int slice = n - (first + i) + d->radius;
            const auto* base = vsapi->getReadPtr(fat, plane);
            const auto* num = base + static_cast<std::ptrdiff_t>(2 * slice) * ph * stride;
            upload_plane(num, stride, row_bytes, ph, staging + region * (2 * i), s.nums[i].get(), row_bytes, s.stream);
            upload_plane(num + static_cast<std::ptrdiff_t>(ph) * stride, stride, row_bytes, ph,
                         staging + region * (2 * i + 1), s.dens[i].get(), row_bytes, s.stream);
        }
        upload_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes, ph,
                     staging + region * (2 * d->centers), s.src.get(), row_bytes, s.stream);
        vaggregate_target(s.num_ptrs.as<const float*>(), s.den_ptrs.as<const float*>(), count, s.src.as<float>(), pw,
                          ph, pw, s.out.as<float>(), s.stream);
        begin_download(s.out.get(), row_bytes, row_bytes, ph, staging + region * (2 * d->centers + 1), s.stream);
        // Staging regions are reused by the next plane.
        s.stream.synchronize();
        finish_download(staging + region * (2 * d->centers + 1), row_bytes, ph, vsapi->getWritePtr(dst, plane),
                        vsapi->getStride(dst, plane));
    }
    if (model > 0) nss::stamp_contribution(dst, 0, n, static_cast<nss::Model>(model), vsapi);
    auto* props = vsapi->getFramePropertiesRW(dst);
    for (const auto* key : {"_NSSFatVersion", "_NSSFatRadius", "_NSSFatCenter", "_NSSFatLayout"})
        vsapi->mapDeleteKey(props, key);
    for (const VSFrame* fat : fats) frames.freeFrame(fat);
    frames.freeFrame(src);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<VAggData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.VAggregate (D14).
    int e_fat = 0, e_src = 0;
    auto fat = nss::get_node(vsapi, in, "clip", 0, &e_fat);
    auto src = nss::get_node(vsapi, in, "src", 0, &e_src);
    if (e_fat || e_src || !fat || !src) throw std::invalid_argument("nss_cuda.VAggregate: clip and src are required");
    auto d = std::make_unique<VAggData>();
    static_cast<nss::VAggregateParams&>(*d) =
        nss::frontend::parse_vaggregate(vsapi, in, *vsapi->getVideoInfo(src), "nss_cuda");
    d->vi = *vsapi->getVideoInfo(src);
    nss::frontend::validate_vaggregate_shape(d->vi, *vsapi->getVideoInfo(fat), d->radius, "nss_cuda");
    d->clip = std::move(fat);
    d->src = std::move(src);
    d->backend = parse_backend_args(vsapi, in, "VAggregate");
    d->device = acquire_device(d->backend.device_id, "VAggregate", core, vsapi);
    d->budget = nss::current_budget();
    d->centers = 2 * d->radius + 1;
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        d->plane_floats = std::max(d->plane_floats, static_cast<std::size_t>(nss::plane_width(d->vi, plane)) *
                                                        nss::plane_height(d->vi, plane));
    }
    DeviceGuard guard(d->device.index);
    // Default streams are lowered to what fits memory_limit_mb, as BM3D.
    const std::size_t per_slot = d->plane_floats * sizeof(float) * (4 * d->centers + 4) + d->plane_floats * 4 * 3;
    const std::size_t limit = d->budget ? d->budget->snapshot().limit : SIZE_MAX;
    if (!d->backend.streams_explicit) {
        d->backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / per_slot, 1, kDefaultStreams));
    }
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));

    VSFilterDependency deps[2]{{d->clip, d->radius ? rpGeneral : rpStrictSpatial}, {d->src, rpStrictSpatial}};
    VAggData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("VAggregate", &raw->vi, nss::checked_frame<getFrame>, freeFilter,
                                            fmParallel, deps, 2, raw, core);
    if (!node) throw std::runtime_error("nss_cuda.VAggregate: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_vaggregate(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kVAggregateSignature);
    vspapi->registerFunction("VAggregate", args.c_str(), "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}

}  // namespace nss_cuda
