// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.DA3D: parse_da3d with the "nss_cuda" prefix around the device
// pipeline of da3d/kernels.hpp. RGB with one sigma is one joint unit; every
// other plane is a unit of its own; a plane with sigma 0 is the guide's.
#include "cuda/da3d/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/da3d_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

struct Slot {
    Stream stream;
    DeviceBuffer noisy, guide, out, transformed, work, stats;
    PinnedBuffer staging;  // noisy, guide and out: frame_floats each
    PinnedBuffer stats_host;
};

struct Da3dData : nss::Da3dParams {
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node, ref;
    VSVideoInfo vi{};
    BackendArgs backend;
    DeviceInfo device;
    bool joint = false;
    std::size_t plane_floats = 0;  // largest plane
    std::size_t frame_floats = 0;  // the planes filtered (joint: three), side by side
    int unit_channels = 1;
    DeviceBuffer tables;
    std::unique_ptr<SlotPool<Slot>> pool;
};

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<Da3dData*>(instance);
    if (activation == arInitial) {
        vsapi->requestFrameFilter(n, d->node, ctx);
        vsapi->requestFrameFilter(n, d->ref, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("da3d.frame");
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* noisy = frames.getFrameFilter(n, d->node, ctx);
    const VSFrame* guide = frames.getFrameFilter(n, d->ref, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, noisy, core);
    unsigned long long patches = 0, guide_only = 0;
    {
        auto slot = d->pool->acquire();
        Slot& s = *slot;
        struct Drain {
            const Slot& s;
            ~Drain() { cudaStreamSynchronize(s.stream.get()); }
        } drain{s};
        auto* staging = s.staging.as<std::uint8_t>();
        const std::size_t frame = d->frame_floats;
        const Da3dTables tables{d->tables.as<float>(), d->tables.as<float>() + 128};
        // One launch: the joint unit, or every plane with a sigma (each a unit,
        // its tiles beside the others'); the planes lie side by side in the
        // buffers (offset[c] floats), and one synchronization ends the frame.
        int planes[3], count = 0;
        std::size_t offset[4] = {};
        Da3dUnit units[3];
        auto add = [&](int plane, int c, Da3dUnit& u) {
            const int w = nss::plane_width(d->vi, plane), h = nss::plane_height(d->vi, plane);
            const std::size_t row = static_cast<std::size_t>(w) * sizeof(float);
            offset[c + 1] = offset[c] + static_cast<std::size_t>(w) * h;
            u.noisy[u.channels] = s.noisy.as<float>() + offset[c];
            u.guide[u.channels] = s.guide.as<float>() + offset[c];
            u.out[u.channels] = s.out.as<float>() + offset[c];
            ++u.channels;
            upload_plane(vsapi->getReadPtr(noisy, plane), vsapi->getStride(noisy, plane), row, h,
                         staging + offset[c] * sizeof(float), s.noisy.as<float>() + offset[c], row, s.stream);
            upload_plane(vsapi->getReadPtr(guide, plane), vsapi->getStride(guide, plane), row, h,
                         staging + (frame + offset[c]) * sizeof(float), s.guide.as<float>() + offset[c], row, s.stream);
        };
        auto unit_of = [&](int plane, float sigma) {
            Da3dUnit u;
            u.channels = 0;
            u.width = nss::plane_width(d->vi, plane);
            u.height = nss::plane_height(d->vi, plane);
            u.sigma = sigma / 255.f;
            u.transformed = s.transformed.as<float>();
            return u;
        };
        if (d->joint) {
            units[0] = unit_of(0, d->sigma[0]);
            units[0].rgb = true;
            for (int c = 0; c < 3; ++c) {
                add(c, c, units[0]);
                planes[c] = c;
            }
            count = 1;
        } else {
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
                if (d->sigma[plane] == 0.f) {
                    vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(guide, plane),
                                vsapi->getStride(guide, plane),
                                static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float),
                                nss::plane_height(d->vi, plane));
                    continue;
                }
                units[count] = unit_of(plane, d->sigma[plane]);
                add(plane, count, units[count]);
                planes[count++] = plane;
            }
        }
        if (count) {
            da3d_run(units, count, tables, s.work.get(), s.work.bytes(), s.stats.as<unsigned long long>(), s.stream);
            const int outputs = d->joint ? 3 : count;
            for (int c = 0; c < outputs; ++c) {
                const int plane = planes[c];
                const std::size_t row = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
                begin_download(s.out.as<float>() + offset[c], row, row, nss::plane_height(d->vi, plane),
                               staging + (2 * frame + offset[c]) * sizeof(float), s.stream);
            }
            NSS_CUDA_CHECK(cudaMemcpyAsync(s.stats_host.get(), s.stats.get(), 2 * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, s.stream));
            s.stream.synchronize();
            for (int c = 0; c < outputs; ++c) {
                const int plane = planes[c];
                const std::size_t row = static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float);
                finish_download(staging + (2 * frame + offset[c]) * sizeof(float), row, nss::plane_height(d->vi, plane),
                                vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane));
            }
            patches = s.stats_host.as<unsigned long long>()[0];
            guide_only = s.stats_host.as<unsigned long long>()[1];
        }
    }
    VSMap* props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_NSSDA3DPatches", static_cast<int64_t>(patches), maReplace);
    vsapi->mapSetInt(props, "_NSSDA3DGuideOnly", static_cast<int64_t>(guide_only), maReplace);
    frames.freeFrame(noisy);
    frames.freeFrame(guide);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<Da3dData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->tables.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<Da3dData>();
    Da3dData& d = *data;
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    static_cast<nss::Da3dParams&>(d) = nss::frontend::parse_da3d(vsapi, in, d.vi, "nss_cuda");
    d.ref = nss::get_node(vsapi, in, "ref", 0, nullptr);
    if (!nss::same_video(d.vi, *vsapi->getVideoInfo(d.ref)))
        throw std::invalid_argument("nss_cuda.DA3D: ref must have the clip's format, size and length");
    d.joint = nss::frontend::da3d_joint_rgb(d.vi, d);
    d.backend = parse_backend_args(vsapi, in, "DA3D");
    d.device = acquire_device(d.backend.device_id, "DA3D", core, vsapi);
    d.budget = nss::current_budget();
    d.unit_channels = d.joint ? 3 : 1;
    std::size_t work = 0;  // a joint unit's, or the sum of the planes' (one launch)
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const int w = nss::plane_width(d.vi, plane), h = nss::plane_height(d.vi, plane);
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(w) * h);
        if (d.sigma[plane] > 0.f) {
            const std::size_t bytes = da3d_work_bytes(w, h, d.unit_channels);
            work = d.joint ? std::max(work, bytes) : work + bytes;
            if (!d.joint) d.frame_floats += static_cast<std::size_t>(w) * h;
        }
    }
    if (d.joint) d.frame_floats = 3 * d.plane_floats;
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t frame_bytes = std::max<std::size_t>(d.frame_floats, 1) * sizeof(float);
    // device: noisy, guide, out (+ the transformed planes) and work; pinned staging
    const std::size_t stream_bytes = 3 * frame_bytes + (d.joint ? 6 * plane_bytes : 0) + work + 3 * frame_bytes;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    // Single-channel tiles run two to an SM, and a third frame in flight
    // fills the GPU (gray 1080p sigma 10: 70.7 to 86.2 fps); joint RGB is
    // saturated with two.
    const std::size_t max_streams = d.joint ? 2 : 3;
    if (!d.backend.streams_explicit)
        d.backend.num_streams =
            static_cast<int>(std::clamp<std::size_t>(limit / std::max<std::size_t>(stream_bytes, 1), 1, max_streams));
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < stream_bytes) {
        throw std::invalid_argument("nss_cuda.DA3D: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((stream_bytes >> 20) + 1) + " MiB");
    }
    {
        DeviceGuard guard(d.device.index);
        std::vector<float> host(da3d_table_floats());
        da3d_fill_tables(host.data());
        d.tables = DeviceBuffer(host.size() * sizeof(float), d.budget);
        NSS_CUDA_CHECK(cudaMemcpy(d.tables.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<std::unique_ptr<Slot>> slots;
        for (int i = 0; i < d.backend.num_streams; ++i) {
            auto s = std::make_unique<Slot>();
            s->noisy = DeviceBuffer(frame_bytes, d.budget);
            s->guide = DeviceBuffer(frame_bytes, d.budget);
            s->out = DeviceBuffer(frame_bytes, d.budget);
            if (d.joint) s->transformed = DeviceBuffer(6 * plane_bytes, d.budget);
            if (work) s->work = DeviceBuffer(work, d.budget);
            s->stats = DeviceBuffer(2 * sizeof(unsigned long long), d.budget);
            s->staging = PinnedBuffer(3 * frame_bytes, d.budget);
            s->stats_host = PinnedBuffer(2 * sizeof(unsigned long long), d.budget);
            slots.push_back(std::move(s));
        }
        d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    }
    VSFilterDependency deps[2]{{d.node, rpStrictSpatial}, {d.ref, rpStrictSpatial}};
    VSNode* node =
        vsapi->createVideoFilter2("DA3D", &d.vi, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps, 2, &d, core);
    if (!node) {
        freeFilter(data.release(), core, vsapi);
        throw std::runtime_error("nss_cuda.DA3D: failed to create filter");
    }
    data.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_da3d(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kDa3dSignature);
    vspapi->registerFunction("DA3D", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr,
                             plugin);
}

}  // namespace nss_cuda
