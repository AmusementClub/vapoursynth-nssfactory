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
    PinnedBuffer staging;  // 3 noisy + 3 guide + 3 out planes
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
        const std::size_t plane_bytes = d->plane_floats * sizeof(float);
        const Da3dTables tables{d->tables.as<float>(), d->tables.as<float>() + 128};
        auto run = [&](const int* planes, int count, float sigma) {
            const int w = nss::plane_width(d->vi, planes[0]), h = nss::plane_height(d->vi, planes[0]);
            const std::size_t row = static_cast<std::size_t>(w) * sizeof(float);
            const std::size_t floats = static_cast<std::size_t>(w) * h;
            Da3dUnit u;
            u.channels = count;
            u.width = w;
            u.height = h;
            u.rgb = count == 3;
            u.sigma = sigma / 255.f;
            u.transformed = s.transformed.as<float>();
            for (int c = 0; c < count; ++c) {
                const int plane = planes[c];
                u.noisy[c] = s.noisy.as<float>() + c * floats;
                u.guide[c] = s.guide.as<float>() + c * floats;
                u.out[c] = s.out.as<float>() + c * floats;
                upload_plane(vsapi->getReadPtr(noisy, plane), vsapi->getStride(noisy, plane), row, h,
                             staging + c * plane_bytes, s.noisy.as<float>() + c * floats, row, s.stream);
                upload_plane(vsapi->getReadPtr(guide, plane), vsapi->getStride(guide, plane), row, h,
                             staging + (3 + c) * plane_bytes, s.guide.as<float>() + c * floats, row, s.stream);
            }
            da3d_run(u, tables, s.work.get(), s.work.bytes(), s.stats.as<unsigned long long>(), s.stream);
            for (int c = 0; c < count; ++c)
                begin_download(s.out.as<float>() + c * floats, row, row, h, staging + (6 + c) * plane_bytes, s.stream);
            NSS_CUDA_CHECK(cudaMemcpyAsync(s.stats_host.get(), s.stats.get(), 2 * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, s.stream));
            s.stream.synchronize();
            for (int c = 0; c < count; ++c)
                finish_download(staging + (6 + c) * plane_bytes, row, h, vsapi->getWritePtr(dst, planes[c]),
                                vsapi->getStride(dst, planes[c]));
            patches += s.stats_host.as<unsigned long long>()[0];
            guide_only += s.stats_host.as<unsigned long long>()[1];
        };
        if (d->joint) {
            const int planes[3] = {0, 1, 2};
            run(planes, 3, d->sigma[0]);
        } else {
            for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
                if (d->sigma[plane] == 0.f) {
                    vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(guide, plane),
                                vsapi->getStride(guide, plane),
                                static_cast<std::size_t>(nss::plane_width(d->vi, plane)) * sizeof(float),
                                nss::plane_height(d->vi, plane));
                    continue;
                }
                run(&plane, 1, d->sigma[plane]);
            }
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
    std::size_t work = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        const int w = nss::plane_width(d.vi, plane), h = nss::plane_height(d.vi, plane);
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(w) * h);
        if (d.sigma[plane] > 0.f) work = std::max(work, da3d_work_bytes(w, h, d.unit_channels));
    }
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t stream_bytes = 3 * 3 * plane_bytes + (d.joint ? 6 * plane_bytes : 0) + work + 9 * plane_bytes;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (!d.backend.streams_explicit)
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / std::max<std::size_t>(stream_bytes, 1), 1, 2));
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
            s->noisy = DeviceBuffer(3 * plane_bytes, d.budget);
            s->guide = DeviceBuffer(3 * plane_bytes, d.budget);
            s->out = DeviceBuffer(3 * plane_bytes, d.budget);
            if (d.joint) s->transformed = DeviceBuffer(6 * plane_bytes, d.budget);
            if (work) s->work = DeviceBuffer(work, d.budget);
            s->stats = DeviceBuffer(2 * sizeof(unsigned long long), d.budget);
            s->staging = PinnedBuffer(9 * plane_bytes, d.budget);
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
