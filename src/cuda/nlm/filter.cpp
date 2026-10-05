// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NLM. Arguments and errors come from parse_nlm with the "nss_cuda"
// prefix (D14); the whole frame (distances, box sums, weights, accumulation,
// normalization) runs on the device (D3). Only the centre frame and the
// current backward/forward pair are resident, so large d does not grow
// device memory.
#include "cuda/nlm/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/nlm_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

using nss::host_detail::temporal_last;
using nss::host_detail::temporal_slot_frame;

// Device copies of one frame's used planes (source and, with rclip, reference).
struct FramePlanes {
    std::array<DeviceBuffer, 3> src, ref;
};

struct Slot {
    Stream stream;
    FramePlanes center, bwd, fwd;
    DeviceBuffer hsum_bwd, hsum_fwd, weight, max_weight;
    std::array<DeviceBuffer, 3> wdst, out;
    PinnedBuffer staging;
};

struct NlmData : nss::NlmParams {
    nss::NodeRef node, rclip;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
    BackendArgs backend;
    DeviceInfo device;
    int width = 0, height = 0, nc = 1, first_plane = 0;
    NlmDistance distance = NlmDistance::Luma;
    std::size_t plane_bytes = 0;
    std::unique_ptr<SlotPool<Slot>> pool;
};

// Staging regions (one plane each): centre, backward, forward uploads for
// src and ref, then the outputs.
int regions_per_frame(const NlmData& d) { return d.nc * (d.rclip ? 2 : 1); }
int staging_regions(const NlmData& d) { return regions_per_frame(d) * (d.d > 0 ? 3 : 1) + d.nc; }

std::size_t slot_bytes(const NlmData& d) {
    const int frames = d.d > 0 ? 3 : 1;
    const int device_planes = regions_per_frame(d) * frames + 4 + 2 * d.nc;
    return d.plane_bytes * (device_planes + staging_regions(d));
}

std::unique_ptr<Slot> make_slot(const NlmData& d) {
    auto s = std::make_unique<Slot>();
    auto frame = [&](FramePlanes& f) {
        for (int c = 0; c < d.nc; ++c) {
            f.src[c] = DeviceBuffer(d.plane_bytes, d.budget);
            if (d.rclip) f.ref[c] = DeviceBuffer(d.plane_bytes, d.budget);
        }
    };
    frame(s->center);
    if (d.d > 0) {
        frame(s->bwd);
        frame(s->fwd);
    }
    s->hsum_bwd = DeviceBuffer(d.plane_bytes, d.budget);
    s->hsum_fwd = DeviceBuffer(d.plane_bytes, d.budget);
    s->weight = DeviceBuffer(d.plane_bytes, d.budget);
    s->max_weight = DeviceBuffer(d.plane_bytes, d.budget);
    for (int c = 0; c < d.nc; ++c) {
        s->wdst[c] = DeviceBuffer(d.plane_bytes, d.budget);
        s->out[c] = DeviceBuffer(d.plane_bytes, d.budget);
    }
    s->staging = PinnedBuffer(d.plane_bytes * staging_regions(d), d.budget);
    return s;
}

NlmPlanes planes_of(const std::array<DeviceBuffer, 3>& buffers, int nc) {
    NlmPlanes p{};
    for (int c = 0; c < nc; ++c) p.p[c] = buffers[c].as<float>();
    return p;
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<NlmData*>(instance);
    if (activation == arInitial) {
        for (int i = std::max(0, n - d->d); i <= temporal_last(n, d->d, d->vi.numFrames); ++i) {
            vsapi->requestFrameFilter(i, d->node, ctx);
            if (d->rclip) vsapi->requestFrameFilter(i, d->rclip, ctx);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("nlm.frame");
    auto slot = d->pool->acquire();
    Slot& s = *slot;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);

    const int ntemp = 2 * d->d + 1;
    std::vector<const VSFrame*> srcf(ntemp), reff(ntemp, nullptr);
    for (int t = 0; t < ntemp; ++t) {
        const int fn = temporal_slot_frame(n, t, d->d, d->vi.numFrames);
        srcf[t] = frames.getFrameFilter(fn, d->node, ctx);
        if (d->rclip) reff[t] = frames.getFrameFilter(fn, d->rclip, ctx);
    }
    const VSFrame* src_center = srcf[d->d];
    VSFrame* dst = nullptr;
    if (d->channels == nss::ChannelMode::Y && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{nullptr, src_center, src_center};
        const int pl[3]{0, 1, 2};
        dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else if (d->channels == nss::ChannelMode::UV && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{src_center, nullptr, nullptr};
        const int pl[3]{0, 1, 2};
        dst = frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else {
        dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src_center, core);
    }

    const int w = d->width, h = d->height, nc = d->nc;
    const std::size_t row_bytes = static_cast<std::size_t>(w) * sizeof(float);
    auto* staging = s.staging.as<std::uint8_t>();
    const int per_frame = regions_per_frame(*d);
    auto upload_frame = [&](FramePlanes& planes, int t, int region) {
        for (int c = 0; c < nc; ++c) {
            const int plane = d->first_plane + c;
            upload_plane(vsapi->getReadPtr(srcf[t], plane), vsapi->getStride(srcf[t], plane), row_bytes, h,
                         staging + d->plane_bytes * (region + c), planes.src[c].get(), row_bytes, s.stream);
            if (d->rclip) {
                upload_plane(vsapi->getReadPtr(reff[t], plane), vsapi->getStride(reff[t], plane), row_bytes, h,
                             staging + d->plane_bytes * (region + nc + c), planes.ref[c].get(), row_bytes, s.stream);
            }
        }
    };
    upload_frame(s.center, d->d, 0);
    float* wdst[3]{s.wdst[0].as<float>(), s.wdst[1].as<float>(), s.wdst[2].as<float>()};
    nlm_reset(s.weight.as<float>(), s.max_weight.as<float>(), wdst, nc, w, h, s.stream);

    const NlmPlanes center_src = planes_of(s.center.src, nc);
    const NlmPlanes center_ref = d->rclip ? planes_of(s.center.ref, nc) : center_src;
    const float h2_inv_norm = (255.0f * 255.0f) / (3.0f * d->h * d->h * (static_cast<float>(2 * d->s + 1) *
                                                                        static_cast<float>(2 * d->s + 1)));
    const int span = 2 * d->a + 1;
    for (int i = -d->d; i <= 0; ++i) {
        NlmPlanes bwd_src = center_src, fwd_src = center_src, bwd_ref = center_ref, fwd_ref = center_ref;
        if (i < 0) {
            // The pair's staging regions are reused: let the previous pair finish first.
            if (i > -d->d) s.stream.synchronize();
            upload_frame(s.bwd, d->d + i, per_frame);
            upload_frame(s.fwd, d->d - i, 2 * per_frame);
            bwd_src = planes_of(s.bwd.src, nc);
            fwd_src = planes_of(s.fwd.src, nc);
            bwd_ref = d->rclip ? planes_of(s.bwd.ref, nc) : bwd_src;
            fwd_ref = d->rclip ? planes_of(s.fwd.ref, nc) : fwd_src;
        }
        for (int oy = -d->a; oy <= d->a; ++oy) {
            for (int ox = -d->a; ox <= d->a; ++ox) {
                if (static_cast<std::int64_t>(i) * span * span + static_cast<std::int64_t>(oy) * span + ox >= 0) continue;
                nlm_distance_hsum(center_ref, bwd_ref, d->distance, ox, oy, d->s, w, h, s.hsum_bwd.as<float>(),
                                  s.stream);
                if (i < 0) {
                    nlm_distance_hsum(fwd_ref, center_ref, d->distance, ox, oy, d->s, w, h, s.hsum_fwd.as<float>(),
                                      s.stream);
                }
                NlmAccumArgs args{};
                args.hsum_bwd = s.hsum_bwd.as<float>();
                args.hsum_fwd = i < 0 ? s.hsum_fwd.as<float>() : s.hsum_bwd.as<float>();
                args.src_bwd = bwd_src;
                args.src_fwd = fwd_src;
                args.channels = nc;
                args.ox = ox;
                args.oy = oy;
                args.s = d->s;
                args.h2_inv_norm = h2_inv_norm;
                args.width = w;
                args.height = h;
                args.weight = s.weight.as<float>();
                args.max_weight = s.max_weight.as<float>();
                for (int c = 0; c < nc; ++c) args.wdst[c] = wdst[c];
                nlm_accumulate(args, s.stream);
            }
        }
    }
    const int out_region = per_frame * (d->d > 0 ? 3 : 1);
    for (int c = 0; c < nc; ++c) {
        nlm_finish(center_src.p[c], s.weight.as<float>(), s.max_weight.as<float>(), wdst[c], d->wref, w, h,
                   s.out[c].as<float>(), s.stream);
        begin_download(s.out[c].get(), row_bytes, row_bytes, h, staging + d->plane_bytes * (out_region + c), s.stream);
    }
    s.stream.synchronize();
    for (int c = 0; c < nc; ++c) {
        const int plane = d->first_plane + c;
        finish_download(staging + d->plane_bytes * (out_region + c), row_bytes, h, vsapi->getWritePtr(dst, plane),
                        vsapi->getStride(dst, plane));
    }
    for (int t = 0; t < ntemp; ++t) {
        frames.freeFrame(srcf[t]);
        if (reff[t]) frames.freeFrame(reff[t]);
    }
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<NlmData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
    }
    delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.NLM (D14).
    auto d = std::make_unique<NlmData>();
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    static_cast<nss::NlmParams&>(*d) = nss::frontend::parse_nlm(vsapi, in, d->vi, "nss_cuda");
    int err = 0;
    d->rclip = nss::get_node(vsapi, in, "rclip", 0, &err);
    if (err) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        throw std::invalid_argument("nss_cuda.NLM: rclip must match clip");
    }
    d->backend = parse_backend_args(vsapi, in, "NLM");
    d->device = acquire_device(d->backend.device_id, "NLM", core, vsapi);
    d->budget = nss::current_budget();

    const bool uv = d->channels == nss::ChannelMode::UV;
    d->width = uv ? d->vi.width >> d->vi.format.subSamplingW : d->vi.width;
    d->height = uv ? d->vi.height >> d->vi.format.subSamplingH : d->vi.height;
    d->nc = d->channels == nss::ChannelMode::Y ? 1 : uv ? 2 : 3;
    d->first_plane = uv ? 1 : 0;
    d->distance = d->channels == nss::ChannelMode::Y     ? NlmDistance::Luma
                  : uv                                   ? NlmDistance::Chroma
                  : d->channels == nss::ChannelMode::YUV ? NlmDistance::Yuv
                                                         : NlmDistance::Rgb;
    d->plane_bytes = static_cast<std::size_t>(d->width) * d->height * sizeof(float);

    // Per slot: device planes, pinned staging and the leased output frame.
    const std::size_t need = slot_bytes(*d) + d->plane_bytes * d->vi.format.numPlanes;
    const std::size_t limit = d->budget ? d->budget->snapshot().limit : SIZE_MAX;
    if (!d->backend.streams_explicit) {
        d->backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / need, 1, kDefaultStreams));
    }
    if (limit / static_cast<std::size_t>(d->backend.num_streams) < need) {
        throw std::invalid_argument("nss_cuda.NLM: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d->backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((need >> 20) + 1) + " MiB");
    }
    DeviceGuard guard(d->device.index);
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d->backend.num_streams; ++i) slots.push_back(make_slot(*d));
    d->pool = std::make_unique<SlotPool<Slot>>(std::move(slots));

    const auto pattern = d->d == 0 ? rpStrictSpatial : rpGeneral;
    VSFilterDependency deps[2]{{d->node, pattern}, {d->rclip, pattern}};
    NlmData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("NLM", &raw->vi, nss::checked_frame<getFrame>, freeFilter, fmParallel,
                                            deps, d->rclip ? 2 : 1, raw, core);
    if (!node) throw std::runtime_error("nss_cuda.NLM: failed to create filter");
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNlmSignature);
    vspapi->registerFunction("NLM", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
