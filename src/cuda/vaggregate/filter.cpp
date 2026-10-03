// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.VAggregate: reduces versioned fat intermediates (from any
// backend's legacy temporal output) to final frames. The fat frames are host
// VSFrames by definition of the two-stage API, so the reduction runs on the
// host: uploading 2(2R+1) planes per frame costs several times more than the
// sum itself (measured 5.8 vs 1.2 ms per 1080p frame at radius 1). The
// device-resident temporal path is BM3D's temporal_mode="rolling" (D8/D19).
//
// For frame n it sums slice n - c + R of every contributing center c in
// [n - R, n + R] in ascending center order and normalizes with the CPU's
// identity rule (den == 1 -> num, den > 1e-12 -> num / den, else src), using
// IEEE division (this TU is not built with fast-math).
#include "cuda/runtime/context.hpp"
#include "frontend/contribution.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/vaggregate_args.hpp"
#include "frontend/validate.hpp"
#include "nss/params.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

using nss::host_detail::temporal_last;

struct VAggData : nss::VAggregateParams {
    nss::NodeRef clip, src;
    VSVideoInfo vi{};
    std::shared_ptr<nss::ResourceBudget> budget;
};

void reduce_row(float* out, const float* const* nums, const float* const* dens, int count, const float* src,
                int width) {
    for (int x = 0; x < width; ++x) {
        float num = 0.f, den = 0.f;
        for (int c = 0; c < count; ++c) {
            num += nums[c][x];
            den += dens[c][x];
        }
        // Disabled planes contribute one exact identity slice (den == 1).
        out[x] = den == 1.f ? num : (den > 1e-12f ? num / den : src[x]);
    }
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
    const float* nums[2 * nss::kBmMaxRadius + 1];
    const float* dens[2 * nss::kBmMaxRadius + 1];
    std::ptrdiff_t strides[2 * nss::kBmMaxRadius + 1];
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi, plane);
        const int ph = nss::plane_height(d->vi, plane);
        const std::ptrdiff_t ss = vsapi->getStride(src, plane), ds = vsapi->getStride(dst, plane);
        const auto* sp = vsapi->getReadPtr(src, plane);
        auto* op = vsapi->getWritePtr(dst, plane);
        if (!d->planes[plane]) {
            vsh::bitblt(op, ds, sp, ss, static_cast<std::size_t>(pw) * sizeof(float), ph);
            continue;
        }
        for (int i = 0; i < count; ++i) {
            strides[i] = vsapi->getStride(fats[i], plane);
            const int slice = n - (first + i) + d->radius;
            const auto* base = vsapi->getReadPtr(fats[i], plane) + static_cast<std::ptrdiff_t>(2 * slice) * ph * strides[i];
            nums[i] = reinterpret_cast<const float*>(base);
            dens[i] = reinterpret_cast<const float*>(base + static_cast<std::ptrdiff_t>(ph) * strides[i]);
        }
        const float* row_nums[2 * nss::kBmMaxRadius + 1];
        const float* row_dens[2 * nss::kBmMaxRadius + 1];
        for (int y = 0; y < ph; ++y) {
            for (int i = 0; i < count; ++i) {
                row_nums[i] = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(nums[i]) + y * strides[i]);
                row_dens[i] = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(dens[i]) + y * strides[i]);
            }
            reduce_row(reinterpret_cast<float*>(op + y * ds), row_nums, row_dens, count,
                       reinterpret_cast<const float*>(sp + y * ss), pw);
        }
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
    delete static_cast<VAggData*>(instance);
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
    // The backend arguments are validated for interface parity; the host
    // reduction itself uses no device.
    (void)parse_backend_args(vsapi, in, "VAggregate");
    d->budget = nss::current_budget();

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
