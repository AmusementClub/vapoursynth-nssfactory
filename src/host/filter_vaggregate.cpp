#include "host/filters.hpp"
#include "host/temporal.hpp"
#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace {

struct VAggData {
    VSNode* clip = nullptr;  // fat intermediate
    VSNode* src = nullptr;
    VSVideoInfo vi_src{};
    int radius = 0;
    int planes[3]{1, 1, 1};
};

const VSFrame* VS_CC vaggGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<VAggData*>(instanceData);
    (void)frameData;
    if (activationReason == arInitial) {
        for(int c=std::max(0,n-d->radius);c<=nss::host_detail::temporal_last(n,d->radius,d->vi_src.numFrames);++c)
            vsapi->requestFrameFilter(c,d->clip,frameCtx);
        vsapi->requestFrameFilter(n, d->src, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }
    const int first=std::max(0,n-d->radius);
    const int last=nss::host_detail::temporal_last(n,d->radius,d->vi_src.numFrames);
    std::vector<const VSFrame*> frames;
    for(int c=first;c<=last;++c) frames.push_back(vsapi->getFrameFilter(c,d->clip,frameCtx));
    const VSFrame* src = vsapi->getFrameFilter(n, d->src, frameCtx);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi_src.format, d->vi_src.width, d->vi_src.height, src, core);

    for (int plane = 0; plane < d->vi_src.format.numPlanes; ++plane) {
        const int pw = nss::plane_width(d->vi_src, plane);
        const int ph = nss::plane_height(d->vi_src, plane);
        const int sstride = static_cast<int>(vsapi->getStride(src, plane) / sizeof(float));

        const int dstride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(float));
        float* outp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane));
        const float* srcp = reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane));

        if (!d->planes[plane]) {
            for (int y = 0; y < ph; ++y) {
                std::memcpy(outp + y * dstride, srcp + y * sstride, static_cast<std::size_t>(pw) * sizeof(float));
            }
            continue;
        }
        const float* nums[2*nss::kBmMaxRadius+1];
        const float* dens[2*nss::kBmMaxRadius+1];
        int strides[2*nss::kBmMaxRadius+1];
        for(int c=first;c<=last;++c) {
            const auto* frame=frames[c-first];
            const int stride=static_cast<int>(vsapi->getStride(frame,plane)/sizeof(float));
            const int slice=n-c+d->radius;
            const float* base=reinterpret_cast<const float*>(vsapi->getReadPtr(frame,plane));
            nums[c-first]=base+static_cast<std::size_t>(slice*2)*ph*stride;
            dens[c-first]=nums[c-first]+static_cast<std::size_t>(ph)*stride;
            strides[c-first]=stride;
        }
        nss::vaggregate_target(outp,nums,dens,strides,last-first+1,srcp,pw,ph,dstride,sstride);
    }
    for(auto* frame:frames) vsapi->freeFrame(frame);
    vsapi->freeFrame(src);
    return dst;
}

void VS_CC vaggFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<VAggData*>(instanceData);
    vsapi->freeNode(d->clip);
    vsapi->freeNode(d->src);
    delete d;
}

}  // namespace

VSNode* nss_create_vaggregate(VSNode* fat, VSNode* src, int radius, const int* planes, VSCore* core,
                              const VSAPI* vsapi, VSMap* err) {
    auto fail = [&](const char* msg) -> VSNode* {
        vsapi->mapSetError(err, msg);
        vsapi->freeNode(fat);
        vsapi->freeNode(src);
        return nullptr;
    };
    if (!nss::cpu_has_avx2()) {
        return fail("nss.VAggregate: AVX2 is required");
    }
    if (!fat || !src) {
        return fail("nss.VAggregate: clip and src are required");
    }
    auto d = std::make_unique<VAggData>();
    d->clip = fat;
    d->src = src;
    d->vi_src = *vsapi->getVideoInfo(src);
    const VSVideoInfo* vi_fat = vsapi->getVideoInfo(fat);
    if (!nss::is_const_32f(d->vi_src) || !nss::is_const_32f(*vi_fat)) {
        d->clip = nullptr;
        d->src = nullptr;
        return fail("nss.VAggregate: constant 32f clips required");
    }
    if (radius < 0 || radius > nss::kBmMaxRadius) {
        d->clip = nullptr;
        d->src = nullptr;
        return fail("nss.VAggregate: radius must be in [0, 16]");
    }
    d->radius = radius;
    const int expect_h = d->vi_src.height * (2 * d->radius + 1) * 2;
    const VSVideoFormat& src_format = d->vi_src.format;
    const VSVideoFormat& fat_format = vi_fat->format;
    const bool same_format = src_format.colorFamily == fat_format.colorFamily &&
                             src_format.sampleType == fat_format.sampleType &&
                             src_format.bitsPerSample == fat_format.bitsPerSample &&
                             src_format.bytesPerSample == fat_format.bytesPerSample &&
                             src_format.subSamplingW == fat_format.subSamplingW &&
                             src_format.subSamplingH == fat_format.subSamplingH &&
                             src_format.numPlanes == fat_format.numPlanes;
    if (!same_format || vi_fat->width != d->vi_src.width || vi_fat->height != expect_h ||
        vi_fat->numFrames != d->vi_src.numFrames) {
        d->clip = nullptr;
        d->src = nullptr;
        return fail("nss.VAggregate: clip must match src format, width, and frame count, with "
                    "height=src.height*(2*radius+1)*2");
    }
    for (int i = 0; i < 3; ++i) {
        d->planes[i] = planes ? (planes[i] != 0) : 1;
    }
    VSFilterDependency deps[2]{{d->clip, d->radius ? rpGeneral : rpStrictSpatial}, {d->src, rpStrictSpatial}};
    VAggData* raw = d.get();
    VSNode* node = vsapi->createVideoFilter2("VAggregate", &raw->vi_src, vaggGetFrame, vaggFree, fmParallel, deps, 2,
                                            raw, core);
    if (!node) {
        d->clip = nullptr;
        d->src = nullptr;
        return fail("nss.VAggregate: failed to create filter");
    }
    d.release();
    return node;
}

void VS_CC vaggCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    int e_fat = 0;
    int e_src = 0;
    VSNode* fat = vsapi->mapGetNode(in, "clip", 0, &e_fat);
    VSNode* src = vsapi->mapGetNode(in, "src", 0, &e_src);
    if (e_fat || e_src || !fat || !src) {
        vsapi->mapSetError(out, "nss.VAggregate: clip and src are required");
        if (fat) {
            vsapi->freeNode(fat);
        }
        if (src) {
            vsapi->freeNode(src);
        }
        return;
    }
    const int radius = nss::map_int(vsapi, in, "radius", 0);
    const VSVideoInfo* vi_src = vsapi->getVideoInfo(src);
    int pl[3]{1, 1, 1};
    const int num_plane_args = vsapi->mapNumElements(in, "planes");
    if (num_plane_args > 0) {
        std::fill_n(pl, 3, 0);
        for (int i = 0; i < num_plane_args; ++i) {
            int err = 0;
            const int plane = static_cast<int>(vsapi->mapGetInt(in, "planes", i, &err));
            if (err || plane < 0 || plane >= vi_src->format.numPlanes || pl[plane]) {
                vsapi->mapSetError(out, "nss.VAggregate: planes must contain unique valid plane indices");
                vsapi->freeNode(fat);
                vsapi->freeNode(src);
                return;
            }
            pl[plane] = 1;
        }
    }
    VSNode* node = nss_create_vaggregate(fat, src, radius, pl, core, vsapi, out);
    if (node) {
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
    }
}

void register_vaggregate(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("VAggregate", "clip:vnode;src:vnode;radius:int:opt;planes:int[]:opt;", "clip:vnode;",
                             vaggCreate, nullptr, plugin);
}
