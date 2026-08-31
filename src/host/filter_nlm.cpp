#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace {

struct NlmData {
    VSNode* node = nullptr;
    VSNode* rclip = nullptr;
    const VSAPI* vsapi = nullptr;
    VSVideoInfo vi{};
    int d = nss::kNlmDefaultD;
    int a = nss::kNlmDefaultA;
    int s = nss::kNlmDefaultS;
    float h = nss::kNlmDefaultH;
    float wref = nss::kNlmDefaultWref;
    nss::ChannelMode channels = nss::ChannelMode::Y;
    nss::Workspace ws;
};

int nlm_width(const NlmData* d) {
    if (d->channels == nss::ChannelMode::UV) {
        return d->vi.width >> d->vi.format.subSamplingW;
    }
    return d->vi.width;
}
int nlm_height(const NlmData* d) {
    if (d->channels == nss::ChannelMode::UV) {
        return d->vi.height >> d->vi.format.subSamplingH;
    }
    return d->vi.height;
}

int num_channels(nss::ChannelMode c) {
    if (c == nss::ChannelMode::Y) {
        return 1;
    }
    if (c == nss::ChannelMode::UV) {
        return 2;
    }
    return 3;
}

const VSFrame* VS_CC nlmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                 VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<NlmData*>(instanceData);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->d);
        const int end = std::min(n + d->d, d->vi.numFrames - 1);
        for (int i = start; i <= end; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
            if (d->rclip) {
                vsapi->requestFrameFilter(i, d->rclip, frameCtx);
            }
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    VSNode* ref_node = d->rclip ? d->rclip : d->node;
    const VSFrame* ref_frame = vsapi->getFrameFilter(n, ref_node, frameCtx);
    const int w = nlm_width(d);
    const int h = nlm_height(d);
    const int plane0 = (d->channels == nss::ChannelMode::UV) ? 1 : 0;
    const int stride = static_cast<int>(vsapi->getStride(ref_frame, plane0) / sizeof(float));
    const int size = h * stride;
    const int nc = num_channels(d->channels);
    const int extra = (d->d != 0) ? 1 : 0;
    const std::size_t floats = static_cast<std::size_t>(size) * static_cast<std::size_t>(4 + nc + extra) +
                               static_cast<std::size_t>(w);

    float* workspace = nullptr;
    try {
        workspace = d->ws.get(floats);
    } catch (...) {
        vsapi->freeFrame(ref_frame);
        vsapi->setFilterError("nss.NLM: workspace allocation failed", frameCtx);
        return nullptr;
    }
    std::memset(workspace, 0, static_cast<std::size_t>(1 + nc) * static_cast<std::size_t>(size) * sizeof(float));
    float* weightp = workspace;
    std::array<float*, 3> wdstp{workspace + size, nc > 1 ? workspace + 2 * size : nullptr,
                                nc > 2 ? workspace + 3 * size : nullptr};
    float* max_weightp = workspace + (1 + nc) * size;
    for (int i = 0; i < size; ++i) {
        max_weightp[i] = std::numeric_limits<float>::epsilon();
    }
    float* temp = workspace + (2 + nc) * size;
    float* temp_bwd = workspace + (3 + nc) * size;
    float* temp_fwd = d->d == 0 ? nullptr : workspace + (4 + nc) * size;
    float* buffer = workspace + (4 + nc + extra) * size;

    auto ptrs = [&](const VSFrame* f) {
        std::array<const float*, 3> p{};
        if (d->channels == nss::ChannelMode::Y) {
            p[0] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 0));
        } else if (d->channels == nss::ChannelMode::UV) {
            p[1] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 1));
            p[2] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 2));
        } else {
            p[0] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 0));
            p[1] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 1));
            p[2] = reinterpret_cast<const float*>(vsapi->getReadPtr(f, 2));
        }
        return p;
    };

    auto distance = [&](float* dst, std::array<const float*, 3> c, std::array<const float*, 3> nb, int ox, int oy) {
        switch (d->channels) {
        case nss::ChannelMode::Y:
            nss::nlm_distance_luma_f32(dst, c[0], nb[0], ox, oy, w, h, stride);
            break;
        case nss::ChannelMode::UV:
            nss::nlm_distance_chroma_f32(dst, c[1], c[2], nb[1], nb[2], ox, oy, w, h, stride);
            break;
        case nss::ChannelMode::YUV:
            nss::nlm_distance_yuv_f32(dst, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, h, stride);
            break;
        case nss::ChannelMode::RGB:
            nss::nlm_distance_rgb_f32(dst, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, h, stride);
            break;
        }
    };

    const float h2_inv_norm = (255.0f * 255.0f) / (3.0f * d->h * d->h * static_cast<float>((2 * d->s + 1) * (2 * d->s + 1)));
    auto refp = ptrs(ref_frame);

    for (int i = -d->d; i <= 0; ++i) {
        const int bwd_n = std::max(n + i, 0);
        const int fwd_n = std::min(n - i, d->vi.numFrames - 1);
        const VSFrame* src_bwd = vsapi->getFrameFilter(bwd_n, d->node, frameCtx);
        const VSFrame* src_fwd = vsapi->getFrameFilter(fwd_n, d->node, frameCtx);
        const VSFrame* ref_bwd = vsapi->getFrameFilter(bwd_n, ref_node, frameCtx);
        const VSFrame* ref_fwd = vsapi->getFrameFilter(fwd_n, ref_node, frameCtx);
        auto srcp_bwd = ptrs(src_bwd);
        auto srcp_fwd = ptrs(src_fwd);
        auto refp_bwd = ptrs(ref_bwd);
        auto refp_fwd = ptrs(ref_fwd);

        for (int oy = -d->a; oy <= d->a; ++oy) {
            for (int ox = -d->a; ox <= d->a; ++ox) {
                if (i * (2 * d->a + 1) * (2 * d->a + 1) + oy * (2 * d->a + 1) + ox >= 0) {
                    continue;
                }
                distance(temp_bwd, refp, refp_bwd, ox, oy);
                nss::nlm_horizontal(temp, temp_bwd, d->s, w, h, stride);
                nss::nlm_vertical_welsch(temp_bwd, temp, d->s, h2_inv_norm, w, h, stride, buffer);
                if (i == 0) {
                    if (d->channels == nss::ChannelMode::Y) {
                        nss::nlm_accum_ch1(weightp, wdstp[0], max_weightp, srcp_bwd[0], srcp_bwd[0], temp_bwd, temp_bwd,
                                           ox, oy, w, h, stride);
                    } else if (d->channels == nss::ChannelMode::UV) {
                        nss::nlm_accum_ch2(weightp, wdstp[0], wdstp[1], max_weightp, srcp_bwd[1], srcp_bwd[2],
                                           srcp_bwd[1], srcp_bwd[2], temp_bwd, temp_bwd, ox, oy, w, h, stride);
                    } else {
                        nss::nlm_accum_ch3(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, srcp_bwd[0], srcp_bwd[1],
                                           srcp_bwd[2], srcp_bwd[0], srcp_bwd[1], srcp_bwd[2], temp_bwd, temp_bwd, ox, oy,
                                           w, h, stride);
                    }
                    continue;
                }
                distance(temp_fwd, refp_fwd, refp, ox, oy);
                nss::nlm_horizontal(temp, temp_fwd, d->s, w, h, stride);
                nss::nlm_vertical_welsch(temp_fwd, temp, d->s, h2_inv_norm, w, h, stride, buffer);
                if (d->channels == nss::ChannelMode::Y) {
                    nss::nlm_accum_ch1(weightp, wdstp[0], max_weightp, srcp_bwd[0], srcp_fwd[0], temp_bwd, temp_fwd, ox,
                                       oy, w, h, stride);
                } else if (d->channels == nss::ChannelMode::UV) {
                    nss::nlm_accum_ch2(weightp, wdstp[0], wdstp[1], max_weightp, srcp_bwd[1], srcp_bwd[2], srcp_fwd[1],
                                       srcp_fwd[2], temp_bwd, temp_fwd, ox, oy, w, h, stride);
                } else {
                    nss::nlm_accum_ch3(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, srcp_bwd[0], srcp_bwd[1],
                                       srcp_bwd[2], srcp_fwd[0], srcp_fwd[1], srcp_fwd[2], temp_bwd, temp_fwd, ox, oy, w,
                                       h, stride);
                }
            }
        }
        vsapi->freeFrame(src_fwd);
        vsapi->freeFrame(src_bwd);
        vsapi->freeFrame(ref_fwd);
        vsapi->freeFrame(ref_bwd);
    }
    vsapi->freeFrame(ref_frame);

    const VSFrame* src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
    VSFrame* dst_frame = nullptr;
    if (d->channels == nss::ChannelMode::Y && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{nullptr, src_frame, src_frame};
        const int pl[3]{0, 1, 2};
        dst_frame = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_frame, core);
    } else if (d->channels == nss::ChannelMode::UV && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{src_frame, nullptr, nullptr};
        const int pl[3]{0, 1, 2};
        dst_frame = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_frame, core);
    } else {
        dst_frame = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src_frame, core);
    }
    auto srcp = ptrs(src_frame);
    std::array<float*, 3> dstp{};
    if (d->channels == nss::ChannelMode::Y) {
        dstp[0] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 0));
        nss::nlm_finish_ch1(dstp[0], srcp[0], weightp, wdstp[0], max_weightp, d->wref, w, h, stride);
    } else if (d->channels == nss::ChannelMode::UV) {
        dstp[1] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 1));
        dstp[2] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 2));
        nss::nlm_finish_ch2(dstp[1], dstp[2], srcp[1], srcp[2], weightp, wdstp[0], wdstp[1], max_weightp, d->wref, w, h,
                            stride);
    } else {
        dstp[0] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 0));
        dstp[1] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 1));
        dstp[2] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, 2));
        nss::nlm_finish_ch3(dstp[0], dstp[1], dstp[2], srcp[0], srcp[1], srcp[2], weightp, wdstp[0], wdstp[1], wdstp[2],
                            max_weightp, d->wref, w, h, stride);
    }
    vsapi->freeFrame(src_frame);
    return dst_frame;
}

void VS_CC nlmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<NlmData*>(instanceData);
    vsapi->freeNode(d->node);
    if (d->rclip) {
        vsapi->freeNode(d->rclip);
    }
    delete d;
}

void VS_CC nlmCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_has_avx2()) {
        vsapi->mapSetError(out, "nss.NLM: AVX2 is required");
        return;
    }
    auto d = std::make_unique<NlmData>();
    d->vsapi = vsapi;
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) {
        vsapi->mapSetError(out, msg);
        vsapi->freeNode(d->node);
        if (d->rclip) {
            vsapi->freeNode(d->rclip);
            d->rclip = nullptr;
        }
    };
    if (!nss::is_const_32f(d->vi)) {
        fail("nss.NLM: constant Gray/YUV/RGB 32-bit float required");
        return;
    }
    d->d = nss::map_int(vsapi, in, "d", nss::kNlmDefaultD);
    d->a = nss::map_int(vsapi, in, "a", nss::kNlmDefaultA);
    d->s = nss::map_int(vsapi, in, "s", nss::kNlmDefaultS);
    d->h = nss::map_float(vsapi, in, "h", nss::kNlmDefaultH);
    d->wref = nss::map_float(vsapi, in, "wref", nss::kNlmDefaultWref);
    if (d->d < 0 || d->a <= 0 || d->s < 0 || d->h <= 0.f) {
        fail("nss.NLM: invalid d/a/s/h");
        return;
    }
    const int wmode = nss::map_int(vsapi, in, "wmode", 0);
    if (wmode != 0) {
        fail("nss.NLM: only wmode=0 (Welsch) is implemented");
        return;
    }
    int err = 0;
    const char* ch = vsapi->mapGetData(in, "channels", 0, &err);
    std::string cs = (!err && ch) ? ch : "AUTO";
    if (cs == "Y") {
        d->channels = nss::ChannelMode::Y;
    } else if (cs == "UV") {
        d->channels = nss::ChannelMode::UV;
    } else if (cs == "YUV") {
        d->channels = nss::ChannelMode::YUV;
    } else if (cs == "RGB") {
        d->channels = nss::ChannelMode::RGB;
    } else if (cs == "AUTO") {
        d->channels = (d->vi.format.colorFamily == cfRGB) ? nss::ChannelMode::RGB : nss::ChannelMode::Y;
    } else {
        fail("nss.NLM: channels must be Y, UV, YUV, RGB, or AUTO");
        return;
    }
    if (d->channels == nss::ChannelMode::UV && d->vi.format.colorFamily != cfYUV) {
        fail("nss.NLM: UV requires YUV");
        return;
    }
    if (d->channels == nss::ChannelMode::YUV &&
        (d->vi.format.colorFamily != cfYUV || d->vi.format.subSamplingW || d->vi.format.subSamplingH)) {
        fail("nss.NLM: YUV requires YUV444");
        return;
    }
    if (d->channels == nss::ChannelMode::RGB && d->vi.format.colorFamily != cfRGB) {
        fail("nss.NLM: RGB requires RGB");
        return;
    }
    err = 0;
    d->rclip = vsapi->mapGetNode(in, "rclip", 0, &err);
    if (err) {
        d->rclip = nullptr;
    } else if (!nss::same_video(d->vi, *vsapi->getVideoInfo(d->rclip))) {
        fail("nss.NLM: rclip must match clip");
        return;
    }

    VSFilterDependency deps[2]{};
    deps[0].source = d->node;
    deps[0].requestPattern = (d->d == 0) ? rpStrictSpatial : rpGeneral;
    if (d->rclip) {
        deps[1].source = d->rclip;
        deps[1].requestPattern = (d->d == 0) ? rpStrictSpatial : rpGeneral;
    }
    const int ndeps = d->rclip ? 2 : 1;
    NlmData* raw = d.get();
    vsapi->createVideoFilter(out, "NLM", &raw->vi, nlmGetFrame, nlmFree, fmParallel, deps, ndeps, raw, core);
    d.release();
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("NLM",
                             "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;wmode:int:opt;"
                             "wref:float:opt;rclip:vnode:opt;",
                             "clip:vnode;", nlmCreate, nullptr, plugin);
}
