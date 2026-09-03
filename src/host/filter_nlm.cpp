#include "host/validate.hpp"
#include "nss/avx2.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

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

constexpr std::size_t kNlmMaxScratch = 1u << 20;

std::size_t nlm_scratch_budget() {
#if defined(_SC_LEVEL2_CACHE_SIZE)
    const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) {
        return std::min(kNlmMaxScratch, static_cast<std::size_t>(l2) / 2u);
    }
#endif
    return kNlmMaxScratch;
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
    const int w = nlm_width(d);
    const int h = nlm_height(d);
    const int nc = num_channels(d->channels);

    const int ntemp = 2 * d->d + 1;
    std::vector<const VSFrame*> src_frames(static_cast<std::size_t>(ntemp));
    std::vector<const VSFrame*> ref_frames(static_cast<std::size_t>(ntemp));
    for (int t = 0; t < ntemp; ++t) {
        const int fn = std::clamp(n - d->d + t, 0, d->vi.numFrames - 1);
        src_frames[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, d->node, frameCtx);
        ref_frames[static_cast<std::size_t>(t)] = vsapi->getFrameFilter(fn, ref_node, frameCtx);
    }
    const VSFrame* src_center = src_frames[static_cast<std::size_t>(d->d)];
    const VSFrame* ref_center = ref_frames[static_cast<std::size_t>(d->d)];
    using PlanePtrs = std::array<const float*, 3>;
    using PlaneStrides = std::array<int, 3>;
    auto plane_strides = [&](const VSFrame* frame) {
        PlaneStrides result{};
        for (int c = 0; c < nc; ++c) {
            const int plane = (d->channels == nss::ChannelMode::UV) ? c + 1 : c;
            result[plane] = static_cast<int>(vsapi->getStride(frame, plane) / sizeof(float));
        }
        return result;
    };
    std::vector<PlaneStrides> src_strides(static_cast<std::size_t>(ntemp));
    std::vector<PlaneStrides> ref_strides(static_cast<std::size_t>(ntemp));
    int stride = w;
    for (int t = 0; t < ntemp; ++t) {
        src_strides[static_cast<std::size_t>(t)] = plane_strides(src_frames[static_cast<std::size_t>(t)]);
        ref_strides[static_cast<std::size_t>(t)] = plane_strides(ref_frames[static_cast<std::size_t>(t)]);
        for (int c = 0; c < nc; ++c) {
            const int plane = (d->channels == nss::ChannelMode::UV) ? c + 1 : c;
            stride = std::max(stride, src_strides[static_cast<std::size_t>(t)][plane]);
            stride = std::max(stride, ref_strides[static_cast<std::size_t>(t)][plane]);
        }
    }
    const int halo = d->a + d->s;
    const int temp_arrays = 2;       // horizontal + one reusable vertical map
    const int accum_arrays = 2 + nc; // weight, max_weight, and channel sums
    const int saved_arrays = 1;      // bwd map for the core rows of a temporal pair
    const auto stripe_bytes = [&](int core) {
        const std::size_t core_rows = static_cast<std::size_t>(std::max(core, 0));
        const std::size_t ext = std::min(static_cast<std::size_t>(h),
                                         core_rows + 2u * static_cast<std::size_t>(halo));
        const std::size_t rows = core_rows * static_cast<std::size_t>(accum_arrays) +
                                 core_rows * static_cast<std::size_t>(saved_arrays) +
                                 ext * static_cast<std::size_t>(temp_arrays);
        return rows * static_cast<std::size_t>(stride) * sizeof(float) + static_cast<std::size_t>(w) * sizeof(float);
    };
    // A very small reported L2 can leave no useful core rows after accounting
    // for the halo. Raise the effective budget (still capped at 1 MiB) enough
    // to keep a minimum tile without making the per-thread scratch unbounded.
    constexpr std::size_t kMinCoreRows = 21;
    const std::size_t min_tile_bytes = stripe_bytes(static_cast<int>(kMinCoreRows));
    if (stripe_bytes(1) > kNlmMaxScratch) {
        for (const VSFrame* frame : src_frames) {
            vsapi->freeFrame(frame);
        }
        for (const VSFrame* frame : ref_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->setFilterError("nss.NLM: image stride/halo exceeds the 1 MiB workspace limit", frameCtx);
        return nullptr;
    }
    const std::size_t effective_budget = std::min(kNlmMaxScratch, std::max(nlm_scratch_budget(), min_tile_bytes));
    int core_rows = 1;
    while (core_rows < h && stripe_bytes(core_rows + 1) <= effective_budget) {
        ++core_rows;
    }
    const std::size_t alloc_ext_rows = std::min(static_cast<std::size_t>(h),
                                                static_cast<std::size_t>(core_rows) +
                                                    2u * static_cast<std::size_t>(halo));
    const std::size_t floats = static_cast<std::size_t>(core_rows) *
                                   static_cast<std::size_t>(accum_arrays + saved_arrays) * static_cast<std::size_t>(stride) +
                               alloc_ext_rows * static_cast<std::size_t>(temp_arrays) *
                                   static_cast<std::size_t>(stride) + static_cast<std::size_t>(w);
    if (floats > kNlmMaxScratch / sizeof(float)) {
        for (const VSFrame* frame : src_frames) {
            vsapi->freeFrame(frame);
        }
        for (const VSFrame* frame : ref_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->setFilterError("nss.NLM: computed workspace exceeds the 1 MiB limit", frameCtx);
        return nullptr;
    }

    float* workspace = nullptr;
    try {
        workspace = d->ws.get(floats);
    } catch (...) {
        for (const VSFrame* frame : src_frames) {
            vsapi->freeFrame(frame);
        }
        for (const VSFrame* frame : ref_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->setFilterError("nss.NLM: workspace allocation failed", frameCtx);
        return nullptr;
    }

    VSFrame* dst_frame = nullptr;
    if (d->channels == nss::ChannelMode::Y && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{nullptr, src_center, src_center};
        const int pl[3]{0, 1, 2};
        dst_frame = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else if (d->channels == nss::ChannelMode::UV && d->vi.format.numPlanes > 1) {
        const VSFrame* fr[3]{src_center, nullptr, nullptr};
        const int pl[3]{0, 1, 2};
        dst_frame = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src_center, core);
    } else {
        dst_frame = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src_center, core);
    }
    const PlaneStrides dst_strides = plane_strides(dst_frame);
    bool mixed_strides = false;
    for (int t = 0; t < ntemp && !mixed_strides; ++t) {
        for (int c = 0; c < nc; ++c) {
            const int plane = (d->channels == nss::ChannelMode::UV) ? c + 1 : c;
            mixed_strides = mixed_strides || src_strides[static_cast<std::size_t>(t)][plane] != stride ||
                            ref_strides[static_cast<std::size_t>(t)][plane] != stride;
        }
    }
    for (int c = 0; c < nc; ++c) {
        const int plane = (d->channels == nss::ChannelMode::UV) ? c + 1 : c;
        mixed_strides = mixed_strides || dst_strides[plane] != stride;
    }

    auto ptrs = [&](const VSFrame* f) {
        PlanePtrs p{};
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

    auto ptrs_at = [&](const VSFrame* frame, int row0) {
        auto p = ptrs(frame);
        const auto st = plane_strides(frame);
        for (int c = 0; c < 3; ++c) {
            if (p[c]) {
                p[c] += static_cast<std::size_t>(row0) * static_cast<std::size_t>(st[c]);
            }
        }
        return p;
    };
    auto compact_ptrs = [&](const PlanePtrs& p) {
        PlanePtrs result{};
        if (d->channels == nss::ChannelMode::UV) {
            result[0] = p[1];
            result[1] = p[2];
        } else {
            result = p;
        }
        return result;
    };
    auto compact_strides = [&](const PlaneStrides& st) {
        PlaneStrides result{};
        if (d->channels == nss::ChannelMode::UV) {
            result[0] = st[1];
            result[1] = st[2];
        } else {
            result = st;
        }
        return result;
    };

    const float h2_inv_norm =
        (255.0f * 255.0f) / (3.0f * d->h * d->h * static_cast<float>((2 * d->s + 1) * (2 * d->s + 1)));

    auto process_stripe = [&](int core0, int core1) {
        const int ext0 = halo >= core0 ? 0 : core0 - halo;
        const int ext1 = halo >= h - core1 ? h : core1 + halo;
        const int ext_h = ext1 - ext0;
        const int core_local = core0 - ext0;
        const int core_height = core1 - core0;
        const std::size_t ext_size = static_cast<std::size_t>(ext_h) * static_cast<std::size_t>(stride);
        const std::size_t core_size = static_cast<std::size_t>(core_height) * static_cast<std::size_t>(stride);
        std::memset(workspace, 0, static_cast<std::size_t>(accum_arrays) * core_size * sizeof(float));
        float* weightp = workspace;
        std::array<float*, 3> wdstp{workspace + core_size, nc > 1 ? workspace + 2 * core_size : nullptr,
                                    nc > 2 ? workspace + 3 * core_size : nullptr};
        float* max_weightp = workspace + (1 + nc) * core_size;
        for (std::size_t i = 0; i < core_size; ++i) {
            max_weightp[i] = std::numeric_limits<float>::epsilon();
        }
        float* saved_bwd = workspace + static_cast<std::size_t>(accum_arrays) * core_size;
        float* temp = saved_bwd + core_size;
        float* temp_bwd = temp + ext_size;
        float* buffer = temp + static_cast<std::size_t>(temp_arrays) * ext_size;
        const auto refp = ptrs_at(ref_center, ext0);
        const auto refp_strides = ref_strides[static_cast<std::size_t>(d->d)];
        const int span = 2 * d->a + 1;

        for (int i = -d->d; i <= 0; ++i) {
            const int bwd_slot = d->d + i;
            const int fwd_slot = d->d - i;
            const auto srcp_bwd = ptrs_at(src_frames[static_cast<std::size_t>(bwd_slot)], ext0);
            const auto srcp_fwd = ptrs_at(src_frames[static_cast<std::size_t>(fwd_slot)], ext0);
            const auto refp_bwd = ptrs_at(ref_frames[static_cast<std::size_t>(bwd_slot)], ext0);
            const auto refp_fwd = ptrs_at(ref_frames[static_cast<std::size_t>(fwd_slot)], ext0);
            const auto srcp_bwd_strides = src_strides[static_cast<std::size_t>(bwd_slot)];
            const auto srcp_fwd_strides = src_strides[static_cast<std::size_t>(fwd_slot)];
            const auto refp_bwd_strides = ref_strides[static_cast<std::size_t>(bwd_slot)];
            const auto refp_fwd_strides = ref_strides[static_cast<std::size_t>(fwd_slot)];

            for (int oy = -d->a; oy <= d->a; ++oy) {
                for (int ox = -d->a; ox <= d->a; ++ox) {
                    if (i * span * span + oy * span + ox >= 0) {
                        continue;
                    }
                    auto distance = [&](float* dst, const PlanePtrs& c, const PlaneStrides& cs,
                                        const PlanePtrs& nb, const PlaneStrides& ns) {
                        if (mixed_strides) {
                            nss::nlm_distance_strided_f32(dst, c.data(), cs.data(), nb.data(), ns.data(), d->channels,
                                                           ox, oy, w, ext_h, stride);
                            return;
                        }
                        switch (d->channels) {
                        case nss::ChannelMode::Y:
                            nss::nlm_distance_luma_f32(dst, c[0], nb[0], ox, oy, w, ext_h, stride);
                            break;
                        case nss::ChannelMode::UV:
                            nss::nlm_distance_chroma_f32(dst, c[1], c[2], nb[1], nb[2], ox, oy, w, ext_h, stride);
                            break;
                        case nss::ChannelMode::YUV:
                            nss::nlm_distance_yuv_f32(dst, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, ext_h,
                                                       stride);
                            break;
                        case nss::ChannelMode::RGB:
                            nss::nlm_distance_rgb_f32(dst, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, ext_h,
                                                       stride);
                            break;
                        }
                    };
                    auto distance_horizontal = [&](const PlanePtrs& c, const PlaneStrides& cs,
                                                   const PlanePtrs& nb, const PlaneStrides& ns) {
                        if (!mixed_strides && d->channels == nss::ChannelMode::Y) {
                            nss::nlm_distance_luma_horizontal_f32(temp, temp_bwd, c[0], nb[0], ox, oy, d->s, w,
                                                                  ext_h, stride);
                            return;
                        }
                        distance(temp_bwd, c, cs, nb, ns);
                        nss::nlm_horizontal(temp, temp_bwd, d->s, w, ext_h, stride);
                    };
                    distance_horizontal(refp, refp_strides, refp_bwd, refp_bwd_strides);
                    nss::nlm_vertical_welsch(temp_bwd, temp, d->s, h2_inv_norm, w, ext_h, stride, buffer);
                    auto accum_range = [&](const PlanePtrs& sb, const PlaneStrides& sbs, const PlanePtrs& sf,
                                           const PlaneStrides& sfs, const float* t1, const float* t2, int y0, int y1,
                                           int temp_base) {
                        if (mixed_strides) {
                            const auto csb = compact_ptrs(sb);
                            const auto csf = compact_ptrs(sf);
                            const auto cssb = compact_strides(sbs);
                            const auto cssf = compact_strides(sfs);
                            nss::nlm_accum_strided(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, csb.data(),
                                                   cssb.data(), csf.data(), cssf.data(), t1, t2, nc, ox, oy, w, ext_h,
                                                   stride, y0, y1, temp_base);
                            return;
                        }
                        if (d->channels == nss::ChannelMode::Y) {
                            nss::nlm_accum_ch1_range(weightp, wdstp[0], max_weightp, sb[0], sf[0], t1, t2, ox, oy, w,
                                                     ext_h, stride, y0, y1);
                        } else if (d->channels == nss::ChannelMode::UV) {
                            nss::nlm_accum_ch2_range(weightp, wdstp[0], wdstp[1], max_weightp, sb[1], sb[2], sf[1], sf[2],
                                                     t1, t2, ox, oy, w, ext_h, stride, y0, y1);
                        } else {
                            nss::nlm_accum_ch3_range(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, sb[0], sb[1],
                                                     sb[2], sf[0], sf[1], sf[2], t1, t2, ox, oy, w, ext_h, stride, y0,
                                                     y1);
                        }
                    };
                    if (i == 0) {
                        accum_range(srcp_bwd, srcp_bwd_strides, srcp_bwd, srcp_bwd_strides, temp_bwd, temp_bwd,
                                     core_local, core_local + core_height, 0);
                        continue;
                    }
                    for (int yy = 0; yy < core_height; ++yy) {
                        std::memcpy(saved_bwd + static_cast<std::size_t>(yy) * static_cast<std::size_t>(stride),
                                    temp_bwd + static_cast<std::size_t>(core_local + yy) * static_cast<std::size_t>(stride),
                                    static_cast<std::size_t>(stride) * sizeof(float));
                    }
                    distance_horizontal(refp_fwd, refp_fwd_strides, refp, refp_strides);
                    const int temp2_base_y = mixed_strides ? 0 : std::clamp(core_local - oy, 0, ext_h - 1);
                    if (mixed_strides) {
                        nss::nlm_vertical_welsch(temp_bwd, temp, d->s, h2_inv_norm, w, ext_h, stride, buffer);
                    } else {
                        const int temp2_end_y = std::clamp(core_local + core_height - 1 - oy, 0, ext_h - 1) + 1;
                        nss::nlm_vertical_welsch_range(temp_bwd, temp, d->s, h2_inv_norm, w, ext_h, stride,
                                                       temp2_base_y, temp2_end_y, buffer);
                    }
                    if (mixed_strides) {
                        const auto csb = compact_ptrs(srcp_bwd);
                        const auto csf = compact_ptrs(srcp_fwd);
                        const auto cssb = compact_strides(srcp_bwd_strides);
                        const auto cssf = compact_strides(srcp_fwd_strides);
                        nss::nlm_accum_strided(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, csb.data(),
                                               cssb.data(), csf.data(), cssf.data(), saved_bwd, temp_bwd, nc, ox, oy,
                                               w, ext_h, stride, core_local, core_local + core_height, core_local);
                    } else if (d->channels == nss::ChannelMode::Y) {
                        nss::nlm_accum_ch1_core_range(weightp, wdstp[0], max_weightp, srcp_bwd[0], srcp_fwd[0],
                                                      saved_bwd, temp_bwd, ox, oy, w, ext_h, stride, core_local,
                                                      core_local + core_height, temp2_base_y);
                    } else if (d->channels == nss::ChannelMode::UV) {
                        nss::nlm_accum_ch2_core_range(weightp, wdstp[0], wdstp[1], max_weightp, srcp_bwd[1], srcp_bwd[2],
                                                      srcp_fwd[1], srcp_fwd[2], saved_bwd, temp_bwd, ox, oy, w, ext_h,
                                                      stride, core_local, core_local + core_height, temp2_base_y);
                    } else {
                        nss::nlm_accum_ch3_core_range(weightp, wdstp[0], wdstp[1], wdstp[2], max_weightp, srcp_bwd[0],
                                                      srcp_bwd[1], srcp_bwd[2], srcp_fwd[0], srcp_fwd[1], srcp_fwd[2],
                                                      saved_bwd, temp_bwd, ox, oy, w, ext_h, stride, core_local,
                                                      core_local + core_height, temp2_base_y);
                    }
                }
            }
        }

        auto srcp = ptrs_at(src_center, ext0);
        const auto src_center_strides = src_strides[static_cast<std::size_t>(d->d)];
        std::array<const float*, 3> src_core{};
        for (int c = 0; c < 3; ++c) {
            if (srcp[c]) {
                src_core[c] = srcp[c] + static_cast<std::size_t>(core_local) *
                                           static_cast<std::size_t>(src_center_strides[c]);
            }
        }
        std::array<float*, 3> dstp{};
        for (int c = 0; c < 3; ++c) {
            if (srcp[c]) {
                dstp[c] = reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, c)) +
                          static_cast<std::size_t>(core0) * static_cast<std::size_t>(dst_strides[c]);
            }
        }
        if (mixed_strides) {
            std::array<float*, 3> compact_dst{};
            std::array<const float*, 3> compact_src{};
            if (d->channels == nss::ChannelMode::UV) {
                compact_dst[0] = dstp[1];
                compact_dst[1] = dstp[2];
                compact_src[0] = src_core[1];
                compact_src[1] = src_core[2];
            } else {
                compact_dst = dstp;
                compact_src = src_core;
            }
            const auto compact_dst_strides = compact_strides(dst_strides);
            const auto compact_src_strides = compact_strides(src_center_strides);
            const std::array<const float*, 3> wdst_const{wdstp[0], wdstp[1], wdstp[2]};
            nss::nlm_finish_strided(compact_dst.data(), compact_dst_strides.data(), compact_src.data(),
                                    compact_src_strides.data(), weightp, wdst_const.data(), max_weightp, d->wref, nc,
                                    w, core_height, stride);
        } else if (d->channels == nss::ChannelMode::Y) {
            nss::nlm_finish_ch1(dstp[0], src_core[0], weightp, wdstp[0], max_weightp, d->wref, w, core_height, stride);
        } else if (d->channels == nss::ChannelMode::UV) {
            nss::nlm_finish_ch2(dstp[1], dstp[2], src_core[1], src_core[2], weightp, wdstp[0], wdstp[1], max_weightp,
                                d->wref, w, core_height, stride);
        } else {
            nss::nlm_finish_ch3(dstp[0], dstp[1], dstp[2], src_core[0], src_core[1], src_core[2], weightp, wdstp[0],
                                wdstp[1], wdstp[2], max_weightp, d->wref, w, core_height, stride);
        }
    };

    for (int y0 = 0; y0 < h; y0 += core_rows) {
        process_stripe(y0, std::min(h, y0 + core_rows));
    }
    for (const VSFrame* frame : src_frames) {
        vsapi->freeFrame(frame);
    }
    for (const VSFrame* frame : ref_frames) {
        vsapi->freeFrame(frame);
    }
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
    constexpr int kMaxSafeRadius = (std::numeric_limits<int>::max() - 1) / 2;
    if (d->d < 0 || d->a <= 0 || d->s < 0 || d->a > kMaxSafeRadius || d->s > kMaxSafeRadius ||
        !std::isfinite(d->h) || d->h <= 0.f || !std::isfinite(d->wref)) {
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
