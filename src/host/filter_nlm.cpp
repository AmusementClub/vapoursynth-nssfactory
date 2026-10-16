// SPDX-License-Identifier: GPL-2.0-only
#include "nss/resources.hpp"
#include "frontend/validate.hpp"
#include "frontend/temporal.hpp"
#include "frontend/nlm_args.hpp"
#include "nss/backend.hpp"
#include "nss/cpu_api.hpp"
#include "nss/params.hpp"
#include "nss/workspace.hpp"

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

struct NlmData : nss::NlmParams {
    std::shared_ptr<nss::ResourceBudget> budget = nss::current_budget();
    nss::NodeRef node;
    nss::NodeRef rclip;
    const VSAPI* vsapi = nullptr;
    VSVideoInfo vi{};
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

// The UV mode filters planes 1 and 2; the others start at plane 0.
int nlm_plane(const NlmData* d, int c) {
    return d->channels == nss::ChannelMode::UV ? c + 1 : c;
}

using PlanePtrs = std::array<const float*, 3>;
using PlaneStrides = std::array<int, 3>;

// Strides (in floats) of the filtered planes; the others stay 0.
PlaneStrides plane_strides(const NlmData* d, int nc, nss::FrameScope& frames, const VSFrame* frame) {
    PlaneStrides result{};
    for (int c = 0; c < nc; ++c) {
        const int plane = nlm_plane(d, c);
        result[plane] = static_cast<int>(frames.getStride(frame, plane) / sizeof(float));
    }
    return result;
}

// The 2d + 1 source and reference frames around n (slot d is n itself), their
// strides, and the widest stride of all (at least the plane width).
struct NlmWindow {
    nss::ResourceVector<const VSFrame*> src, ref;
    nss::ResourceVector<PlaneStrides> src_strides, ref_strides;
    int stride = 0;
};

NlmWindow load_window(const NlmData* d, int n, int w, int nc, nss::FrameScope& frames, VSFrameContext* frameCtx) {
    VSNode* ref_node = d->rclip ? d->rclip : d->node;
    const std::size_t ntemp = static_cast<std::size_t>(2 * d->d + 1);
    NlmWindow window{nss::ResourceVector<const VSFrame*>(ntemp), nss::ResourceVector<const VSFrame*>(ntemp),
                     nss::ResourceVector<PlaneStrides>(ntemp), nss::ResourceVector<PlaneStrides>(ntemp), w};
    for (std::size_t t = 0; t < ntemp; ++t) {
        const int fn = nss::host_detail::temporal_slot_frame(n, static_cast<int>(t), d->d, d->vi.numFrames);
        window.src[t] = frames.getFrameFilter(fn, d->node, frameCtx);
        window.ref[t] = frames.getFrameFilter(fn, ref_node, frameCtx);
    }
    for (std::size_t t = 0; t < ntemp; ++t) {
        window.src_strides[t] = plane_strides(d, nc, frames, window.src[t]);
        window.ref_strides[t] = plane_strides(d, nc, frames, window.ref[t]);
        for (int c = 0; c < nc; ++c) {
            const int plane = nlm_plane(d, c);
            window.stride = std::max(window.stride, window.src_strides[t][plane]);
            window.stride = std::max(window.stride, window.ref_strides[t][plane]);
        }
    }
    return window;
}

// Rows are filtered in stripes of core_rows, each with `halo` rows of context
// on both sides, so that one stripe's scratch fits half the L2 cache (at most
// 1 MiB). Per stripe: weight, max weight and the channel sums over the core
// rows, one saved map of the core rows (the backward half of a temporal
// pair), and two maps over the rows with context (horizontal and vertical).
struct NlmStripes {
    static constexpr int kTempArrays = 2;
    static constexpr int kSavedArrays = 1;
    int halo = 0;
    int accum_arrays = 0;
    int core_rows = 1;
    std::size_t floats = 0;
};

NlmStripes plan_stripes(const NlmData* d, int w, int h, int nc, int stride) {
    NlmStripes plan;
    plan.halo = d->a + d->s;
    plan.accum_arrays = 2 + nc;
    const auto stripe_bytes = [&](int core) {
        const std::size_t core_rows = static_cast<std::size_t>(std::max(core, 0));
        const std::size_t ext = std::min(static_cast<std::size_t>(h),
                                         core_rows + 2u * static_cast<std::size_t>(plan.halo));
        const std::size_t rows = core_rows * static_cast<std::size_t>(plan.accum_arrays) +
                                 core_rows * static_cast<std::size_t>(NlmStripes::kSavedArrays) +
                                 ext * static_cast<std::size_t>(NlmStripes::kTempArrays);
        return rows * static_cast<std::size_t>(stride) * sizeof(float) + static_cast<std::size_t>(w) * sizeof(float);
    };
    // A very small reported L2 can leave no useful core rows after accounting
    // for the halo. Raise the effective budget (still capped at 1 MiB) enough
    // to keep a minimum tile without making the per-thread scratch unbounded.
    constexpr std::size_t kMinCoreRows = 21;
    const std::size_t min_tile_bytes = stripe_bytes(static_cast<int>(kMinCoreRows));
    if (stripe_bytes(1) > kNlmMaxScratch) {
        throw std::runtime_error("nss.NLM: image stride/halo exceeds the 1 MiB workspace limit");
    }
    const std::size_t effective_budget = std::min(kNlmMaxScratch, std::max(nlm_scratch_budget(), min_tile_bytes));
    while (plan.core_rows < h && stripe_bytes(plan.core_rows + 1) <= effective_budget) {
        ++plan.core_rows;
    }
    const std::size_t alloc_ext_rows = std::min(static_cast<std::size_t>(h),
                                                static_cast<std::size_t>(plan.core_rows) +
                                                    2u * static_cast<std::size_t>(plan.halo));
    plan.floats = static_cast<std::size_t>(plan.core_rows) *
                      static_cast<std::size_t>(plan.accum_arrays + NlmStripes::kSavedArrays) *
                      static_cast<std::size_t>(stride) +
                  alloc_ext_rows * static_cast<std::size_t>(NlmStripes::kTempArrays) * static_cast<std::size_t>(stride) +
                  static_cast<std::size_t>(w);
    if (plan.floats > kNlmMaxScratch / sizeof(float)) {
        throw std::runtime_error("nss.NLM: computed workspace exceeds the 1 MiB limit");
    }
    return plan;
}

// The output frame; planes that are not filtered are taken from the source.
VSFrame* new_output_frame(const NlmData* d, nss::FrameScope& frames, const VSFrame* src, VSCore* core) {
    const bool planar = d->vi.format.numPlanes > 1;
    const int pl[3]{0, 1, 2};
    if (d->channels == nss::ChannelMode::Y && planar) {
        const VSFrame* fr[3]{nullptr, src, src};
        return frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src, core);
    }
    if (d->channels == nss::ChannelMode::UV && planar) {
        const VSFrame* fr[3]{src, nullptr, nullptr};
        return frames.newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, fr, pl, src, core);
    }
    return frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
}

// The UV mode's planes 1 and 2 moved to slots 0 and 1, for the strided kernels.
template <class T>
std::array<T, 3> compact(const NlmData* d, const std::array<T, 3>& planes) {
    if (d->channels != nss::ChannelMode::UV) return planes;
    return {planes[1], planes[2], T{}};
}

// One stripe's work: for every offset of the search window, the patch
// distances to the center frame (and, for a temporal pair, to its mirror
// frame), their Welsch weights, and the weighted sums; then the finish.
// Frames whose strides differ from `stride` take the strided kernels.
struct NlmPass {
    const NlmData* d;
    const VSAPI* vsapi;
    nss::FrameScope& frames;
    const NlmWindow& window;
    const NlmStripes& plan;
    int w, h, nc, stride;
    bool mixed_strides;
    float h2_inv_norm;
    float* workspace;
    VSFrame* dst;
    PlaneStrides dst_strides;

    // The stripe: its rows with context [ext0, ext0 + ext_h), its core rows
    // [core0, core0 + core_height) at core_local within them, and its scratch.
    struct Stripe {
        int ext0, ext_h, core0, core_local, core_height;
        float* weight;
        std::array<float*, 3> wdst;
        float* max_weight;
        float* saved_bwd;
        float* temp;
        float* temp_bwd;
        float* buffer;
    };

    PlanePtrs ptrs_at(const VSFrame* frame, int row0) const {
        PlanePtrs p{};
        for (int c = 0; c < nc; ++c) {
            const int plane = nlm_plane(d, c);
            p[plane] = reinterpret_cast<const float*>(vsapi->getReadPtr(frame, plane));
        }
        const auto st = plane_strides(d, nc, frames, frame);
        for (int c = 0; c < 3; ++c) {
            if (p[c]) p[c] += static_cast<std::size_t>(row0) * static_cast<std::size_t>(st[c]);
        }
        return p;
    }

    Stripe stripe(int core0, int core1) const {
        Stripe s{};
        s.ext0 = plan.halo >= core0 ? 0 : core0 - plan.halo;
        const int ext1 = plan.halo >= h - core1 ? h : core1 + plan.halo;
        s.ext_h = ext1 - s.ext0;
        s.core0 = core0;
        s.core_local = core0 - s.ext0;
        s.core_height = core1 - core0;
        const std::size_t ext_size = static_cast<std::size_t>(s.ext_h) * static_cast<std::size_t>(stride);
        const std::size_t core_size = static_cast<std::size_t>(s.core_height) * static_cast<std::size_t>(stride);
        std::memset(workspace, 0, static_cast<std::size_t>(plan.accum_arrays) * core_size * sizeof(float));
        s.weight = workspace;
        s.wdst = {workspace + core_size, nc > 1 ? workspace + 2 * core_size : nullptr,
                  nc > 2 ? workspace + 3 * core_size : nullptr};
        s.max_weight = workspace + (1 + nc) * core_size;
        for (std::size_t i = 0; i < core_size; ++i) s.max_weight[i] = std::numeric_limits<float>::epsilon();
        s.saved_bwd = workspace + static_cast<std::size_t>(plan.accum_arrays) * core_size;
        s.temp = s.saved_bwd + core_size;
        s.temp_bwd = s.temp + ext_size;
        s.buffer = s.temp + static_cast<std::size_t>(NlmStripes::kTempArrays) * ext_size;
        return s;
    }

    // Squared patch differences between frame c and frame nb shifted by (ox, oy).
    void distance(const Stripe& s, float* out, const PlanePtrs& c, const PlaneStrides& cs, const PlanePtrs& nb,
                  const PlaneStrides& ns, int ox, int oy) const {
        if (mixed_strides) {
            nss::nlm_distance_strided_f32(out, c.data(), cs.data(), nb.data(), ns.data(), d->channels, ox, oy, w,
                                           s.ext_h, stride);
            return;
        }
        switch (d->channels) {
        case nss::ChannelMode::Y:
            nss::nlm_distance_luma_f32(out, c[0], nb[0], ox, oy, w, s.ext_h, stride);
            break;
        case nss::ChannelMode::UV:
            nss::nlm_distance_chroma_f32(out, c[1], c[2], nb[1], nb[2], ox, oy, w, s.ext_h, stride);
            break;
        case nss::ChannelMode::YUV:
            nss::nlm_distance_yuv_f32(out, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, s.ext_h, stride);
            break;
        case nss::ChannelMode::RGB:
            nss::nlm_distance_rgb_f32(out, c[0], c[1], c[2], nb[0], nb[1], nb[2], ox, oy, w, s.ext_h, stride);
            break;
        }
    }

    // The distances summed over the patch rows, into s.temp.
    void distance_horizontal(const Stripe& s, const PlanePtrs& c, const PlaneStrides& cs, const PlanePtrs& nb,
                             const PlaneStrides& ns, int ox, int oy) const {
        if (!mixed_strides && d->channels == nss::ChannelMode::Y) {
            nss::nlm_distance_luma_horizontal_f32(s.temp, s.temp_bwd, c[0], nb[0], ox, oy, d->s, w, s.ext_h, stride);
            return;
        }
        distance(s, s.temp_bwd, c, cs, nb, ns, ox, oy);
        nss::nlm_horizontal(s.temp, s.temp_bwd, d->s, w, s.ext_h, stride);
    }

    // An offset within the center frame: one weight map serves both sides.
    void accumulate_center(const Stripe& s, const PlanePtrs& src, const PlaneStrides& src_st, int ox, int oy) const {
        const int core_end = s.core_local + s.core_height;
        if (mixed_strides) {
            nss::nlm_vertical_welsch(s.temp_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, s.buffer);
            const auto cs = compact(d, src);
            const auto css = compact(d, src_st);
            nss::nlm_accum_strided(s.weight, s.wdst[0], s.wdst[1], s.wdst[2], s.max_weight, cs.data(), css.data(),
                                   cs.data(), css.data(), s.temp_bwd, s.temp_bwd, nc, ox, oy, w, s.ext_h, stride,
                                   s.core_local, core_end, 0);
            return;
        }
        // Only the core rows plus the |oy| halo are consumed; the range pass
        // replays the same sliding trajectory as a full-ext pass, so values
        // are bit-identical with dead rows skipped.
        const int abs_oy = std::abs(oy);
        const int base_y = std::max(0, s.core_local - abs_oy);
        const int end_y = std::min(s.ext_h, core_end + abs_oy);
        nss::nlm_vertical_welsch_range(s.temp_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, base_y, end_y,
                                       s.buffer);
        if (d->channels == nss::ChannelMode::Y) {
            nss::nlm_accum_ch1_base_range(s.weight, s.wdst[0], s.max_weight, src[0], src[0], s.temp_bwd, s.temp_bwd,
                                          ox, oy, w, s.ext_h, stride, s.core_local, core_end, base_y);
        } else if (d->channels == nss::ChannelMode::UV) {
            nss::nlm_accum_ch2_base_range(s.weight, s.wdst[0], s.wdst[1], s.max_weight, src[1], src[2], src[1], src[2],
                                          s.temp_bwd, s.temp_bwd, ox, oy, w, s.ext_h, stride, s.core_local, core_end,
                                          base_y);
        } else {
            nss::nlm_accum_ch3_base_range(s.weight, s.wdst[0], s.wdst[1], s.wdst[2], s.max_weight, src[0], src[1],
                                          src[2], src[0], src[1], src[2], s.temp_bwd, s.temp_bwd, ox, oy, w, s.ext_h,
                                          stride, s.core_local, core_end, base_y);
        }
    }

    // A temporal pair: the backward frame's weights (already in s.temp) are
    // kept in s.saved_bwd while the forward frame's are computed, then both
    // sides are added.
    void accumulate_pair(const Stripe& s, const PlanePtrs& ref, const PlaneStrides& ref_st, const PlanePtrs& ref_fwd,
                         const PlaneStrides& ref_fwd_st, const PlanePtrs& src_bwd, const PlaneStrides& src_bwd_st,
                         const PlanePtrs& src_fwd, const PlaneStrides& src_fwd_st, int ox, int oy) const {
        const int core_end = s.core_local + s.core_height;
        if (mixed_strides) {
            nss::nlm_vertical_welsch(s.temp_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, s.buffer);
            for (int yy = 0; yy < s.core_height; ++yy) {
                std::memcpy(s.saved_bwd + static_cast<std::size_t>(yy) * static_cast<std::size_t>(stride),
                            s.temp_bwd + static_cast<std::size_t>(s.core_local + yy) * static_cast<std::size_t>(stride),
                            static_cast<std::size_t>(stride) * sizeof(float));
            }
        } else {
            // Write the consumed core rows directly into saved_bwd: same
            // sliding trajectory as a full-ext pass, with the dead rows and
            // the core-row memcpy skipped.
            nss::nlm_vertical_welsch_range(s.saved_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, s.core_local,
                                           core_end, s.buffer);
        }
        distance_horizontal(s, ref_fwd, ref_fwd_st, ref, ref_st, ox, oy);
        if (mixed_strides) {
            nss::nlm_vertical_welsch(s.temp_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, s.buffer);
            const auto csb = compact(d, src_bwd);
            const auto csf = compact(d, src_fwd);
            const auto cssb = compact(d, src_bwd_st);
            const auto cssf = compact(d, src_fwd_st);
            nss::nlm_accum_strided(s.weight, s.wdst[0], s.wdst[1], s.wdst[2], s.max_weight, csb.data(), cssb.data(),
                                   csf.data(), cssf.data(), s.saved_bwd, s.temp_bwd, nc, ox, oy, w, s.ext_h, stride,
                                   s.core_local, core_end, s.core_local);
            return;
        }
        const int base_y = std::clamp(s.core_local - oy, 0, s.ext_h - 1);
        const int end_y = std::clamp(core_end - 1 - oy, 0, s.ext_h - 1) + 1;
        nss::nlm_vertical_welsch_range(s.temp_bwd, s.temp, d->s, h2_inv_norm, w, s.ext_h, stride, base_y, end_y,
                                       s.buffer);
        if (d->channels == nss::ChannelMode::Y) {
            nss::nlm_accum_ch1_core_range(s.weight, s.wdst[0], s.max_weight, src_bwd[0], src_fwd[0], s.saved_bwd,
                                          s.temp_bwd, ox, oy, w, s.ext_h, stride, s.core_local, core_end, base_y);
        } else if (d->channels == nss::ChannelMode::UV) {
            nss::nlm_accum_ch2_core_range(s.weight, s.wdst[0], s.wdst[1], s.max_weight, src_bwd[1], src_bwd[2],
                                          src_fwd[1], src_fwd[2], s.saved_bwd, s.temp_bwd, ox, oy, w, s.ext_h, stride,
                                          s.core_local, core_end, base_y);
        } else {
            nss::nlm_accum_ch3_core_range(s.weight, s.wdst[0], s.wdst[1], s.wdst[2], s.max_weight, src_bwd[0],
                                          src_bwd[1], src_bwd[2], src_fwd[0], src_fwd[1], src_fwd[2], s.saved_bwd,
                                          s.temp_bwd, ox, oy, w, s.ext_h, stride, s.core_local, core_end, base_y);
        }
    }

    // The weighted mean of the core rows into the output frame.
    void finish(const Stripe& s) const {
        const VSFrame* src_center = window.src[static_cast<std::size_t>(d->d)];
        const PlanePtrs srcp = ptrs_at(src_center, s.ext0);
        const PlaneStrides src_st = window.src_strides[static_cast<std::size_t>(d->d)];
        PlanePtrs src{};
        std::array<float*, 3> out{};
        for (int c = 0; c < 3; ++c) {
            if (!srcp[c]) continue;
            src[c] = srcp[c] + static_cast<std::size_t>(s.core_local) * static_cast<std::size_t>(src_st[c]);
            out[c] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, c)) +
                     static_cast<std::size_t>(s.core0) * static_cast<std::size_t>(dst_strides[c]);
        }
        if (mixed_strides) {
            const auto out_c = compact(d, out);
            const auto src_c = compact(d, src);
            const auto out_st = compact(d, dst_strides);
            const auto src_st_c = compact(d, src_st);
            const std::array<const float*, 3> wdst{s.wdst[0], s.wdst[1], s.wdst[2]};
            nss::nlm_finish_strided(out_c.data(), out_st.data(), src_c.data(), src_st_c.data(), s.weight, wdst.data(),
                                    s.max_weight, d->wref, nc, w, s.core_height, stride);
        } else if (d->channels == nss::ChannelMode::Y) {
            nss::nlm_finish_ch1(out[0], src[0], s.weight, s.wdst[0], s.max_weight, d->wref, w, s.core_height, stride);
        } else if (d->channels == nss::ChannelMode::UV) {
            nss::nlm_finish_ch2(out[1], out[2], src[1], src[2], s.weight, s.wdst[0], s.wdst[1], s.max_weight, d->wref,
                                w, s.core_height, stride);
        } else {
            nss::nlm_finish_ch3(out[0], out[1], out[2], src[0], src[1], src[2], s.weight, s.wdst[0], s.wdst[1],
                                s.wdst[2], s.max_weight, d->wref, w, s.core_height, stride);
        }
    }

    // Core rows [core0, core1). Frame pairs (n - k, n + k) for k = d .. 0;
    // offsets in raster order, each pair of opposite offsets once (the one
    // before the center in (frame, row, column) order).
    void run(int core0, int core1) const {
        const Stripe s = stripe(core0, core1);
        const std::size_t center = static_cast<std::size_t>(d->d);
        const PlanePtrs ref = ptrs_at(window.ref[center], s.ext0);
        const PlaneStrides ref_st = window.ref_strides[center];
        const int span = 2 * d->a + 1;
        for (int i = -d->d; i <= 0; ++i) {
            const std::size_t bwd = static_cast<std::size_t>(d->d + i);
            const std::size_t fwd = static_cast<std::size_t>(d->d - i);
            const PlanePtrs src_bwd = ptrs_at(window.src[bwd], s.ext0);
            const PlanePtrs src_fwd = ptrs_at(window.src[fwd], s.ext0);
            const PlanePtrs ref_bwd = ptrs_at(window.ref[bwd], s.ext0);
            const PlanePtrs ref_fwd = ptrs_at(window.ref[fwd], s.ext0);
            for (int oy = -d->a; oy <= d->a; ++oy) {
                for (int ox = -d->a; ox <= d->a; ++ox) {
                    // int64 every factor: oy*span overflows int for
                    // a >= 32769 (reachable under the a < width contract with
                    // small frame heights), i*span*span for smaller a.
                    if (static_cast<std::int64_t>(i) * span * span + static_cast<std::int64_t>(oy) * span + ox >= 0) {
                        continue;
                    }
                    distance_horizontal(s, ref, ref_st, ref_bwd, window.ref_strides[bwd], ox, oy);
                    if (i == 0) {
                        accumulate_center(s, src_bwd, window.src_strides[bwd], ox, oy);
                    } else {
                        accumulate_pair(s, ref, ref_st, ref_fwd, window.ref_strides[fwd], src_bwd,
                                        window.src_strides[bwd], src_fwd, window.src_strides[fwd], ox, oy);
                    }
                }
            }
        }
        finish(s);
    }
};

bool has_mixed_strides(const NlmData* d, int nc, const NlmWindow& window, const PlaneStrides& dst_strides) {
    for (std::size_t t = 0; t < window.src.size(); ++t) {
        for (int c = 0; c < nc; ++c) {
            const int plane = nlm_plane(d, c);
            if (window.src_strides[t][plane] != window.stride || window.ref_strides[t][plane] != window.stride) {
                return true;
            }
        }
    }
    for (int c = 0; c < nc; ++c) {
        if (dst_strides[nlm_plane(d, c)] != window.stride) return true;
    }
    return false;
}

const VSFrame* VS_CC nlmGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                 VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<NlmData*>(instanceData);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames_owned(vsapi);
    (void)frameData;
    if (activationReason == arInitial) {
        const int start = std::max(0, n - d->d);
        const int end = nss::host_detail::temporal_last(n, d->d, d->vi.numFrames);
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

    const int w = nlm_width(d);
    const int h = nlm_height(d);
    const int nc = num_channels(d->channels);
    const NlmWindow window = load_window(d, n, w, nc, frames_owned, frameCtx);
    const NlmStripes plan = plan_stripes(d, w, h, nc, window.stride);
    float* workspace = nullptr;
    try {
        workspace = d->ws.get(plan.floats);
    } catch (...) {
        throw std::runtime_error("nss.NLM: workspace allocation failed");
    }
    VSFrame* dst = new_output_frame(d, frames_owned, window.src[static_cast<std::size_t>(d->d)], core);
    const PlaneStrides dst_strides = plane_strides(d, nc, frames_owned, dst);
    const float h2_inv_norm =
        (255.0f * 255.0f) / (3.0f * d->h * d->h * (static_cast<float>(2 * d->s + 1) * static_cast<float>(2 * d->s + 1)));
    const NlmPass pass{d, vsapi, frames_owned, window, plan, w, h, nc, window.stride,
                       has_mixed_strides(d, nc, window, dst_strides), h2_inv_norm, workspace, dst, dst_strides};
    for (int y0 = 0; y0 < h; y0 += plan.core_rows) {
        pass.run(y0, std::min(h, y0 + plan.core_rows));
    }
    for (const VSFrame* frame : window.src) {
        frames_owned.freeFrame(frame);
    }
    for (const VSFrame* frame : window.ref) {
        frames_owned.freeFrame(frame);
    }
    return frames_owned.keep(dst);
}

void VS_CC nlmFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    (void)core;
    auto* d = static_cast<NlmData*>(instanceData);
    d->node.reset();
    if (d->rclip) {
        d->rclip.reset();
    }
    delete d;
}

void VS_CC nlmCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    (void)userData;
    if (!nss::cpu_backend_available()) {
        vsapi->mapSetError(out, "nss.NLM: no executable CPU backend (AVX2 on x86; NEON on AArch64)");
        return;
    }
    auto d = std::make_unique<NlmData>();
    d->vsapi = vsapi;
    d->node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const char* msg) {
        vsapi->mapSetError(out, msg);
        d->node.reset();
        if (d->rclip) {
            d->rclip.reset();
            d->rclip = nullptr;
        }
    };
    static_cast<nss::NlmParams&>(*d) = nss::frontend::parse_nlm(vsapi, in, d->vi, "nss");
    int err = 0;
    d->rclip = nss::get_node(vsapi, in, "rclip", 0, &err);
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
    VSNode* node = vsapi->createVideoFilter2("NLM", &raw->vi, nss::checked_frame<nlmGetFrame>, nlmFree,
                                            fmParallel, deps, ndeps, raw, core);
    if (!node) { vsapi->mapSetError(out, "nss.NLM: failed to create filter"); return; }
    d.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlm(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->registerFunction("NLM", nss::frontend::kNlmSignature,
                             "clip:vnode;", nss::checked_create<nlmCreate>, nullptr, plugin);
}
