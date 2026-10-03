// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NLH: parse_full_image with the "nss_cuda" prefix (D14) around a
// device-resident full-image pipeline (the CPU nlh_image model):
//   - upload the request window, RGB -> YUV, optional blind noise estimate,
//   - Basic rounds over every frame of the window (mix with the input, match
//     on the luma guide, hard threshold, per-pixel aggregation, finish),
//   - one Wiener round for the center frame against the Basic estimate,
//   - YUV -> RGB on the numerators, then the final frame or the fat
//     intermediate.
// Planes of equal geometry share matching and pixel selection; subsampled
// chroma matches on the area-averaged luma.
#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "cuda/nlh/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/contribution.hpp"
#include "frontend/full_image_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/temporal.hpp"
#include "frontend/validate.hpp"
#include "nss/checked.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nss_cuda {
namespace {

// Upper bound on the per-batch group workspace.
constexpr std::size_t kArenaBytes = 192u << 20;
// Aggregation capacity per arena byte (patches); small groups are capped by it.
constexpr std::size_t kArenaBytesPerPatch = 1024;

struct Shape {
    int block, group, q, nch;
    bool wiener;
};

// Device bytes one group needs in the arena for `s` (see NlhGroupArgs).
std::size_t group_bytes(const Shape& s) {
    const std::size_t m = static_cast<std::size_t>(s.block) * s.block;
    const std::size_t matrices = static_cast<std::size_t>(s.nch) * m * s.q * s.group;
    const std::size_t coef = nlh_filter_fused(s.block, s.group, s.q)
                                 ? 0
                                 : matrices * (s.wiener && !nlh_local_matrix(s.group, s.q) ? 2 : 1);
    const std::size_t guide = nlh_prepare_shared(s.block, s.group) ? 0 : m * s.group;
    const std::size_t floats = guide + coef + static_cast<std::size_t>(s.nch) * s.group * m + m;
    return floats * sizeof(float) + m * s.q * sizeof(int) + sizeof(int) + (s.nch + 1) * sizeof(double) +
           static_cast<std::size_t>(s.group) * (sizeof(DeviceMatch) + sizeof(AggregatePatch));
}

struct Slot {
    Stream stream;
    DeviceBuffer input, basic, reference, resized, num, den;  // planes of `plane_floats`
    DeviceBuffer guide_ptrs, data_ptrs, basic_ptrs;           // device pointer arrays
    DeviceBuffer arena, totals;
    PinnedBuffer staging;
    std::unique_ptr<OrderedAggregator> aggregator;
};

struct NlhData : nss::FullImageParams {
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node, reference;
    VSVideoInfo vi{}, output{};
    BackendArgs backend;
    DeviceInfo device;
    bool rgb = false;
    int planes = 1, max_frames = 1, reference_planes = 0;
    int width[3]{}, height[3]{};
    std::size_t plane_floats = 0, arena_bytes = 0, max_patches = 0;
    std::unique_ptr<SlotPool<Slot>> pool;
    int radius() const { return nlh.radius; }
};

// Per-frame state on one slot.
struct Frame {
    const NlhData& d;
    Slot& s;
    int count;   // frames in the request window
    int center;  // index of the output frame in it
    std::array<std::array<float, 3>, 33> sigma{};
    std::array<std::array<double, 3>, 33> units{};
    std::uint64_t groups = 0;

    float* plane(const DeviceBuffer& buffer, int t, int c) const {
        return buffer.as<float>() + (static_cast<std::size_t>(t) * 3 + c) * d.plane_floats;
    }
    // Numerator/denominator of channel c: `count` contiguous slices.
    float* accum(const DeviceBuffer& buffer, int c, int t = 0) const {
        return buffer.as<float>() + (static_cast<std::size_t>(c) * d.max_frames + t) * d.plane_floats;
    }
    std::uint8_t* staging(int region) const {
        return s.staging.as<std::uint8_t>() + static_cast<std::size_t>(region) * d.plane_floats * sizeof(float);
    }
    std::size_t floats(int c) const { return static_cast<std::size_t>(d.width[c]) * d.height[c]; }
};

// Planes of equal geometry starting at `first` (the CPU grouping order).
int geometry_group(const NlhData& d, int first, std::array<bool, 3>& used, std::array<int, 3>& channels) {
    int nch = 0;
    for (int c = first; c < d.planes; ++c) {
        if (!used[c] && d.width[c] == d.width[first] && d.height[c] == d.height[first]) {
            channels[nch++] = c;
            used[c] = true;
        }
    }
    return nch;
}

// Device pointer array from host pointers (pageable source: staged before
// cudaMemcpyAsync returns, stream-ordered after earlier kernels).
template <class T>
void set_pointers(const DeviceBuffer& buffer, const std::vector<T*>& host, cudaStream_t stream) {
    NSS_CUDA_CHECK(cudaMemcpyAsync(buffer.get(), host.data(), host.size() * sizeof(T*), cudaMemcpyHostToDevice, stream));
}

// Matching planes of the window for a geometry: the luma of `luma_source`
// (reference clip when present), area-averaged for subsampled planes.
std::vector<const float*> guides(Frame& f, const DeviceBuffer& luma_source, int width, int height) {
    const NlhData& d = f.d;
    std::vector<const float*> out(f.count);
    for (int t = 0; t < f.count; ++t) {
        const float* luma = d.reference ? f.plane(f.s.reference, t, 0) : f.plane(luma_source, t, 0);
        if (width == d.width[0] && height == d.height[0]) {
            out[t] = luma;
        } else {
            if (d.width[0] % width || d.height[0] % height) throw std::invalid_argument("nss: unsupported luminance guide grid");
            float* guide = f.s.resized.as<float>() + static_cast<std::size_t>(t) * d.plane_floats;
            nlh_area_guide(luma, d.width[0], guide, width, height, d.width[0] / width, d.height[0] / height, f.s.stream);
            out[t] = guide;
        }
    }
    return out;
}

// Splits the arena for `batch` groups of `shape`.
NlhGroupArgs layout(const Frame& f, const Shape& shape, int batch, double** partial, double** sums) {
    const std::size_t m = static_cast<std::size_t>(shape.block) * shape.block;
    std::uint8_t* at = f.s.arena.as<std::uint8_t>();
    const auto take = [&](std::size_t bytes) {
        std::uint8_t* p = at;
        at += (bytes + 15) & ~std::size_t{15};
        return p;
    };
    NlhGroupArgs a{};
    const std::size_t b = static_cast<std::size_t>(batch);
    if (partial) *partial = reinterpret_cast<double*>(take(b * shape.nch * sizeof(double)));
    if (sums) *sums = reinterpret_cast<double*>(take((b / 256 + 1) * shape.nch * sizeof(double)));
    a.matches = reinterpret_cast<DeviceMatch*>(take(b * shape.group * sizeof(DeviceMatch)));
    a.patches = reinterpret_cast<AggregatePatch*>(take(b * shape.group * sizeof(AggregatePatch)));
    a.counts = reinterpret_cast<int*>(take(b * sizeof(int)));
    a.indices = reinterpret_cast<int*>(take(b * m * shape.q * sizeof(int)));
    if (!nlh_prepare_shared(shape.block, shape.group)) {
        a.guide_group = reinterpret_cast<float*>(take(b * m * shape.group * sizeof(float)));
    }
    if (!nlh_filter_fused(shape.block, shape.group, shape.q)) {
        const std::size_t coef = b * shape.nch * m * shape.q * shape.group * sizeof(float);
        a.coef = reinterpret_cast<float*>(take(coef));
        if (shape.wiener && !nlh_local_matrix(shape.group, shape.q)) a.ref_coef = reinterpret_cast<float*>(take(coef));
    }
    a.values = reinterpret_cast<float*>(take(b * shape.nch * shape.group * m * sizeof(float)));
    a.den = reinterpret_cast<float*>(take(b * m * sizeof(float)));
    if (static_cast<std::size_t>(at - f.s.arena.as<std::uint8_t>()) > f.d.arena_bytes) {
        throw std::logic_error("nss_cuda.NLH: group arena overflow");
    }
    a.batch = batch;
    a.block = shape.block;
    a.group = shape.group;
    a.q = shape.q;
    a.nch = shape.nch;
    a.wiener = shape.wiener;
    return a;
}

int batch_for(const NlhData& d, const Shape& shape, int total) {
    // Alignment slack of the arena split: 16 bytes per region.
    const std::size_t fit = (d.arena_bytes - 16 * 12) / (group_bytes(shape) + 16);
    const std::size_t cap = d.max_patches / shape.group;
    return static_cast<int>(std::max<std::size_t>(1, std::min({fit, cap, static_cast<std::size_t>(total)})));
}

MatchGeometry match_geometry(int width, int height, int block, int window, int group) {
    MatchGeometry g{width, height, width, block, window / 2, group};
    g.range_hi = window - window / 2 - 1;
    return g;
}

// nss::nlh_estimate_frame_sigma on the device: fixed 8x8 / 16 / q4 / W40 /
// step 1; planes of equal geometry share matching and pixel selection.
void estimate_sigma(Frame& f, int t) {
    const NlhData& d = f.d;
    Slot& s = f.s;
    std::array<bool, 3> used{};
    for (int first = 0; first < d.planes; ++first) {
        if (used[first]) continue;
        std::array<int, 3> channels{};
        const int nch = geometry_group(d, first, used, channels);
        const int width = d.width[first], height = d.height[first];
        if (width < 8 || height < 8) {
            throw std::invalid_argument("nss: blind noise estimation requires an 8x8 or larger plane");
        }
        // The fixed bootstrap search of the estimate (nss::ImageSearch defaults).
        const struct {
            int block = 8, step = 1, group = 16, window = 40;
        } search;
        const Shape shape{search.block, search.group, 4, nch, false};
        // Single-frame matching on this frame's guide.
        const float* luma = d.reference ? f.plane(s.reference, t, 0) : f.plane(s.input, t, 0);
        const float* guide = luma;
        if (width != d.width[0] || height != d.height[0]) {
            if (d.width[0] % width || d.height[0] % height) throw std::invalid_argument("nss: unsupported luminance guide grid");
            float* resized = s.resized.as<float>();
            nlh_area_guide(luma, d.width[0], resized, width, height, d.width[0] / width, d.height[0] / height, s.stream);
            guide = resized;
        }
        std::vector<const float*> guide_ptr{guide}, data_ptr(nch);
        for (int c = 0; c < nch; ++c) data_ptr[c] = f.plane(s.input, t, channels[c]);
        set_pointers(s.guide_ptrs, guide_ptr, s.stream);
        set_pointers(s.data_ptrs, data_ptr, s.stream);
        NSS_CUDA_CHECK(cudaMemsetAsync(s.totals.get(), 0, 3 * sizeof(double), s.stream));
        const RasterGrid grid = make_raster_grid(width, height, shape.block, search.step);
        const MatchGeometry geometry = match_geometry(width, height, shape.block, search.window, shape.group);
        const int batch = batch_for(d, shape, grid.count());
        for (int begin = 0; begin < grid.count(); begin += batch) {
            const int n = std::min(batch, grid.count() - begin);
            double *partial = nullptr, *sums = nullptr;
            NlhGroupArgs a = layout(f, shape, n, &partial, &sums);
            a.guides = s.guide_ptrs.as<const float*>();
            a.data = s.data_ptrs.as<const float*>();
            a.width = width;
            a.pow2 = false;
            spatial_match(guide, geometry, grid, begin, n, const_cast<DeviceMatch*>(a.matches),
                          const_cast<int*>(a.counts), s.stream);
            nlh_prepare_groups(a, s.stream);
            nlh_sigma_groups(a, partial, sums, s.totals.as<double>(), s.stream);
        }
        double totals[3]{};
        NSS_CUDA_CHECK(cudaMemcpyAsync(totals, s.totals.get(), sizeof(totals), cudaMemcpyDeviceToHost, s.stream));
        s.stream.synchronize();
        for (int c = 0; c < nch; ++c) {
            const double sigma = totals[c] / static_cast<double>(grid.count());
            if (!std::isfinite(sigma)) throw std::runtime_error("nss: nonfinite blind noise estimate");
            f.sigma[t][channels[c]] = static_cast<float>(sigma);
            f.units[t][channels[c]] = static_cast<double>(f.sigma[t][channels[c]]) * 255;
        }
    }
}

// One Basic or Wiener round (nss::nlh_pass): groups of every reference frame
// (Basic) or of the center (Wiener) into the slot's num/den slices.
void run_pass(Frame& f, const DeviceBuffer& data, const nss::NlhImageOptions& o, int stage, bool wiener) {
    const NlhData& d = f.d;
    Slot& s = f.s;
    std::array<bool, 3> used{};
    for (int first = 0; first < d.planes; ++first) {
        if (used[first]) continue;
        std::array<int, 3> channels{};
        const int nch = geometry_group(d, first, used, channels);
        const int width = d.width[first], height = d.height[first];
        bool all_off = true;
        for (int t = 0; t < f.count; ++t) {
            for (int c = 0; c < nch; ++c) all_off = all_off && f.sigma[t][channels[c]] == 0;
        }
        if (all_off) {
            // Nothing is aggregated: the finish falls back to the source.
            for (int c = 0; c < nch; ++c) {
                const std::size_t bytes = static_cast<std::size_t>(d.max_frames) * d.plane_floats * sizeof(float);
                NSS_CUDA_CHECK(cudaMemsetAsync(f.accum(s.num, channels[c]), 0, bytes, s.stream));
                NSS_CUDA_CHECK(cudaMemsetAsync(f.accum(s.den, channels[c]), 0, bytes, s.stream));
            }
            continue;
        }
        const Shape shape{o.block[stage], o.group[stage], o.q[stage], nch, wiener};
        if (width < shape.block || height < shape.block || o.step[stage] < 1 || o.step[stage] > shape.block) {
            throw std::invalid_argument("nss: selected plane is smaller than block_size or invalid step");
        }
        const std::vector<const float*> guide_ptr = guides(f, s.basic, width, height);
        std::vector<const float*> data_ptr(static_cast<std::size_t>(f.count) * nch), basic_ptr(data_ptr.size());
        for (int t = 0; t < f.count; ++t) {
            for (int c = 0; c < nch; ++c) {
                data_ptr[t * nch + c] = f.plane(data, t, channels[c]);
                basic_ptr[t * nch + c] = f.plane(s.basic, t, channels[c]);
            }
        }
        set_pointers(s.guide_ptrs, guide_ptr, s.stream);
        set_pointers(s.data_ptrs, data_ptr, s.stream);
        set_pointers(s.basic_ptrs, basic_ptr, s.stream);
        const RasterGrid grid = make_raster_grid(width, height, shape.block, o.step[stage]);
        MatchGeometry geometry = match_geometry(width, height, shape.block, o.window[stage], shape.group);
        TemporalWindow window{};
        window.frames = s.guide_ptrs.as<const float*>();
        window.ntemp = f.count;
        window.radius = o.radius;
        window.valid_begin = 0;
        window.valid_end = f.count;
        window.ps_num = o.ps_num;
        window.ps_range = o.ps_range;
        const int batch = batch_for(d, shape, grid.count());
        bool written = false;
        for (int t0 = wiener ? f.center : 0; t0 < (wiener ? f.center + 1 : f.count); ++t0) {
            window.t0 = t0;
            for (int begin = 0; begin < grid.count(); begin += batch) {
                const int n = std::min(batch, grid.count() - begin);
                NlhGroupArgs a = layout(f, shape, n, nullptr, nullptr);
                a.guides = s.guide_ptrs.as<const float*>();
                a.data = s.data_ptrs.as<const float*>();
                a.reference = wiener ? s.basic_ptrs.as<const float*>() : nullptr;
                a.width = width;
                a.pow2 = true;
                a.wiener_iterations = o.wiener_iterations;
                for (int c = 0; c < nch; ++c) {
                    const float sigma = f.sigma[t0][channels[c]];
                    a.threshold[c] = nlh_float_threshold(nss::kNlhHardCoefficient * o.hard_strength * sigma);
                    a.noise[c] = (o.wiener_sigma_scale * sigma) * (o.wiener_sigma_scale * sigma);
                    a.identity[c] = sigma == 0;
                }
                {
                    NSS_CUDA_RANGE("nlh.match");
                    auto* matches = const_cast<DeviceMatch*>(a.matches);
                    auto* counts = const_cast<int*>(a.counts);
                    if (o.radius > 0 && f.count > 1) {
                        predictive_match(geometry, window, grid, begin, n, matches, counts, s.stream);
                    } else {
                        spatial_match(guide_ptr[t0], geometry, grid, begin, n, matches, counts, s.stream);
                    }
                }
                {
                    NSS_CUDA_RANGE("nlh.filter");
                    nlh_prepare_groups(a, s.stream);
                    nlh_filter_groups(a, s.stream);
                }
                NSS_CUDA_RANGE("nlh.aggregate");
                const std::size_t channel_values = static_cast<std::size_t>(n) * shape.group * shape.block * shape.block;
                for (int c = 0; c < nch; ++c) {
                    const AggregateTarget target{f.accum(s.num, channels[c]), f.accum(s.den, channels[c]), width, height,
                                                 width, f.count, d.plane_floats};
                    s.aggregator->run(a.values + c * channel_values, a.patches, n * shape.group, shape.block, target,
                                      s.stream, written, a.den, shape.group);
                }
                written = true;
            }
            f.groups += static_cast<std::uint64_t>(grid.count()) * nch;
        }
    }
}

void diagnostic_ints(const VSAPI* api, VSMap* props, const char* key, const int* values, int count) {
    std::int64_t data[2]{values[0], count > 1 ? values[1] : 0};
    if (api->mapSetIntArray(props, key, data, count)) throw std::bad_alloc();
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* api) {
    auto& d = *static_cast<NlhData*>(instance);
    const int radius = d.radius();
    const int first = nss::host_detail::temporal_first(n, radius);
    const int last = nss::host_detail::temporal_last(n, radius, d.vi.numFrames);
    if (activation == arInitial) {
        for (int frame = first; frame <= last; ++frame) {
            api->requestFrameFilter(frame, d.node, ctx);
            if (d.reference) api->requestFrameFilter(frame, d.reference, ctx);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("nlh.frame");
    auto slot = d.pool->acquire();
    Slot& s = *slot;
    DeviceGuard guard(d.device.index);
    nss::ResourceScope scope(d.budget);
    nss::FrameScope owned(api);
    Frame f{d, s, last - first + 1, n - first};
    const int planes = d.planes;

    // Upload the window: every source plane, and the reference planes the
    // luma guide needs.
    std::vector<const VSFrame*> source(f.count);
    int region = 0;
    const auto upload = [&](const VSFrame* frame, int c, float* device) {
        const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
        std::uint8_t* staging = f.staging(region++);
        upload_plane(api->getReadPtr(frame, c), api->getStride(frame, c), row_bytes, d.height[c], staging, device,
                     row_bytes, s.stream);
        const float* samples = reinterpret_cast<const float*>(staging);
        for (std::size_t i = 0; i < f.floats(c); ++i) {
            if (!std::isfinite(samples[i])) throw std::invalid_argument("nss: nonfinite source/reference sample");
        }
    };
    try {
        for (int t = 0; t < f.count; ++t) {
            source[t] = owned.getFrameFilter(first + t, d.node, ctx);
            for (int c = 0; c < planes; ++c) {
                upload(source[t], c, f.plane(s.input, t, c));
                f.sigma[t][c] = d.sigma[c] / 255.f;
                f.units[t][c] = d.sigma_units[c];
            }
            if (d.reference) {
                const VSFrame* frame = owned.getFrameFilter(first + t, d.reference, ctx);
                for (int c = 0; c < d.reference_planes; ++c) upload(frame, c, f.plane(s.reference, t, c));
                owned.freeFrame(frame);
            }
        }
    } catch (...) {
        s.stream.synchronize();  // staged copies must not outlive this frame
        throw;
    }
    for (int t = 0; t < f.count && d.rgb; ++t) {
        nlh_rgb_to_yuv(f.plane(s.input, t, 0), f.plane(s.input, t, 1), f.plane(s.input, t, 2), d.plane_floats, false,
                       s.stream);
        if (!d.estimate) nss::nlh_rgb_sigma_to_yuv(f.sigma[t], f.units[t]);
        if (d.reference) {
            nlh_rgb_to_yuv(f.plane(s.reference, t, 0), f.plane(s.reference, t, 1), f.plane(s.reference, t, 2),
                           d.plane_floats, true, s.stream);
        }
    }
    if (d.estimate) {
        NSS_CUDA_RANGE("nlh.estimate");
        for (int t = 0; t < f.count; ++t) estimate_sigma(f, t);
    }
    const auto sigma_units = [&](int t, int c) {
        return f.units[t][c] >= 0 ? f.units[t][c] : static_cast<double>(f.sigma[t][c]) * 255;
    };
    bool bypass = true;
    for (int c = 0; c < planes; ++c) bypass = bypass && f.sigma[f.center][c] == 0;
    // nss::nlh_resolve_options: a bypassed center resolves from itself alone.
    double max_sigma = 0;
    int available = 16;
    for (int c = 0; c < planes; ++c) {
        const double units = sigma_units(f.center, c);
        if (!(units >= 0) || !std::isfinite(units)) throw std::invalid_argument("nss.NLH: invalid preset sigma");
        max_sigma = std::max(max_sigma, units);
        for (int t = bypass ? f.center : 0; t < (bypass ? f.center + 1 : f.count); ++t) {
            if (f.sigma[t][c] > 0) available = std::min({available, d.width[c], d.height[c]});
        }
    }
    const nss::NlhImageOptions o = nss::nlh_resolve_preset(max_sigma, available, d.nlh);

    VSFrame* dst = owned.newVideoFrame(&d.output.format, d.output.width, d.output.height, source[f.center], core);
    nss::stamp_contribution(dst, radius, n, nss::Model::NLH, api);
    const int slices = 2 * radius + 1;
    const int rows = radius ? 2 * slices : 1;
    const int downloads = region;
    if (!bypass) {
        const std::size_t window_bytes = static_cast<std::size_t>(f.count) * 3 * d.plane_floats * sizeof(float);
        NSS_CUDA_CHECK(cudaMemcpyAsync(s.basic.get(), s.input.get(), window_bytes, cudaMemcpyDeviceToDevice, s.stream));
        for (int iteration = 0; iteration < o.basic_iterations; ++iteration) {
            nlh_mix(s.basic.as<float>(), s.input.as<float>(), static_cast<std::size_t>(f.count) * 3 * d.plane_floats,
                    o.basic_mix, s.stream);
            run_pass(f, s.basic, o, 0, false);
            for (int t = 0; t < f.count; ++t) {
                for (int c = 0; c < planes; ++c) {
                    aggregate_finish(f.accum(s.num, c, t), f.accum(s.den, c, t), f.plane(s.input, t, c), d.width[c],
                                     d.height[c], d.width[c], f.plane(s.basic, t, c), s.stream);
                }
            }
        }
        run_pass(f, s.input, o, 1, true);
        for (int t = 0; t < f.count && d.rgb; ++t) {
            // Shared pixel indices give all three channels identical counts.
            nlh_yuv_to_rgb(f.accum(s.num, 0, t), f.accum(s.num, 1, t), f.accum(s.num, 2, t), d.plane_floats, s.stream);
        }
        for (int c = 0; c < planes; ++c) {
            const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
            if (!d.estimate && d.sigma[c] == 0) continue;
            if (radius == 0) {
                // The fallback is only reached by planes no group covered,
                // which are never RGB (see run_pass), so the input is the source.
                aggregate_finish(f.accum(s.num, c, f.center), f.accum(s.den, c, f.center), f.plane(s.input, f.center, c),
                                 d.width[c], d.height[c], d.width[c], f.plane(s.basic, f.center, c), s.stream);
                begin_download(f.plane(s.basic, f.center, c), row_bytes, row_bytes, d.height[c],
                               f.staging(downloads + c * rows), s.stream);
                continue;
            }
            for (int sl = 0; sl < slices; ++sl) {
                const int t = n - radius + sl - first;
                if (t < 0 || t >= f.count) continue;
                begin_download(f.accum(s.num, c, t), row_bytes, row_bytes, d.height[c],
                               f.staging(downloads + c * rows + 2 * sl), s.stream);
                begin_download(f.accum(s.den, c, t), row_bytes, row_bytes, d.height[c],
                               f.staging(downloads + c * rows + 2 * sl + 1), s.stream);
            }
        }
    }
    s.stream.synchronize();
    for (int c = 0; c < planes; ++c) {
        const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
        auto* out = api->getWritePtr(dst, c);
        const std::ptrdiff_t stride = api->getStride(dst, c);
        const auto* original = api->getReadPtr(source[f.center], c);
        const std::ptrdiff_t src_stride = api->getStride(source[f.center], c);
        if (bypass || (!d.estimate && d.sigma[c] == 0)) {
            if (radius) {
                nss::host_detail::temporal_identity(reinterpret_cast<float*>(out), static_cast<int>(stride / sizeof(float)),
                                                    reinterpret_cast<const float*>(original),
                                                    static_cast<int>(src_stride / sizeof(float)), d.width[c], d.height[c],
                                                    radius);
            } else {
                vsh::bitblt(out, stride, original, src_stride, row_bytes, d.height[c]);
            }
            continue;
        }
        for (int k = 0; k < rows; ++k) {
            auto* target = out + static_cast<std::ptrdiff_t>(k) * d.height[c] * stride;
            const int t = n - radius + k / 2 - first;
            if (radius && (t < 0 || t >= f.count)) {
                for (int y = 0; y < d.height[c]; ++y) std::memset(target + y * stride, 0, row_bytes);
            } else {
                finish_download(f.staging(downloads + c * rows + k), row_bytes, d.height[c], target, stride);
            }
        }
    }

    auto* props = api->getFramePropertiesRW(dst);
    double sigmas[3]{};
    for (int c = 0; c < planes; ++c) sigmas[c] = sigma_units(f.center, c);
    if (api->mapSetFloatArray(props, "_NSSSigma", sigmas, planes) ||
        api->mapSetInt(props, "_NSSGroups", static_cast<std::int64_t>(f.groups), maReplace) ||
        api->mapSetInt(props, "_NSSADMMMaxIterGroups", 0, maReplace) ||
        api->mapSetInt(props, "_NSSSvdDoubleGroups", 0, maReplace) ||
        api->mapSetFloat(props, "_NSSSylvesterResidual", 0, maReplace)) {
        throw std::bad_alloc();
    }
    const int iterations[2]{o.basic_iterations, o.wiener_iterations};
    diagnostic_ints(api, props, "_NSSBlockSize", o.block.data(), 2);
    diagnostic_ints(api, props, "_NSSGroupSize", o.group.data(), 2);
    diagnostic_ints(api, props, "_NSSIterations", iterations, 2);
    diagnostic_ints(api, props, "_NSSSearchWindow", o.window.data(), 2);
    diagnostic_ints(api, props, "_NSSBlockStep", o.step.data(), 2);
    diagnostic_ints(api, props, "_NSSQ", o.q.data(), 2);
    if (api->mapSetFloat(props, "_NSSLambdaBasic", o.basic_mix, maReplace) ||
        api->mapSetFloat(props, "_NSSHardStrength", o.hard_strength, maReplace) ||
        api->mapSetFloat(props, "_NSSHardCoefficient", nss::kNlhHardCoefficient * o.hard_strength, maReplace) ||
        api->mapSetFloat(props, "_NSSWienerSigmaScale", o.wiener_sigma_scale, maReplace)) {
        throw std::bad_alloc();
    }
    for (const VSFrame* frame : source) owned.freeFrame(frame);
    return owned.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<NlhData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
    }
    delete d;
}

std::unique_ptr<Slot> make_slot(const NlhData& d, int staging_regions) {
    auto slot = std::make_unique<Slot>();
    const std::size_t window = static_cast<std::size_t>(d.max_frames) * d.plane_floats * sizeof(float);
    slot->input = DeviceBuffer(window * 3, d.budget);
    slot->basic = DeviceBuffer(window * 3, d.budget);
    if (d.reference) slot->reference = DeviceBuffer(window * 3, d.budget);
    slot->resized = DeviceBuffer(window, d.budget);
    slot->num = DeviceBuffer(window * 3, d.budget);
    slot->den = DeviceBuffer(window * 3, d.budget);
    const std::size_t pointers = static_cast<std::size_t>(d.max_frames) * 3 * sizeof(float*);
    slot->guide_ptrs = DeviceBuffer(pointers, d.budget);
    slot->data_ptrs = DeviceBuffer(pointers, d.budget);
    slot->basic_ptrs = DeviceBuffer(pointers, d.budget);
    slot->arena = DeviceBuffer(d.arena_bytes, d.budget);
    slot->totals = DeviceBuffer(3 * sizeof(double), d.budget);
    slot->staging = PinnedBuffer(static_cast<std::size_t>(staging_regions) * d.plane_floats * sizeof(float), d.budget);
    slot->aggregator = std::make_unique<OrderedAggregator>(d.width[0], d.height[0], d.max_frames, d.max_patches, d.budget);
    return slot;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    // Same validation order and text as nss.NLH (D14).
    auto data = std::make_unique<NlhData>();
    NlhData& d = *data;
    d.node = nss::get_node(api, in, "clip", 0, nullptr);
    d.vi = *api->getVideoInfo(d.node);
    if (api->mapNumElements(in, "rclip") >= 0) d.reference = nss::get_node(api, in, "rclip", 0, nullptr);
    static_cast<nss::FullImageParams&>(d) = nss::frontend::parse_full_image(
        api, in, d.vi, d.reference ? api->getVideoInfo(d.reference) : nullptr, nss::Model::NLH, "nss_cuda");
    try {
        nss::frontend::validate_full_image_geometry(d, d.vi, nss::Model::NLH, "nss_cuda");
    } catch (const std::invalid_argument& error) {
        // The shared preset resolution names the CPU plugin.
        const std::string text = error.what();
        if (text.rfind("nss.NLH:", 0) != 0) throw;
        throw std::invalid_argument("nss_cuda" + text.substr(3));
    }
    d.output = d.vi;
    if (d.radius()) d.output.height = nss::checked_fat_height(d.vi.height, d.radius());
    d.backend = parse_backend_args(api, in, "NLH");
    d.device = acquire_device(d.backend.device_id, "NLH", core, api);
    d.budget = nss::current_budget();
    d.rgb = d.vi.format.colorFamily == cfRGB;
    d.planes = d.vi.format.numPlanes;
    d.max_frames = std::min(2 * d.radius() + 1, d.vi.numFrames);
    d.reference_planes = d.reference ? (d.rgb ? 3 : 1) : 0;
    for (int c = 0; c < d.planes; ++c) {
        d.width[c] = nss::plane_width(d.vi, c);
        d.height[c] = nss::plane_height(d.vi, c);
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(d.width[c]) * d.height[c]);
    }

    // Memory plan: the window planes, staging and the output frame are fixed
    // per slot; the group arena takes what the budget leaves, up to
    // kArenaBytes, and must hold one group of the largest possible shape.
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const int rows = d.radius() ? 2 * (2 * d.radius() + 1) : 1;
    const int staging_regions = d.max_frames * (d.planes + d.reference_planes) + d.planes * rows;
    const std::size_t device_planes = static_cast<std::size_t>(d.max_frames) * (3 * 4 + (d.reference ? 3 : 0) + 1);
    const std::size_t fixed = plane_bytes * (device_planes + staging_regions) + plane_bytes * d.planes * rows;
    // The largest group over the possible resolved shapes (block <= 16).
    const int max_group = std::max({d.nlh.group[0], d.nlh.group[1], 16});
    const int max_q = std::max({d.nlh.q[0], d.nlh.q[1], 4});
    std::size_t one_group = 0;
    for (int block = 2; block <= 16; ++block) {
        one_group = std::max(one_group, group_bytes(Shape{block, max_group, max_q, d.planes, true}) + 16 * 13);
    }
    const Shape worst{16, max_group, max_q, d.planes, true};
    // Arena plus its aggregation buffers.
    const auto slot_bytes = [&](std::size_t arena) {
        return fixed + arena + arena / kArenaBytesPerPatch * OrderedAggregator::kBytesPerPatch + (1u << 16);
    };
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / slot_bytes(kArenaBytes), 1, kDefaultStreams));
    }
    const std::size_t share = limit == SIZE_MAX ? SIZE_MAX : limit / static_cast<std::size_t>(d.backend.num_streams);
    if (share < slot_bytes(one_group)) {
        throw std::invalid_argument("nss_cuda.NLH: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((slot_bytes(one_group) >> 20) + 1) + " MiB");
    }
    d.arena_bytes = kArenaBytes;
    if (share != SIZE_MAX && slot_bytes(kArenaBytes) > share) {
        // slot_bytes is affine in the arena: solve for the largest that fits.
        const double per_byte = 1.0 + static_cast<double>(OrderedAggregator::kBytesPerPatch) / kArenaBytesPerPatch;
        d.arena_bytes = std::max(one_group, static_cast<std::size_t>((share - slot_bytes(0)) / per_byte));
    }
    d.max_patches = std::max<std::size_t>(d.arena_bytes / kArenaBytesPerPatch, worst.group);

    {
        DeviceGuard guard(d.device.index);
        std::vector<std::unique_ptr<Slot>> slots;
        for (int i = 0; i < d.backend.num_streams; ++i) slots.push_back(make_slot(d, staging_regions));
        d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    }
    const auto pattern = d.radius() ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2]{{d.node, pattern}, {d.reference, pattern}};
    VSNode* node = api->createVideoFilter2("NLH", &d.output, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps,
                                          d.reference ? 2 : 1, &d, core);
    if (!node) {
        freeFilter(data.release(), core, api);
        throw std::runtime_error("nss_cuda: failed to create full image filter");
    }
    data.release();
    api->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_nlh(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNlhSignature);
    vspapi->registerFunction("NLH", args.c_str(), "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}

}  // namespace nss_cuda
