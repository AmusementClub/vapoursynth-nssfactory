// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.NLH and nss_cuda.TWSC: parse_full_image with the "nss_cuda" prefix
// around a device-resident full-image pipeline (the CPU nlh_image /
// twsc_image models). Both upload the request window, optionally estimate the
// noise blind, iterate over every frame of the window and return the final
// frame or the fat intermediate. The rolling mode (the default with a radius)
// runs that work for every center of a chunk and adds the centers' slices on
// the device (see compute_chunk).
//
// NLH: RGB -> YUV, Basic rounds (mix with the input, match on the luma guide,
// hard threshold, per-pixel aggregation, finish), one Wiener round for the
// center against the Basic estimate, YUV -> RGB on the numerators. Planes of
// equal geometry share matching and pixel selection; subsampled chroma
// matches on the area-averaged luma. The filter kernels aggregate their own
// output into fixed-point accumulators (TWSC uses the ordered aggregator).
//
// TWSC: rounds of joint matching over the noisy planes of equal geometry (on
// the estimate, or on rclip), the trilateral weighted sparse coding of every
// group, weighted aggregation and finish; the last round runs for the center
// only.
#include "cuda/common/aggregate.hpp"
#include "cuda/common/chunk_cache.hpp"
#include "cuda/common/match.hpp"
#include "cuda/nlh/kernels.hpp"
#include "cuda/twsc/kernels.hpp"
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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_cuda {
namespace {

// Upper bound on the per-batch group workspace.
constexpr std::size_t kArenaBytes = 192u << 20;
// NLH keeps matches and pixel indices there only (its kernels aggregate):
// half holds every pass in a few batches (96 and 192 MiB measured the same).
constexpr std::size_t kNlhArenaBytes = 96u << 20;
// Frames in flight per instance (staging sets; see create()). Two saturate
// one stream (1080p TWSC, block_step 8 / group 8 / 2 rounds on an RTX 5080:
// 87 fps with one set, 97 with two, 96 with three).
constexpr std::size_t kFramesInFlight = 2;
// Aggregation capacity per arena byte (patches); small groups are capped by
// it. TWSC sizes it to the patches its arena holds when that is more: at
// block 4 / group 32, 9300 instead of 6144 groups a batch, about 5% at 1080p
// for 10 MiB of sort arrays (RTX 5080).
constexpr std::size_t kArenaBytesPerPatch = 1024;

struct Shape {
    int block, group, q, nch;
    bool wiener;
};

// Device bytes one NLH group needs in the arena for `s` (see NlhGroupArgs).
std::size_t group_bytes(const Shape& s) {
    const std::size_t m = static_cast<std::size_t>(s.block) * s.block;
    const std::size_t matrices = static_cast<std::size_t>(s.nch) * m * s.q * s.group;
    const std::size_t coef = nlh_filter_fused(s.block, s.group, s.q)
                                 ? 0
                                 : matrices * (s.wiener && !nlh_local_matrix(s.group, s.q) ? 2 : 1);
    const std::size_t guide = nlh_prepare_shared(s.block, s.group) ? 0 : m * s.group;
    return (guide + coef) * sizeof(float) + m * s.q * sizeof(int) + sizeof(int) + (s.nch + 1) * sizeof(double) +
           static_cast<std::size_t>(s.group) * sizeof(DeviceMatch);
}

// Device bytes one TWSC group needs in the arena (see TwscGroupArgs).
std::size_t twsc_group_bytes(int block, int group, int nch) {
    const std::size_t m = static_cast<std::size_t>(block) * block * nch;
    return 2 * m * group * sizeof(float) + twsc_work_reals(static_cast<int>(m), group) * twsc_real_bytes() + sizeof(int) +
           static_cast<std::size_t>(group) * (sizeof(DeviceMatch) + sizeof(AggregatePatch));
}

struct Slot {
    Stream stream;
    DeviceBuffer input, basic, reference, resized, num, den;  // planes of `plane_floats`
    DeviceBuffer fixed_num, fixed_den;  // NLH accumulators: 64-bit sums per plane, 32-bit counts per geometry
    DeviceBuffer guide_ptrs, data_ptrs, basic_ptrs;           // device pointer arrays
    DeviceBuffer arena, totals, stalled;
    std::unique_ptr<OrderedAggregator> aggregator;
    // Rolling: the source (and reference) frames of a ring of 2R+1 (frame f
    // in cell f % (2R+1), its planes at a plane_floats stride, as the source
    // has them), each plane's sums of the frames still open (a ring of 2R+1
    // slices of num, then as many of den) and the frame being finished.
    DeviceBuffer ring, reference_ring, sums, out;
    // Guarded by ImageData::roll_mu. Chunks take a ticket when they are
    // planned and run on the slot in ticket order, so a chunk can carry on
    // from the one planned before it.
    int next_start = -1;  // the chunk that can carry on after the planned ones
    std::uint64_t issued = 0, serving = 0;
    std::uint64_t chain = 0;  // counts the chunks that started afresh
    std::uint64_t failed_chain = ~0ull;
};

// Pinned staging of one frame in flight: its uploads, then its downloads. A
// frame is copied in before it takes a slot and copied out after it has
// given the slot back, so a stream works while other threads copy.
struct Staging {
    PinnedBuffer bytes;
    cudaEvent_t done = nullptr;  // recorded behind the frame's last download
    Staging() = default;
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    ~Staging() {
        if (done) cudaEventDestroy(done);
    }
};

struct ImageData : nss::FullImageParams {
    nss::Model model = nss::Model::NLH;
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node, reference;
    VSVideoInfo vi{}, output{};
    BackendArgs backend;
    DeviceInfo device;
    bool rgb = false;
    bool split = false;  // TWSC cfast: three opponent planes, each a unit of its own
    int planes = 1, max_frames = 1, reference_planes = 0;
    int width[3]{}, height[3]{};
    int geometries = 1, geometry[3]{};  // planes of equal size share their selection counts
    std::size_t plane_floats = 0, arena_bytes = 0, max_patches = 0;
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
    // Rolling (temporal_mode="rolling" with a radius): chunks run on the
    // slots by turn instead of from the pool.
    bool rolling = false;
    nss::RollingParams roll{};
    std::vector<Slot*> slots;
    std::mutex roll_mu;
    std::condition_variable roll_cv;
    PlanOrder order;
    std::size_t frame_bytes = 0;  // a finished frame, its planes back to back (tight rows)
    std::array<std::size_t, 3> plane_offset{};
    ChunkCache cache;
    int radius() const { return model == nss::Model::TWSC ? twsc.radius : nlh.radius; }
    const char* name() const { return model == nss::Model::TWSC ? "TWSC" : "NLH"; }
};

// Per-frame state on one slot.
struct Frame {
    const ImageData& d;
    Slot& s;
    Staging& g;
    int count;   // frames in the request window
    int center;  // index of the output frame in it
    std::array<std::array<float, 3>, 33> sigma{};
    std::array<std::array<double, 3>, 33> units{};
    std::uint64_t groups = 0;
    int stalled = 0;

    float* plane(const DeviceBuffer& buffer, int t, int c) const {
        return buffer.as<float>() + (static_cast<std::size_t>(t) * 3 + c) * d.plane_floats;
    }
    // Numerator/denominator of channel c: `count` contiguous slices.
    float* accum(const DeviceBuffer& buffer, int c, int t = 0) const {
        return buffer.as<float>() + (static_cast<std::size_t>(c) * d.max_frames + t) * d.plane_floats;
    }
    std::uint8_t* staging(int region) const {
        return g.bytes.as<std::uint8_t>() + static_cast<std::size_t>(region) * d.plane_floats * sizeof(float);
    }
    unsigned long long* sums(int c) const {
        return s.fixed_num.as<unsigned long long>() + static_cast<std::size_t>(c) * d.max_frames * d.plane_floats;
    }
    unsigned* counts(int c) const {
        return s.fixed_den.as<unsigned>() + static_cast<std::size_t>(d.geometry[c]) * d.max_frames * d.plane_floats;
    }
    std::size_t floats(int c) const { return static_cast<std::size_t>(d.width[c]) * d.height[c]; }
    // The parts of a window buffer that hold samples, as (offset, floats) runs
    // with neighbours merged: a frame's slots are 3 planes of plane_floats,
    // and plane c fills the first floats(c) of its slot (one run for RGB, a
    // run per frame for gray). Whole-window copies and elementwise passes
    // take these, so they never read the unused slots (gray: two of three).
    std::vector<std::pair<std::size_t, std::size_t>> runs() const {
        std::vector<std::pair<std::size_t, std::size_t>> out;
        for (int t = 0; t < count; ++t) {
            for (int c = 0; c < d.planes; ++c) {
                const std::size_t at = (static_cast<std::size_t>(t) * 3 + c) * d.plane_floats, n = floats(c);
                if (!out.empty() && out.back().first + out.back().second == at) {
                    out.back().second += n;
                } else {
                    out.emplace_back(at, n);
                }
            }
        }
        return out;
    }
    // basic = input over the window's samples.
    void copy_window(const DeviceBuffer& to, const DeviceBuffer& from) const {
        for (const auto& [at, n] : runs()) {
            NSS_CUDA_CHECK(cudaMemcpyAsync(to.as<float>() + at, from.as<float>() + at, n * sizeof(float),
                                           cudaMemcpyDeviceToDevice, s.stream));
        }
    }
};

// Planes of equal geometry starting at `first` (the CPU grouping order).
int geometry_group(const ImageData& d, int first, std::array<bool, 3>& used, std::array<int, 3>& channels) {
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
    const ImageData& d = f.d;
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
    if (static_cast<std::size_t>(at - f.s.arena.as<std::uint8_t>()) > f.d.arena_bytes) {
        throw std::logic_error("nss_cuda: group arena overflow");
    }
    a.batch = batch;
    a.block = shape.block;
    a.group = shape.group;
    a.q = shape.q;
    a.nch = shape.nch;
    a.wiener = shape.wiener;
    return a;
}

int batch_for(const ImageData& d, const Shape& shape, int total) {
    // Alignment slack of the arena split: 16 bytes per region.
    const std::size_t fit = (d.arena_bytes - 16 * 12) / (group_bytes(shape) + 16);
    return static_cast<int>(std::clamp<std::size_t>(fit, 1, static_cast<std::size_t>(total)));
}

MatchGeometry match_geometry(int width, int height, int block, int window, int group) {
    MatchGeometry g{width, height, width, block, window / 2, group};
    g.range_hi = window - window / 2 - 1;
    return g;
}

// Blind noise estimate on the device: fixed 8x8 / 16 / q4 / W40 over every
// kNoiseEstimateStep-th reference.
// NLH (nss::nlh_estimate_frame_sigma): planes of equal geometry share the
// luma-guided matching and pixel selection. TWSC (nss::nlh_estimate_sigma per
// plane): every plane is matched on itself (or on its rclip plane).
void estimate_sigma(Frame& f, int t) {
    const ImageData& d = f.d;
    Slot& s = f.s;
    const bool own_guide = d.model == nss::Model::TWSC;
    std::array<bool, 3> used{};
    for (int first = 0; first < d.planes; ++first) {
        if (used[first]) continue;
        std::array<int, 3> channels{first, 0, 0};
        const int nch = own_guide ? 1 : geometry_group(d, first, used, channels);
        const int width = d.width[first], height = d.height[first];
        if (width < 8 || height < 8) {
            throw std::invalid_argument("nss: blind noise estimation requires an 8x8 or larger plane");
        }
        // The fixed bootstrap search of the estimate (nss::ImageSearch defaults).
        const struct {
            int block = 8, step = nss::kNoiseEstimateStep, group = 16, window = 40;
        } search;
        const Shape shape{search.block, search.group, 4, nch, false};
        // Single-frame matching on this frame's guide.
        const int guide_plane = own_guide ? first : 0;
        const float* luma = d.reference ? f.plane(s.reference, t, guide_plane) : f.plane(s.input, t, guide_plane);
        const float* guide = luma;
        if (!own_guide && (width != d.width[0] || height != d.height[0])) {
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
        // Only matches and the statistic live in the arena: batches are far
        // larger than those of the filter passes.
        const std::size_t per_group = shape.group * sizeof(DeviceMatch) + sizeof(int) + nch * sizeof(double);
        const std::size_t fit = (d.arena_bytes - 16 * 4 - 2 * nch * sizeof(double)) / (per_group + 1);
        const int batch = static_cast<int>(std::clamp<std::size_t>(fit, 1, static_cast<std::size_t>(grid.count())));
        for (int begin = 0; begin < grid.count(); begin += batch) {
            const int n = std::min(batch, grid.count() - begin);
            std::uint8_t* at = s.arena.as<std::uint8_t>();
            const auto take = [&](std::size_t bytes) {
                std::uint8_t* p = at;
                at += (bytes + 15) & ~std::size_t{15};
                return p;
            };
            const std::size_t b = static_cast<std::size_t>(n);
            auto* partial = reinterpret_cast<double*>(take(b * nch * sizeof(double)));
            auto* sums = reinterpret_cast<double*>(take((b / 256 + 1) * nch * sizeof(double)));
            NlhGroupArgs a{};
            a.matches = reinterpret_cast<DeviceMatch*>(take(b * shape.group * sizeof(DeviceMatch)));
            a.counts = reinterpret_cast<int*>(take(b * sizeof(int)));
            if (static_cast<std::size_t>(at - s.arena.as<std::uint8_t>()) > d.arena_bytes) {
                throw std::logic_error("nss_cuda: group arena overflow");
            }
            a.batch = n;
            a.nch = nch;
            a.guides = s.guide_ptrs.as<const float*>();
            a.data = s.data_ptrs.as<const float*>();
            a.width = width;
            for (int c = 0; c < nch; ++c) a.guided[c] = data_ptr[c] == guide;
            spatial_match(guide, geometry, grid, begin, n, const_cast<DeviceMatch*>(a.matches),
                          const_cast<int*>(a.counts), s.stream);
            nlh_estimate_groups(a, partial, sums, s.totals.as<double>(), s.stream);
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
    const ImageData& d = f.d;
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
        // The kernels add into the accumulators of the geometry's planes.
        const std::size_t cells = static_cast<std::size_t>(f.count) * d.plane_floats;
        for (int c = 0; c < nch; ++c) {
            NSS_CUDA_CHECK(cudaMemsetAsync(f.sums(channels[c]), 0, cells * sizeof(unsigned long long), s.stream));
        }
        NSS_CUDA_CHECK(cudaMemsetAsync(f.counts(first), 0, cells * sizeof(unsigned), s.stream));
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
                    nlh_set_noise(a, c, (o.wiener_sigma_scale * sigma) * (o.wiener_sigma_scale * sigma));
                    a.identity[c] = sigma == 0;
                    a.num[c] = f.sums(channels[c]);
                }
                a.den = f.counts(first);
                a.slice_step = d.plane_floats;
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
            }
            f.groups += static_cast<std::uint64_t>(grid.count()) * nch;
        }
        for (int c = 0; c < nch; ++c) {
            nlh_finish_sums(f.sums(channels[c]), f.counts(first), f.accum(s.num, channels[c]),
                            f.accum(s.den, channels[c]), cells, s.stream);
        }
    }
}

// Splits the arena for `batch` TWSC groups.
TwscGroupArgs twsc_layout(const Frame& f, int block, int group, int nch, int batch) {
    const std::size_t m = static_cast<std::size_t>(block) * block * nch;
    std::uint8_t* at = f.s.arena.as<std::uint8_t>();
    const auto take = [&](std::size_t bytes) {
        std::uint8_t* p = at;
        at += (bytes + 15) & ~std::size_t{15};
        return p;
    };
    TwscGroupArgs a{};
    const std::size_t b = static_cast<std::size_t>(batch);
    a.work = take(b * twsc_work_reals(static_cast<int>(m), group) * twsc_real_bytes());
    a.matches = reinterpret_cast<DeviceMatch*>(take(b * group * sizeof(DeviceMatch)));
    a.patches = reinterpret_cast<AggregatePatch*>(take(b * group * sizeof(AggregatePatch)));
    a.counts = reinterpret_cast<int*>(take(b * sizeof(int)));
    a.centered = reinterpret_cast<float*>(take(b * m * group * sizeof(float)));
    a.values = reinterpret_cast<float*>(take(b * m * group * sizeof(float)));
    if (static_cast<std::size_t>(at - f.s.arena.as<std::uint8_t>()) > f.d.arena_bytes) {
        throw std::logic_error("nss_cuda: group arena overflow");
    }
    a.batch = batch;
    a.block = block;
    a.group = group;
    a.nch = nch;
    return a;
}

// nss::twsc_image: every round re-matches on the estimate (or on rclip),
// codes the groups of every frame of the window (the last round: of the
// center) and finishes the estimate.
void run_twsc(Frame& f) {
    const ImageData& d = f.d;
    Slot& s = f.s;
    const nss::TwscImageOptions& o = d.twsc;
    const std::size_t channel_bytes = static_cast<std::size_t>(d.max_frames) * d.plane_floats * sizeof(float);
    const auto runs = f.runs();
    f.copy_window(s.basic, s.input);
    NSS_CUDA_CHECK(cudaMemsetAsync(s.stalled.get(), 0, sizeof(int), s.stream));
    for (int iteration = 0; iteration < o.iterations; ++iteration) {
        const bool last = iteration + 1 == o.iterations;
        if (iteration > 0 && o.delta != 0) {
            for (const auto& [at, n] : runs) {
                iter_regularize(s.basic.as<float>() + at, s.input.as<float>() + at, n, static_cast<float>(o.delta), s.stream);
            }
        }
        std::array<bool, 3> used{};
        for (int first = 0; first < d.planes; ++first) {
            if (used[first]) continue;
            if (f.sigma[f.center][first] == 0) {
                // Not coded: the finish falls back to the source.
                NSS_CUDA_CHECK(cudaMemsetAsync(f.accum(s.num, first), 0, channel_bytes, s.stream));
                NSS_CUDA_CHECK(cudaMemsetAsync(f.accum(s.den, first), 0, channel_bytes, s.stream));
                continue;
            }
            const int width = d.width[first], height = d.height[first];
            std::array<int, 3> channels{};
            int nch = 0;
            for (int c = first; c < d.planes; ++c) {
                if (!used[c] && f.sigma[f.center][c] > 0 && (c == first || !d.split) && d.width[c] == width &&
                    d.height[c] == height) {
                    channels[nch++] = c;
                    used[c] = true;
                }
            }
            if (width < o.block || height < o.block) {
                throw std::invalid_argument("nss: selected plane is smaller than block_size or invalid step");
            }
            // Guides: channel 0 of the unit per frame; the other channels follow
            // at a constant plane distance.
            const DeviceBuffer& guide_source = d.reference ? s.reference : s.basic;
            std::vector<const float*> guide_ptr(f.count);
            std::vector<const float*> estimate_ptr(static_cast<std::size_t>(f.count) * nch), original_ptr(estimate_ptr.size());
            for (int t = 0; t < f.count; ++t) {
                guide_ptr[t] = f.plane(guide_source, t, channels[0]);
                for (int c = 0; c < nch; ++c) {
                    estimate_ptr[t * nch + c] = f.plane(s.basic, t, channels[c]);
                    original_ptr[t * nch + c] = f.plane(s.input, t, channels[c]);
                }
            }
            set_pointers(s.guide_ptrs, guide_ptr, s.stream);
            set_pointers(s.basic_ptrs, estimate_ptr, s.stream);
            set_pointers(s.data_ptrs, original_ptr, s.stream);
            const RasterGrid grid = make_raster_grid(width, height, o.block, o.step);
            MatchGeometry geometry = match_geometry(width, height, o.block, o.window, o.group);
            geometry.channels = nch;
            geometry.channel_step = nch > 1 ? static_cast<long long>(channels[1] - channels[0]) * d.plane_floats : 0;
            TemporalWindow window{};
            window.frames = s.guide_ptrs.as<const float*>();
            window.ntemp = f.count;
            window.radius = o.radius;
            window.valid_begin = 0;
            window.valid_end = f.count;
            window.ps_num = o.ps_num;
            window.ps_range = o.ps_range;
            const std::size_t per_group = twsc_group_bytes(o.block, o.group, nch) + 16;
            int batch = static_cast<int>(std::max<std::size_t>(
                1, std::min({(d.arena_bytes - 16 * 8) / per_group, d.max_patches / o.group,
                             static_cast<std::size_t>(grid.count())})));
            // Whole waves of blocks (see twsc_resident_groups) when a batch is a few of them. A partial
            // wave among many costs little, and other batch sizes suit the matcher better there (light
            // 1080p shape, 21 waves: 2% slower rounded).
            const int resident = twsc_resident_groups(o.block, o.group, nch);
            if (batch > resident && batch < 8 * resident) batch -= batch % resident;
            bool written = false;
            for (int t0 = last ? f.center : 0; t0 < (last ? f.center + 1 : f.count); ++t0) {
                window.t0 = t0;
                for (int begin = 0; begin < grid.count(); begin += batch) {
                    const int n = std::min(batch, grid.count() - begin);
                    TwscGroupArgs a = twsc_layout(f, o.block, o.group, nch, n);
                    a.estimate = s.basic_ptrs.as<const float*>();
                    a.original = s.data_ptrs.as<const float*>();
                    a.width = width;
                    for (int t = 0; t < f.count; ++t) {
                        for (int c = 0; c < nch; ++c) a.sigma[t * nch + c] = f.sigma[t][channels[c]];
                    }
                    for (int c = 0; c < nch; ++c) a.row_sigma[c] = f.sigma[t0][channels[c]];
                    a.residual = iteration > 0;
                    a.lambda2 = o.lambda2;
                    a.iterations = o.solver.iterations;
                    a.rho = o.solver.rho;
                    a.mu = o.solver.mu;
                    a.tolerance = o.solver.tolerance;
                    a.stalled = s.stalled.as<int>();
                    {
                        NSS_CUDA_RANGE("twsc.match");
                        auto* matches = const_cast<DeviceMatch*>(a.matches);
                        auto* counts = const_cast<int*>(a.counts);
                        if (o.radius > 0 && f.count > 1) {
                            predictive_match(geometry, window, grid, begin, n, matches, counts, s.stream);
                        } else {
                            spatial_match(guide_ptr[t0], geometry, grid, begin, n, matches, counts, s.stream);
                        }
                    }
                    {
                        NSS_CUDA_RANGE("twsc.solve");
                        twsc_filter_groups(a, s.stream);
                    }
                    NSS_CUDA_RANGE("twsc.aggregate");
                    // The rows a spatial batch's patches lie within: the windows of its first and last
                    // reference rows (clipped, not shifted, at the edges). Predictive matches may drift.
                    int row_first = 0, row_last = -1;
                    if (!(o.radius > 0 && f.count > 1)) {
                        row_first = std::max(grid.y(begin) - geometry.lo(), 0);
                        row_last = std::min(grid.y(begin + n - 1) + geometry.hi(), height - o.block) + o.block - 1;
                    }
                    const std::size_t channel_values = static_cast<std::size_t>(n) * o.group * o.block * o.block;
                    for (int c = 0; c < nch; ++c) {
                        const AggregateTarget target{f.accum(s.num, channels[c]), f.accum(s.den, channels[c]), width,
                                                     height, width, f.count, d.plane_floats};
                        s.aggregator->run(a.values + c * channel_values, a.patches, n * o.group, o.block, target, s.stream,
                                          written, nullptr, 1, row_first, row_last);
                    }
                    written = true;
                }
                f.groups += static_cast<std::uint64_t>(grid.count());
            }
        }
        if (last) break;
        for (int t = 0; t < f.count; ++t) {
            for (int c = 0; c < d.planes; ++c) {
                aggregate_finish(f.accum(s.num, c, t), f.accum(s.den, c, t), f.plane(s.input, t, c), d.width[c], d.height[c],
                                 d.width[c], f.plane(s.basic, t, c), s.stream);
            }
        }
    }
    NSS_CUDA_CHECK(cudaMemcpyAsync(&f.stalled, s.stalled.get(), sizeof(int), cudaMemcpyDeviceToHost, s.stream));
}

// nss::nlh_image: Basic rounds over the window (mix with the input, filter,
// finish), one Wiener round for the center against the Basic estimate, and
// for RGB the numerators back from YUV.
void run_nlh(Frame& f, const nss::NlhImageOptions& o) {
    const ImageData& d = f.d;
    Slot& s = f.s;
    const auto runs = f.runs();
    f.copy_window(s.basic, s.input);
    for (int iteration = 0; iteration < o.basic_iterations; ++iteration) {
        for (const auto& [at, n] : runs) nlh_mix(s.basic.as<float>() + at, s.input.as<float>() + at, n, o.basic_mix, s.stream);
        run_pass(f, s.basic, o, 0, false);
        for (int t = 0; t < f.count; ++t) {
            for (int c = 0; c < d.planes; ++c) {
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
}

// Stages plane c of a source or reference frame into pinned memory (tight
// rows) and rejects it if any sample is an infinity or a NaN.
void stage_checked(const ImageData& d, const VSAPI* api, const VSFrame* frame, int c, std::uint8_t* staging) {
    const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
    stage_plane(api->getReadPtr(frame, c), api->getStride(frame, c), row_bytes, d.height[c], staging);
    const auto* samples = reinterpret_cast<const std::uint32_t*>(staging);
    const std::size_t floats = static_cast<std::size_t>(d.width[c]) * d.height[c];
    std::uint32_t bad = 0;  // an all-ones exponent is an infinity or a NaN
    for (std::size_t i = 0; i < floats; ++i) bad |= static_cast<std::uint32_t>((samples[i] & 0x7f800000u) == 0x7f800000u);
    if (bad) throw std::invalid_argument("nss: nonfinite source/reference sample");
}

double sigma_units(const Frame& f, int t, int c) {
    return f.units[t][c] >= 0 ? f.units[t][c] : static_cast<double>(f.sigma[t][c]) * 255;
}

struct CenterWork {
    bool bypass = true;  // every plane of the center has sigma 0: nothing ran
    nss::NlhImageOptions o;
};

// The work of one center once its window is on the device (s.input and
// s.reference, slot t = frame first + t, as the source has it): the colour
// transforms, the blind estimate, and the model's rounds, which leave the
// center's slices of every window frame in s.num / s.den (f.accum).
CenterWork run_center(Frame& f) {
    const ImageData& d = f.d;
    Slot& s = f.s;
    const int planes = d.planes;
    for (int t = 0; t < f.count; ++t) {
        for (int c = 0; c < planes; ++c) {
            f.sigma[t][c] = d.sigma[c] / 255.f;
            f.units[t][c] = d.sigma_units[c];
        }
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
    for (int t = 0; t < f.count && d.split; ++t) {
        twsc_opponent(f.plane(s.input, t, 0), f.plane(s.input, t, 1), f.plane(s.input, t, 2), d.plane_floats, false, s.stream);
        if (d.reference) {
            twsc_opponent(f.plane(s.reference, t, 0), f.plane(s.reference, t, 1), f.plane(s.reference, t, 2), d.plane_floats,
                          false, s.stream);
        }
    }
    if (d.estimate) {
        NSS_CUDA_RANGE("image.estimate");
        for (int t = 0; t < f.count; ++t) estimate_sigma(f, t);
    }
    CenterWork w;
    for (int c = 0; c < planes; ++c) w.bypass = w.bypass && f.sigma[f.center][c] == 0;
    const bool twsc = d.model == nss::Model::TWSC;
    if (!twsc) {
        // nss::nlh_resolve_options: a bypassed center resolves from itself alone.
        double max_sigma = 0;
        int available = 16;
        for (int c = 0; c < planes; ++c) {
            const double units = sigma_units(f, f.center, c);
            if (!(units >= 0) || !std::isfinite(units)) throw std::invalid_argument("nss.NLH: invalid preset sigma");
            max_sigma = std::max(max_sigma, units);
            for (int t = w.bypass ? f.center : 0; t < (w.bypass ? f.center + 1 : f.count); ++t) {
                if (f.sigma[t][c] > 0) available = std::min({available, d.width[c], d.height[c]});
            }
        }
        w.o = nss::nlh_resolve_preset(max_sigma, available, d.nlh);
    }
    if (!w.bypass) {
        if (twsc) run_twsc(f);
        else run_nlh(f, w.o);
    }
    return w;
}

void diagnostic_ints(const VSAPI* api, VSMap* props, const char* key, const int* values, int count) {
    std::int64_t data[2]{values[0], count > 1 ? values[1] : 0};
    if (api->mapSetIntArray(props, key, data, count)) throw std::bad_alloc();
}

// Rolling. A chunk [start, start + count) needs every center whose window
// reaches it, [start - R, start + count - 1 + R], each run as a legacy center
// is (run_center); its slices are added to the sums of the frames from the
// chunk's first on, in ascending center order, and a frame is finished as
// soon as its last center has run: nss.VAggregate's sums and rule (an IEEE
// division; den == 1 divides exactly) on the legacy slices, so the output is
// the legacy intermediate reduced, bit for bit. A chunk that follows the last
// one planned on a slot carries on from it: the centers up to start - 1 + R
// have run and the frames up to start - 1 + 2R are in the ring, so it runs R
// centers past its end and uploads each of its frames once. Any other chunk
// starts afresh 2R centers (and frames) earlier.
constexpr std::uint64_t kCarryBacklog = 3;  // chunks waiting on a slot before another slot is preferred

int roll_ring(const ImageData& d) { return 2 * d.radius() + 1; }
int roll_reach(const ImageData& d) { return 2 * d.radius(); }

// Plane c of `frame` in the source ring (planes planes a cell) or the
// reference ring (reference_planes a cell).
float* ring_plane(const ImageData& d, const Slot& s, int frame, int c) {
    return s.ring.as<float>() + (static_cast<std::size_t>(frame % roll_ring(d)) * d.planes + c) * d.plane_floats;
}
float* reference_ring_plane(const ImageData& d, const Slot& s, int frame, int c) {
    return s.reference_ring.as<float>() +
           (static_cast<std::size_t>(frame % roll_ring(d)) * d.reference_planes + c) * d.plane_floats;
}
// The open sums of plane c of `frame`.
float* roll_num(const ImageData& d, const Slot& s, int frame, int c) {
    return s.sums.as<float>() + (static_cast<std::size_t>(c) * roll_ring(d) + frame % roll_ring(d)) * d.plane_floats;
}
float* roll_den(const ImageData& d, const Slot& s, int frame, int c) {
    return roll_num(d, s, frame, c) + static_cast<std::size_t>(d.planes) * roll_ring(d) * d.plane_floats;
}

// A chunk's turn on a slot: planned at construction (the slot, whether it
// carries on, its centers and the frames to upload), run between enter() and
// leave(). If a chunk fails, those planned to carry on from it fail too.
struct RollTurn {
    ImageData& d;
    Slot* slot = nullptr;
    bool carry = false;
    int first_center = 0, last_center = -1;
    std::vector<int> uploads;  // in upload order
    std::uint64_t ticket = 0, chain = 0;
    bool entered = false, left = false;

    RollTurn(ImageData& data, int start, int count) : d(data) {
        const int radius = d.radius(), nframes = d.vi.numFrames;
        // Everything that can throw comes before the slot's state changes: a
        // ticket nobody holds would stop the chunks behind it.
        last_center = std::min(start + count - 1 + radius, nframes - 1);
        const int last_frame = std::min(start + count - 1 + 2 * radius, nframes - 1);
        uploads.reserve(static_cast<std::size_t>(count + 2 * roll_reach(d)));
        std::lock_guard lock(d.roll_mu);
        const auto backlog = [](const Slot* s) { return s->issued - s->serving; };
        Slot* idle = d.slots.front();
        for (Slot* s : d.slots) {
            if (s->next_start == start && s->failed_chain != s->chain && !slot) slot = s;
            if (backlog(s) < backlog(idle)) idle = s;
        }
        // Carrying on saves 2R centers, unless the slot is far behind.
        carry = slot && (backlog(slot) < kCarryBacklog || backlog(idle) >= backlog(slot));
        if (!carry) slot = idle;
        first_center = carry ? std::min(start + radius, last_center + 1) : std::max(0, start - radius);
        const int first_frame = carry ? std::min(start - 1 + 2 * radius, nframes - 1) + 1 : std::max(0, start - 2 * radius);
        for (int frame = first_frame; frame <= last_frame; ++frame) uploads.push_back(frame);
        if (!carry) ++slot->chain;
        ticket = slot->issued++;
        chain = slot->chain;
        slot->next_start = start + count;
    }
    RollTurn(const RollTurn&) = delete;
    RollTurn& operator=(const RollTurn&) = delete;

    void enter() {
        std::unique_lock lock(d.roll_mu);
        d.roll_cv.wait(lock, [&] { return slot->serving == ticket; });
        entered = true;
        if (slot->failed_chain == chain) {
            throw std::runtime_error(std::string("nss_cuda.") + d.name() + ": an earlier chunk of this run failed");
        }
    }
    void leave() { finish(false); }
    ~RollTurn() {
        if (left) return;
        if (!entered) {
            std::unique_lock lock(d.roll_mu);
            d.roll_cv.wait(lock, [&] { return slot->serving == ticket; });
        }
        finish(true);
    }

private:
    void finish(bool failed) {
        {
            std::lock_guard lock(d.roll_mu);
            if (failed) {
                slot->failed_chain = chain;
                if (slot->chain == chain) slot->next_start = -1;
            }
            ++slot->serving;
            left = true;
        }
        d.roll_cv.notify_all();
    }
};

// The chunk [start, start + count): its frames are staged before the turn and
// the result is in the store once the staging's event has passed.
std::shared_ptr<ChunkStore> compute_chunk(ImageData& d, int start, int count, VSFrameContext* ctx, const VSAPI* api) {
    d.order.wait_for_previous(start, d.roll.rolling_chunk, d.cache);
    auto staging = [&] {
        NSS_CUDA_RANGE("image.staging");
        return d.staging->acquire();
    }();
    Staging& g = *staging;
    DeviceGuard guard(d.device.index);
    nss::FrameScope owned(api);
    auto store = d.cache.make_store(start, count);
    const int radius = d.radius(), nframes = d.vi.numFrames;
    const std::size_t per_frame = static_cast<std::size_t>(d.planes + d.reference_planes);
    const auto region = [&](std::size_t upload, int k) {
        return g.bytes.as<std::uint8_t>() + (upload * per_frame + k) * d.plane_floats * sizeof(float);
    };
    RollTurn turn(d, start, count);
    d.order.planned(start, d.roll.rolling_chunk);
    {
        NSS_CUDA_RANGE("image.stage_in");
        for (std::size_t i = 0; i < turn.uploads.size(); ++i) {
            const VSFrame* frame = owned.getFrameFilter(turn.uploads[i], d.node, ctx);
            for (int c = 0; c < d.planes; ++c) stage_checked(d, api, frame, c, region(i, c));
            owned.freeFrame(frame);
            if (d.reference) {
                const VSFrame* ref = owned.getFrameFilter(turn.uploads[i], d.reference, ctx);
                for (int c = 0; c < d.reference_planes; ++c) stage_checked(d, api, ref, c, region(i, d.planes + c));
                owned.freeFrame(ref);
            }
        }
    }
    NSS_CUDA_RANGE("image.queue");
    turn.enter();
    Slot& s = *turn.slot;
    struct Drain {
        cudaStream_t stream;
        bool armed = true;
        ~Drain() {
            if (armed) cudaStreamSynchronize(stream);
        }
    } drain{s.stream};
    const auto floats = [&](int c) { return static_cast<std::size_t>(d.width[c]) * d.height[c]; };
    const auto row_bytes = [&](int c) { return static_cast<std::size_t>(d.width[c]) * sizeof(float); };
    if (!turn.carry) NSS_CUDA_CHECK(cudaMemsetAsync(s.sums.get(), 0, s.sums.bytes(), s.stream));
    std::size_t uploaded = 0;
    const auto upload_through = [&](int last) {
        for (; uploaded < turn.uploads.size() && turn.uploads[uploaded] <= last; ++uploaded) {
            const int frame = turn.uploads[uploaded];
            for (int c = 0; c < d.planes; ++c) {
                upload_staged(region(uploaded, c), row_bytes(c), d.height[c], ring_plane(d, s, frame, c), row_bytes(c),
                              s.stream);
            }
            for (int c = 0; c < d.reference_planes; ++c) {
                upload_staged(region(uploaded, d.planes + c), row_bytes(c), d.height[c],
                              reference_ring_plane(d, s, frame, c), row_bytes(c), s.stream);
            }
        }
    };
    // A finished frame goes through s.out into the store (stream order keeps
    // the next finish behind the download); its sums are a later frame's from
    // here on.
    const auto finish = [&](int frame) {
        for (int c = 0; c < d.planes; ++c) {
            float* result = s.out.as<float>() + static_cast<std::size_t>(c) * d.plane_floats;
            aggregate_finish(roll_num(d, s, frame, c), roll_den(d, s, frame, c), ring_plane(d, s, frame, c), d.width[c],
                             d.height[c], d.width[c], result, s.stream);
            begin_download(result, row_bytes(c), row_bytes(c), d.height[c], store->plane(frame - start, c), s.stream);
            NSS_CUDA_CHECK(cudaMemsetAsync(roll_num(d, s, frame, c), 0, floats(c) * sizeof(float), s.stream));
            NSS_CUDA_CHECK(cudaMemsetAsync(roll_den(d, s, frame, c), 0, floats(c) * sizeof(float), s.stream));
        }
    };
    int out = start;
    for (int center = turn.first_center; center <= turn.last_center; ++center) {
        const int first = nss::host_detail::temporal_first(center, radius);
        const int last = nss::host_detail::temporal_last(center, radius, nframes);
        upload_through(last);
        // The window as a legacy center has it: slot t is frame first + t.
        Frame f{d, s, g, last - first + 1, center - first};
        for (int t = 0; t < f.count; ++t) {
            for (int c = 0; c < d.planes; ++c) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(f.plane(s.input, t, c), ring_plane(d, s, first + t, c),
                                               floats(c) * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
            }
            for (int c = 0; c < d.reference_planes; ++c) {
                NSS_CUDA_CHECK(cudaMemcpyAsync(f.plane(s.reference, t, c), reference_ring_plane(d, s, first + t, c),
                                               floats(c) * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
            }
        }
        const CenterWork work = run_center(f);
        // The center's slice of every open frame (the legacy fat rows). A
        // plane left alone has the identity slices: the source with weight 1
        // for its own center, zeros (which change no sum) for the others.
        for (int t = 0; t < f.count; ++t) {
            const int frame = first + t;
            if (frame < out) continue;
            for (int c = 0; c < d.planes; ++c) {
                if (work.bypass || (!d.estimate && d.sigma[c] == 0)) {
                    if (frame == center) {
                        accumulate_identity(roll_num(d, s, frame, c), roll_den(d, s, frame, c),
                                            ring_plane(d, s, frame, c), floats(c), s.stream);
                    }
                    continue;
                }
                accumulate_slice(roll_num(d, s, frame, c), roll_den(d, s, frame, c), f.accum(s.num, c, t),
                                 f.accum(s.den, c, t), floats(c), s.stream);
            }
        }
        for (; out < start + count && std::min(out + radius, nframes - 1) <= center; ++out) finish(out);
    }
    for (; out < start + count; ++out) finish(out);
    NSS_CUDA_CHECK(cudaEventRecord(g.done, s.stream));
    drain.armed = false;
    turn.leave();
    NSS_CUDA_RANGE("image.wait");
    NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
    return store;
}

// A frame of the rolling output: from its chunk, with the source frame's
// properties and the contribution stamp (as nss.VAggregate leaves them).
const VSFrame* rolling_frame(ImageData& d, int n, VSFrameContext* ctx, VSCore* core, const VSAPI* api) {
    const int start = n / d.roll.rolling_chunk * d.roll.rolling_chunk;
    const int count = std::min(d.roll.rolling_chunk, d.vi.numFrames - start);
    auto chunk = d.cache.get(start, [&] { return compute_chunk(d, start, count, ctx, api); });
    // The output frame is planned per staging set: holding one bounds the
    // frames being copied out (and the stores they read) under the limit.
    auto lease = d.staging->acquire();
    nss::ResourceScope scope(d.budget);
    nss::FrameScope owned(api);
    const VSFrame* src = owned.getFrameFilter(n, d.node, ctx);
    VSFrame* dst = owned.newVideoFrame(&d.vi.format, d.vi.width, d.vi.height, src, core);
    nss::stamp_contribution(dst, 0, n, d.model, api);
    owned.freeFrame(src);
    for (int c = 0; c < d.planes; ++c) {
        const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
        vsh::bitblt(api->getWritePtr(dst, c), api->getStride(dst, c), chunk->plane(n - chunk->start, c), row_bytes,
                    row_bytes, d.height[c]);
    }
    return owned.keep(dst);
}

// Queues the downloads of a frame's result into its staging, behind the
// uploads (regions from `downloads` on): per plane, `rows` regions from
// downloads + c * rows, the finished plane (spatial) or the window's num and
// den slices (legacy). Planes that are not filtered download nothing.
void queue_downloads(const ImageData& d, Frame& f, Slot& s, int n, int first, int downloads, int rows) {
    const int radius = d.radius();
    if (d.split) {
        // Each plane's estimate, then back to RGB before the planes are copied out.
        for (int c = 0; c < d.planes; ++c) {
            aggregate_finish(f.accum(s.num, c, f.center), f.accum(s.den, c, f.center), f.plane(s.input, f.center, c),
                             d.width[c], d.height[c], d.width[c], f.plane(s.basic, f.center, c), s.stream);
        }
        twsc_opponent(f.plane(s.basic, f.center, 0), f.plane(s.basic, f.center, 1), f.plane(s.basic, f.center, 2),
                      d.plane_floats, true, s.stream);
        for (int c = 0; c < d.planes; ++c) {
            const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
            begin_download(f.plane(s.basic, f.center, c), row_bytes, row_bytes, d.height[c],
                           f.staging(downloads + c * rows), s.stream);
        }
        return;
    }
    for (int c = 0; c < d.planes; ++c) {
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
        for (int sl = 0; sl < 2 * radius + 1; ++sl) {
            const int t = n - radius + sl - first;
            if (t < 0 || t >= f.count) continue;
            begin_download(f.accum(s.num, c, t), row_bytes, row_bytes, d.height[c],
                           f.staging(downloads + c * rows + 2 * sl), s.stream);
            begin_download(f.accum(s.den, c, t), row_bytes, row_bytes, d.height[c],
                           f.staging(downloads + c * rows + 2 * sl + 1), s.stream);
        }
    }
}

// Plane c of dst once the downloads have landed: the source when the plane
// is not filtered (its identity slices when legacy), else the staged rows
// (zero slices for window frames outside the clip).
void write_plane(const ImageData& d, Frame& f, const VSAPI* api, VSFrame* dst, const VSFrame* source, int c,
                 bool filtered, int n, int first, int downloads, int rows) {
    const int radius = d.radius();
    const std::size_t row_bytes = static_cast<std::size_t>(d.width[c]) * sizeof(float);
    auto* out = api->getWritePtr(dst, c);
    const std::ptrdiff_t stride = api->getStride(dst, c);
    if (!filtered) {
        const auto* original = api->getReadPtr(source, c);
        const std::ptrdiff_t src_stride = api->getStride(source, c);
        if (radius) {
            nss::host_detail::temporal_identity(reinterpret_cast<float*>(out), static_cast<int>(stride / sizeof(float)),
                                                reinterpret_cast<const float*>(original),
                                                static_cast<int>(src_stride / sizeof(float)), d.width[c], d.height[c],
                                                radius);
        } else {
            vsh::bitblt(out, stride, original, src_stride, row_bytes, d.height[c]);
        }
        return;
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

// The resolved values and statistics as _NSS* frame properties (as the CPU
// plugin stamps them; the device has no double-precision SVD and no
// Sylvester residual).
void stamp_diagnostics(const ImageData& d, const Frame& f, const nss::NlhImageOptions& o, VSFrame* dst,
                       const VSAPI* api) {
    const bool twsc = d.model == nss::Model::TWSC;
    auto* props = api->getFramePropertiesRW(dst);
    double sigmas[3]{};
    for (int c = 0; c < d.planes; ++c) sigmas[c] = sigma_units(f, f.center, c);
    if (api->mapSetFloatArray(props, "_NSSSigma", sigmas, d.planes) ||
        api->mapSetInt(props, "_NSSGroups", static_cast<std::int64_t>(f.groups), maReplace) ||
        api->mapSetInt(props, "_NSSADMMMaxIterGroups", f.stalled, maReplace) ||
        api->mapSetInt(props, "_NSSSvdDoubleGroups", 0, maReplace) ||
        api->mapSetFloat(props, "_NSSSylvesterResidual", 0, maReplace)) {
        throw std::bad_alloc();
    }
    if (twsc && api->mapSetInt(props, "_NSSCFast", d.split, maReplace)) throw std::bad_alloc();
    const int stages = twsc ? 1 : 2;
    const int blocks[2]{twsc ? d.twsc.block : o.block[0], o.block[1]};
    const int groups[2]{twsc ? d.twsc.group : o.group[0], o.group[1]};
    const int iterations[2]{twsc ? d.twsc.iterations : o.basic_iterations, o.wiener_iterations};
    const int windows[2]{twsc ? d.twsc.window : o.window[0], o.window[1]};
    const int steps[2]{twsc ? d.twsc.step : o.step[0], o.step[1]};
    diagnostic_ints(api, props, "_NSSBlockSize", blocks, stages);
    diagnostic_ints(api, props, "_NSSGroupSize", groups, stages);
    diagnostic_ints(api, props, "_NSSIterations", iterations, stages);
    diagnostic_ints(api, props, "_NSSSearchWindow", windows, stages);
    diagnostic_ints(api, props, "_NSSBlockStep", steps, stages);
    if (!twsc) {
        diagnostic_ints(api, props, "_NSSQ", o.q.data(), 2);
        if (api->mapSetFloat(props, "_NSSLambdaBasic", o.basic_mix, maReplace) ||
            api->mapSetFloat(props, "_NSSHardStrength", o.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSHardCoefficient", nss::kNlhHardCoefficient * o.hard_strength, maReplace) ||
            api->mapSetFloat(props, "_NSSWienerSigmaScale", o.wiener_sigma_scale, maReplace)) {
            throw std::bad_alloc();
        }
    }
}

// A frame of the spatial or legacy output: the window [first, last] around n.
const VSFrame* window_frame(ImageData& d, int n, int first, int last, VSFrameContext* ctx, VSCore* core,
                            const VSAPI* api) {
    // The staging set is leased before the output frame is allocated, so at
    // most that many frames are in flight.
    auto staging = d.staging->acquire();
    Staging& g = *staging;
    DeviceGuard guard(d.device.index);
    nss::ResourceScope scope(d.budget);
    nss::FrameScope owned(api);
    const int count = last - first + 1;
    const auto region_at = [&](int region) {
        return g.bytes.as<std::uint8_t>() + static_cast<std::size_t>(region) * d.plane_floats * sizeof(float);
    };

    // Stage the window: every source plane, and the reference planes the
    // luma guide needs.
    struct Upload {
        int region, t, c;
        bool reference;
    };
    std::vector<Upload> uploads;
    std::vector<const VSFrame*> source(count);
    const auto stage = [&](const VSFrame* frame, int t, int c, bool reference) {
        const int region = static_cast<int>(uploads.size());
        stage_checked(d, api, frame, c, region_at(region));
        uploads.push_back({region, t, c, reference});
    };
    for (int t = 0; t < count; ++t) {
        source[t] = owned.getFrameFilter(first + t, d.node, ctx);
        for (int c = 0; c < d.planes; ++c) stage(source[t], t, c, false);
        if (d.reference) {
            const VSFrame* frame = owned.getFrameFilter(first + t, d.reference, ctx);
            for (int c = 0; c < d.reference_planes; ++c) stage(frame, t, c, true);
            owned.freeFrame(frame);
        }
    }

    // The slot is held while the frame's work is queued (the blind estimate
    // waits for its result in between): what the next frame queues on the
    // stream runs after it, and the staging belongs to this frame until its
    // event has passed.
    std::optional<SlotPool<Slot>::Lease> slot(d.pool->acquire());
    Slot& s = **slot;
    // On an error the queued copies must not outlive the staging lease.
    struct Drain {
        cudaStream_t stream;
        bool armed = true;
        ~Drain() {
            if (armed) cudaStreamSynchronize(stream);
        }
    } drain{s.stream};
    Frame f{d, s, g, count, n - first};
    for (const Upload& u : uploads) {
        const std::size_t row_bytes = static_cast<std::size_t>(d.width[u.c]) * sizeof(float);
        upload_staged(region_at(u.region), row_bytes, d.height[u.c], f.plane(u.reference ? s.reference : s.input, u.t, u.c),
                      row_bytes, s.stream);
    }
    const int downloads = static_cast<int>(uploads.size());
    const int rows = d.radius() ? 2 * (2 * d.radius() + 1) : 1;
    VSFrame* dst = owned.newVideoFrame(&d.output.format, d.output.width, d.output.height, source[f.center], core);
    nss::stamp_contribution(dst, d.radius(), n, d.model, api);
    const CenterWork work = run_center(f);
    if (!work.bypass) queue_downloads(d, f, s, n, first, downloads, rows);
    NSS_CUDA_CHECK(cudaEventRecord(g.done, s.stream));
    drain.armed = false;
    slot.reset();
    NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
    for (int c = 0; c < d.planes; ++c) {
        const bool filtered = !work.bypass && (d.estimate || d.sigma[c] != 0);
        write_plane(d, f, api, dst, source[f.center], c, filtered, n, first, downloads, rows);
    }
    stamp_diagnostics(d, f, work.o, dst, api);
    for (const VSFrame* frame : source) owned.freeFrame(frame);
    return owned.keep(dst);
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* api) {
    auto& d = *static_cast<ImageData*>(instance);
    const int radius = d.radius();
    int first = nss::host_detail::temporal_first(n, radius);
    int last = nss::host_detail::temporal_last(n, radius, d.vi.numFrames);
    if (activation == arInitial) {
        if (d.rolling) {
            // Always the full chunk window: a cache hit seen now may be evicted before arAllFramesReady.
            const int start = n / d.roll.rolling_chunk * d.roll.rolling_chunk;
            const int count = std::min(d.roll.rolling_chunk, d.vi.numFrames - start);
            first = std::max(0, start - roll_reach(d));
            last = nss::host_detail::temporal_last(start + count - 1, roll_reach(d), d.vi.numFrames);
        }
        for (int frame = first; frame <= last; ++frame) {
            api->requestFrameFilter(frame, d.node, ctx);
            if (d.reference) api->requestFrameFilter(frame, d.reference, ctx);
        }
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("image.frame");
    if (d.rolling) return rolling_frame(d, n, ctx, core, api);
    return window_frame(d, n, first, last, ctx, core, api);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<ImageData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->staging.reset();
        d->cache.clear();
    }
    delete d;
}

// TWSC's 4x4 patches on 4x4 aggregation tiles: a pixel tests about 49
// entries a 16-patch hit instead of 121 on 8x8 tiles.
bool twsc_micro_tiles(const ImageData& d) { return d.model == nss::Model::TWSC && d.twsc.block <= 4; }

std::unique_ptr<Slot> make_slot(const ImageData& d) {
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
    slot->stalled = DeviceBuffer(sizeof(int), d.budget);
    if (d.rolling) {
        const std::size_t plane_bytes = d.plane_floats * sizeof(float);
        const std::size_t ring = static_cast<std::size_t>(roll_ring(d));
        slot->ring = DeviceBuffer(ring * d.planes * plane_bytes, d.budget);
        if (d.reference) slot->reference_ring = DeviceBuffer(ring * d.reference_planes * plane_bytes, d.budget);
        slot->sums = DeviceBuffer(2 * ring * d.planes * plane_bytes, d.budget);
        slot->out = DeviceBuffer(d.planes * plane_bytes, d.budget);
    }
    if (d.model == nss::Model::TWSC) {
        slot->aggregator =
            std::make_unique<OrderedAggregator>(d.width[0], d.height[0], d.max_frames, d.max_patches, d.budget, true,
                                                twsc_micro_tiles(d));
    } else {
        const std::size_t cells = static_cast<std::size_t>(d.max_frames) * d.plane_floats;
        slot->fixed_num = DeviceBuffer(cells * d.planes * sizeof(unsigned long long), d.budget);
        slot->fixed_den = DeviceBuffer(cells * d.geometries * sizeof(unsigned), d.budget);
    }
    return slot;
}

// The planes' sizes and offsets in a frame, the planes the reference
// supplies, and the planes' geometry groups (planes of one shape share passes).
void describe_planes(ImageData& d) {
    const bool twsc = d.model == nss::Model::TWSC;
    d.planes = d.vi.format.numPlanes;
    d.max_frames = std::min(2 * d.radius() + 1, d.vi.numFrames);
    // NLH matches on the reference luma (which needs all of RGB); TWSC on
    // every reference plane.
    d.reference_planes = d.reference ? (twsc || d.rgb ? d.planes : 1) : 0;
    for (int c = 0; c < d.planes; ++c) {
        d.width[c] = nss::plane_width(d.vi, c);
        d.height[c] = nss::plane_height(d.vi, c);
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(d.width[c]) * d.height[c]);
        d.plane_offset[c] = d.frame_bytes;
        d.frame_bytes += static_cast<std::size_t>(d.width[c]) * d.height[c] * sizeof(float);
    }
    std::array<bool, 3> used{};
    d.geometries = 0;
    for (int first = 0; first < d.planes; ++first) {
        if (used[first]) continue;
        std::array<int, 3> channels{};
        const int nch = geometry_group(d, first, used, channels);
        for (int c = 0; c < nch; ++c) d.geometry[channels[c]] = d.geometries;
        ++d.geometries;
    }
}

// What the memory plan settles beside the device buffers: the pinned
// staging regions per set, the sets, and how far the rolling cache may grow.
struct MemoryPlan {
    int staging_regions = 0;
    int staging_sets = 0;
    int cache_cap = 1;
};

// Sets the streams (unless given), the group arena and the patch capacity
// within memory_limit_mb, and returns the rest of the plan.
MemoryPlan plan_memory(ImageData& d) {
    const bool twsc = d.model == nss::Model::TWSC;
    MemoryPlan plan;
    // Memory plan: the window planes, staging and the output frame are fixed
    // per slot; the group arena takes what the budget leaves, up to
    // kArenaBytes, and must hold one group of the largest possible shape.
    // Rolling stages a chunk's frames instead of a window and the fat
    // download, keeps the ring, its sums and the finished frame on the
    // device, and holds host chunk stores: one being built per slot, the
    // cache (beside the slots) and the output frames copied out of them.
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const int rows = d.radius() ? 2 * (2 * d.radius() + 1) : 1;
    const int ring = 2 * d.radius() + 1;
    const std::size_t chunk_bytes = d.rolling ? d.frame_bytes * d.roll.rolling_chunk : 0;
    plan.staging_regions = d.rolling ? (d.roll.rolling_chunk + 4 * d.radius()) * (d.planes + d.reference_planes)
                                          : d.max_frames * (d.planes + d.reference_planes) + d.planes * rows;
    const std::size_t device_planes =
        static_cast<std::size_t>(d.max_frames) * (3 * 4 + (d.reference ? 3 : 0) + 1) +
        (d.rolling ? static_cast<std::size_t>(ring) * (3 * d.planes + d.reference_planes) + d.planes : 0);
    const std::size_t output_bytes = d.rolling ? 2 * d.frame_bytes + chunk_bytes : plane_bytes * d.planes * rows;
    // NLH accumulators: 64-bit sums per plane and 32-bit counts per geometry.
    const std::size_t accumulators = twsc ? 0 : static_cast<std::size_t>(d.max_frames) * (2 * d.planes + d.geometries);
    const std::size_t fixed = plane_bytes * (device_planes + plan.staging_regions + accumulators) + output_bytes;
    // The largest group over the possible resolved shapes (NLH: block <= 16).
    const int max_group = twsc ? d.twsc.group : std::max({d.nlh.group[0], d.nlh.group[1], 16});
    const int max_q = std::max({d.nlh.q[0], d.nlh.q[1], 4});
    std::size_t one_group = d.estimate ? group_bytes(Shape{8, 16, 4, d.planes, false}) + 16 * 13 : 0;
    if (twsc) {
        one_group = std::max(one_group, twsc_group_bytes(d.twsc.block, d.twsc.group, d.split ? 1 : d.planes) + 16 * 9);
    } else {
        for (int block = 2; block <= 16; ++block) {
            one_group = std::max(one_group, group_bytes(Shape{block, max_group, max_q, d.planes, true}) + 16 * 13);
        }
    }
    const std::size_t arena_cap = twsc ? kArenaBytes : kNlhArenaBytes;
    // Arena plus, for TWSC, its aggregation buffers (sort arrays and the tile bins).
    const std::size_t bins =
        twsc ? OrderedAggregator::bin_bytes(d.width[0], d.height[0], d.max_frames, true, twsc_micro_tiles(d)) : 0;
    const std::size_t sort_bytes = twsc ? OrderedAggregator::kBytesPerPatch : 0;
    const std::size_t bytes_per_patch =
        twsc ? std::clamp<std::size_t>(
                   twsc_group_bytes(d.twsc.block, d.twsc.group, d.split ? 1 : d.planes) / d.twsc.group, 1,
                   kArenaBytesPerPatch)
             : kArenaBytesPerPatch;
    const auto slot_bytes = [&](std::size_t arena) {
        return fixed + arena + arena / bytes_per_patch * sort_bytes + bins + (1u << 16);
    };
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const std::size_t shared_cache = chunk_bytes * (d.rolling ? d.roll.cache_chunks : 0);
    if (total != SIZE_MAX && total <= shared_cache) {
        throw std::invalid_argument(std::string("nss_cuda.") + d.name() + ": memory_limit_mb is too small for the rolling cache");
    }
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total - shared_cache;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / slot_bytes(arena_cap), 1, kDefaultStreams));
    }
    const std::size_t share = limit == SIZE_MAX ? SIZE_MAX : limit / static_cast<std::size_t>(d.backend.num_streams);
    if (share < slot_bytes(one_group)) {
        throw std::invalid_argument(std::string("nss_cuda.") + d.name() +
                                    ": memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((slot_bytes(one_group) >> 20) + 1) + " MiB" +
                                    (shared_cache ? ", plus " + std::to_string((shared_cache >> 20) + 1) +
                                                        " MiB for the rolling cache"
                                                  : ""));
    }
    d.arena_bytes = arena_cap;
    if (share != SIZE_MAX && slot_bytes(arena_cap) > share) {
        // slot_bytes is affine in the arena: solve for the largest that fits.
        const double per_byte = 1.0 + static_cast<double>(sort_bytes) / bytes_per_patch;
        d.arena_bytes = std::max(one_group, static_cast<std::size_t>((share - slot_bytes(0)) / per_byte));
    }
    d.max_patches = std::max<std::size_t>(d.arena_bytes / bytes_per_patch, max_group);
    // A staging set and the output frame it feeds per frame in flight. One
    // per stream is in `fixed`; more keep a stream busy while other threads
    // copy, where the limit leaves room beside the full arena.
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    const std::size_t set_bytes = plane_bytes * plan.staging_regions + output_bytes;
    std::size_t extra = std::max(streams, kFramesInFlight) - streams;
    if (limit != SIZE_MAX) {
        const std::size_t used = streams * slot_bytes(d.arena_bytes);
        extra = std::min(extra, limit > used ? (limit - used) / set_bytes : 0);
    }
    plan.staging_sets = static_cast<int>(streams + extra);
    // Under a limit the rolling cache grows only into what the plan leaves over.
    plan.cache_cap = d.roll.cache_limit;
    if (d.rolling && limit != SIZE_MAX) {
        const std::size_t used = streams * slot_bytes(d.arena_bytes) + extra * set_bytes;
        const std::size_t spare = limit > used ? (limit - used) / chunk_bytes : 0;
        plan.cache_cap = static_cast<int>(std::min<std::size_t>(d.roll.cache_limit, d.roll.cache_chunks + spare));
    }

    return plan;
}

// The slots, the staging sets and the rolling cache, as planned.
void allocate(ImageData& d, const MemoryPlan& plan) {
    DeviceGuard guard(d.device.index);
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < d.backend.num_streams; ++i) slots.push_back(make_slot(d));
    for (const auto& slot : slots) d.slots.push_back(slot.get());
    if (d.rolling) {
        d.cache.configure(d.roll.rolling_chunk, d.frame_bytes, d.plane_offset, d.roll.cache_chunks, plan.cache_cap, d.budget);
    }
    std::vector<std::unique_ptr<Staging>> sets;
    for (int i = 0; i < plan.staging_sets; ++i) {
        auto set = std::make_unique<Staging>();
        set->bytes = PinnedBuffer(static_cast<std::size_t>(plan.staging_regions) * d.plane_floats * sizeof(float), d.budget);
        NSS_CUDA_CHECK(cudaEventCreateWithFlags(&set->done, cudaEventDisableTiming));
        sets.push_back(std::move(set));
    }
    d.staging = std::make_unique<SlotPool<Staging>>(std::move(sets));
    d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
}

void create(const VSMap* in, VSMap* out, VSCore* core, const VSAPI* api, nss::Model model) {
    // Same validation order and text as nss.NLH / nss.TWSC: the
    // temporal arguments first.
    const auto temporal = nss::frontend::parse_temporal(api, in, 0, model == nss::Model::TWSC ? "TWSC" : "NLH", "nss_cuda",
                                                        kDefaultTemporalMode, kRollingCache);
    auto data = std::make_unique<ImageData>();
    ImageData& d = *data;
    d.model = model;
    d.node = nss::get_node(api, in, "clip", 0, nullptr);
    d.vi = *api->getVideoInfo(d.node);
    if (api->mapNumElements(in, "rclip") >= 0) d.reference = nss::get_node(api, in, "rclip", 0, nullptr);
    static_cast<nss::FullImageParams&>(d) = nss::frontend::parse_full_image(
        api, in, d.vi, d.reference ? api->getVideoInfo(d.reference) : nullptr, model, "nss_cuda");
    try {
        nss::frontend::validate_full_image_geometry(d, d.vi, model, "nss_cuda");
    } catch (const std::invalid_argument& error) {
        // The shared preset resolution names the CPU plugin.
        const std::string text = error.what();
        if (text.rfind("nss.NLH:", 0) != 0) throw;
        throw std::invalid_argument("nss_cuda" + text.substr(3));
    }
    d.rolling = temporal.rolling;
    d.roll = temporal.params;
    d.output = d.vi;
    if (d.radius() && !d.rolling) d.output.height = nss::checked_fat_height(d.vi.height, d.radius());
    d.backend = parse_backend_args(api, in, d.name());
    d.device = acquire_device(d.backend.device_id, d.name(), core, api);
    d.budget = nss::current_budget();
    const bool twsc = model == nss::Model::TWSC;
    d.rgb = !twsc && d.vi.format.colorFamily == cfRGB;  // NLH works in YUV
    d.split = twsc && nss::twsc_color_split(d.twsc, d.vi.format.colorFamily == cfRGB, d.estimate, d.sigma_units.data());
    describe_planes(d);
    const MemoryPlan plan = plan_memory(d);
    allocate(d, plan);
    const auto pattern = d.radius() ? rpGeneral : rpStrictSpatial;
    VSFilterDependency deps[2]{{d.node, pattern}, {d.reference, pattern}};
    VSNode* node = api->createVideoFilter2(d.name(), &d.output, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps,
                                          d.reference ? 2 : 1, &d, core);
    if (!node) {
        freeFilter(data.release(), core, api);
        throw std::runtime_error("nss_cuda: failed to create full image filter");
    }
    data.release();
    api->mapConsumeNode(out, "clip", node, maAppend);
}

void VS_CC create_nlh(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    create(in, out, core, api, nss::Model::NLH);
}
void VS_CC create_twsc(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* api) {
    create(in, out, core, api, nss::Model::TWSC);
}

}  // namespace

void register_nlh(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kNlhSignature);
    vspapi->registerFunction("NLH", args.c_str(), "clip:vnode;", nss::checked_create<create_nlh, kDefaultMemoryLimitMb>, nullptr, plugin);
}

void register_twsc(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kTwscSignature);
    vspapi->registerFunction("TWSC", args.c_str(), "clip:vnode;", nss::checked_create<create_twsc, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
