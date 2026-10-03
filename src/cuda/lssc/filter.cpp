// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.LSSC: parse_lssc with the "nss_cuda" prefix (D14) around the
// device pipeline of lssc/kernels.hpp, one plane at a time. The host only
// sequences the steps and handles index bookkeeping: the k-means round loop
// with its empty-cluster steals (from downloaded member counts), the member
// lists of the clusters, the LCG sample positions and the constant DCT atoms.
#include "cuda/common/aggregate.hpp"
#include "cuda/lssc/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/nvtx.hpp"
#include "cuda/runtime/stream_pool.hpp"
#include "frontend/lssc_args.hpp"
#include "frontend/ownership.hpp"
#include "frontend/validate.hpp"
#include "nss/checked.hpp"

#include <VSHelper4.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_cuda {
namespace {

constexpr int kAtoms = 256;     // nss::kLsscDefaultAtoms
constexpr int kClusters = 64;   // nss::kLsscDefaultClusters
constexpr int kKmeansRounds = 8;
constexpr unsigned kSeed = 0x4C535343u;

struct Slot {
    Stream stream;
    DeviceBuffer plane, num, den, out;
    DeviceBuffer patches, values, coef, mean, dict, centroids, assign, counts, members, offsets, scale, flags, distance;
    DeviceBuffer sample_y, sample_a, sample_r, update, meta, sources, power;
    PinnedBuffer staging;  // upload, then download
    std::unique_ptr<OrderedAggregator> aggregator;
};

struct PlaneShape {
    int width = 0, height = 0, np = 0;
    RasterGrid grid{};
};

struct LsscData : nss::LsscParams {
    std::shared_ptr<nss::ResourceBudget> budget;
    nss::NodeRef node;
    VSVideoInfo vi{};
    BackendArgs backend;
    DeviceInfo device;
    PlaneShape planes[3];
    std::size_t plane_floats = 0;
    int max_np = 0;
    std::vector<float> dct;  // min(kAtoms, m) orthonormal 2D DCT atoms, low frequencies first
    std::unique_ptr<SlotPool<Slot>> pool;
};

// nss fill_dct_atoms: unit impulses through the orthonormal inverse 2D DCT,
// ordered by u + v, then by position.
std::vector<float> dct_atoms(int block) {
    const int m = block * block;
    const int count = std::min(kAtoms, m);
    std::vector<std::pair<int, int>> order;
    for (int v = 0; v < block; ++v) {
        for (int u = 0; u < block; ++u) order.emplace_back(u + v, u + v * block);
    }
    std::sort(order.begin(), order.end());
    const double pi = 3.14159265358979323846;
    const auto basis = [&](int k, int x) {
        return (k == 0 ? std::sqrt(1.0 / block) : std::sqrt(2.0 / block)) * std::cos((2 * x + 1) * k * pi / (2 * block));
    };
    std::vector<float> atoms(static_cast<std::size_t>(count) * m);
    for (int a = 0; a < count; ++a) {
        const int u = order[a].second % block, v = order[a].second / block;
        for (int y = 0; y < block; ++y) {
            for (int x = 0; x < block; ++x) atoms[static_cast<std::size_t>(a) * m + y * block + x] = static_cast<float>(basis(v, y) * basis(u, x));
        }
    }
    return atoms;
}

LsscWork make_work(const LsscData& d, Slot& s, int plane) {
    const PlaneShape& p = d.planes[plane];
    LsscWork w{};
    w.plane = s.plane.as<float>();
    w.width = p.width;
    w.grid = p.grid;
    w.block = d.block_size;
    w.m = d.block_size * d.block_size;
    w.np = p.np;
    w.atoms = std::min(kAtoms, p.np);
    w.clusters = std::min(kClusters, p.np);
    w.samples = std::min(kLsscSamples, p.np);
    w.patches = s.patches.as<float>();
    w.values = s.values.as<float>();
    w.coef = s.coef.as<float>();
    w.mean = s.mean.as<float>();
    w.dict = s.dict.as<float>();
    w.centroids = s.centroids.as<float>();
    w.assign = s.assign.as<int>();
    w.counts = s.counts.as<int>();
    w.members = s.members.as<int>();
    w.offsets = s.offsets.as<int>();
    w.scale = s.scale.as<float>();
    w.flags = s.flags.as<int>();
    w.distance = s.distance.as<float>();
    w.sample_y = s.sample_y.as<float>();
    w.sample_a = s.sample_a.as<float>();
    w.sample_r = s.sample_r.as<float>();
    w.update = s.update.as<float>();
    w.meta = s.meta.as<AggregatePatch>();
    return w;
}

template <class T>
void fetch(const void* device, T* host, std::size_t count, const Stream& stream) {
    NSS_CUDA_CHECK(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();
}

// nss::lssc_cluster_workspace: assignments in w.assign; returns the member
// counts.
std::vector<int> cluster(const LsscWork& w, Slot& s) {
    cudaStream_t stream = s.stream;
    std::vector<int> counts(w.clusters, 0);
    int* scratch = w.flags + 2 * w.clusters;  // device ints: changed, steal
    if (w.clusters == 1) {
        NSS_CUDA_CHECK(cudaMemsetAsync(w.assign, 0, static_cast<std::size_t>(w.np) * sizeof(int), stream));
        counts[0] = w.np;
        return counts;
    }
    lssc_seed_centroids(w, stream);
    for (int round = 0; round < kKmeansRounds; ++round) {
        NSS_CUDA_CHECK(cudaMemsetAsync(scratch, 0, 2 * sizeof(int), stream));
        lssc_assign(w, round == 0, scratch, stream);
        lssc_accumulate(w, stream);
        int changed = 0;
        NSS_CUDA_CHECK(cudaMemcpyAsync(&changed, scratch, sizeof(int), cudaMemcpyDeviceToHost, stream));
        fetch(w.counts, counts.data(), counts.size(), s.stream);
        for (int c = 0; c < w.clusters; ++c) {
            if (counts[c] > 0) continue;
            // An empty cluster takes the member farthest from its centroid.
            lssc_farthest(w, scratch + 1, stream);
            int steal = -1;
            fetch(scratch + 1, &steal, 1, s.stream);
            if (steal < 0) steal = c % w.np;
            lssc_steal(w, c, steal, stream);
            fetch(w.counts, counts.data(), counts.size(), s.stream);
            changed = 1;
        }
        if (!changed && round > 0) break;
    }
    lssc_count(w, stream);
    fetch(w.counts, counts.data(), counts.size(), s.stream);
    return counts;
}

void denoise_plane(const LsscData& d, Slot& s, int plane, float sigma) {
    cudaStream_t stream = s.stream;
    const PlaneShape& p = d.planes[plane];
    const LsscWork w = make_work(d, s, plane);
    lssc_pack(w, stream);
    {
        NSS_CUDA_RANGE("lssc.cluster");
        const std::vector<int> counts = cluster(w, s);
        // Member lists in (cluster, patch) order.
        std::vector<int> assign(w.np), offsets(w.clusters + 1, 0), members(w.np);
        fetch(w.assign, assign.data(), assign.size(), s.stream);
        for (int c = 0; c < w.clusters; ++c) offsets[c + 1] = offsets[c] + counts[c];
        std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
        for (int j = 0; j < w.np; ++j) members[cursor[assign[j]]++] = j;
        NSS_CUDA_CHECK(cudaMemcpyAsync(w.offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        NSS_CUDA_CHECK(cudaMemcpyAsync(w.members, members.data(), members.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    {
        NSS_CUDA_RANGE("lssc.dictionary");
        // DCT atoms first, the rest from LCG-sampled patches.
        const int dct = std::min(w.atoms, w.m);
        NSS_CUDA_CHECK(cudaMemcpyAsync(w.dict, d.dct.data(), static_cast<std::size_t>(dct) * w.m * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        std::vector<int> sources(std::max(1, w.atoms - dct));
        unsigned state = kSeed;
        for (int a = dct; a < w.atoms; ++a) {
            state = state * 1664525u + 1013904223u;
            sources[a - dct] = static_cast<int>(state % static_cast<unsigned>(w.np));
        }
        NSS_CUDA_CHECK(cudaMemcpyAsync(s.sources.get(), sources.data(), sources.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        lssc_patch_atoms(w, s.sources.as<int>(), dct, stream);
        if (w.np >= 2) lssc_ksvd(w, stream);
    }
    float lipschitz = 1.f;
    lssc_lipschitz(w, s.power.as<float>(), s.power.as<float>() + kAtoms + 256, stream);
    fetch(s.power.as<float>() + kAtoms + 256, &lipschitz, 1, s.stream);
    const float mu = 0.9f / lipschitz;
    {
        NSS_CUDA_RANGE("lssc.code");
        lssc_center(w, stream);
        NSS_CUDA_CHECK(cudaMemsetAsync(w.flags, 0, 2 * static_cast<std::size_t>(w.clusters) * sizeof(int), stream));
        for (int round = 0; round < kLsscRounds; ++round) lssc_round(w, round, mu, sigma, stream);
        lssc_reconstruct(w, stream);
    }
    NSS_CUDA_RANGE("lssc.aggregate");
    const AggregateTarget target{s.num.as<float>(), s.den.as<float>(), p.width, p.height, p.width, 1, d.plane_floats};
    s.aggregator->run(w.values, w.meta, w.np, w.block, target, stream);
    aggregate_finish(s.num.as<float>(), s.den.as<float>(), s.plane.as<float>(), p.width, p.height, p.width,
                     s.out.as<float>(), stream);
}

const VSFrame* getFrame(int n, int activation, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                        const VSAPI* vsapi) {
    auto* d = static_cast<LsscData*>(instance);
    if (activation == arInitial) {
        vsapi->requestFrameFilter(n, d->node, ctx);
        return nullptr;
    }
    if (activation != arAllFramesReady) return nullptr;
    NSS_CUDA_RANGE("lssc.frame");
    auto slot = d->pool->acquire();
    Slot& s = *slot;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
        const PlaneShape& p = d->planes[plane];
        const std::size_t row_bytes = static_cast<std::size_t>(p.width) * sizeof(float);
        if (d->sigma[plane] == 0.f) {
            vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                        vsapi->getStride(src, plane), row_bytes, p.height);
            continue;
        }
        upload_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes, p.height, s.staging.get(),
                     s.plane.get(), row_bytes, s.stream);
        denoise_plane(*d, s, plane, d->sigma[plane] / 255.f);
        // The staged upload has been consumed by now (stream order), so the
        // same region takes the result.
        begin_download(s.out.get(), row_bytes, row_bytes, p.height, s.staging.get(), s.stream);
        s.stream.synchronize();
        finish_download(s.staging.get(), row_bytes, p.height, vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane));
    }
    frames.freeFrame(src);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<LsscData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
    }
    delete d;
}

std::unique_ptr<Slot> make_slot(const LsscData& d) {
    auto slot = std::make_unique<Slot>();
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t np = static_cast<std::size_t>(d.max_np), m = static_cast<std::size_t>(d.block_size) * d.block_size;
    const std::size_t atoms = std::min<std::size_t>(kAtoms, np), clusters = std::min<std::size_t>(kClusters, np);
    const std::size_t samples = std::min<std::size_t>(kLsscSamples, np);
    const auto floats = [&](std::size_t count) { return DeviceBuffer(count * sizeof(float), d.budget); };
    const auto ints = [&](std::size_t count) { return DeviceBuffer(count * sizeof(int), d.budget); };
    slot->plane = DeviceBuffer(plane_bytes, d.budget);
    slot->num = DeviceBuffer(plane_bytes, d.budget);
    slot->den = DeviceBuffer(plane_bytes, d.budget);
    slot->out = DeviceBuffer(plane_bytes, d.budget);
    slot->patches = floats(np * m);
    slot->values = floats(np * m);
    slot->coef = floats(np * atoms);
    slot->mean = floats(np);
    slot->dict = floats(atoms * m);
    slot->centroids = floats(clusters * m);
    slot->assign = ints(np);
    slot->counts = ints(clusters);
    slot->members = ints(np);
    slot->offsets = ints(clusters + 1);
    slot->scale = floats(clusters * atoms);
    slot->flags = ints(2 * clusters + 4);
    slot->distance = floats(np);
    slot->sample_y = floats(samples * m);
    slot->sample_a = floats(atoms * samples);
    slot->sample_r = floats(samples * m);
    slot->update = floats((kLsscSupport + 2) * m);
    slot->meta = DeviceBuffer(np * sizeof(AggregatePatch), d.budget);
    slot->sources = ints(kAtoms);
    slot->power = floats(kAtoms + 256 + 1);
    slot->staging = PinnedBuffer(plane_bytes, d.budget);
    int max_w = 1, max_h = 1;
    for (const PlaneShape& p : d.planes) {
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
    }
    slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, 1, np, d.budget);
    return slot;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.LSSC (D14).
    auto data = std::make_unique<LsscData>();
    LsscData& d = *data;
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    static_cast<nss::LsscParams&>(d) = nss::frontend::parse_lssc(vsapi, in, d.vi, "nss_cuda");
    nss::validate_group_planes(d.vi, d.sigma, d.block_size);
    d.backend = parse_backend_args(vsapi, in, "LSSC");
    d.device = acquire_device(d.backend.device_id, "LSSC", core, vsapi);
    d.budget = nss::current_budget();
    d.dct = dct_atoms(d.block_size);
    std::size_t frame_bytes = 0;
    for (int plane = 0; plane < d.vi.format.numPlanes; ++plane) {
        PlaneShape& p = d.planes[plane];
        p.width = nss::plane_width(d.vi, plane);
        p.height = nss::plane_height(d.vi, plane);
        frame_bytes += static_cast<std::size_t>(p.width) * p.height * sizeof(float);
        d.plane_floats = std::max(d.plane_floats, static_cast<std::size_t>(p.width) * p.height);
        if (d.sigma[plane] == 0.f) continue;
        p.grid = make_raster_grid(p.width, p.height, d.block_size, d.block_step);
        p.np = p.grid.count();
        d.max_np = std::max(d.max_np, p.np);
    }
    // Memory plan: every stream holds the work arrays of the largest plane.
    const std::size_t slot_bytes = 5 * d.plane_floats * sizeof(float) + frame_bytes +
                                   lssc_work_bytes(d.block_size * d.block_size, d.max_np) +
                                   static_cast<std::size_t>(d.max_np) * OrderedAggregator::kBytesPerPatch + (1u << 16);
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / slot_bytes, 1, kDefaultStreams));
    }
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < slot_bytes) {
        throw std::invalid_argument("nss_cuda.LSSC: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((slot_bytes >> 20) + 1) + " MiB");
    }
    {
        DeviceGuard guard(d.device.index);
        std::vector<std::unique_ptr<Slot>> slots;
        for (int i = 0; i < d.backend.num_streams && d.max_np > 0; ++i) slots.push_back(make_slot(d));
        if (slots.empty()) slots.push_back(std::make_unique<Slot>());  // every plane bypassed
        d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
    }
    VSFilterDependency deps[1]{{d.node, rpStrictSpatial}};
    VSNode* node = vsapi->createVideoFilter2("LSSC", &d.vi, nss::checked_frame<getFrame>, freeFilter, fmParallel, deps, 1,
                                            &d, core);
    if (!node) {
        freeFilter(data.release(), core, vsapi);
        throw std::runtime_error("nss_cuda.LSSC: failed to create filter");
    }
    data.release();
    vsapi->mapConsumeNode(out, "clip", node, maAppend);
}

}  // namespace

void register_lssc(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    static const std::string args = signature(nss::frontend::kLsscSignature);
    vspapi->registerFunction("LSSC", args.c_str(), "clip:vnode;", nss::checked_create<create>, nullptr, plugin);
}

}  // namespace nss_cuda
