// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.LSSC: parse_lssc with the "nss_cuda" prefix (D14) around the
// device pipeline of lssc/kernels.hpp, one plane at a time. The host only
// sequences the steps and handles index bookkeeping: the LCG sample positions
// and the constant DCT atoms.
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
constexpr int kMaxDepth = 3;  // buffer sets per slot
// Frames in flight per instance (staging sets) where the limit leaves room:
// three for one denoised plane (gray 1080p, one stream: 1.045 against two;
// four measured the same as three), two for more planes (RGB measured no gain).
constexpr std::size_t frames_in_flight(int denoised) { return denoised == 1 ? 3 : 2; }

class Event {
public:
    Event() { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() {
        if (event_) cudaEventDestroy(event_);
    }
    operator cudaEvent_t() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// A plane runs on the streams of its slot: `load` uploads it, a `front`
// stream packs and clusters it (k-means, queued without host round trips),
// `side` builds the dictionary (atoms, K-SVD, Lipschitz estimate) meanwhile,
// and `stream` codes and aggregates once both are done; `copy` downloads the
// result. Where the limit leaves room the buffers the front end writes and the
// coding reads (input, patches, assignments and member lists, dictionary,
// Lipschitz estimate, k-means scratch) are held up to three times, each set
// with a front stream of its own, and planes take them in turn (`depth`): the
// front ends of the next planes then run while the stream still codes, and
// two planes can cluster at once (k-means is a chain of small kernels that
// wait for room beside the coding). A plane waits for the coding of the plane
// that last used its buffers (`coded`), and for the download before it
// rewrites `out`.
struct Slot {
    Stream stream, side, copy, load;
    Stream front[kMaxDepth];  // a front end per buffer set: planes cluster concurrently
    Event packed, clustered, dictionary, downloaded;
    Event uploaded[kMaxDepth], coded[kMaxDepth];
    DeviceBuffer input[kMaxDepth], patches[kMaxDepth], dict[kMaxDepth], assign[kMaxDepth], members[kMaxDepth],
        offsets[kMaxDepth], power[kMaxDepth];
    DeviceBuffer centroids[kMaxDepth], counts[kMaxDepth], chunks[kMaxDepth];  // k-means scratch; counts + 2 state ints
    DeviceBuffer num, den, out;
    DeviceBuffer values, coef, mean, scale, flags;
    DeviceBuffer sample_y, sample_a, sample_r, ksvd_sync, tiles, partial, meta, sources;
    int turn = 0;  // the buffer set of the next plane
    std::unique_ptr<OrderedAggregator> aggregator;
};

// Pinned staging of one frame in flight: a region per denoised plane, each
// its upload and then its download. With a region for every plane the frame
// is copied in before it takes a slot and copied out after it has given the
// slot back, so the stream works while other threads copy; with one region
// (a tight memory limit) the planes are copied one by one inside the slot.
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
    int denoised = 0;  // planes with sigma > 0
    int regions = 1;   // staging regions per set: 1 or denoised
    int depth = 1;     // buffer sets per slot: 1 to kMaxDepth
    std::vector<float> dct;  // min(kAtoms, m) orthonormal 2D DCT atoms, low frequencies first
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
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

LsscWork make_work(const LsscData& d, Slot& s, int plane, int b) {
    const PlaneShape& p = d.planes[plane];
    LsscWork w{};
    w.plane = s.input[b].as<float>();
    w.width = p.width;
    w.grid = p.grid;
    w.block = d.block_size;
    w.m = d.block_size * d.block_size;
    w.np = p.np;
    w.atoms = std::min(kAtoms, p.np);
    w.clusters = std::min(kClusters, p.np);
    w.samples = std::min(kLsscSamples, p.np);
    w.patches = s.patches[b].as<float>();
    w.values = s.values.as<float>();
    w.coef = s.coef.as<float>();
    w.mean = s.mean.as<float>();
    w.dict = s.dict[b].as<float>();
    w.centroids = s.centroids[b].as<float>();
    w.assign = s.assign[b].as<int>();
    w.counts = s.counts[b].as<int>();
    w.members = s.members[b].as<int>();
    w.offsets = s.offsets[b].as<int>();
    w.scale = s.scale.as<float>();
    w.flags = s.flags.as<int>();
    w.sample_y = s.sample_y.as<float>();
    w.sample_a = s.sample_a.as<float>();
    w.sample_r = s.sample_r.as<float>();
    w.ksvd_sync = s.ksvd_sync.as<int>();
    w.tiles = s.tiles.as<int>();
    w.chunks = s.chunks[b].as<int>();
    w.partial = s.partial.as<float>();
    w.meta = s.meta.as<AggregatePatch>();
    return w;
}

void denoise_plane(const LsscData& d, Slot& s, int plane, float sigma, int b) {
    cudaStream_t stream = s.stream, front = s.front[b], side = s.side;
    const PlaneShape& p = d.planes[plane];
    const LsscWork w = make_work(d, s, plane, b);
    NSS_CUDA_CHECK(cudaStreamWaitEvent(front, s.uploaded[b], 0));
    NSS_CUDA_CHECK(cudaStreamWaitEvent(front, s.coded[b], 0));
    lssc_pack(w, front);
    NSS_CUDA_CHECK(cudaEventRecord(s.packed, front));
    NSS_CUDA_CHECK(cudaStreamWaitEvent(side, s.packed, 0));
    float* power = s.power[b].as<float>();
    float* lipschitz = power + kAtoms + 256;
    {
        NSS_CUDA_RANGE("lssc.dictionary");
        // DCT atoms first, the rest from LCG-sampled patches.
        const int dct = std::min(w.atoms, w.m);
        NSS_CUDA_CHECK(cudaMemcpyAsync(w.dict, d.dct.data(), static_cast<std::size_t>(dct) * w.m * sizeof(float),
                                       cudaMemcpyHostToDevice, side));
        std::vector<int> sources(std::max(1, w.atoms - dct));
        unsigned state = kSeed;
        for (int a = dct; a < w.atoms; ++a) {
            state = state * 1664525u + 1013904223u;
            sources[a - dct] = static_cast<int>(state % static_cast<unsigned>(w.np));
        }
        NSS_CUDA_CHECK(cudaMemcpyAsync(s.sources.get(), sources.data(), sources.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, side));
        lssc_patch_atoms(w, s.sources.as<int>(), dct, side);
        if (w.np >= 2) lssc_ksvd(w, side);
        lssc_lipschitz(w, power, lipschitz, side);
        NSS_CUDA_CHECK(cudaEventRecord(s.dictionary, side));
    }
    {
        NSS_CUDA_RANGE("lssc.cluster");
        lssc_cluster(w, kKmeansRounds, w.counts + w.clusters, front);
        NSS_CUDA_CHECK(cudaEventRecord(s.clustered, front));
    }
    NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, s.clustered, 0));
    NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, s.dictionary, 0));
    {
        NSS_CUDA_RANGE("lssc.code");
        lssc_center(w, stream);
        NSS_CUDA_CHECK(cudaMemsetAsync(w.flags, 0, 2 * static_cast<std::size_t>(w.clusters) * sizeof(int), stream));
        for (int round = 0; round < kLsscRounds; ++round) lssc_round(w, round, lipschitz, sigma, stream);
        lssc_reconstruct(w, stream);
    }
    NSS_CUDA_RANGE("lssc.aggregate");
    const AggregateTarget target{s.num.as<float>(), s.den.as<float>(), p.width, p.height, p.width, 1, d.plane_floats};
    s.aggregator->run(w.values, w.meta, w.np, w.block, target, stream);
    NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, s.downloaded, 0));
    aggregate_finish(s.num.as<float>(), s.den.as<float>(), w.plane, p.width, p.height, p.width, s.out.as<float>(),
                     stream);
    NSS_CUDA_CHECK(cudaEventRecord(s.coded[b], stream));
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
    // The staging set is leased before the output frame is allocated, so at
    // most that many frames are in flight.
    auto staging = d->staging->acquire();
    Staging& g = *staging;
    DeviceGuard guard(d->device.index);
    nss::ResourceScope resource_scope(d->budget);
    nss::FrameScope frames(vsapi);
    const VSFrame* src = frames.getFrameFilter(n, d->node, ctx);
    VSFrame* dst = frames.newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    const bool whole = d->regions == d->denoised;
    const auto region = [&](int r) {
        return g.bytes.as<std::uint8_t>() + static_cast<std::size_t>(r) * d->plane_floats * sizeof(float);
    };
    const auto row_bytes = [&](int plane) { return static_cast<std::size_t>(d->planes[plane].width) * sizeof(float); };
    for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
        const PlaneShape& p = d->planes[plane];
        if (d->sigma[plane] == 0.f) {
            vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                        vsapi->getStride(src, plane), row_bytes(plane), p.height);
        } else if (whole) {
            stage_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes(plane), p.height, region(r++));
        }
    }
    {
        auto slot = d->pool->acquire();
        Slot& s = *slot;
        // On an error the queued copies must not outlive the staging lease.
        struct Drain {
            const Slot& s;
            bool armed = true;
            ~Drain() {
                if (!armed) return;
                for (cudaStream_t stream : {s.load.get(), s.front[0].get(), s.front[1].get(), s.front[2].get(), s.side.get(),
                                            s.stream.get(), s.copy.get()}) {
                    cudaStreamSynchronize(stream);
                }
            }
        } drain{s};
        for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
            if (d->sigma[plane] == 0.f) continue;
            const PlaneShape& p = d->planes[plane];
            std::uint8_t* staged = region(whole ? r++ : 0);
            if (!whole) {
                stage_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes(plane), p.height, staged);
            }
            const int b = s.turn;
            s.turn = (s.turn + 1) % d->depth;
            NSS_CUDA_CHECK(cudaStreamWaitEvent(s.load, s.coded[b], 0));
            upload_staged(staged, row_bytes(plane), p.height, s.input[b].get(), row_bytes(plane), s.load);
            NSS_CUDA_CHECK(cudaEventRecord(s.uploaded[b], s.load));
            denoise_plane(*d, s, plane, d->sigma[plane] / 255.f, b);
            // The staged upload has been consumed by now (stream order), so
            // the same region takes the result.
            if (whole) {
                NSS_CUDA_CHECK(cudaStreamWaitEvent(s.copy, s.coded[b], 0));
                begin_download(s.out.get(), row_bytes(plane), row_bytes(plane), p.height, staged, s.copy);
                NSS_CUDA_CHECK(cudaEventRecord(s.downloaded, s.copy));
            } else {
                begin_download(s.out.get(), row_bytes(plane), row_bytes(plane), p.height, staged, s.stream);
                s.stream.synchronize();
                finish_download(staged, row_bytes(plane), p.height, vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane));
            }
        }
        if (whole) NSS_CUDA_CHECK(cudaEventRecord(g.done, s.copy));
        drain.armed = false;
    }
    if (whole) {
        NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
        for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
            if (d->sigma[plane] == 0.f) continue;
            finish_download(region(r++), row_bytes(plane), d->planes[plane].height, vsapi->getWritePtr(dst, plane),
                            vsapi->getStride(dst, plane));
        }
    }
    frames.freeFrame(src);
    return frames.keep(dst);
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI*) {
    auto* d = static_cast<LsscData*>(instance);
    {
        DeviceGuard guard(d->device.index);
        d->pool.reset();
        d->staging.reset();
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
    for (int b = 0; b < d.depth; ++b) {
        slot->input[b] = DeviceBuffer(plane_bytes, d.budget);
        slot->patches[b] = floats(np * m);
        slot->dict[b] = floats(atoms * m);
        slot->assign[b] = ints(np);
        slot->members[b] = ints(np);
        slot->offsets[b] = ints(clusters + 1);
        slot->power[b] = floats(kAtoms + 256 + 1);
        slot->centroids[b] = floats(clusters * m);
        slot->counts[b] = ints(clusters + 2);
        if (m <= 64) slot->chunks[b] = ints(np + 64);
    }
    slot->num = DeviceBuffer(plane_bytes, d.budget);
    slot->den = DeviceBuffer(plane_bytes, d.budget);
    slot->out = DeviceBuffer(plane_bytes, d.budget);
    slot->values = floats(np * m);
    slot->coef = floats(np * atoms);
    slot->mean = floats(np);
    slot->scale = floats(clusters * atoms);
    slot->flags = ints(2 * clusters + 4);
    slot->sample_y = floats(samples * m);
    slot->sample_a = floats(atoms * samples);
    slot->sample_r = floats(samples * m);
    slot->ksvd_sync = ints(atoms * (kLsscSamples / 32 + 1));
    if (m <= 64) {
        const std::size_t tiles = static_cast<std::size_t>(lssc_tile_capacity(static_cast<int>(np), static_cast<int>(clusters)));
        slot->tiles = ints(1 + 2 * tiles + clusters + 1 + clusters * atoms);
        slot->partial = floats(tiles * atoms);
    }
    slot->meta = DeviceBuffer(np * sizeof(AggregatePatch), d.budget);
    slot->sources = ints(kAtoms);
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
        ++d.denoised;
        p.grid = make_raster_grid(p.width, p.height, d.block_size, d.block_step);
        p.np = p.grid.count();
        d.max_np = std::max(d.max_np, p.np);
    }
    // Memory plan: every stream holds the work arrays of the largest plane,
    // every frame in flight its output frame and pinned staging. The smallest
    // plan is one staging set of one region and one buffer set per stream; the
    // limit's room beyond it goes first to a region per plane, then to a
    // second buffer set, then to more staging sets.
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t slot_bytes = 4 * plane_bytes + lssc_work_bytes(d.block_size * d.block_size, d.max_np) +
                                   static_cast<std::size_t>(d.max_np) * OrderedAggregator::kBytesPerPatch + (1u << 16);
    const std::size_t set_bytes = frame_bytes + plane_bytes;
    const std::size_t stream_bytes = slot_bytes + set_bytes;
    const std::size_t limit = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    if (!d.backend.streams_explicit) {
        d.backend.num_streams = static_cast<int>(std::clamp<std::size_t>(limit / stream_bytes, 1, kDefaultStreams));
    }
    if (limit / static_cast<std::size_t>(d.backend.num_streams) < stream_bytes) {
        throw std::invalid_argument("nss_cuda.LSSC: memory_limit_mb is too small for this clip: each of the " +
                                    std::to_string(d.backend.num_streams) + " stream(s) needs at least " +
                                    std::to_string((stream_bytes >> 20) + 1) + " MiB");
    }
    const std::size_t streams = static_cast<std::size_t>(d.backend.num_streams);
    std::size_t room = limit == SIZE_MAX ? SIZE_MAX : limit - streams * stream_bytes;
    const std::size_t more_regions = static_cast<std::size_t>(std::max(d.denoised - 1, 0)) * plane_bytes;
    if (d.denoised > 1 && room / streams >= more_regions) {
        d.regions = d.denoised;
        if (room != SIZE_MAX) room -= streams * more_regions;
    }
    const std::size_t np = static_cast<std::size_t>(d.max_np), m = static_cast<std::size_t>(d.block_size) * d.block_size;
    const std::size_t atoms = std::min<std::size_t>(kAtoms, np), clusters = std::min<std::size_t>(kClusters, np);
    const std::size_t extra_set = plane_bytes + (np * m + atoms * m + kAtoms + 257 + clusters * m) * sizeof(float) +
                                  (2 * np + clusters + 1 + clusters + 2 + (m <= 64 ? np + 64 : 0)) * sizeof(int);
    while (d.max_np > 0 && d.depth < kMaxDepth && room / streams >= extra_set) {
        ++d.depth;
        if (room != SIZE_MAX) room -= streams * extra_set;
    }
    const std::size_t full_set = set_bytes + more_regions * (d.regions > 1);
    const std::size_t in_flight = frames_in_flight(d.denoised);
    const std::size_t sets = streams + std::min(std::max(streams, in_flight) - streams, room / full_set);
    {
        DeviceGuard guard(d.device.index);
        std::vector<std::unique_ptr<Slot>> slots;
        std::vector<std::unique_ptr<Staging>> staging;
        if (d.max_np > 0) {
            for (int i = 0; i < d.backend.num_streams; ++i) slots.push_back(make_slot(d));
        } else {
            slots.push_back(std::make_unique<Slot>());  // every plane bypassed
        }
        for (std::size_t i = 0; i < sets; ++i) {
            auto set = std::make_unique<Staging>();
            if (d.max_np > 0) set->bytes = PinnedBuffer(static_cast<std::size_t>(d.regions) * plane_bytes, d.budget);
            NSS_CUDA_CHECK(cudaEventCreateWithFlags(&set->done, cudaEventDisableTiming));
            staging.push_back(std::move(set));
        }
        d.pool = std::make_unique<SlotPool<Slot>>(std::move(slots));
        d.staging = std::make_unique<SlotPool<Staging>>(std::move(staging));
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
    vspapi->registerFunction("LSSC", args.c_str(), "clip:vnode;", nss::checked_create<create, kDefaultMemoryLimitMb>, nullptr, plugin);
}

}  // namespace nss_cuda
