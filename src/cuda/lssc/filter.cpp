// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.LSSC: parse_lssc with the "nss_cuda" prefix around the
// device pipeline of lssc/kernels.hpp, one plane at a time. The host only
// sequences the steps and handles index bookkeeping: the LCG sample positions
// and the constant DCT atoms.
#include "cuda/common/aggregate.hpp"
#include "cuda/lssc/kernels.hpp"
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/frame_memory.hpp"
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
constexpr unsigned kSeed = 0x4C535343u;
// Buffer sets per slot: three, and a fourth for subsampled clips of several planes (a chroma plane codes in a
// fraction of its dictionary's time, so the next frame's first plane must not wait for this frame's last set:
// YUV420 1080p 1.15x for one set, 42 MiB; more sets measured the same; gray and RGB gain nothing).
constexpr int kMaxDepth = 4;
// Frames in flight per instance (staging sets) where the limit leaves room:
// three for one denoised plane (gray 1080p, one stream: 1.045 against two;
// four measured the same as three), two for more planes (RGB measured no gain).
constexpr std::size_t frames_in_flight(int denoised) { return denoised == 1 ? 3 : 2; }

class Event {
public:
    Event() { NSS_CUDA_CHECK(cudaEventCreateWithFlags(&event_, kStreamEvent)); }
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
// stream packs its patches, a `side` stream builds the dictionary and its Gram
// matrix, and `stream` codes and aggregates once both are done; `copy`
// downloads the result. Where the limit leaves room the buffers the front end
// and side write and the coding reads (input, patches, dictionary, K-SVD
// samples) are held up to three times, each set with a front and a side
// stream of its own, and planes take them in turn (`depth`): the front end of
// the next plane then runs while the stream still codes, and the dictionaries
// of a frame's planes are built at the same time (K-SVD runs on 16 blocks; on
// one side stream the three chains of a YUV420 frame were its critical path). A plane waits for the coding of the plane that last used its
// buffers (`coded`), and for the download before it rewrites `out`.
struct Slot {
    Stream stream, copy, load;
    Stream front[kMaxDepth], side[kMaxDepth];
    Event downloaded;
    Event packed[kMaxDepth], dictionary[kMaxDepth], uploaded[kMaxDepth], coded[kMaxDepth];
    DeviceBuffer input[kMaxDepth], patches[kMaxDepth], dict[kMaxDepth], transpose[kMaxDepth], gram[kMaxDepth],
        inv_norm[kMaxDepth], sample_y[kMaxDepth], sample_a[kMaxDepth], sample_r[kMaxDepth], ksvd_sync[kMaxDepth],
        sources[kMaxDepth];
    DeviceBuffer num, den, out;
    DeviceBuffer values, mean, yy, flags, alpha0, ubuf, counters, meta;
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
    int code_slots = 1;  // coding warps (their u rows in each slot)
    std::vector<float> dct;  // min(kAtoms, m) orthonormal 2D DCT atoms, low frequencies first
    DeviceBuffer dct_device; // the same on the device, the source of every dictionary's first atoms
    std::unique_ptr<SlotPool<Slot>> pool;
    std::unique_ptr<SlotPool<Staging>> staging;
    // Direct DMA to and from page-locked VS frame planes (vapoursynth-classic).
    std::unique_ptr<FrameMemory> frame_memory;
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
    w.samples = std::min(kLsscSamples, p.np);
    w.cap = nss::lssc_omp_cap(w.m, w.atoms);
    w.patches = s.patches[b].as<float>();
    w.values = s.values.as<float>();
    w.mean = s.mean.as<float>();
    w.yy = s.yy.as<float>();
    w.flags = s.flags.as<int>();
    w.alpha0 = s.alpha0.as<float>();
    w.batch = std::min(kLsscBatch, p.np);
    w.ubuf = s.ubuf.as<float>();
    w.code_slots = d.code_slots;
    w.counters = s.counters.as<int>();
    w.dict = s.dict[b].as<float>();
    w.transpose = s.transpose[b].as<float>();
    w.gram = s.gram[b].as<float>();
    w.inv_norm = s.inv_norm[b].as<float>();
    w.sample_y = s.sample_y[b].as<float>();
    w.sample_a = s.sample_a[b].as<float>();
    w.sample_r = s.sample_r[b].as<float>();
    w.ksvd_sync = s.ksvd_sync[b].as<int>();
    w.meta = s.meta.as<AggregatePatch>();
    return w;
}

void denoise_plane(const LsscData& d, Slot& s, int plane, float sigma, int b) {
    cudaStream_t stream = s.stream, front = s.front[b], side = s.side[b];
    const PlaneShape& p = d.planes[plane];
    LsscWork w = make_work(d, s, plane, b);
    NSS_CUDA_CHECK(cudaStreamWaitEvent(front, s.uploaded[b], 0));
    NSS_CUDA_CHECK(cudaStreamWaitEvent(front, s.coded[b], 0));
    lssc_pack(w, front);
    NSS_CUDA_CHECK(cudaEventRecord(s.packed[b], front));
    NSS_CUDA_CHECK(cudaStreamWaitEvent(side, s.packed[b], 0));
    {
        NSS_CUDA_RANGE("lssc.dictionary");
        // DCT atoms first, the rest from LCG-sampled patches.
        const int dct = std::min(w.atoms, w.m);
        // From device memory: no host buffer whose lifetime the copies depend
        // on (a pageable source is neither promised to be captured when
        // cudaMemcpyAsync returns nor to leave the call asynchronous).
        NSS_CUDA_CHECK(cudaMemcpyAsync(w.dict, d.dct_device.get(), static_cast<std::size_t>(dct) * w.m * sizeof(float),
                                       cudaMemcpyDeviceToDevice, side));
        lssc_lcg_sources(s.sources[b].as<int>(), dct, w.atoms, w.np, kSeed, side);
        lssc_patch_atoms(w, s.sources[b].as<int>(), dct, side);
        if (w.np >= 2) lssc_ksvd(w, side);
        lssc_prepare(w, side);
        NSS_CUDA_CHECK(cudaEventRecord(s.dictionary[b], side));
    }
    NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, s.dictionary[b], 0));
    {
        NSS_CUDA_RANGE("lssc.code");
        lssc_code(w, sigma, stream);
    }
    NSS_CUDA_RANGE("lssc.aggregate");
    const AggregateTarget target{s.num.as<float>(), s.den.as<float>(), p.width, p.height, p.width, 1, d.plane_floats};
    s.aggregator->run(w.values, w.meta, w.np, w.block, target, stream);
    NSS_CUDA_CHECK(cudaStreamWaitEvent(stream, s.downloaded, 0));
    lssc_finish(s.num.as<float>(), s.den.as<float>(), w.plane, p.width, p.height, p.width,
                nss::lssc_image_weight(d.block_size, d.block_step, sigma), s.out.as<float>(), stream);
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
    // Planes page-locked for direct DMA skip their staging copy in
    // (direct_in) or out (direct_out); src and dst are held until the copies
    // have finished.
    bool direct_in[3]{}, direct_out[3]{};
    for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
        const PlaneShape& p = d->planes[plane];
        if (d->sigma[plane] == 0.f) {
            vsh::bitblt(vsapi->getWritePtr(dst, plane), vsapi->getStride(dst, plane), vsapi->getReadPtr(src, plane),
                        vsapi->getStride(src, plane), row_bytes(plane), p.height);
            continue;
        }
        direct_in[plane] = d->frame_memory->pinned(src, plane);
        direct_out[plane] = d->frame_memory->pinned(dst, plane);
        if (whole && !direct_in[plane]) {
            stage_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes(plane), p.height, region(r));
        }
        r += whole;
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
                for (cudaStream_t stream : {s.load.get(), s.stream.get(), s.copy.get()}) cudaStreamSynchronize(stream);
                for (int b = 0; b < kMaxDepth; ++b) {
                    cudaStreamSynchronize(s.front[b].get());
                    cudaStreamSynchronize(s.side[b].get());
                }
            }
        } drain{s};
        for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
            if (d->sigma[plane] == 0.f) continue;
            const PlaneShape& p = d->planes[plane];
            std::uint8_t* staged = region(whole ? r++ : 0);
            if (!whole && !direct_in[plane]) {
                stage_plane(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes(plane), p.height, staged);
            }
            const int b = s.turn;
            s.turn = (s.turn + 1) % d->depth;
            NSS_CUDA_CHECK(cudaStreamWaitEvent(s.load, s.coded[b], 0));
            if (direct_in[plane]) {
                upload_direct(vsapi->getReadPtr(src, plane), vsapi->getStride(src, plane), row_bytes(plane), p.height,
                              s.input[b].get(), row_bytes(plane), s.load);
            } else {
                upload_staged(staged, row_bytes(plane), p.height, s.input[b].get(), row_bytes(plane), s.load);
            }
            NSS_CUDA_CHECK(cudaEventRecord(s.uploaded[b], s.load));
            denoise_plane(*d, s, plane, d->sigma[plane] / 255.f, b);
            // The staged upload has been consumed by now (stream order), so
            // the same region takes the result.
            cudaStream_t out_stream = whole ? s.copy.get() : s.stream.get();
            if (whole) NSS_CUDA_CHECK(cudaStreamWaitEvent(s.copy, s.coded[b], 0));
            if (direct_out[plane]) {
                download_direct(s.out.get(), row_bytes(plane), row_bytes(plane), p.height, vsapi->getWritePtr(dst, plane),
                                vsapi->getStride(dst, plane), out_stream);
            } else {
                begin_download(s.out.get(), row_bytes(plane), row_bytes(plane), p.height, staged, out_stream);
            }
            if (whole) {
                NSS_CUDA_CHECK(cudaEventRecord(s.downloaded, s.copy));
            } else {
                s.stream.synchronize();
                if (!direct_out[plane]) {
                    finish_download(staged, row_bytes(plane), p.height, vsapi->getWritePtr(dst, plane),
                                    vsapi->getStride(dst, plane));
                }
            }
        }
        if (whole) NSS_CUDA_CHECK(cudaEventRecord(g.done, s.copy));
        drain.armed = false;
    }
    if (whole) {
        NSS_CUDA_CHECK(cudaEventSynchronize(g.done));
        for (int plane = 0, r = 0; plane < d->vi.format.numPlanes; ++plane) {
            if (d->sigma[plane] == 0.f) continue;
            std::uint8_t* staged = region(r++);
            if (direct_out[plane]) continue;  // the DMA wrote it
            finish_download(staged, row_bytes(plane), d->planes[plane].height, vsapi->getWritePtr(dst, plane),
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
        d->frame_memory.reset();
    }
    delete d;
}

std::unique_ptr<Slot> make_slot(const LsscData& d) {
    auto slot = std::make_unique<Slot>();
    const std::size_t plane_bytes = d.plane_floats * sizeof(float);
    const std::size_t np = static_cast<std::size_t>(d.max_np), m = static_cast<std::size_t>(d.block_size) * d.block_size;
    const std::size_t atoms = std::min<std::size_t>(kAtoms, np);
    const std::size_t samples = std::min<std::size_t>(kLsscSamples, np);
    const auto floats = [&](std::size_t count) { return DeviceBuffer(count * sizeof(float), d.budget); };
    const auto ints = [&](std::size_t count) { return DeviceBuffer(count * sizeof(int), d.budget); };
    for (int b = 0; b < d.depth; ++b) {
        slot->input[b] = DeviceBuffer(plane_bytes, d.budget);
        slot->patches[b] = floats(np * m);
        slot->dict[b] = floats(atoms * m);
        slot->transpose[b] = floats(atoms * m);
        slot->gram[b] = floats(atoms * atoms);
        slot->inv_norm[b] = floats(atoms);
        slot->sample_y[b] = floats(samples * m);
        slot->sample_a[b] = floats(atoms * samples);
        slot->sample_r[b] = floats(samples * m);
        slot->ksvd_sync[b] = ints(atoms * (kLsscSamples / 32 + 1));
        slot->sources[b] = ints(kAtoms);
    }
    slot->num = DeviceBuffer(plane_bytes, d.budget);
    slot->den = DeviceBuffer(plane_bytes, d.budget);
    slot->out = DeviceBuffer(plane_bytes, d.budget);
    slot->values = floats(np * m);
    slot->mean = floats(np);
    slot->yy = floats(np);
    slot->flags = ints(np);
    slot->alpha0 = floats(std::min<std::size_t>(np, kLsscBatch) * atoms);
    slot->ubuf = DeviceBuffer(lssc_code_scratch_bytes(d.code_slots, static_cast<int>(m), static_cast<int>(atoms)), d.budget);
    slot->counters = ints((np + kLsscBatch - 1) / kLsscBatch + 1);
    slot->meta = DeviceBuffer(np * sizeof(AggregatePatch), d.budget);
    int max_w = 1, max_h = 1;
    for (const PlaneShape& p : d.planes) {
        max_w = std::max(max_w, p.width);
        max_h = std::max(max_h, p.height);
    }
    slot->aggregator = std::make_unique<OrderedAggregator>(max_w, max_h, 1, np, d.budget);
    return slot;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
    // Same validation order and text as nss.LSSC.
    auto data = std::make_unique<LsscData>();
    LsscData& d = *data;
    d.node = nss::get_node(vsapi, in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);
    static_cast<nss::LsscParams&>(d) = nss::frontend::parse_lssc(vsapi, in, d.vi, "nss_cuda");
    nss::validate_group_planes(d.vi, d.sigma, d.block_size);
    d.backend = parse_backend_args(vsapi, in, "LSSC");
    d.device = acquire_device(d.backend.device_id, "LSSC", core, vsapi);
    d.budget = nss::current_budget();
    d.frame_memory = std::make_unique<FrameMemory>(core, vsapi);
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
    const int pm = d.block_size * d.block_size, plane_atoms = std::min(kAtoms, std::max(d.max_np, 1));
    const std::size_t base_bytes = 4 * plane_bytes + lssc_work_bytes(pm, d.max_np) +
                                   static_cast<std::size_t>(d.max_np) * OrderedAggregator::kBytesPerPatch + (1u << 16);
    // The DCT atoms are one device copy beside the streams.
    const std::size_t dct_bytes = std::max<std::size_t>(d.dct.size(), 1) * sizeof(float);
    const std::size_t total = d.budget ? d.budget->snapshot().limit : SIZE_MAX;
    const std::size_t limit = total == SIZE_MAX ? SIZE_MAX : total > dct_bytes ? total - dct_bytes : 0;
    // Coding warps: one per 32 patches up to the resident ones, halved while
    // a stream would not fit the limit (the output does not change).
    if (d.max_np > 0) {
        d.code_slots = lssc_code_slots(d.device.index, d.max_np);
        const std::size_t one_stream = base_bytes + frame_bytes + plane_bytes;
        while (d.code_slots > 1 && limit != SIZE_MAX &&
               one_stream + lssc_code_scratch_bytes(d.code_slots, pm, plane_atoms) > limit)
            d.code_slots /= 2;
    }
    const std::size_t slot_bytes = base_bytes + (d.max_np > 0 ? lssc_code_scratch_bytes(d.code_slots, pm, plane_atoms) : 0);
    const std::size_t set_bytes = frame_bytes + plane_bytes;
    const std::size_t stream_bytes = slot_bytes + set_bytes;
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
    const std::size_t atoms = std::min<std::size_t>(kAtoms, np);
    const std::size_t samples = std::min<std::size_t>(kLsscSamples, np);
    const std::size_t extra_set = plane_bytes +
                                  (np * m + 2 * atoms * m + atoms * atoms + atoms + 2 * samples * m + atoms * samples) * sizeof(float) +
                                  (atoms * (kLsscSamples / 32 + 1) + kAtoms) * sizeof(int);
    const bool subsampled = d.vi.format.subSamplingW > 0 || d.vi.format.subSamplingH > 0;
    const int max_depth = d.denoised > 1 && subsampled ? kMaxDepth : 3;
    while (d.max_np > 0 && d.depth < max_depth && room / streams >= extra_set) {
        ++d.depth;
        if (room != SIZE_MAX) room -= streams * extra_set;
    }
    const std::size_t full_set = set_bytes + more_regions * (d.regions > 1);
    const std::size_t in_flight = frames_in_flight(d.denoised);
    const std::size_t sets = streams + std::min(std::max(streams, in_flight) - streams, room / full_set);
    {
        DeviceGuard guard(d.device.index);
        // Synchronous from pageable memory: returns once the source is staged
        // (CUDA Runtime API, "API synchronization behavior").
        d.dct_device = DeviceBuffer(dct_bytes, d.budget);
        if (!d.dct.empty()) {
            NSS_CUDA_CHECK(cudaMemcpy(d.dct_device.get(), d.dct.data(), d.dct.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
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
            NSS_CUDA_CHECK(cudaEventCreateWithFlags(&set->done, kHostWaitEvent));
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
