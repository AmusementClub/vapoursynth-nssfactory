// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM4D: the CPU model (include/nss/params/bm4d.hpp, src/cpu/bm4d)
// operation for operation. Matching: one block per reference cube; every
// candidate's distance is summed in the CPU's order (along z by fused
// multiply-adds, then y, then x) and the closest are taken by a block radix
// sort of (distance, offset in scan order), the CPU's insertion order. Groups:
// one block per reference, Haar along the group then the 3D DCT of every
// cube with the CPU's arithmetic, then the fixed-point sums by 64-bit integer
// atomics (exact in any order). Built with --fmad=false.
#include "cuda/bm4d/kernels.hpp"
#include "cuda/runtime/error.hpp"
#include "nss/params/bm4d.hpp"


#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nss_cuda {
namespace {

constexpr int kSearch = nss::kBm4dSearch;
constexpr int kSide = 2 * kSearch + 1;
constexpr int kMatchThreads = 128;
constexpr int kGroupThreads = 256;
constexpr int kMaxGroup = 32;
constexpr unsigned long long kNoKey = ~0ull;

std::size_t align256(std::size_t n) { return (n + 255) & ~std::size_t{255}; }

int position_count(int n, int L, int step) {
    if (n < L) return 0;
    const int regular = (n - L) / step + 1;
    return regular + ((n - L) % step != 0 ? 1 : 0);
}
__host__ __device__ __forceinline__ int position(int n, int L, int step, int i) {
    const int p = i * step;
    return p > n - L ? n - L : p;
}

struct Volume {
    int t, h, w;
    __device__ std::size_t at(int z, int y, int x) const { return (static_cast<std::size_t>(z) * h + y) * w + x; }
};

struct Refs {
    int nz, ny, nx, L, step;
};

// One block a reference. The search region ((L + 14)^3 voxels around the
// reference cube) is staged in shared memory; every offset's distance is
// summed in the CPU's order (along z by fused multiply-adds, then y, then x);
// the closest are the group - 1 smallest keys (distance bits, offset in scan
// order), the CPU's insertion order, found by a radix select (8 bits a round
// over the 44-bit keys) and ranked among themselves.
template <int L>
__global__ void __launch_bounds__(kMatchThreads) match_kernel(const float* V, Volume vol, Refs refs, float limit, int group,
                                                              int* pos, int* count) {
    constexpr int R = L + 2 * kSearch;  // region edge
    __shared__ float region[R * R * R];
    __shared__ unsigned hist[256];
    __shared__ unsigned long long chosen[kMaxGroup];
    __shared__ unsigned long long prefix;
    __shared__ int need, taken;
    const int r = blockIdx.x;
    const int iz = r / (refs.ny * refs.nx), iy = (r / refs.nx) % refs.ny, ix = r % refs.nx;
    const int z0 = position(vol.t, L, refs.step, iz), y0 = position(vol.h, L, refs.step, iy),
              x0 = position(vol.w, L, refs.step, ix);
    for (int e = threadIdx.x; e < R * R * R; e += kMatchThreads) {
        const int z = z0 - kSearch + e / (R * R), y = y0 - kSearch + (e / R) % R, x = x0 - kSearch + e % R;
        region[e] = (z >= 0 && z < vol.t && y >= 0 && y < vol.h && x >= 0 && x < vol.w) ? V[vol.at(z, y, x)] : 0.f;
    }
    __syncthreads();
    // Work items: (dz, dy, a run of kRun neighbouring dx); a thread takes
    // kItems of them (item it * threads + thread). For every (y, z) row of the
    // cube it loads the candidates' rows once for its kRun offsets; each
    // offset's distance keeps the CPU's order: over z by fused multiply-adds
    // (A), then over y (B), then over x.
    constexpr int kRun = 4;
    constexpr int kRuns = (kSide + kRun - 1) / kRun;  // 4 runs of dx
    constexpr int kWork = kSide * kSide * kRuns;
    constexpr int kItems = (kWork + kMatchThreads - 1) / kMatchThreads;
    unsigned long long keys[kItems][kRun];
    int valid = 0;
    const float* ref = region + (kSearch * R + kSearch) * R + kSearch;
#pragma unroll
    for (int it = 0; it < kItems; ++it) {
        const int work = it * kMatchThreads + threadIdx.x;
        const int run = work % kRuns, dyi = (work / kRuns) % kSide, dzi = work / (kRuns * kSide);
        const int dz = dzi - kSearch, dy = dyi - kSearch, dx0 = run * kRun - kSearch;
        const bool row_ok = work < kWork && z0 + dz >= 0 && z0 + dz <= vol.t - L && y0 + dy >= 0 && y0 + dy <= vol.h - L;
        float B[kRun][L];
        if (row_ok) {
            const float* cand = region + ((kSearch + dz) * R + kSearch + dy) * R + kSearch + dx0;
#pragma unroll
            for (int j = 0; j < L; ++j) {
                float A[kRun][L];
#pragma unroll
                for (int u = 0; u < kRun; ++u)
#pragma unroll
                    for (int i = 0; i < L; ++i) A[u][i] = 0.f;
#pragma unroll
                for (int k = 0; k < L; ++k) {
                    const float* rrow = ref + (k * R + j) * R;
                    const float* crow = cand + (k * R + j) * R;
                    float rv[L], cv[L + kRun - 1];
#pragma unroll
                    for (int i = 0; i < L; ++i) rv[i] = rrow[i];
#pragma unroll
                    for (int i = 0; i < L + kRun - 1; ++i) cv[i] = (kSearch + dx0 + i < R) ? crow[i] : 0.f;  // past the row: offsets beyond the search, unused
#pragma unroll
                    for (int u = 0; u < kRun; ++u)
#pragma unroll
                        for (int i = 0; i < L; ++i) {
                            const float e = rv[i] - cv[u + i];
                            A[u][i] = fmaf(e, e, A[u][i]);
                        }
                }
#pragma unroll
                for (int u = 0; u < kRun; ++u)
#pragma unroll
                    for (int i = 0; i < L; ++i) B[u][i] = j == 0 ? A[u][i] : B[u][i] + A[u][i];
            }
        }
#pragma unroll
        for (int u = 0; u < kRun; ++u) {
            keys[it][u] = kNoKey;
            const int dx = dx0 + u;
            const int cx = x0 + dx;
            if (!row_ok || dx > kSearch || cx < 0 || cx > vol.w - L || (dz == 0 && dy == 0 && dx == 0)) continue;
            float s = B[u][0];
#pragma unroll
            for (int i = 1; i < L; ++i) s = s + B[u][i];
            if (s <= limit) {
                const int o = (dzi * kSide + dyi) * kSide + dx + kSearch;
                keys[it][u] = static_cast<unsigned long long>(__float_as_uint(s)) << 12 | static_cast<unsigned>(o);
                ++valid;
            }
        }
    }
    // the number wanted: group - 1 or every valid key
    if (threadIdx.x == 0) {
        need = 0;
        prefix = 0;
        taken = 0;
    }
    __syncthreads();
    atomicAdd(&need, valid);
    __syncthreads();
    const int k_want = min(group - 1, need);
    // every valid key when they all fit, else those up to the k_want-th
    // smallest, found by a radix select (44-bit keys: 6 rounds of 8 bits from
    // the top; warp 0 locates the bin by a prefix sum)
    unsigned long long last = kNoKey - 1;
    if (k_want > 0 && need > k_want) {
        int rank = k_want;  // 1-based rank among keys with the current prefix
        int shift = 40;
        unsigned long long mask = 0;
#pragma unroll 1
        for (int round = 0; round < 6; ++round) {
            for (int b = threadIdx.x; b < 256; b += kMatchThreads) hist[b] = 0;
            __syncthreads();
            const unsigned long long pre = prefix;
#pragma unroll
            for (int it = 0; it < kItems; ++it)
#pragma unroll
                for (int u = 0; u < kRun; ++u) {
                    const unsigned long long key = keys[it][u];
                    if (key != kNoKey && (key & mask) == pre) atomicAdd(&hist[(key >> shift) & 255u], 1u);
                }
            __syncthreads();
            if (threadIdx.x < 32) {
                const int lane = threadIdx.x;
                unsigned local[8];
                unsigned sum = 0;
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    local[b] = hist[lane * 8 + b];
                    sum += local[b];
                }
                unsigned inclusive = sum;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const unsigned v = __shfl_up_sync(0xffffffffu, inclusive, d);
                    if (lane >= d) inclusive += v;
                }
                const unsigned before = inclusive - sum;
                const unsigned hit = __ballot_sync(0xffffffffu, before < static_cast<unsigned>(rank) &&
                                                                   static_cast<unsigned>(rank) <= inclusive);
                if (lane == __ffs(hit) - 1) {
                    unsigned acc = before;
                    int bin = lane * 8;
#pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (acc + local[b] >= static_cast<unsigned>(rank)) {
                            bin = lane * 8 + b;
                            break;
                        }
                        acc += local[b];
                    }
                    need = rank - static_cast<int>(acc);  // rank within the chosen bin
                    prefix = pre | static_cast<unsigned long long>(bin) << shift;
                }
            }
            __syncthreads();
            rank = need;
            mask |= 255ull << shift;
            shift -= 8;
        }
        last = prefix;  // the k_want-th smallest key itself (keys are unique)
    }
    if (k_want > 0) {
#pragma unroll
        for (int it = 0; it < kItems; ++it)
#pragma unroll
            for (int u = 0; u < kRun; ++u)
                if (keys[it][u] <= last) chosen[atomicAdd(&taken, 1)] = keys[it][u];
        __syncthreads();
    }
    int* p = pos + static_cast<std::size_t>(r) * kMaxGroup;
    if (threadIdx.x < k_want) {  // ranked among themselves
        const unsigned long long key = chosen[threadIdx.x];
        int rank = 0;
        for (int i = 0; i < k_want; ++i) rank += chosen[i] < key;
        const int o = static_cast<int>(key & 0xfffu);
        const int dz = o / (kSide * kSide) - kSearch, dy = (o / kSide) % kSide - kSearch, dx = o % kSide - kSearch;
        p[1 + rank] = static_cast<int>(vol.at(z0 + dz, y0 + dy, x0 + dx));
    }
    if (threadIdx.x == 0) {
        p[0] = static_cast<int>(vol.at(z0, y0, x0));
        count[r] = 1 + k_want;
    }
}

__device__ __forceinline__ long long fixed(float x) { return __float2ll_rn(x * nss::kBm4dFixedScale); }

// Haar along the group (m a power of two) of n-element vectors, as the CPU's,
// in place: a level's pairs are read into registers, then written.
template <int N>
__device__ void haar(float* g, int m, bool inverse) {
    const float c = 0.70710678118654752f;
    constexpr int kMaxPairs = (kMaxGroup / 2 * N + kGroupThreads - 1) / kGroupThreads;
    for (int step = 0, len = inverse ? 2 : m; inverse ? len <= m : len > 1; ++step, len = inverse ? len * 2 : len / 2) {
        const int pairs = (len / 2) * N;
        float u[kMaxPairs], v[kMaxPairs];
#pragma unroll
        for (int p = 0; p < kMaxPairs; ++p) {
            const int it = p * kGroupThreads + threadIdx.x;
            if (it >= pairs) continue;
            const int i = it / N, e = it % N;
            const float a = inverse ? g[i * N + e] : g[(2 * i) * N + e];
            const float b = inverse ? g[(len / 2 + i) * N + e] : g[(2 * i + 1) * N + e];
            u[p] = (a + b) * c;
            v[p] = (a - b) * c;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < kMaxPairs; ++p) {
            const int it = p * kGroupThreads + threadIdx.x;
            if (it >= pairs) continue;
            const int i = it / N, e = it % N;
            if (inverse) {
                g[(2 * i) * N + e] = u[p];
                g[(2 * i + 1) * N + e] = v[p];
            } else {
                g[i * N + e] = u[p];
                g[(len / 2 + i) * N + e] = v[p];
            }
        }
        __syncthreads();
    }
}

// The 3D DCT of m cubes (x, then y, then z), forward or inverse, in place:
// one thread a line; each output is D(k, 0) in(0) then fused terms in order,
// as the CPU's.
template <int L>
__device__ void dct3(float* g, int m, const float* D, bool inverse) {
    constexpr int L2 = L * L, L3 = L2 * L;
    for (int axis = 0; axis < 3; ++axis) {
        const int unit = axis == 0 ? 1 : axis == 1 ? L : L2;
        for (int ln = threadIdx.x; ln < m * L2; ln += kGroupThreads) {
            const int q = ln / L2, a = (ln / L) % L, b = ln % L;
            const int base = q * L3 + (axis == 0 ? (a * L + b) * L : axis == 1 ? a * L2 + b : a * L + b);
            float in[L];
#pragma unroll
            for (int i = 0; i < L; ++i) in[i] = g[base + i * unit];
#pragma unroll
            for (int k = 0; k < L; ++k) {
                float acc = (inverse ? D[0 * L + k] : D[k * L + 0]) * in[0];
#pragma unroll
                for (int i = 1; i < L; ++i) acc = fmaf(inverse ? D[i * L + k] : D[k * L + i], in[i], acc);
                g[base + k * unit] = acc;
            }
        }
        __syncthreads();
    }
}

template <int L, bool Wiener>
__global__ void __launch_bounds__(kGroupThreads) group_kernel(const float* noisy, const float* basic, Volume vol,
                                                              const int* pos, const int* count, float sigma,
                                                              const float* D, const float* kaiser,
                                                              unsigned long long* num, unsigned long long* den) {
    constexpr int L3 = L * L * L;
    extern __shared__ float smem[];
    float* g = smem;
    float* b = g + kMaxGroup * L3;
    __shared__ float weight;
    __shared__ int kept_count;
    const int r = blockIdx.x;
    int m = 1;
    while (2 * m <= count[r]) m *= 2;
    const int* p = pos + static_cast<std::size_t>(r) * kMaxGroup;
    const std::size_t plane = static_cast<std::size_t>(vol.h) * vol.w;
    for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
        const int q = it / L3, e = it % L3;
        const std::size_t o = p[q] + (e / (L * L)) * plane + static_cast<std::size_t>((e / L) % L) * vol.w + e % L;
        g[it] = noisy[o];
        if (Wiener) b[it] = basic[o];
    }
    if (threadIdx.x == 0) kept_count = 0;
    __syncthreads();
    haar<L3>(g, m, false);
    dct3<L>(g, m, D, false);
    const float sigma2 = sigma * sigma;
    if (!Wiener) {
        const float threshold = nss::kBm4dLambda * sigma;
        int kept = 0;
        for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
            if (fabsf(g[it]) >= threshold) ++kept;
            else g[it] = 0.f;
        }
        atomicAdd(&kept_count, kept);
        __syncthreads();
        if (threadIdx.x == 0) weight = 1.f / static_cast<float>(max(kept_count, 1));
    } else {
        haar<L3>(b, m, false);
        dct3<L>(b, m, D, false);
        for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
            const float b2 = b[it] * b[it];
            const float wv = b2 / (b2 + sigma2);
            g[it] = g[it] * wv;
            b[it] = wv;
        }
        __syncthreads();
        if (threadIdx.x < 32) {  // the 32-lane sum over t = e m + q
            const int lane = threadIdx.x;
            float acc = 0.f;
            for (int t = lane; t < L3 * m; t += 32) {
                const float wv = b[(t % m) * L3 + t / m];
                acc = fmaf(wv, wv, acc);
            }
#pragma unroll
            for (int half = 16; half > 0; half >>= 1) acc = acc + __shfl_down_sync(0xffffffffu, acc, half);
            if (lane == 0) weight = 1.f / fmaxf(acc, nss::kBm4dEnergyFloor);
        }
    }
    __syncthreads();
    dct3<L>(g, m, D, true);
    haar<L3>(g, m, true);
    const float w = weight;
    for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
        const int q = it / L3, e = it % L3;
        const int z = e / (L * L), y = (e / L) % L, x = e % L;
        const float window = kaiser[z] * kaiser[y] * kaiser[x];
        const float wv = w * window;
        const std::size_t o = p[q] + z * plane + static_cast<std::size_t>(y) * vol.w + x;
        atomicAdd(num + o, static_cast<unsigned long long>(fixed(wv * g[it])));
        atomicAdd(den + o, static_cast<unsigned long long>(fixed(wv)));
    }
}

__global__ void ratio_kernel(const long long* num, const long long* den, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(static_cast<double>(num[i]) / static_cast<double>(den[i]));
}

__global__ void gather_kernel(const float* const* frames, std::size_t pitch, int t, int y0, int x0, int h, int w,
                              float* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    if (x >= w) return;
    dst[(static_cast<std::size_t>(z) * h + y) * w + x] = frames[z][static_cast<std::size_t>(y0 + y) * pitch + x0 + x];
}

__global__ void scatter_kernel(const float* src, int h, int w, int y0, int x0, int z0, int ky0, int kx0, int kw,
                               float* const* frames, std::size_t pitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y + ky0;
    const int f = blockIdx.z;
    if (x >= kw) return;
    const int px = kx0 + x;
    frames[f][static_cast<std::size_t>(y) * pitch + px] =
        src[(static_cast<std::size_t>(z0 + f) * h + (y - y0)) * w + (px - x0)];
}

struct Layout {
    std::size_t basic, num, den, pos, count, total;
    int refs;
};

Layout layout(int t, int h, int w) {
    Layout l{};
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    const int R = std::max(1, position_count(t, 4, 3)) * std::max(1, position_count(h, 4, 3)) *
                  std::max(1, position_count(w, 4, 3));
    l.refs = R;
    l.basic = align256(n * sizeof(float));
    l.num = align256(n * sizeof(long long));
    l.den = l.num;
    l.pos = align256(static_cast<std::size_t>(R) * kMaxGroup * sizeof(int));
    l.count = align256(static_cast<std::size_t>(R) * sizeof(int));
    l.total = l.basic + l.num + l.den + l.pos + l.count;
    return l;
}

template <int L>
void run_stage(const float* noisy, const float* basic, const Volume& vol, float sigma, const nss::Bm4dStage& st,
               const float* D, const float* kaiser, int* pos, int* count, unsigned long long* num,
               unsigned long long* den, cudaStream_t stream) {
    const Refs refs{position_count(vol.t, L, st.step), position_count(vol.h, L, st.step), position_count(vol.w, L, st.step),
                    L, st.step};
    const int R = refs.nz * refs.ny * refs.nx;
    const float sigma2 = sigma * sigma;
    const float limit = st.tau * static_cast<float>(L * L * L) * sigma2;
    match_kernel<L><<<R, kMatchThreads, 0, stream>>>(basic ? basic : noisy, vol, refs, limit, st.group, pos, count);
    NSS_CUDA_CHECK_LAUNCH();
    const std::size_t n = static_cast<std::size_t>(vol.t) * vol.h * vol.w;
    NSS_CUDA_CHECK(cudaMemsetAsync(num, 0, n * sizeof(long long), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(den, 0, n * sizeof(long long), stream));
    const std::size_t smem = 2 * kMaxGroup * L * L * L * sizeof(float);
    if (basic) {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(group_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        group_kernel<L, true><<<R, kGroupThreads, smem, stream>>>(noisy, basic, vol, pos, count, sigma, D, kaiser, num, den);
    } else {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(group_kernel<L, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        group_kernel<L, false><<<R, kGroupThreads, smem, stream>>>(noisy, nullptr, vol, pos, count, sigma, D, kaiser, num, den);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace

std::size_t bm4d_volume_work_bytes(int t, int h, int w) { return layout(t, h, w).total; }

// D4 (16), D5 (25), kaiser4 (4), kaiser5 (5), as the CPU builds them
std::size_t bm4d_table_floats() { return 16 + 25 + 4 + 5; }

void bm4d_fill_tables(float* host) {
    const double pi = 3.14159265358979323846;
    float* d4 = host;
    float* d5 = host + 16;
    float* k4 = host + 41;
    float* k5 = host + 45;
    auto i0 = [](double x) {
        double sum = 1, term = 1;
        for (int k = 1; k < 64; ++k) {
            term *= (x / (2 * k)) * (x / (2 * k));
            sum += term;
        }
        return sum;
    };
    for (int s = 0; s < 2; ++s) {
        const int L = s ? 5 : 4;
        float* D = s ? d5 : d4;
        float* K = s ? k5 : k4;
        for (int k = 0; k < L; ++k)
            for (int i = 0; i < L; ++i) {
                double v = std::cos(pi * (2 * i + 1) * k / (2.0 * L)) * std::sqrt(2.0 / L);
                if (k == 0) v /= std::sqrt(2.0);
                D[k * L + i] = static_cast<float>(v);
            }
        for (int n = 0; n < L; ++n) {
            const double r = 2.0 * n / (L - 1) - 1.0;
            K[n] = static_cast<float>(i0(nss::kBm4dKaiserBeta * std::sqrt(1 - r * r)) / i0(nss::kBm4dKaiserBeta));
        }
    }
}

void bm4d_run_volume(const float* noisy, int t, int h, int w, float sigma, float* out, const float* tables, void* work,
                     std::size_t work_bytes, cudaStream_t stream) {
    const Layout l = layout(t, h, w);
    if (work_bytes < l.total) throw std::invalid_argument("nss_cuda: BM4D work buffer too small");
    auto* base = static_cast<std::uint8_t*>(work);
    auto* basic = reinterpret_cast<float*>(base);
    auto* num = reinterpret_cast<unsigned long long*>(base + l.basic);
    auto* den = reinterpret_cast<unsigned long long*>(base + l.basic + l.num);
    auto* pos = reinterpret_cast<int*>(base + l.basic + l.num + l.den);
    auto* count = reinterpret_cast<int*>(base + l.basic + l.num + l.den + l.pos);
    const Volume vol{t, h, w};
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    auto fits = [&](const nss::Bm4dStage& st) { return t >= st.cube && h >= st.cube && w >= st.cube; };
    if (fits(nss::kBm4dHard)) {
        run_stage<4>(noisy, nullptr, vol, sigma, nss::kBm4dHard, tables, tables + 41, pos, count, num, den, stream);
        ratio_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const long long*>(num),
                                                 reinterpret_cast<const long long*>(den), basic, n);
        NSS_CUDA_CHECK_LAUNCH();
    } else {
        NSS_CUDA_CHECK(cudaMemcpyAsync(basic, noisy, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    if (fits(nss::kBm4dWiener)) {
        run_stage<5>(noisy, basic, vol, sigma, nss::kBm4dWiener, tables + 16, tables + 45, pos, count, num, den, stream);
        ratio_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const long long*>(num),
                                                 reinterpret_cast<const long long*>(den), out, n);
        NSS_CUDA_CHECK_LAUNCH();
    } else {
        NSS_CUDA_CHECK(cudaMemcpyAsync(out, basic, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

void bm4d_gather(const float* const* frames, std::size_t pitch, int t, int y0, int x0, int h, int w, float* dst,
                 cudaStream_t stream) {
    gather_kernel<<<dim3((w + 127) / 128, h, t), 128, 0, stream>>>(frames, pitch, t, y0, x0, h, w, dst);
    NSS_CUDA_CHECK_LAUNCH();
}

void bm4d_scatter(const float* src, int h, int w, int y0, int x0, int z0, int count, int ky0, int ky1, int kx0, int kx1,
                  float* const* frames, std::size_t pitch, cudaStream_t stream) {
    const int kw = kx1 - kx0;
    scatter_kernel<<<dim3((kw + 127) / 128, ky1 - ky0, count), 128, 0, stream>>>(src, h, w, y0, x0, z0, ky0, kx0, kw,
                                                                                frames, pitch);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
