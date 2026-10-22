// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.BM4D: the CPU model (include/nss/params/bm4d.hpp, src/cpu/bm4d).
// Matching: on sm_80+ the distances are FP16 tensor-core products
// (tensor_match_kernel: not the CPU's distances, so not its matches near
// ties; the output's PSNR is the CPU's); otherwise, and for the references
// it leaves, match_kernel: one block per reference cube, every candidate's
// distance summed in the CPU's order (along z by fused multiply-adds, then y,
// then x), the closest taken by a block radix select of (distance, offset in
// scan order), the CPU's insertion order. Groups:
// one block per reference, Haar along the group then the 3D DCT of every
// cube with the CPU's arithmetic, then the fixed-point sums by 64-bit integer
// atomics (exact in any order). Built with --fmad=false.
#include "cuda/bm4d/kernels.hpp"
#include "cuda/runtime/error.hpp"
#include "nss/params/bm4d.hpp"


#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nss_cuda {
namespace {

constexpr int kSearch = nss::kBm4dSearch;
constexpr int kSide = 2 * kSearch + 1;
constexpr int kMatchThreads = 128;  // 256 measured the same
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

// Whether any cube of a reference's window (cubes within kSearch) has a voxel
// in keep: one that has none ends in a group that the Wiener stage skips.
__device__ __forceinline__ bool window_reaches(const Bm4dKeep& keep, int L, int z0, int y0, int x0) {
    const int span = 2 * kSearch + L;  // the window's cubes cover [o - kSearch, o + kSearch + L)
    return z0 - kSearch < keep.z1 && z0 - kSearch + span > keep.z0 && y0 - kSearch < keep.y1 &&
           y0 - kSearch + span > keep.y0 && x0 - kSearch < keep.x1 && x0 - kSearch + span > keep.x0;
}

// One block a reference. The search region ((L + 14)^3 voxels around the
// reference cube) is staged in shared memory; every offset's distance is
// summed in the CPU's order (along z by fused multiply-adds, then y, then x);
// the closest are the group - 1 smallest keys (distance bits, offset in scan
// order), the CPU's insertion order, found by a radix select (8 bits a round
// over the 44-bit keys, until the bin is small) and ranked among themselves.
template <int L>
__global__ void __launch_bounds__(kMatchThreads) match_kernel(const float* V, Volume vol, Refs refs, float limit, int group,
                                                              int* pos, int* count, bool fallback, Bm4dKeep keep) {
    constexpr int R = L + 2 * kSearch;  // region edge
    __shared__ float region[R * R * R];
    __shared__ unsigned hist[256];
    __shared__ unsigned long long chosen[2 * kMaxGroup];
    __shared__ unsigned long long prefix;
    __shared__ int need, taken, bin_size;
    const int r = blockIdx.x;
    if (fallback && count[r] >= 0) return;  // matched by tensor_match_kernel
    const int iz = r / (refs.ny * refs.nx), iy = (r / refs.nx) % refs.ny, ix = r % refs.nx;
    const int z0 = position(vol.t, L, refs.step, iz), y0 = position(vol.h, L, refs.step, iy),
              x0 = position(vol.w, L, refs.step, ix);
    if (!window_reaches(keep, L, z0, y0, x0)) {  // its group is skipped: the reference alone
        if (threadIdx.x == 0) {
            pos[static_cast<std::size_t>(r) * kMaxGroup] = static_cast<int>(vol.at(z0, y0, x0));
            count[r] = 1;
        }
        return;
    }
    {
        // (z, y, x) of element e advanced by counters (no division a voxel)
        constexpr int sx = kMatchThreads % R, sy = (kMatchThreads / R) % R, sz = kMatchThreads / (R * R);
        int ex = threadIdx.x % R, ey = (threadIdx.x / R) % R, ez = threadIdx.x / (R * R);
#pragma unroll 4
        for (int e = threadIdx.x; e < R * R * R; e += kMatchThreads) {
            const int z = z0 - kSearch + ez, y = y0 - kSearch + ey, x = x0 - kSearch + ex;
            region[e] = (z >= 0 && z < vol.t && y >= 0 && y < vol.h && x >= 0 && x < vol.w) ? V[vol.at(z, y, x)] : 0.f;
            ex += sx;
            ey += sy;
            ez += sz;
            if (ex >= R) {
                ex -= R;
                ++ey;
            }
            if (ey >= R) {
                ey -= R;
                ++ez;
            }
        }
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
            auto row_j = [&](int j) {
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
            };
            if constexpr (L == 5) {  // 5 x 5 rows unrolled is 13k instructions (instruction-fetch stalls)
#pragma unroll 1
                for (int j = 0; j < L; ++j) row_j(j);
            } else {
#pragma unroll
                for (int j = 0; j < L; ++j) row_j(j);
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
    // The keys kept are those whose top bits (mask) are at most prefix: every
    // valid key when they all fit; else a radix select (44-bit keys, 8 bits a
    // round from bit 36; warp 0 locates the bin by a prefix sum) narrows the
    // k_want-th smallest key to a bin and stops once the bin holds at most
    // kMaxGroup keys: then the keys at or below it are fewer than 2 kMaxGroup
    // and are ranked among themselves.
    unsigned long long mask = 0, top = 0;
    if (k_want > 0 && need > k_want) {
        int rank = k_want;  // 1-based rank among keys with the current prefix
        int shift = 36;
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
                            bin_size = static_cast<int>(local[b]);
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
            shift = max(shift - 8, 0);  // the last round (bits 0-7) overlaps the one before; its bits 4-7 are fixed
            if (bin_size <= kMaxGroup) break;
        }
        top = prefix;
    } else {
        mask = 0;  // every valid key
        top = 0;
    }
    if (k_want > 0) {
#pragma unroll
        for (int it = 0; it < kItems; ++it)
#pragma unroll
            for (int u = 0; u < kRun; ++u) {
                const unsigned long long key = keys[it][u];
                if (key != kNoKey && (key & mask) <= top) chosen[atomicAdd(&taken, 1)] = key;
            }
        __syncthreads();
    }
    int* p = pos + static_cast<std::size_t>(r) * kMaxGroup;
    if (threadIdx.x < taken) {  // the k_want smallest, ranked among themselves
        const unsigned long long key = chosen[threadIdx.x];
        int rank = 0;
        for (int i = 0; i < taken; ++i) rank += chosen[i] < key;
        if (rank < k_want) {
            const int o = static_cast<int>(key & 0xfffu);
            const int dz = o / (kSide * kSide) - kSearch, dy = (o / kSide) % kSide - kSearch, dx = o % kSide - kSearch;
            p[1 + rank] = static_cast<int>(vol.at(z0 + dz, y0 + dy, x0 + dx));
        }
    }
    if (threadIdx.x == 0) {
        p[0] = static_cast<int>(vol.at(z0, y0, x0));
        count[r] = 1 + k_want;
    }
}

// ---------------------------------------------------------------------------
// Tensor-core matching (sm_80+). The squared distance of the reference r and
// a candidate c is taken as d' = |r'|^2 + |c'|^2 - 2 r'.c' over FP16 values
// v' = fp16(v - mu), mu the volume's mean (FP32 accumulation), and the K =
// group - 1 smallest d' within the limit are the matches, ties to the lower
// offset in scan order. Not match_kernel's exact distances: the owner
// accepted it on 2026-10-22 (1080p gray 1.16x; PSNR to the clean source
// -0.0004 dB, mean difference 0.05 level). The K-th d' is found by a
// histogram of the d' within the limit (64 bins an octave), the candidates up
// to its bin's upper edge are kept and ranked. A reference with more than
// kTensorList of them, or every reference of a volume that FP16 cannot hold,
// is left to match_kernel (count -1), as on GPUs without sm_80 code.
// (The exact variant with rounding bounds and an exact re-rank of the
// survivors is in the history: docs/perf-da3d-bm4d-r3-20261022.md §1.)
//
// The dot products of a reference are one product of matrices per (k, j)
// row of the cube: rows (dz, dy) of the window (225, in 15 tiles of 16), the
// region row's 16 values from column 8 h as K, and dx = 8 h + n as N, with
// the reference row placed at i' - dx (a band; the same B fragment for both
// halves h of dx).
[[maybe_unused]] constexpr int kTensorPitch = 24;   // FP16 region row: 16-byte aligned rows for ldmatrix
[[maybe_unused]] constexpr int kTensorList = 512;   // candidates a reference ranks
constexpr int kTensorThreads = 128;

constexpr int kTensorPairPad = 16;  // pair columns before and after each row
struct TensorVolume {
    const unsigned* pairs;       // (fp16(v(x) - mu), fp16(v(x + 1) - mu)), rows of w + 2 pad, x at pad + x; 0 outside
    const float* norm;           // |c'|^2 of the cube at each voxel (positions that fit a cube)
    const unsigned* header;      // [0] FP16 cannot hold the volume, [1] mu (float bits)
};

// The histogram of d': 64 bins an octave of float bits below the limit's (8
// octaves; lower values in bin 0).
[[maybe_unused]] constexpr int kTensorBins = 512;
[[maybe_unused]] __device__ __forceinline__ int tensor_bin(float hi, unsigned top) {
    const int b = static_cast<int>(__float_as_uint(hi) >> 17) - static_cast<int>(top) + kTensorBins - 1;
    return max(b, 0);
}

template <int L>
__global__ void __launch_bounds__(kTensorThreads) tensor_match_kernel(const float* V, Volume vol, Refs refs, float limit,
                                                                     int group, int* pos, int* count, TensorVolume tv,
                                                                     Bm4dKeep keep) {
#if __CUDA_ARCH__ >= 800
    constexpr int R = L + 2 * kSearch;
    constexpr int P = kTensorPitch;
    constexpr int kChunks = L * L;              // (k, j) rows of the cube
    constexpr int kRows = kSide * kSide;        // (dz, dy) of the window
    constexpr int kTiles = (kRows + 15) / 16;   // 15
    constexpr int kTilesPerWarp = (kTiles + 3) / 4;
    constexpr int kSlots = kRows * 16;          // candidates (dzi, dyi) x 16 dxi
    static_assert(R <= P - 4 && 8 + 16 <= P, "region row");
    // a region row reads the pairs from x0 - kSearch to x0 - kSearch + P - 1
    // (x0 <= w - L): all within the row's padding
    static_assert(kTensorPairPad >= kSearch && kTensorPairPad >= P - L - kSearch, "pair padding");
    // the FP16 region; after the products the distances; then the kept keys
    constexpr int kRegionBytes = R * R * P * 2;
    static_assert(kRegionBytes >= kSlots * 4 && kRegionBytes >= kTensorList * 8, "aliases of the region");
    __shared__ __align__(16) unsigned char buf[kRegionBytes];
    unsigned short* region = reinterpret_cast<unsigned short*>(buf);
    float* dist = reinterpret_cast<float*>(buf);
    unsigned long long* keys = reinterpret_cast<unsigned long long*>(buf);
    // the histogram; then the kept candidates
    __shared__ unsigned hist[kTensorBins];
    unsigned short* surv = reinterpret_cast<unsigned short*>(hist);
    static_assert(kTensorBins * 4 >= kTensorList * 2, "kept candidates in the histogram");
    __shared__ int nsurv, nvalid;
    __shared__ float ubound;
    const int r = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tv.header[0]) {
        if (tid == 0) count[r] = -1;
        return;
    }
    const int iz = r / (refs.ny * refs.nx), iy = (r / refs.nx) % refs.ny, ix = r % refs.nx;
    const int z0 = position(vol.t, L, refs.step, iz), y0 = position(vol.h, L, refs.step, iy),
              x0 = position(vol.w, L, refs.step, ix);
    if (!window_reaches(keep, L, z0, y0, x0)) {  // its group is skipped: the reference alone
        if (tid == 0) {
            pos[static_cast<std::size_t>(r) * kMaxGroup] = static_cast<int>(vol.at(z0, y0, x0));
            count[r] = 1;
        }
        return;
    }
    {
        // the FP16 region by asynchronous 4-byte copies of value pairs (0
        // outside the volume), (zy row, pair) by counters
        constexpr int kPairs = P / 2;
        constexpr int sc = kTensorThreads % kPairs, sr = kTensorThreads / kPairs;
        int c = tid % kPairs, row = tid / kPairs;
        const int pw = vol.w + kTensorPairPad * 2;
        const unsigned dst0 = static_cast<unsigned>(__cvta_generic_to_shared(region));
        for (int e = tid; e < R * R * kPairs; e += kTensorThreads) {
            const int vz = z0 - kSearch + row / R, vy = y0 - kSearch + row % R;
            const bool in = vz >= 0 && vz < vol.t && vy >= 0 && vy < vol.h;
            const unsigned* src = tv.pairs + (in ? (static_cast<std::size_t>(vz) * vol.h + vy) * pw + kTensorPairPad + x0 - kSearch + 2 * c : 0);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst0 + 4u * static_cast<unsigned>(e)), "l"(src),
                         "r"(in ? 4 : 0));
            c += sc;
            row += sr;
            if (c >= kPairs) {
                c -= kPairs;
                ++row;
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int b = tid; b < kTensorBins; b += kTensorThreads) hist[b] = 0;
    if (tid == 0) {
        nsurv = 0;
        nvalid = 0;
    }
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
    // the dot products: warp w takes the tiles w, w + 4, ...
    float acc[kTilesPerWarp][2][4];
    int rowbase[kTilesPerWarp];
    const int row_in_tile = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int tt = 0; tt < kTilesPerWarp; ++tt) {
        const int m = min((warp + 4 * tt) * 16 + row_in_tile, kRows - 1);
        rowbase[tt] = ((m / kSide) * R + m % kSide) * P;
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[tt][h][e] = 0.f;
    }
    const unsigned rbase = static_cast<unsigned>(__cvta_generic_to_shared(region));
    const int colsel = (lane >> 4) * 8;
#pragma unroll 1
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        const int off = ((chunk / L) * R + chunk % L) * P;
        // this lane's B fragment of the reference row (k, j): element (kk, n) is r'[k][j][kk - n]
        uint2 b;
        {
            const unsigned short* rrow = region + ((kSearch + chunk / L) * R + kSearch + chunk % L) * P + kSearch;
            const int t = lane & 3, n = lane >> 2;
            auto v = [&](int kk) -> unsigned { const int ii = kk - n; return ii >= 0 && ii < L ? rrow[ii] : 0u; };
            b = make_uint2(v(2 * t) | v(2 * t + 1) << 16, v(2 * t + 8) | v(2 * t + 9) << 16);
        }
#pragma unroll
        for (int tt = 0; tt < kTilesPerWarp; ++tt) {
            if (warp + 4 * tt >= kTiles) continue;
            unsigned a0, a1, a2, a3, a4, a5;
            const unsigned at = rbase + 2u * static_cast<unsigned>(rowbase[tt] + off);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(at + 2u * colsel));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(a4), "=r"(a5) : "r"(at + 32u));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[tt][0][0]), "+f"(acc[tt][0][1]), "+f"(acc[tt][0][2]), "+f"(acc[tt][0][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[tt][1][0]), "+f"(acc[tt][1][1]), "+f"(acc[tt][1][2]), "+f"(acc[tt][1][3])
                         : "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(b.x), "r"(b.y));
        }
    }
    __syncthreads();  // the region is read: its buffer takes the products
#pragma unroll
    for (int tt = 0; tt < kTilesPerWarp; ++tt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = (warp + 4 * tt) * 16 + (lane >> 2) + (e >= 2 ? 8 : 0);
                if (warp + 4 * tt < kTiles && row < kRows) dist[row * 16 + 8 * h + 2 * (lane & 3) + (e & 1)] = acc[tt][h][e];
            }
    __syncthreads();
    // d' of every candidate (neighbouring threads, neighbouring dx; inf
    // where no cube fits), those within the limit to the histogram
    const std::size_t ref_at = vol.at(z0, y0, x0);
    const unsigned top = __float_as_uint(limit) >> 17;
    {
        // thread (dxi, first row): rows first, first + 8, ... by counters
        const float nr = tv.norm[ref_at];
        constexpr int kStep = kTensorThreads / 16;
        static_assert(kStep < kSide, "one wrap a step");
        const int dxi = tid & 15, cx = x0 + dxi - kSearch;
        const bool x_ok = dxi < kSide && cx >= 0 && cx <= vol.w - L;
        int row = tid >> 4, dzi = row / kSide, dyi = row % kSide;
        constexpr int kAhead = 4;  // rows whose loads are in flight together
        for (; row < kRows; row += kAhead * kStep) {
            float nc[kAhead];
            int at_e[kAhead];
            bool ok[kAhead];
#pragma unroll
            for (int u = 0; u < kAhead; ++u) {
                const int rr = row + u * kStep;
                const int cz = z0 + dzi - kSearch, cy = y0 + dyi - kSearch;
                ok[u] = rr < kRows && x_ok && cz >= 0 && cz <= vol.t - L && cy >= 0 && cy <= vol.h - L &&
                        !(dzi == kSearch && dyi == kSearch && dxi == kSearch);
                at_e[u] = rr < kRows ? rr * 16 + dxi : -1;
                nc[u] = ok[u] ? __ldg(tv.norm + vol.at(cz, cy, cx)) : 0.f;
                dyi += kStep;
                if (dyi >= kSide) {
                    dyi -= kSide;
                    ++dzi;
                }
            }
#pragma unroll
            for (int u = 0; u < kAhead; ++u) {
                if (at_e[u] < 0) continue;
                const int e = at_e[u];
                float d = INFINITY;
                if (ok[u]) {
                    d = fmaxf(nr + nc[u] - 2.f * dist[e], 0.f);  // NaN (never: the header checks the volume) -> 0
                    if (d <= limit) atomicAdd(&hist[tensor_bin(d, top)], 1u);
                }
                dist[e] = d;
            }
        }
    }
    __syncthreads();
    // U: the upper edge of the bin holding the K-th d', else the limit
    if (warp == 0) {
        const int K = group - 1;
        constexpr int kPer = kTensorBins / 32;
        unsigned local[kPer], sum = 0;
#pragma unroll
        for (int b = 0; b < kPer; ++b) {
            local[b] = hist[lane * kPer + b];
            sum += local[b];
        }
        unsigned inclusive = sum;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inclusive, d);
            if (lane >= d) inclusive += v;
        }
        const unsigned before = inclusive - sum;
        const unsigned hit =
            __ballot_sync(0xffffffffu, K > 0 && before < static_cast<unsigned>(K) && static_cast<unsigned>(K) <= inclusive);
        float u = limit;
        if (hit) {
            const int owner = __ffs(hit) - 1;
            unsigned acc2 = __shfl_sync(0xffffffffu, before, owner);
            int bin = owner * kPer + kPer - 1;
            for (int b = 0; b < kPer; ++b) {
                const unsigned v = __shfl_sync(0xffffffffu, local[b], owner);
                if (acc2 + v >= static_cast<unsigned>(K)) {
                    bin = owner * kPer + b;
                    break;
                }
                acc2 += v;
            }
            // bin b holds float bits [(top + b - 511) << 17, (top + b - 510) << 17)
            u = fminf(limit, __uint_as_float((top + static_cast<unsigned>(bin) + 1u - (kTensorBins - 1)) << 17));
        }
        if (lane == 0) ubound = u;
    }
    __syncthreads();  // the histogram is read: its buffer takes the kept candidates
    const float U = ubound;
    for (int e = tid; e < kSlots; e += kTensorThreads) {
        if (!(dist[e] <= U)) continue;
        const int at = atomicAdd(&nsurv, 1);
        if (at < kTensorList) surv[at] = static_cast<unsigned short>((e >> 4) * kSide + (e & 15));  // (dzi * 15 + dyi) * 15 + dxi
    }
    __syncthreads();
    const int ns = nsurv;
    if (ns > kTensorList) {
        if (tid == 0) count[r] = -1;
        return;
    }
    // their keys (d' bits, offset in scan order): the distances are read
    // before the keys take their buffer
    {
        constexpr int kPerThread = (kTensorList + kTensorThreads - 1) / kTensorThreads;
        float dv[kPerThread];
        int ov[kPerThread];
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int i = tid + j * kTensorThreads;
            ov[j] = i < ns ? surv[i] : -1;
            dv[j] = ov[j] >= 0 ? dist[(ov[j] / kSide) * 16 + ov[j] % kSide] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kPerThread; ++j)
            if (ov[j] >= 0)
                keys[atomicAdd(&nvalid, 1)] = static_cast<unsigned long long>(__float_as_uint(dv[j])) << 12 | static_cast<unsigned>(ov[j]);
    }
    __syncthreads();
    const int nv = nvalid;
    const int k_want = min(group - 1, nv);
    int* p = pos + static_cast<std::size_t>(r) * kMaxGroup;
    for (int i = tid; i < nv; i += kTensorThreads) {
        const unsigned long long key = keys[i];
        int rank = 0;
        for (int q = 0; q < nv && rank < k_want; ++q) rank += keys[q] < key;
        if (rank < k_want) {
            const int o = static_cast<int>(key & 0xfffu);
            const int dz = o / (kSide * kSide) - kSearch, dy = (o / kSide) % kSide - kSearch, dx = o % kSide - kSearch;
            p[1 + rank] = static_cast<int>(vol.at(z0 + dz, y0 + dy, x0 + dx));
        }
    }
    if (tid == 0) {
        p[0] = static_cast<int>(ref_at);
        count[r] = 1 + k_want;
    }
#else
    __trap();
#endif
}

// The volume's mean: a double per (z, y) row in a fixed order, then the rows.
constexpr int kMeanThreads = 256;
__device__ __forceinline__ double block_sum(double v) {
    __shared__ double part[kMeanThreads / 32];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    double total = 0;
    if (threadIdx.x == 0)
        for (int w = 0; w < kMeanThreads / 32; ++w) total += part[w];
    return total;
}
__global__ void __launch_bounds__(kMeanThreads) row_sum_kernel(const float* V, int w, double* rows) {
    const float* row = V + static_cast<std::size_t>(blockIdx.x) * w;
    double v = 0;
    for (int x = threadIdx.x; x < w; x += kMeanThreads) v += row[x];
    const double total = block_sum(v);
    if (threadIdx.x == 0) rows[blockIdx.x] = total;
}
__global__ void __launch_bounds__(kMeanThreads) mean_kernel(const double* rows, int n_rows, double voxels, unsigned* header) {
    double v = 0;
    for (int y = threadIdx.x; y < n_rows; y += kMeanThreads) v += rows[y];
    const double total = block_sum(v);
    if (threadIdx.x == 0) {
        header[0] = 0;
        header[1] = __float_as_uint(static_cast<float>(total / voxels));
    }
}
__device__ __forceinline__ unsigned short tensor_half(float v, float mu) { return __half_as_ushort(__float2half_rn(v - mu)); }
__global__ void pairs_kernel(const float* V, Volume vol, unsigned* pairs, unsigned* header) {
    const int pw = vol.w + 2 * kTensorPairPad;
    const int xx = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (xx >= pw) return;
    const float mu = __uint_as_float(header[1]);
    const int x = xx - kTensorPairPad;
    const float* row = V + vol.at(z, y, 0);
    const bool in0 = x >= 0 && x < vol.w, in1 = x + 1 >= 0 && x + 1 < vol.w;
    if (in0 && !(fabsf(row[x] - mu) <= 60000.f)) atomicOr(header, 1u);  // NaN too
    const unsigned lo = in0 ? tensor_half(row[x], mu) : 0u, hi = in1 ? tensor_half(row[x + 1], mu) : 0u;
    pairs[(static_cast<std::size_t>(z) * vol.h + y) * pw + xx] = lo | hi << 16;
}
// |c'|^2 of the cube at every position that fits one
template <int L>
__global__ void norms_kernel(const float* V, Volume vol, const unsigned* header, float* norm) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x > vol.w - L || y > vol.h - L || z > vol.t - L) return;
    const float mu = __uint_as_float(header[1]);
    float n = 0.f;
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j)
#pragma unroll
            for (int i = 0; i < L; ++i) {
                const float h = __half2float(__ushort_as_half(tensor_half(V[vol.at(z + k, y + j, x + i)], mu)));
                n = fmaf(h, h, n);
            }
    norm[vol.at(z, y, x)] = n;
}

__device__ __forceinline__ long long fixed(float x) { return __float2ll_rn(x * nss::kBm4dFixedScale); }

// Haar along the group (m a power of two) of n-element vectors, as the CPU's:
// a thread holds element e of all m vectors in registers and takes every
// level there (no barrier between levels). With b, threads N to 2N - 1 take
// b's elements at the same time.
template <int M, bool Inverse>
__device__ __forceinline__ void haar_line(float* v) {
    const float c = 0.70710678118654752f;
    float t[M];
    if (!Inverse) {
#pragma unroll
        for (int len = M; len > 1; len /= 2) {
#pragma unroll
            for (int i = 0; i < len / 2; ++i) {
                const float a = v[2 * i], bb = v[2 * i + 1];
                t[i] = (a + bb) * c;
                t[len / 2 + i] = (a - bb) * c;
            }
#pragma unroll
            for (int i = 0; i < len; ++i) v[i] = t[i];
        }
    } else {
#pragma unroll
        for (int len = 2; len <= M; len *= 2) {
#pragma unroll
            for (int i = 0; i < len / 2; ++i) {
                const float a = v[i], bb = v[len / 2 + i];
                t[2 * i] = (a + bb) * c;
                t[2 * i + 1] = (a - bb) * c;
            }
#pragma unroll
            for (int i = 0; i < len; ++i) v[i] = t[i];
        }
    }
}
template <int N, int M, bool Inverse>
__device__ __forceinline__ void haar_column(float* g, int e) {
    float v[M];
#pragma unroll
    for (int q = 0; q < M; ++q) v[q] = g[q * N + e];
    haar_line<M, Inverse>(v);
#pragma unroll
    for (int q = 0; q < M; ++q) g[q * N + e] = v[q];
}
// T0: the first thread taking part (threads T0 and up)
template <int N, bool Inverse, int T0 = 0>
__device__ void haar(float* g, float* b, int m) {
    int e = static_cast<int>(threadIdx.x) - T0;
    float* a = g;
    if (e >= N) {
        if (!b || e >= 2 * N) return;
        e -= N;
        a = b;
    }
    switch (m) {
        case 2: haar_column<N, 2, Inverse>(a, e); break;
        case 4: haar_column<N, 4, Inverse>(a, e); break;
        case 8: haar_column<N, 8, Inverse>(a, e); break;
        case 16: haar_column<N, 16, Inverse>(a, e); break;
        case 32: haar_column<N, 32, Inverse>(a, e); break;
        default: break;  // m = 1
    }
}

// The DCT matrices D4 and D5 (the first 41 floats of bm4d_fill_tables) in
// constant memory: unrolled, every coefficient is an operand of its fused
// multiply-add instead of a load (global loads were 13% of the transforms'
// instructions).
__constant__ float c_dct[16 + 25];
__constant__ float c_kaiser[4 + 5];  // the Kaiser windows of both cube sizes

// The 3D DCT of m cubes (x, then y, then z), forward or inverse, in place:
// one thread a line; each output is D(k, 0) in(0) then fused terms in order,
// as the CPU's.
// Threads T0 and up (the others must not call it), apart by a named barrier
// when T0 > 0.
template <int T0>
__device__ __forceinline__ void part_sync() {
    if constexpr (T0 == 0) __syncthreads();
    else asm volatile("bar.sync 1, %0;" ::"n"(kGroupThreads - T0) : "memory");
}
template <int L, int T0 = 0>
__device__ void dct3(float* g, float* g2, int m, bool inverse) {
    constexpr int L2 = L * L, L3 = L2 * L;
    constexpr int kThreads = kGroupThreads - T0;
    const int lt = static_cast<int>(threadIdx.x) - T0;
    const float* D = c_dct + (L == 5 ? 16 : 0);
    const int cubes = (g2 ? 2 : 1) * m;  // g2: a second array of m cubes, or null
    auto coef = [&](int k, int i) { return inverse ? D[i * L + k] : D[k * L + i]; };
    int first_axis = 0;
    if constexpr (L == 5) {
        // x and y of a z slice on one thread, in registers (a slice's 25
        // words at a stride of 25 between threads: no bank conflict)
        for (int task = lt; task < cubes * L; task += kThreads) {
            const int q = task / L, z = task % L;
            float* const sl = (q < m ? g + q * L3 : g2 + (q - m) * L3) + z * L2;
            float v[L2];
#pragma unroll
            for (int i = 0; i < L2; ++i) v[i] = sl[i];
#pragma unroll
            for (int y = 0; y < L; ++y) {  // along x
                float out[L];
#pragma unroll
                for (int k = 0; k < L; ++k) {
                    float acc = coef(k, 0) * v[y * L];
#pragma unroll
                    for (int i = 1; i < L; ++i) acc = fmaf(coef(k, i), v[y * L + i], acc);
                    out[k] = acc;
                }
#pragma unroll
                for (int k = 0; k < L; ++k) v[y * L + k] = out[k];
            }
#pragma unroll
            for (int x = 0; x < L; ++x) {  // along y
                float out[L];
#pragma unroll
                for (int k = 0; k < L; ++k) {
                    float acc = coef(k, 0) * v[x];
#pragma unroll
                    for (int i = 1; i < L; ++i) acc = fmaf(coef(k, i), v[i * L + x], acc);
                    out[k] = acc;
                }
#pragma unroll
                for (int k = 0; k < L; ++k) v[k * L + x] = out[k];
            }
#pragma unroll
            for (int i = 0; i < L2; ++i) sl[i] = v[i];
        }
        part_sync<T0>();
        first_axis = 2;
    }
    for (int axis = first_axis; axis < 3; ++axis) {
        const int unit = axis == 0 ? 1 : axis == 1 ? L : L2;
        for (int ln = lt; ln < cubes * L2; ln += kThreads) {
            const int q = ln / L2, a = (ln / L) % L, b = ln % L;
            float* const arr = q < m ? g : g2;
            const int base = (q < m ? q : q - m) * L3 + (axis == 0 ? (a * L + b) * L : axis == 1 ? a * L2 + b : a * L + b);
            float in[L];
#pragma unroll
            for (int i = 0; i < L; ++i) in[i] = arr[base + i * unit];
#pragma unroll
            for (int k = 0; k < L; ++k) {
                float acc = coef(k, 0) * in[0];
#pragma unroll
                for (int i = 1; i < L; ++i) acc = fmaf(coef(k, i), in[i], acc);
                arr[base + k * unit] = acc;
            }
        }
        part_sync<T0>();
    }
}

template <int L, bool Wiener>
__global__ void __launch_bounds__(kGroupThreads) group_kernel(const float* noisy, const float* basic, Volume vol,
                                                              const int* pos, const int* count, float sigma,
                                                              Bm4dKeep keep, unsigned long long* num, unsigned long long* den) {
    constexpr int L3 = L * L * L;
    extern __shared__ float smem[];
    float* g = smem;
    float* b = g + kMaxGroup * L3;
    __shared__ float weight;
    __shared__ int kept_count;
    __shared__ int p[kMaxGroup];  // the cubes' first voxels
    const int r = blockIdx.x;
    int m = 1;
    while (2 * m <= count[r]) m *= 2;
    bool touches = false;  // the cube has a voxel in keep
    if (threadIdx.x < m) {
        const int at = pos[static_cast<std::size_t>(r) * kMaxGroup + threadIdx.x];
        p[threadIdx.x] = at;
        const int cx = at % vol.w, cy = (at / vol.w) % vol.h, cz = at / (vol.w * vol.h);
        touches = cz < keep.z1 && cz + L > keep.z0 && cy < keep.y1 && cy + L > keep.y0 && cx < keep.x1 && cx + L > keep.x0;
    }
    if (threadIdx.x == 0) kept_count = 0;
    // The Wiener stage's sums outside keep are never used: a group without a
    // cube there adds nothing that is kept (the hard stage's basic estimate is
    // needed everywhere)
    if (!__syncthreads_or(touches) && Wiener) return;
    const std::size_t plane = static_cast<std::size_t>(vol.h) * vol.w;
    for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
        const int q = it / L3, e = it % L3;
        const std::size_t o = p[q] + (e / (L * L)) * plane + static_cast<std::size_t>((e / L) % L) * vol.w + e % L;
#if __CUDA_ARCH__ >= 800
        // asynchronous copies: all of a thread's loads in flight at once
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(g + it))),
                     "l"(noisy + o));
        if (Wiener)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(b + it))),
                         "l"(basic + o));
#else
        g[it] = noisy[o];
        if (Wiener) b[it] = basic[o];
#endif
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
    __syncthreads();
    haar<L3, false>(g, Wiener ? b : nullptr, m);
    __syncthreads();
    dct3<L>(g, Wiener ? b : nullptr, m, false);
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
        for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
            const float b2 = b[it] * b[it];
            const float wv = b2 / (b2 + sigma2);
            g[it] = g[it] * wv;
            b[it] = wv;
        }
        __syncthreads();
        // warp 0: the energy (a serial chain of 125 m / 32 terms a lane);
        // warps 1 to 7 meanwhile take the inverse transforms of g
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
        } else {
            dct3<L, 32>(g, nullptr, m, true);
            haar<L3, true, 32>(g, nullptr, m);
        }
    }
    if (!Wiener) {
        __syncthreads();
        dct3<L>(g, nullptr, m, true);
        haar<L3, true>(g, nullptr, m);
    }
    __syncthreads();
    const float w = weight;
    for (int it = threadIdx.x; it < m * L3; it += kGroupThreads) {
        const int q = it / L3, e = it % L3;
        const int z = e / (L * L), y = (e / L) % L, x = e % L;
        const float* kaiser = c_kaiser + (L == 5 ? 4 : 0);
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
    std::size_t basic, num, den, pos, count, half, norm, err, rows, header, total;
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
    // tensor matching: the FP16 volume, cube norms and errors, the mean's row sums
    l.half = align256(static_cast<std::size_t>(t) * h * (w + 2 * kTensorPairPad) * sizeof(unsigned));
    l.norm = align256(n * sizeof(float));
    l.err = 0;
    l.rows = align256(static_cast<std::size_t>(t) * h * sizeof(double));
    l.header = 256;
    l.total = l.basic + l.num + l.den + l.pos + l.count + l.half + l.norm + l.err + l.rows + l.header;
    return l;
}

struct TensorWork {
    unsigned* pairs;
    float* norm;
    double* rows;
    unsigned* header;
};

// Whether the current device runs tensor_match_kernel's sm_80 code (the
// compute_75 PTX would trap; mma.m16n8k16 and cp.async need sm_80).
bool tensor_device() {
    cudaFuncAttributes a{};
    return cudaFuncGetAttributes(&a, tensor_match_kernel<5>) == cudaSuccess && a.ptxVersion >= 80 && a.binaryVersion >= 80;
}

template <int L>
void run_stage(const float* noisy, const float* basic, const Volume& vol, float sigma, const Bm4dKeep& keep, const nss::Bm4dStage& st,
               int* pos, int* count, unsigned long long* num,
               unsigned long long* den, const TensorWork& tw, cudaStream_t stream) {
    const Refs refs{position_count(vol.t, L, st.step), position_count(vol.h, L, st.step), position_count(vol.w, L, st.step),
                    L, st.step};
    const int R = refs.nz * refs.ny * refs.nx;
    const float sigma2 = sigma * sigma;
    const float limit = st.tau * static_cast<float>(L * L * L) * sigma2;
    const float* V = basic ? basic : noisy;
    // the hard stage's basic estimate is needed everywhere: every reference matches
    const Bm4dKeep reach = basic ? keep : Bm4dKeep{0, vol.t, 0, vol.h, 0, vol.w};
    const bool tensor = tensor_device();
    if (tensor) {
        const std::size_t n = static_cast<std::size_t>(vol.t) * vol.h * vol.w;
        row_sum_kernel<<<static_cast<unsigned>(vol.t * vol.h), kMeanThreads, 0, stream>>>(V, vol.w, tw.rows);
        NSS_CUDA_CHECK_LAUNCH();
        mean_kernel<<<1, kMeanThreads, 0, stream>>>(tw.rows, vol.t * vol.h, static_cast<double>(n), tw.header);
        NSS_CUDA_CHECK_LAUNCH();
        pairs_kernel<<<dim3((vol.w + 2 * kTensorPairPad + 127) / 128, vol.h, vol.t), 128, 0, stream>>>(V, vol, tw.pairs, tw.header);
        NSS_CUDA_CHECK_LAUNCH();
        norms_kernel<L><<<dim3((vol.w + 127) / 128, vol.h, vol.t), 128, 0, stream>>>(V, vol, tw.header, tw.norm);
        NSS_CUDA_CHECK_LAUNCH();
        tensor_match_kernel<L><<<R, kTensorThreads, 0, stream>>>(V, vol, refs, limit, st.group, pos, count,
                                                                  TensorVolume{tw.pairs, tw.norm, tw.header}, reach);
        NSS_CUDA_CHECK_LAUNCH();
    }
    match_kernel<L><<<R, kMatchThreads, 0, stream>>>(V, vol, refs, limit, st.group, pos, count, tensor, reach);
    NSS_CUDA_CHECK_LAUNCH();
    const std::size_t n = static_cast<std::size_t>(vol.t) * vol.h * vol.w;
    NSS_CUDA_CHECK(cudaMemsetAsync(num, 0, n * sizeof(long long), stream));
    NSS_CUDA_CHECK(cudaMemsetAsync(den, 0, n * sizeof(long long), stream));
    // the noisy group and, Wiener, the basic group; the hard stage's groups
    // have at most st.group cubes and no basic group (6 blocks an SM, not 5)
    const std::size_t smem = basic ? 2 * kMaxGroup * L * L * L * sizeof(float)
                                   : static_cast<std::size_t>(st.group) * L * L * L * sizeof(float);
    if (basic) {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(group_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        group_kernel<L, true><<<R, kGroupThreads, smem, stream>>>(noisy, basic, vol, pos, count, sigma, keep, num, den);
    } else {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(group_kernel<L, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        group_kernel<L, false><<<R, kGroupThreads, smem, stream>>>(noisy, nullptr, vol, pos, count, sigma, keep, num, den);
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

void bm4d_upload_constants(const float* host) {
    NSS_CUDA_CHECK(cudaMemcpyToSymbol(c_dct, host, sizeof(c_dct)));
    NSS_CUDA_CHECK(cudaMemcpyToSymbol(c_kaiser, host + 41, sizeof(c_kaiser)));
}

void bm4d_run_volume(const float* noisy, int t, int h, int w, float sigma, const Bm4dKeep& keep, float* out, void* work,
                     std::size_t work_bytes, cudaStream_t stream) {
    const Layout l = layout(t, h, w);
    if (work_bytes < l.total) throw std::invalid_argument("nss_cuda: BM4D work buffer too small");
    auto* base = static_cast<std::uint8_t*>(work);
    auto* basic = reinterpret_cast<float*>(base);
    auto* num = reinterpret_cast<unsigned long long*>(base + l.basic);
    auto* den = reinterpret_cast<unsigned long long*>(base + l.basic + l.num);
    auto* pos = reinterpret_cast<int*>(base + l.basic + l.num + l.den);
    auto* count = reinterpret_cast<int*>(base + l.basic + l.num + l.den + l.pos);
    std::uint8_t* tb = base + l.basic + l.num + l.den + l.pos + l.count;
    const TensorWork tw{reinterpret_cast<unsigned*>(tb), reinterpret_cast<float*>(tb + l.half),
                        reinterpret_cast<double*>(tb + l.half + l.norm + l.err),
                        reinterpret_cast<unsigned*>(tb + l.half + l.norm + l.err + l.rows)};
    const Volume vol{t, h, w};
    const std::size_t n = static_cast<std::size_t>(t) * h * w;
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    auto fits = [&](const nss::Bm4dStage& st) { return t >= st.cube && h >= st.cube && w >= st.cube; };
    if (fits(nss::kBm4dHard)) {
        run_stage<4>(noisy, nullptr, vol, sigma, keep, nss::kBm4dHard, pos, count, num, den, tw, stream);
        ratio_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const long long*>(num),
                                                 reinterpret_cast<const long long*>(den), basic, n);
        NSS_CUDA_CHECK_LAUNCH();
    } else {
        NSS_CUDA_CHECK(cudaMemcpyAsync(basic, noisy, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    if (fits(nss::kBm4dWiener)) {
        run_stage<5>(noisy, basic, vol, sigma, keep, nss::kBm4dWiener, pos, count, num, den, tw, stream);
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
