// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/bm3d/kernels.hpp"
#include "cuda/runtime/error.hpp"

#include <cmath>
#include <mutex>
#include <set>
#include <vector>

namespace nss_cuda {
namespace {

// Orthonormal DCT-II matrices T_n[k][j] for every block and group length.
// Threads of a warp read the same entry at the same time (broadcast).
constexpr int kTableSizes[] = {1, 2, 4, 8, 12, 16, 32, 64};
constexpr int kTableFloats = 1 + 4 + 16 + 64 + 144 + 256 + 1024 + 4096;
__constant__ float c_dct[kTableFloats];

// Running sum of n^2 over kTableSizes (spelled out: device code cannot read
// the host array).
__host__ __device__ constexpr int table_offset(int n) {
    switch (n) {
    case 1: return 0;
    case 2: return 1;
    case 4: return 5;
    case 8: return 21;
    case 12: return 85;
    case 16: return 229;
    case 32: return 485;
    case 64: return 1509;
    default: return -1;
    }
}
static_assert(table_offset(64) + 64 * 64 == kTableFloats, "DCT table layout");

constexpr float kHardLambda = 2.7f;  // nss::kBmHardLambda
constexpr int kThreads = 128;

// In-place transform of one line of N samples (stride apart).
// Short lines unroll fully; 32/64 (large blocks, long groups) stay compact.
template <int N>
__device__ __forceinline__ void dct_line(float* base, int stride, bool inverse) {
    constexpr int kOffset = table_offset(N);
    constexpr int kUnroll = N <= 16 ? N : 4;
    const float* t = c_dct + kOffset;
    float in[N];
#pragma unroll
    for (int j = 0; j < N; ++j) in[j] = base[j * stride];
#pragma unroll kUnroll
    for (int k = 0; k < N; ++k) {
        float acc = 0.f;
#pragma unroll kUnroll
        for (int j = 0; j < N; ++j) acc = fmaf(inverse ? t[j * N + k] : t[k * N + j], in[j], acc);
        base[k * stride] = acc;
    }
}

__device__ void dct_line_n(int n, float* base, int stride, bool inverse) {
    switch (n) {
    case 1: return;
    case 2: dct_line<2>(base, stride, inverse); return;
    case 4: dct_line<4>(base, stride, inverse); return;
    case 8: dct_line<8>(base, stride, inverse); return;
    case 12: dct_line<12>(base, stride, inverse); return;
    case 16: dct_line<16>(base, stride, inverse); return;
    case 32: dct_line<32>(base, stride, inverse); return;
    default: dct_line<64>(base, stride, inverse); return;
    }
}

// 2D transform of `group` patches (rows then columns), cube = [group][block][block].
__device__ void dct2d(float* cube, int block, int group, bool inverse) {
    const int lines = group * block;
    for (int i = threadIdx.x; i < lines; i += blockDim.x) dct_line_n(block, cube + i * block, 1, inverse);
    __syncthreads();
    for (int i = threadIdx.x; i < lines; i += blockDim.x) {
        const int g = i / block, c = i % block;
        dct_line_n(block, cube + g * block * block + c, block, inverse);
    }
    __syncthreads();
}

__device__ void dct_group(float* cube, int area, int group, bool inverse) {
    for (int f = threadIdx.x; f < area; f += blockDim.x) dct_line_n(group, cube + f, area, inverse);
    __syncthreads();
}

// Fixed-shape tree reduction: deterministic for a fixed blockDim.
template <class T>
__device__ T block_sum(T value, T* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) scratch[threadIdx.x] += scratch[threadIdx.x + half];
        __syncthreads();
    }
    const T total = scratch[0];
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(kThreads) group_filter_kernel(Bm3dGroupArgs a) {
    __shared__ float fsum[kThreads];
    __shared__ int isum[kThreads];
    const int r = blockIdx.x;
    const int area = a.block * a.block;
    const int cube_n = a.group * area;
    float* cube = a.values + static_cast<long long>(r) * cube_n;
    float* refc = a.ref ? a.ref_cube + static_cast<long long>(r) * cube_n : nullptr;
    const DeviceMatch* m = a.matches + static_cast<long long>(r) * a.group;
    const int kk = min(a.counts[r], a.group);

    for (int i = threadIdx.x; i < cube_n; i += blockDim.x) {
        const int g = i / area, p = i % area;
        float v = 0.f, rv = 0.f;
        if (g < kk) {
            const long long offset = static_cast<long long>(m[g].y + p / a.block) * a.pitch + m[g].x + p % a.block;
            v = a.src[m[g].t][offset];
            if (refc) rv = a.ref[m[g].t][offset];
        }
        cube[i] = v;
        if (refc) refc[i] = rv;
    }
    __syncthreads();
    dct2d(cube, a.block, a.group, false);
    if (refc) dct2d(refc, a.block, a.group, false);
    dct_group(cube, area, a.group, false);
    if (refc) dct_group(refc, area, a.group, false);

    float weight;
    if (refc) {
        const float sig2 = a.sigma * a.sigma;
        float w2 = 0.f;
        for (int i = threadIdx.x; i < cube_n; i += blockDim.x) {
            float w = 1.f;
            if (i != 0) {
                const float q = refc[i] * refc[i];
                w = q / (q + sig2);
            }
            cube[i] *= w;
            w2 = fmaf(w, w, w2);
        }
        weight = 1.f / fmaxf(block_sum(w2, fsum), 1e-12f);
    } else {
        const float thr = kHardLambda * a.sigma;
        int kept = 0;
        for (int i = threadIdx.x; i < cube_n; i += blockDim.x) {
            if (i != 0 && fabsf(cube[i]) < thr) {
                cube[i] = 0.f;
            } else {
                ++kept;
            }
        }
        weight = 1.f / static_cast<float>(max(block_sum(kept, isum), 1));
    }
    __syncthreads();
    dct_group(cube, area, a.group, true);
    dct2d(cube, a.block, a.group, true);

    for (int g = threadIdx.x; g < a.group; g += blockDim.x) {
        a.patches[static_cast<long long>(r) * a.group + g] =
            g < kk ? AggregatePatch{m[g].x, m[g].y, m[g].t, weight} : AggregatePatch{0, 0, -1, 0.f};
    }
}


// --- Block 8, group 8: one warp filters four groups in registers -----------
//
// Lane l serves group l / 8 with sub-lane j = l % 8 and holds 64 samples
// v[p * 8 + r]: patch p, row r, column j. The transforms along r and p run in
// registers; the one along the columns runs after an 8 x 8 transposition
// across the group's lanes through a small shared buffer.
//
// 8-point DCT-II / DCT-III butterflies (FFTW e10_8 / e01_8, the arithmetic of
// the CPU's bm3d_filter8). Each is 4x the orthonormal transform, so the 3-D
// forward scales coefficients by 64 and the round trip by 4096: both exact
// powers of two, applied to the threshold and to the result.
constexpr float kDctP414 = 0.414213562373095048801688724209698078569671875f;
constexpr float kDctP1847 = 1.847759065022573512256366378793576573644833252f;
constexpr float kDctP198 = 0.198912367379658006911597622644676228597850501f;
constexpr float kDctP1961 = 1.961570560806460898252364472268478073947867462f;
constexpr float kDctP1414 = 1.414213562373095048801688724209698078569671875f;
constexpr float kDctP668 = 0.668178637919298919997757686523080761552472251f;
constexpr float kDctP1662 = 1.662939224605090474157576755235811513477121624f;
constexpr float kDctP707 = 0.707106781186547524400844362104849039284835938f;

__device__ __forceinline__ void dct8_forward(float* b, int s) {
    const float t1 = b[0], t2 = b[7 * s];
    const float t3 = t1 - t2, tj = t1 + t2;
    const float tc = b[4 * s], td = b[3 * s];
    const float te = tc - td, tk = tc + td;
    const float t4 = b[2 * s], t5 = b[5 * s];
    const float t6 = t4 - t5;
    const float t7 = b[s], t8 = b[6 * s];
    const float t9 = t7 - t8;
    const float ta = t6 + t9, tn = t7 + t8, tf = t6 - t9, tm = t4 + t5;
    const float tb = fmaf(-kDctP707, ta, t3);
    const float tg = fmaf(-kDctP707, tf, te);
    b[3 * s] = kDctP1662 * fmaf(kDctP668, tg, tb);
    b[5 * s] = -(kDctP1662 * fmaf(-kDctP668, tb, tg));
    const float tp = tj + tk, tq = tm + tn;
    b[4 * s] = kDctP1414 * (tp - tq);
    b[0] = kDctP1414 * (tp + tq);
    const float th = fmaf(kDctP707, ta, t3);
    const float ti = fmaf(kDctP707, tf, te);
    b[s] = kDctP1961 * fmaf(-kDctP198, ti, th);
    b[7 * s] = kDctP1961 * fmaf(kDctP198, th, ti);
    const float tl = tj - tk, to = tm - tn;
    b[2 * s] = kDctP1847 * fmaf(-kDctP414, to, tl);
    b[6 * s] = kDctP1847 * fmaf(kDctP414, tl, to);
}

__device__ __forceinline__ void dct8_inverse(float* b, int s) {
    const float t1 = kDctP1414 * b[0];
    const float t2 = b[4 * s];
    const float t3 = fmaf(kDctP1414, t2, t1);
    const float tj = fmaf(-kDctP1414, t2, t1);
    const float t4 = b[2 * s], t5 = b[6 * s];
    const float t6 = fmaf(kDctP414, t5, t4);
    const float tk = kDctP414 * t4 - t5;
    const float t8 = b[s], td = b[7 * s], t9 = b[5 * s], ta = b[3 * s];
    const float tb = t9 + ta, te = ta - t9;
    const float tc = fmaf(kDctP707, tb, t8);
    const float tn = fmaf(-kDctP707, te, td);
    const float tf = fmaf(kDctP707, te, td);
    const float tm = fmaf(-kDctP707, tb, t8);
    const float t7 = fmaf(kDctP1847, t6, t3);
    const float tg = fmaf(kDctP198, tf, tc);
    b[7 * s] = fmaf(-kDctP1961, tg, t7);
    b[0] = fmaf(kDctP1961, tg, t7);
    const float tp = fmaf(-kDctP1847, tk, tj);
    const float tq = fmaf(kDctP668, tm, tn);
    b[5 * s] = fmaf(-kDctP1662, tq, tp);
    b[2 * s] = fmaf(kDctP1662, tq, tp);
    const float th = fmaf(-kDctP1847, t6, t3);
    const float ti = fmaf(-kDctP198, tc, tf);
    b[3 * s] = fmaf(-kDctP1961, ti, th);
    b[4 * s] = fmaf(kDctP1961, ti, th);
    const float tl = fmaf(kDctP1847, tk, tj);
    const float to = fmaf(-kDctP668, tn, tm);
    b[6 * s] = fmaf(-kDctP1662, to, tl);
    b[s] = fmaf(kDctP1662, to, tl);
}

constexpr int kWarpStride = 33;  // one more than the warp: conflict-free banks both ways

// Transposes, for every p, the 8 x 8 matrix whose columns are the lanes of a
// group: v[p * 8 + i] of sub-lane j <-> v[p * 8 + j] of sub-lane i.
__device__ __forceinline__ void transpose8(float* v, float* buffer, int lane) {
    const int sub = lane & 7, base = lane & ~7;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
#pragma unroll
        for (int i = 0; i < 8; ++i) buffer[i * kWarpStride + lane] = v[p * 8 + i];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) v[p * 8 + i] = buffer[sub * kWarpStride + base + i];
        __syncwarp();
    }
}

// Loads the lane's column of the group's patches from `planes` (zero for the
// slots beyond the group's count) and applies the forward transform. On
// return the lane is the row frequency and the index within a patch the
// column frequency: the DC coefficient is v[0] of sub-lane 0.
__device__ __forceinline__ void forward8(float* v, const float* const* planes, const DeviceMatch* m, int kk, int pitch,
                                         float* buffer, int lane) {
    const int sub = lane & 7;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const DeviceMatch& mp = m[p < kk ? p : 0];
        const float* at = planes[mp.t] + static_cast<long long>(mp.y) * pitch + mp.x + sub;
#pragma unroll
        for (int row = 0; row < 8; ++row) v[p * 8 + row] = p < kk ? at[row * pitch] : 0.f;
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) dct8_forward(v + p * 8, 1);  // rows of the patch
#pragma unroll
    for (int row = 0; row < 8; ++row) dct8_forward(v + row, 8);  // group axis
    transpose8(v, buffer, lane);
#pragma unroll
    for (int p = 0; p < 8; ++p) dct8_forward(v + p * 8, 1);  // columns of the patch
}

template <bool Wiener>
__global__ void __launch_bounds__(32) group_filter8_kernel(Bm3dGroupArgs a) {
    __shared__ float buffer[8 * kWarpStride];
    const int lane = threadIdx.x, sub = lane & 7;
    const int r = blockIdx.x * 4 + lane / 8;
    const bool live = r < a.batch;
    const int rr = live ? r : a.batch - 1;  // idle lanes mirror the last group and store nothing
    const DeviceMatch* m = a.matches + static_cast<long long>(rr) * 8;
    const int kk = min(a.counts[rr], 8);
    float v[64];
    float weight;
    if constexpr (Wiener) {
        // Gains from the reference cube, parked in the lane's slice of ref_cube
        // while the registers transform the noisy cube.
        float* gain = a.ref_cube + static_cast<long long>(rr) * 512 + sub * 64;
        forward8(v, a.ref, m, kk, a.pitch, buffer, lane);
        const float sig2 = 4096.f * a.sigma * a.sigma;
        float w2 = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const float q = v[i] * v[i];
            const float w = (i == 0 && sub == 0) ? 1.f : q / (q + sig2);
            if (live) gain[i] = w;
            w2 = fmaf(w, w, w2);
        }
        w2 += __shfl_xor_sync(0xffffffffu, w2, 1);
        w2 += __shfl_xor_sync(0xffffffffu, w2, 2);
        w2 += __shfl_xor_sync(0xffffffffu, w2, 4);
        weight = 1.f / fmaxf(w2, 1e-12f);
        forward8(v, a.src, m, kk, a.pitch, buffer, lane);
        if (live) {
#pragma unroll
            for (int i = 0; i < 64; ++i) v[i] *= gain[i];
        }
    } else {
        forward8(v, a.src, m, kk, a.pitch, buffer, lane);
        const float thr = 64.f * kHardLambda * a.sigma;
        int kept = 0;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            const bool keep = (i == 0 && sub == 0) || !(fabsf(v[i]) < thr);
            v[i] = keep ? v[i] : 0.f;
            kept += keep;
        }
        kept += __shfl_xor_sync(0xffffffffu, kept, 1);
        kept += __shfl_xor_sync(0xffffffffu, kept, 2);
        kept += __shfl_xor_sync(0xffffffffu, kept, 4);
        weight = 1.f / static_cast<float>(max(kept, 1));
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) dct8_inverse(v + p * 8, 1);
    transpose8(v, buffer, lane);
#pragma unroll
    for (int row = 0; row < 8; ++row) dct8_inverse(v + row, 8);
#pragma unroll
    for (int p = 0; p < 8; ++p) dct8_inverse(v + p * 8, 1);
    if (!live) return;
    float* cube = a.values + static_cast<long long>(r) * 512 + sub;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
#pragma unroll
        for (int row = 0; row < 8; ++row) cube[p * 64 + row * 8] = v[p * 8 + row] * (1.f / 4096.f);
    }
    a.patches[static_cast<long long>(r) * 8 + sub] =
        sub < kk ? AggregatePatch{m[sub].x, m[sub].y, m[sub].t, weight} : AggregatePatch{0, 0, -1, 0.f};
}

}  // namespace

constexpr double kPi = 3.14159265358979323846;

void bm3d_init_tables(int device) {
    static std::mutex mutex;
    static std::set<int> ready;
    std::lock_guard lock(mutex);
    if (ready.count(device)) return;
    std::vector<float> table(kTableFloats);
    for (const int n : kTableSizes) {
        const int offset = table_offset(n);
        for (int k = 0; k < n; ++k) {
            const double scale = std::sqrt((k == 0 ? 1.0 : 2.0) / n);
            for (int j = 0; j < n; ++j) {
                table[offset + k * n + j] = static_cast<float>(scale * std::cos(kPi * (2 * j + 1) * k / (2.0 * n)));
            }
        }
    }
    NSS_CUDA_CHECK(cudaMemcpyToSymbol(c_dct, table.data(), table.size() * sizeof(float)));
    // The copy from pageable memory may complete after the call returns;
    // filter streams are non-blocking and would not wait for it.
    NSS_CUDA_CHECK(cudaDeviceSynchronize());
    ready.insert(device);
}

void bm3d_filter_groups(const Bm3dGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    if (args.block == 8 && args.group == 8) {
        if (args.ref) {
            group_filter8_kernel<true><<<(args.batch + 3) / 4, 32, 0, stream>>>(args);
        } else {
            group_filter8_kernel<false><<<(args.batch + 3) / 4, 32, 0, stream>>>(args);
        }
    } else {
        group_filter_kernel<<<args.batch, kThreads, 0, stream>>>(args);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
