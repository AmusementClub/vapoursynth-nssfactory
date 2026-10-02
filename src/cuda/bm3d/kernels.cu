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

__global__ void finish_kernel(const float* num, const float* den, const float* src, int width, int height, int pitch,
                              float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long i = static_cast<long long>(y) * pitch + x;
    const float d = den[i];
    out[i] = d > 1e-12f ? num[i] / d : src[i];
}

__global__ void accumulate_kernel(float* acc_num, float* acc_den, const float* num, const float* den,
                                  std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    acc_num[i] += num[i];
    acc_den[i] += den[i];
}

__global__ void vaggregate_kernel(const float* const* nums, const float* const* dens, int count, const float* src,
                                  int width, int height, int pitch, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long i = static_cast<long long>(y) * pitch + x;
    float num = 0.f, den = 0.f;
    for (int c = 0; c < count; ++c) {
        num += nums[c][i];
        den += dens[c][i];
    }
    // Disabled planes contribute one exact identity slice (den == 1).
    out[i] = den == 1.f ? num : (den > 1e-12f ? num / den : src[i]);
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
    ready.insert(device);
}

void bm3d_filter_groups(const Bm3dGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    group_filter_kernel<<<args.batch, kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void accumulate_slice(float* acc_num, float* acc_den, const float* num, const float* den, std::size_t count,
                      cudaStream_t stream) {
    if (count == 0) return;
    accumulate_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(acc_num, acc_den, num, den, count);
    NSS_CUDA_CHECK_LAUNCH();
}

void vaggregate_target(const float* const* nums, const float* const* dens, int count, const float* src, int width,
                       int height, int pitch, float* out, cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    vaggregate_kernel<<<blocks, threads, 0, stream>>>(nums, dens, count, src, width, height, pitch, out);
    NSS_CUDA_CHECK_LAUNCH();
}

void bm3d_finish(const float* num, const float* den, const float* src, int width, int height, int pitch, float* out,
                 cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    finish_kernel<<<blocks, threads, 0, stream>>>(num, den, src, width, height, pitch, out);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
