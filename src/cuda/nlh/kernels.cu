// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/nlh/kernels.hpp"
#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

constexpr int kThreads = 128;
constexpr float kInvSqrt2 = 0.7071067811865475244f;

__device__ __forceinline__ int group_columns(int count, int group, bool pow2) {
    int n = min(count, group);
    if (pow2) {
        int p = 1;
        while (p * 2 <= n) p *= 2;
        n = p;
    }
    return n;
}

// nss::pixel_match for row r of a guide group stored [row][column]: slot 0 is
// the row itself, the others are the nearest rows by (distance, index).
// With N > 0 the rows are N wide and zero beyond the group's n columns, so
// the distance loop has a fixed bound (zero terms leave the sum unchanged).
template <int N>
__device__ __forceinline__ void select_rows(const float* rows, int m, int n, int stride, int q, int r, int* out) {
    const int qq = min(q, m);
    constexpr float kInf = 1.0e30f;
    float best_d[16];
    int best_i[16];
    for (int k = 0; k < qq; ++k) {
        best_d[k] = k == 0 ? 0.f : kInf;
        best_i[k] = r;
    }
    const float* mine = rows + r * stride;
    float fixed[N > 0 ? N : 1];
#pragma unroll
    for (int c = 0; c < N; ++c) fixed[c] = mine[c];
    for (int s = 0; s < m && qq > 1; ++s) {
        if (s == r) continue;
        const float* other = rows + s * stride;
        float d = 0.f;
        if (N > 0) {
#pragma unroll
            for (int c = 0; c < N; ++c) {
                const float e = other[c] - fixed[c];
                d = fmaf(e, e, d);
            }
        } else {
            for (int c = 0; c < n; ++c) {
                const float e = other[c] - mine[c];
                d = fmaf(e, e, d);
            }
        }
        int pos = qq - 1;
        if (!(d < best_d[pos] || (d == best_d[pos] && s < best_i[pos]))) continue;
        while (pos > 1 && (d < best_d[pos - 1] || (d == best_d[pos - 1] && s < best_i[pos - 1]))) {
            best_d[pos] = best_d[pos - 1];
            best_i[pos] = best_i[pos - 1];
            --pos;
        }
        best_d[pos] = d;
        best_i[pos] = s;
    }
    for (int k = 0; k < q; ++k) out[k] = k < qq ? best_i[k] : r;
}

__device__ __forceinline__ AggregatePatch patch_record(const DeviceMatch* match, int j, int n) {
    return j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t, 1.f} : AggregatePatch{0, 0, -1, 0.f};
}

// One block per group, one thread per patch pixel row (and per patch): the
// guide group is staged in shared memory, then every row selects its pixels.
__global__ void prepare_kernel(NlhGroupArgs a) {
    extern __shared__ float rows[];  // m * group, [row][column]
    const int m = a.block * a.block;
    const int g = blockIdx.x, r = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    if (r < m) {
        const int offset = (r / a.block) * a.width + r % a.block;
        for (int j = 0; j < a.group; ++j) {
            rows[r * a.group + j] =
                j < n ? a.guides[match[j].t][static_cast<long long>(match[j].y) * a.width + match[j].x + offset] : 0.f;
        }
    }
    if (r < a.group) a.patches[static_cast<long long>(g) * a.group + r] = patch_record(match, r, n);
    __syncthreads();
    if (r >= m) return;
    int* out = a.indices + (static_cast<long long>(g) * m + r) * a.q;
    if (a.group == 16) select_rows<16>(rows, m, n, 16, a.q, r, out);
    else select_rows<0>(rows, m, n, a.group, a.q, r, out);
}

// Global-memory variant for guide groups that do not fit shared memory: one
// thread per (group, row) packs, then one per (group, row) selects.
__global__ void pack_kernel(NlhGroupArgs a) {
    const int m = a.block * a.block;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int rows = max(m, a.group);
    if (id >= static_cast<long long>(a.batch) * rows) return;
    const int g = static_cast<int>(id / rows), row = static_cast<int>(id % rows);
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    if (row < m) {
        float* out = a.guide_group + (static_cast<long long>(g) * m + row) * a.group;
        const int offset = (row / a.block) * a.width + row % a.block;
        for (int j = 0; j < n; ++j) {
            out[j] = a.guides[match[j].t][static_cast<long long>(match[j].y) * a.width + match[j].x + offset];
        }
    }
    if (row < a.group) a.patches[static_cast<long long>(g) * a.group + row] = patch_record(match, row, n);
}

__global__ void pixel_match_kernel(NlhGroupArgs a) {
    const int m = a.block * a.block;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(a.batch) * m) return;
    const int g = static_cast<int>(id / m), r = static_cast<int>(id % m);
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    select_rows<0>(a.guide_group + static_cast<long long>(g) * m * a.group, m, n, a.group, a.q, r,
                   a.indices + (static_cast<long long>(g) * m + r) * a.q);
}

// Multi-level orthonormal Haar of n (power of two) strided values, in place.
__device__ void haar(float* data, int stride, int n, bool inverse) {
    float a[64], b[64];
    for (int i = 0; i < n; ++i) a[i] = data[i * stride];
    if (!inverse) {
        for (int len = n; len >= 2; len /= 2) {
            const int h = len / 2;
            for (int i = 0; i < h; ++i) {
                b[i] = (a[2 * i] + a[2 * i + 1]) * kInvSqrt2;
                b[h + i] = (a[2 * i] - a[2 * i + 1]) * kInvSqrt2;
            }
            for (int i = 0; i < len; ++i) a[i] = b[i];
        }
    } else {
        for (int len = 2; len <= n; len *= 2) {
            const int h = len / 2;
            for (int i = 0; i < h; ++i) {
                b[2 * i] = (a[i] + a[h + i]) * kInvSqrt2;
                b[2 * i + 1] = (a[i] - a[h + i]) * kInvSqrt2;
            }
            for (int i = 0; i < len; ++i) a[i] = b[i];
        }
    }
    for (int i = 0; i < n; ++i) data[i * stride] = a[i];
}

// The same transform with compile-time sizes (kept in registers).
template <int N>
__device__ __forceinline__ void haar_fixed(float* data, int stride, bool inverse) {
    float a[N], b[N];
#pragma unroll
    for (int i = 0; i < N; ++i) a[i] = data[i * stride];
    if (!inverse) {
#pragma unroll
        for (int len = N; len >= 2; len /= 2) {
            const int h = len / 2;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                b[i] = (a[2 * i] + a[2 * i + 1]) * kInvSqrt2;
                b[h + i] = (a[2 * i] - a[2 * i + 1]) * kInvSqrt2;
            }
#pragma unroll
            for (int i = 0; i < len; ++i) a[i] = b[i];
        }
    } else {
#pragma unroll
        for (int len = 2; len <= N; len *= 2) {
            const int h = len / 2;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                b[2 * i] = (a[i] + a[h + i]) * kInvSqrt2;
                b[2 * i + 1] = (a[i] - a[h + i]) * kInvSqrt2;
            }
#pragma unroll
            for (int i = 0; i < len; ++i) a[i] = b[i];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) data[i * stride] = a[i];
}

template <int Q, int N>
__device__ __forceinline__ void haar2d_fixed(float* matrix, bool inverse) {
    if (inverse) {
#pragma unroll
        for (int k = 0; k < Q; ++k) haar_fixed<N>(matrix + k, Q, true);
#pragma unroll
        for (int j = 0; j < N; ++j) haar_fixed<Q>(matrix + j * Q, 1, true);
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) haar_fixed<Q>(matrix + j * Q, 1, false);
#pragma unroll
        for (int k = 0; k < Q; ++k) haar_fixed<N>(matrix + k, Q, false);
    }
}

// Forward: columns (length q) then rows (length n); inverse: the reverse.
__device__ void haar2d(float* matrix, int q, int n, bool inverse) {
    if (q == 4 && n == 16) return haar2d_fixed<4, 16>(matrix, inverse);
    if (q == 2 && n == 16) return haar2d_fixed<2, 16>(matrix, inverse);
    if (inverse) {
        for (int k = 0; k < q; ++k) haar(matrix + k, q, n, true);
        for (int j = 0; j < n; ++j) haar(matrix + j * q, 1, q, true);
    } else {
        for (int j = 0; j < n; ++j) haar(matrix + j * q, 1, q, false);
        for (int k = 0; k < q; ++k) haar(matrix + k, q, n, false);
    }
}

// One row's q x n matrix (matrix[k + j * q]) of channel c: gather the selected
// pixels, Haar, Basic threshold or Wiener gain, inverse Haar. matrix and ref
// (Wiener) are q * n floats of workspace.
__device__ void shrink_row(const NlhGroupArgs& a, const DeviceMatch* match, const int* idx, int n, int c,
                           float* matrix, float* ref) {
    const int q = a.q;
    for (int k = 0; k < q; ++k) {
        const int offset = (idx[k] / a.block) * a.width + idx[k] % a.block;
        for (int j = 0; j < n; ++j) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
            matrix[k + j * q] = a.data[match[j].t * a.nch + c][at];
            if (a.wiener) ref[k + j * q] = a.reference[match[j].t * a.nch + c][at];
        }
    }
    haar2d(matrix, q, n, false);
    if (a.wiener) {
        haar2d(ref, q, n, false);
        const double noise = a.noise[c];
        for (int i = 0; i < q * n; ++i) {
            const double r = ref[i];
            const double r2 = r * r;
            // The zero-noise limit is identity, including 0/0 coefficients.
            const double gain = noise == 0.0 ? 1.0 : r2 / (r2 + noise);
            double value = matrix[i];
            for (int it = 0; it < a.wiener_iterations; ++it) value *= gain;
            matrix[i] = static_cast<float>(value);
        }
    } else {
        const float threshold = a.threshold[c];
        const int structural = max(0, q - 2);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < q; ++k) {
                if (fabsf(matrix[k + j * q]) < threshold || (j > 0 && k >= structural)) matrix[k + j * q] = 0.f;
            }
        }
    }
    haar2d(matrix, q, n, true);
}

// Sum for `pixel` of the shrunk values of every (row, slot) that selected it,
// rows ascending (the CPU order); coef holds the rows' matrices back to back
// with `stride` floats per row. Returns the selection count.
__device__ __forceinline__ int gather_pixel(const int* idx, const float* coef, int stride, int m, int q, int n,
                                            int pixel, bool sum, float* num) {
    for (int j = 0; j < n; ++j) num[j] = 0.f;
    int count = 0;
    for (int i = 0; i < m * q; ++i) {
        if (idx[i] != pixel) continue;
        ++count;
        if (!sum) continue;
        const float* matrix = coef + (i / q) * stride;
        for (int j = 0; j < n; ++j) num[j] += matrix[i % q + j * q];
    }
    return count;
}

__device__ __forceinline__ void store_pixel(const NlhGroupArgs& a, const DeviceMatch* match, int g, int c, int n,
                                            int pixel, int count, float* num) {
    const int m = a.block * a.block;
    float* out = a.values + (static_cast<long long>(c) * a.batch + g) * a.group * m;
    const int offset = (pixel / a.block) * a.width + pixel % a.block;
    for (int j = 0; j < n; ++j) {
        if (a.identity[c]) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
            num[j] = static_cast<float>(count) * a.data[match[j].t * a.nch + c][at];
        }
        out[j * m + pixel] = num[j];
    }
    if (c == 0) a.den[static_cast<long long>(g) * m + pixel] = static_cast<float>(count);
}

// Fused filter for groups whose row matrices fit shared memory: one block per
// group, one thread per row. Per channel every thread shrinks its row's
// matrix into shared memory, then gathers its pixel's sum from it.
__global__ void filter_fused_kernel(NlhGroupArgs a) {
    extern __shared__ unsigned char shared[];
    const int m = a.block * a.block;
    const int q = a.q;
    const int stride = q * a.group;
    int* idx = reinterpret_cast<int*>(shared);
    float* coef = reinterpret_cast<float*>(idx + m * q);
    const int g = blockIdx.x, row = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int* source = a.indices + (static_cast<long long>(g) * m + row) * q;
    for (int k = 0; k < q; ++k) idx[row * q + k] = source[k];
    for (int c = 0; c < a.nch; ++c) {
        if (!a.identity[c]) {
            float matrix[64], ref[64];
            shrink_row(a, match, source, n, c, matrix, ref);
            for (int i = 0; i < q * n; ++i) coef[row * stride + i] = matrix[i];
        }
        __syncthreads();
        float num[32];
        const int count = gather_pixel(idx, coef, stride, m, q, n, row, !a.identity[c], num);
        store_pixel(a, match, g, c, n, row, count, num);
        __syncthreads();
    }
}

// Global-memory variant: one thread per (group, channel, row) shrinks into
// coef, then one per (group, channel, pixel) gathers.
__global__ void shrink_kernel(NlhGroupArgs a, bool local) {
    const int m = a.block * a.block;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(a.batch) * a.nch * m) return;
    const int row = static_cast<int>(id % m);
    const int c = static_cast<int>((id / m) % a.nch);
    const int g = static_cast<int>(id / m / a.nch);
    if (a.identity[c]) return;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const int q = a.q;
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int* idx = a.indices + (static_cast<long long>(g) * m + row) * q;
    const long long base = ((static_cast<long long>(g) * a.nch + c) * m + row) * q * a.group;
    if (local) {
        float matrix[64], ref[64];
        shrink_row(a, match, idx, n, c, matrix, ref);
        for (int i = 0; i < q * n; ++i) a.coef[base + i] = matrix[i];
    } else {
        shrink_row(a, match, idx, n, c, a.coef + base, a.wiener ? a.ref_coef + base : nullptr);
    }
}

__global__ void gather_kernel(NlhGroupArgs a) {
    const int m = a.block * a.block;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(a.batch) * a.nch * m) return;
    const int pixel = static_cast<int>(id % m);
    const int c = static_cast<int>((id / m) % a.nch);
    const int g = static_cast<int>(id / m / a.nch);
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const int q = a.q;
    float num[64];
    const int count = gather_pixel(a.indices + static_cast<long long>(g) * m * q,
                                   a.coef + (static_cast<long long>(g) * a.nch + c) * m * q * a.group, q * a.group, m, q,
                                   n, pixel, !a.identity[c], num);
    store_pixel(a, a.matches + static_cast<long long>(g) * a.group, g, c, n, pixel, count, num);
}

// One block per group, one thread per pixel row (m = 64): the group's noise
// statistic per channel, rows summed in order by thread 0.
__global__ void sigma_kernel(NlhGroupArgs a, double* partial) {
    __shared__ double local[64];
    const int m = a.block * a.block;
    const int g = blockIdx.x, row = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int* idx = a.indices + (static_cast<long long>(g) * m + row) * a.q;
    const int offset = (row / a.block) * a.width + row % a.block;
    for (int c = 0; c < a.nch; ++c) {
        double sum = 0.0;
        for (int k = 1; k < a.q; ++k) {
            const int other = (idx[k] / a.block) * a.width + idx[k] % a.block;
            double d2 = 0.0;
            for (int j = 0; j < n; ++j) {
                const float* plane = a.data[match[j].t * a.nch + c] + static_cast<long long>(match[j].y) * a.width + match[j].x;
                const double d = static_cast<double>(plane[offset]) - plane[other];
                d2 += d * d;
            }
            sum += sqrt(d2 / n);
        }
        local[row] = sum;
        __syncthreads();
        if (row == 0) {
            double total = 0.0;
            for (int i = 0; i < m; ++i) total += local[i];
            partial[static_cast<long long>(g) * a.nch + c] = total / (m * (a.q - 1));
        }
        __syncthreads();
    }
}

// Fixed two-level sum of the per-group statistics into totals[nch].
__global__ void sigma_sum_kernel(const double* partial, int batch, int nch, double* sums) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunks = (batch + 255) / 256;
    if (i >= chunks * nch) return;
    const int chunk = i / nch, c = i % nch;
    double total = 0.0;
    for (int g = chunk * 256; g < min(batch, (chunk + 1) * 256); ++g) total += partial[static_cast<long long>(g) * nch + c];
    sums[i] = total;
}
__global__ void sigma_total_kernel(const double* sums, int chunks, int nch, double* totals) {
    const int c = threadIdx.x;
    if (c >= nch) return;
    double total = totals[c];
    for (int i = 0; i < chunks; ++i) total += sums[i * nch + c];
    totals[c] = total;
}

__global__ void rgb_to_yuv_kernel(float* r, float* g, float* b, std::size_t count, bool luma_only) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double rv = r[i], gv = g[i], bv = b[i];
    r[i] = static_cast<float>(0.299 * rv + 0.587 * gv + 0.114 * bv);
    if (luma_only) return;
    g[i] = static_cast<float>(-0.168736607142857 * rv + -0.331263392857143 * gv + 0.5 * bv);
    b[i] = static_cast<float>(0.5 * rv + -0.4186875 * gv + -0.0813125 * bv);
}

__global__ void yuv_to_rgb_kernel(float* yp, float* up, float* vp, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double y = yp[i], u = up[i], v = vp[i];
    yp[i] = static_cast<float>(y + 1.402 * v);
    up[i] = static_cast<float>(y - (0.114 * 1.772 / 0.587) * u - (0.299 * 1.402 / 0.587) * v);
    vp[i] = static_cast<float>(y + 1.772 * u);
}

__global__ void mix_kernel(float* basic, const float* input, std::size_t count, double mix) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    basic[i] = static_cast<float>(mix * basic[i] + (1 - mix) * input[i]);
}

__global__ void area_guide_kernel(const float* luma, int luma_width, float* guide, int width, int height, int sx,
                                  int sy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    double sum = 0.0;
    for (int dy = 0; dy < sy; ++dy) {
        for (int dx = 0; dx < sx; ++dx) sum += luma[static_cast<long long>(y * sy + dy) * luma_width + x * sx + dx];
    }
    guide[static_cast<long long>(y) * width + x] = static_cast<float>(sum / (sx * sy));
}

unsigned blocks_for(long long items, int threads) { return static_cast<unsigned>((items + threads - 1) / threads); }

}  // namespace

void nlh_prepare_groups(const NlhGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block;
    if (nlh_prepare_shared(args.block, args.group)) {
        prepare_kernel<<<args.batch, m > args.group ? m : args.group, m * args.group * sizeof(float), stream>>>(args);
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    const long long rows = static_cast<long long>(args.batch) * (m > args.group ? m : args.group);
    pack_kernel<<<blocks_for(rows, kThreads), kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
    pixel_match_kernel<<<blocks_for(static_cast<long long>(args.batch) * m, kThreads), kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_filter_groups(const NlhGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block;
    if (nlh_filter_fused(args.block, args.group, args.q)) {
        filter_fused_kernel<<<args.batch, m, nlh_fused_bytes(args.block, args.group, args.q), stream>>>(args);
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    const long long items = static_cast<long long>(args.batch) * args.nch * m;
    shrink_kernel<<<blocks_for(items, kThreads), kThreads, 0, stream>>>(args, nlh_local_matrix(args.group, args.q));
    NSS_CUDA_CHECK_LAUNCH();
    gather_kernel<<<blocks_for(items, kThreads), kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_sigma_groups(const NlhGroupArgs& args, double* partial, double* sums, double* totals, cudaStream_t stream) {
    if (args.batch <= 0) return;
    sigma_kernel<<<args.batch, args.block * args.block, 0, stream>>>(args, partial);
    NSS_CUDA_CHECK_LAUNCH();
    const int chunks = (args.batch + 255) / 256;
    sigma_sum_kernel<<<blocks_for(static_cast<long long>(chunks) * args.nch, kThreads), kThreads, 0, stream>>>(
        partial, args.batch, args.nch, sums);
    NSS_CUDA_CHECK_LAUNCH();
    sigma_total_kernel<<<1, 4, 0, stream>>>(sums, chunks, args.nch, totals);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_rgb_to_yuv(float* r, float* g, float* b, std::size_t count, bool luma_only, cudaStream_t stream) {
    rgb_to_yuv_kernel<<<blocks_for(static_cast<long long>(count), 256), 256, 0, stream>>>(r, g, b, count, luma_only);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_yuv_to_rgb(float* y, float* u, float* v, std::size_t count, cudaStream_t stream) {
    yuv_to_rgb_kernel<<<blocks_for(static_cast<long long>(count), 256), 256, 0, stream>>>(y, u, v, count);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_mix(float* basic, const float* input, std::size_t count, double mix, cudaStream_t stream) {
    mix_kernel<<<blocks_for(static_cast<long long>(count), 256), 256, 0, stream>>>(basic, input, count, mix);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_area_guide(const float* luma, int luma_width, float* guide, int width, int height, int sx, int sy,
                    cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    area_guide_kernel<<<blocks, threads, 0, stream>>>(luma, luma_width, guide, width, height, sx, sy);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
