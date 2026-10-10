// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/nlh/kernels.hpp"
#include "cuda/common/fixed_accumulate.cuh"
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
    // Up to four slots the list stays in registers: three sorted candidates,
    // of which the first qq - 1 are the answer (the order is total).
    const bool few = qq <= 4;
    float d1 = kInf, d2 = kInf, d3 = kInf;
    int i1 = r, i2 = r, i3 = r;
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
        if (few) {
            if (!(d < d3 || (d == d3 && s < i3))) continue;
            d3 = d, i3 = s;
            if (d3 < d2 || (d3 == d2 && i3 < i2)) {
                const float td = d2;
                const int ti = i2;
                d2 = d3, i2 = i3, d3 = td, i3 = ti;
            }
            if (d2 < d1 || (d2 == d1 && i2 < i1)) {
                const float td = d1;
                const int ti = i1;
                d1 = d2, i1 = i2, d2 = td, i2 = ti;
            }
            continue;
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
    if (few) {
        const int best[4] = {r, i1, i2, i3};
        for (int k = 0; k < q; ++k) out[k] = k < qq ? best[k] : r;
        return;
    }
    for (int k = 0; k < q; ++k) out[k] = k < qq ? best_i[k] : r;
}

// One block per group, one thread per patch pixel row: the guide group is
// staged in shared memory, then every row selects its pixels. With `den`
// (the lane filter follows) it also adds every pixel's selection count to
// the count plane.
__global__ void prepare_kernel(NlhGroupArgs a, int den) {
    extern __shared__ float rows[];  // m * group, [row][column]
    const int m = a.block * a.block;
    const int g = blockIdx.x, r = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int offset = (r / a.block) * a.width + r % a.block;
    for (int j = 0; j < a.group; ++j) {
        rows[r * a.group + j] =
            j < n ? a.guides[match[j].t][static_cast<long long>(match[j].y) * a.width + match[j].x + offset] : 0.f;
    }
    __syncthreads();
    int* out = a.indices + (static_cast<long long>(g) * m + r) * a.q;
    if (a.group == 16) select_rows<16>(rows, m, n, 16, a.q, r, out);
    else select_rows<0>(rows, m, n, a.group, a.q, r, out);
    if (!den) return;
    // The selections of every pixel, counted where the guide rows were.
    auto* hits = reinterpret_cast<unsigned*>(rows);
    __syncthreads();
    hits[r] = 0;
    __syncthreads();
    for (int k = 0; k < a.q; ++k) atomicAdd(hits + out[k], 1u);
    __syncthreads();
    const unsigned count = hits[r];
    if (count == 0) return;
    for (int j = 0; j < n; ++j) {
        const long long cell = match[j].t * static_cast<long long>(a.slice_step) +
                               static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
        atomicAdd(a.den + cell, count);
    }
}

// Global-memory variant for guide groups that do not fit shared memory: one
// thread per (group, row) packs, then one per (group, row) selects.
__global__ void pack_kernel(NlhGroupArgs a) {
    const int m = a.block * a.block;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(a.batch) * m) return;
    const int g = static_cast<int>(id / m), row = static_cast<int>(id % m);
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    float* out = a.guide_group + (static_cast<long long>(g) * m + row) * a.group;
    const int offset = (row / a.block) * a.width + row % a.block;
    for (int j = 0; j < n; ++j) {
        out[j] = a.guides[match[j].t][static_cast<long long>(match[j].y) * a.width + match[j].x + offset];
    }
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
__device__ __forceinline__ void haar2d_fixed(float* matrix, bool inverse, int es) {
    if (inverse) {
#pragma unroll
        for (int k = 0; k < Q; ++k) haar_fixed<N>(matrix + k * es, Q * es, true);
#pragma unroll
        for (int j = 0; j < N; ++j) haar_fixed<Q>(matrix + j * Q * es, es, true);
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) haar_fixed<Q>(matrix + j * Q * es, es, false);
#pragma unroll
        for (int k = 0; k < Q; ++k) haar_fixed<N>(matrix + k * es, Q * es, false);
    }
}

// Forward: columns (length q) then rows (length n); inverse: the reverse.
__device__ void haar2d(float* matrix, int q, int n, bool inverse, int es = 1) {
    if (q == 4 && n == 16) return haar2d_fixed<4, 16>(matrix, inverse, es);
    if (q == 2 && n == 16) return haar2d_fixed<2, 16>(matrix, inverse, es);
    if (inverse) {
        for (int k = 0; k < q; ++k) haar(matrix + k * es, q * es, n, true);
        for (int j = 0; j < n; ++j) haar(matrix + j * q * es, es, q, true);
    } else {
        for (int j = 0; j < n; ++j) haar(matrix + j * q * es, es, q, false);
        for (int k = 0; k < q; ++k) haar(matrix + k * es, q * es, n, false);
    }
}

// A value as an unevaluated sum of two floats (hi + lo, |lo| well below an
// ulp of hi): about 48 significant bits from FP32 operations alone.
struct FloatPair {
    float hi, lo;
};

// a + b with its rounding error.
__device__ __forceinline__ FloatPair two_sum(float a, float b) {
    const float s = a + b;
    const float v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}

__device__ __forceinline__ FloatPair pair_add(FloatPair sum, FloatPair term) {
    const FloatPair s = two_sum(sum.hi, term.hi);
    return {s.hi, sum.lo + (s.lo + term.lo)};
}

// sqrt(x / n) for x >= 0.
__device__ __forceinline__ FloatPair pair_root_mean(FloatPair x, float n) {
    const float q = x.hi / n;
    const float q_lo = (fmaf(-q, n, x.hi) + x.lo) / n;
    if (!(q > 0.f)) return {0.f, 0.f};
    const float root = sqrtf(q);
    return {root, (fmaf(-root, root, q) + q_lo) / (2.f * root)};
}

// value * (r^2 / (r^2 + noise))^iterations rounded to float, for noise > 0
// given as a pair: the FP64 Wiener gain of the CPU from FP32 operations.
__device__ __forceinline__ float pair_wiener(float value, float r, FloatPair noise, int iterations) {
    const float r2 = r * r;
    const float r2_lo = fmaf(r, r, -r2);
    const FloatPair sum = two_sum(r2, noise.hi);
    const FloatPair den = two_sum(sum.hi, sum.lo + (r2_lo + noise.lo));
    const float gain = r2 / den.hi;
    const float gain_lo = (fmaf(-gain, den.hi, r2) + (r2_lo - gain * den.lo)) / den.hi;
    FloatPair v{value, 0.f};
    for (int it = 0; it < iterations; ++it) {
        const float hi = v.hi * gain;
        v = {hi, fmaf(v.hi, gain, -hi) + (v.hi * gain_lo + v.lo * gain)};
    }
    return v.hi + v.lo;
}

// One row's q x n matrix (element k + j * q at matrix[(k + j * q) * es]) of
// channel c: gather the selected pixels, Haar, Basic threshold or Wiener
// gain, inverse Haar. ref (Wiener) is q * n floats of workspace.
__device__ __forceinline__ void shrink_row(const NlhGroupArgs& a, bool wiener, const DeviceMatch* match,
                                           const int* idx, int n, int c, float* matrix, float* ref, int es = 1) {
    const int q = a.q;
    for (int k = 0; k < q; ++k) {
        const int offset = (idx[k] / a.block) * a.width + idx[k] % a.block;
        for (int j = 0; j < n; ++j) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
            matrix[(k + j * q) * es] = a.data[match[j].t * a.nch + c][at];
            if (wiener) ref[k + j * q] = a.reference[match[j].t * a.nch + c][at];
        }
    }
    haar2d(matrix, q, n, false, es);
    if (wiener) {
        haar2d(ref, q, n, false);
        const double noise = a.noise[c];
        const FloatPair noise_pair{a.noise_hi[c], a.noise_lo[c]};
        // Operands whose squares leave the float range keep the FP64 form.
        const bool pairs = noise_pair.hi > 1e-20f && noise_pair.hi < 1e30f;
        for (int i = 0; i < q * n; ++i) {
            if (pairs && fabsf(ref[i]) < 1e15f) {
                matrix[i * es] = pair_wiener(matrix[i * es], ref[i], noise_pair, a.wiener_iterations);
                continue;
            }
            const double r = ref[i];
            const double r2 = r * r;
            // The zero-noise limit is identity, including 0/0 coefficients.
            const double gain = noise == 0.0 ? 1.0 : r2 / (r2 + noise);
            double value = matrix[i * es];
            for (int it = 0; it < a.wiener_iterations; ++it) value *= gain;
            matrix[i * es] = static_cast<float>(value);
        }
    } else {
        const float threshold = a.threshold[c];
        const int structural = max(0, q - 2);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < q; ++k) {
                if (fabsf(matrix[(k + j * q) * es]) < threshold || (j > 0 && k >= structural)) matrix[(k + j * q) * es] = 0.f;
            }
        }
    }
    haar2d(matrix, q, n, true, es);
}

// Adds a pixel's sums (num[j] for patch j; `count` selections each) to the
// accumulators: exact integer sums, so the order of the atomics is free. An
// identity channel adds the samples themselves.
__device__ __forceinline__ void store_pixel(const NlhGroupArgs& a, const DeviceMatch* match, int c, int n, int pixel,
                                            int count, const float* num) {
    const int offset = (pixel / a.block) * a.width + pixel % a.block;
    for (int j = 0; j < n; ++j) {
        const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
        const long long cell = match[j].t * static_cast<long long>(a.slice_step) + at;
        fixed_add(a.num[c], cell,
                  a.identity[c] ? static_cast<float>(count) * a.data[match[j].t * a.nch + c][at] : num[j]);
        if (c == 0) atomicAdd(a.den + cell, static_cast<unsigned>(count));
    }
}

// Inverts the selections of a group for a block with one thread per pixel:
// every pixel gets the (row, slot) pairs that selected it, rows ascending
// (the CPU order of its sum), as row << 8 | slot in list[begin .. begin +
// count). list (m * q) and cursor (m) are shared workspace. The rows count
// and place their own selections with atomics, then every pixel sorts its
// short list: scanning all selections per pixel cost m * q compares twice.
__device__ __forceinline__ void invert_selection(const int* source, int m, int q, unsigned short* list,
                                                 unsigned* cursor, int& begin, int& count) {
    const int pixel = threadIdx.x;
    cursor[pixel] = 0;
    __syncthreads();
    for (int k = 0; k < q; ++k) atomicAdd(cursor + source[k], 1u);
    __syncthreads();
    count = static_cast<int>(cursor[pixel]);
    begin = 0;
    for (int p = 0; p < pixel; ++p) begin += static_cast<int>(cursor[p]);
    __syncthreads();
    cursor[pixel] = static_cast<unsigned>(begin);
    __syncthreads();
    for (int k = 0; k < q; ++k) list[atomicAdd(cursor + source[k], 1u)] = static_cast<unsigned short>(pixel << 8 | k);
    __syncthreads();
    for (int i = begin + 1; i < begin + count; ++i) {
        const unsigned short entry = list[i];
        int at = i;
        for (; at > begin && list[at - 1] > entry; --at) list[at] = list[at - 1];
        list[at] = entry;
    }
}

// Sum for a pixel of the shrunk values of its list; coef holds the rows'
// matrices back to back with `stride` floats per row.
__device__ __forceinline__ void gather_list(const unsigned short* list, int count, const float* coef, int stride, int q,
                                            int n, float* num) {
    for (int j = 0; j < n; ++j) num[j] = 0.f;
    for (int i = 0; i < count; ++i) {
        const int entry = list[i];
        const float* matrix = coef + (entry >> 8) * stride + (entry & 255);
        for (int j = 0; j < n; ++j) num[j] += matrix[j * q];
    }
}

// Fused filter: one block per group, one thread per row. Per channel the
// rows pass through shared memory `rows` at a time: the threads of the rows
// present shrink their matrices in place, then every thread adds the entries
// of its pixel's list that lie in them (the list is in row order, so the sum
// keeps the CPU order). Element i of row r is coef[i * rows + r]: the
// threads of a warp touch consecutive words, and a matrix kept in a thread's
// local storage instead missed the cache on most loads.
// The launch bound holds the kernel to 80 registers: unbounded it takes 168
// and an SM holds a fifth of its warps (the spills cost less than that).
template <bool Wiener>
__global__ void __launch_bounds__(256, 3) filter_fused_kernel(NlhGroupArgs a, int rows) {
    extern __shared__ unsigned char shared[];
    const int m = a.block * a.block;
    const int q = a.q;
    const int stride = q * a.group;
    float* coef = reinterpret_cast<float*>(shared);
    auto* cursor = reinterpret_cast<unsigned*>(coef + rows * stride);
    auto* list = reinterpret_cast<unsigned short*>(cursor + m);
    const int g = blockIdx.x, row = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int* source = a.indices + (static_cast<long long>(g) * m + row) * q;
    int begin, count;
    invert_selection(source, m, q, list, cursor, begin, count);
    for (int c = 0; c < a.nch; ++c) {
        const bool identity = a.identity[c];
        float ref[64];
        float num[32];
        for (int j = 0; j < n; ++j) num[j] = 0.f;
        int at = begin;
        const int end = identity ? begin : begin + count;
        for (int first = 0; first < m; first += rows) {
            if (!identity && row >= first && row < first + rows) {
                shrink_row(a, Wiener, match, source, n, c, coef + (row - first), ref, rows);
            }
            __syncthreads();
            for (; at < end && (list[at] >> 8) < first + rows; ++at) {
                const int entry = list[at];
                const float* from = coef + (entry & 255) * rows + ((entry >> 8) - first);
                for (int j = 0; j < n; ++j) num[j] += from[j * q * rows];
            }
            __syncthreads();
        }
        store_pixel(a, match, c, n, row, count, num);
    }
}

// Column transform (length Q) of the lane filter: the Q lanes of a row hold
// one element each and exchange by shuffles. The forward transform leaves
// the coefficients permuted over the lanes (lane_index); the inverse takes
// them from there. Same pairs and operand order as haar_fixed<Q>.
template <int Q>
__device__ __forceinline__ float lane_forward(float x, int k) {
    const float p = __shfl_xor_sync(0xffffffffu, x, 1);
    x = ((k & 1) ? p - x : x + p) * kInvSqrt2;
    if constexpr (Q == 4) {
        const float p2 = __shfl_xor_sync(0xffffffffu, x, 2);
        if (!(k & 1)) x = ((k & 2) ? p2 - x : x + p2) * kInvSqrt2;
    }
    return x;
}
template <int Q>
__device__ __forceinline__ float lane_inverse(float x, int k) {
    if constexpr (Q == 4) {
        const float p2 = __shfl_xor_sync(0xffffffffu, x, 2);
        if (!(k & 1)) x = ((k & 2) ? p2 - x : x + p2) * kInvSqrt2;
    }
    const float p = __shfl_xor_sync(0xffffffffu, x, 1);
    return ((k & 1) ? p - x : x + p) * kInvSqrt2;
}
// Coefficient index (along q) that lane k holds after lane_forward.
template <int Q>
__device__ __forceinline__ int lane_index(int k) {
    return Q == 4 ? (k == 1 ? 2 : k == 2 ? 1 : k) : k;
}
// Row transform (length n) of the 16 values a lane holds.
__device__ __forceinline__ void haar_columns(float* v, int n, bool inverse) {
    switch (n) {
        case 16: haar_fixed<16>(v, 1, inverse); break;
        case 8: haar_fixed<8>(v, 1, inverse); break;
        case 4: haar_fixed<4>(v, 1, inverse); break;
        case 2: haar_fixed<2>(v, 1, inverse); break;
        default: break;
    }
}

// The 16 filtered values of row slot k (sample `offset` of every patch) of
// channel c: lane k's part of the row's q x n matrix after Haar, shrink and
// inverse Haar. The q lanes of the row must call it together.
template <bool Wiener, int Q>
__device__ __forceinline__ void lane_values(const NlhGroupArgs& a, const DeviceMatch* match, int n, int offset, int k,
                                            int c, float* v) {
    const int kk = lane_index<Q>(k);
    float r[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int jj = min(j, n - 1);
        const long long at = static_cast<long long>(match[jj].y) * a.width + match[jj].x + offset;
        const int plane = match[jj].t * a.nch + c;
        v[j] = j < n ? a.data[plane][at] : 0.f;
        if (Wiener) r[j] = j < n && !a.identity[c] ? a.reference[plane][at] : 0.f;
    }
    if (!a.identity[c]) {
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = lane_forward<Q>(v[j], k);
        haar_columns(v, n, false);
        if (Wiener) {
#pragma unroll
            for (int j = 0; j < 16; ++j) r[j] = lane_forward<Q>(r[j], k);
            haar_columns(r, n, false);
            const double noise = a.noise[c];
            const FloatPair noise_pair{a.noise_hi[c], a.noise_lo[c]};
            const bool pairs = noise_pair.hi > 1e-20f && noise_pair.hi < 1e30f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (pairs && fabsf(r[j]) < 1e15f) {
                    v[j] = pair_wiener(v[j], r[j], noise_pair, a.wiener_iterations);
                } else {
                    const double ref = r[j];
                    const double r2 = ref * ref;
                    const double gain = noise == 0.0 ? 1.0 : r2 / (r2 + noise);
                    double value = v[j];
                    for (int it = 0; it < a.wiener_iterations; ++it) value *= gain;
                    v[j] = static_cast<float>(value);
                }
            }
        } else {
            const float threshold = a.threshold[c];
            const bool structural = kk >= max(0, Q - 2);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (fabsf(v[j]) < threshold || (j > 0 && structural)) v[j] = 0.f;
            }
        }
        haar_columns(v, n, true);
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = lane_inverse<Q>(v[j], k);
    }
}

// Lane filter (group 16, q = Q; nlh_filter_lanes): one block per group, one
// thread per (row, slot). A thread holds the slot's 16 values in registers
// and the q slots of a row are neighbouring lanes, so the kernel has no
// lists, no local storage and no shared matrices. The lanes add their
// values to the group's sums in shared memory (32-bit fixed-point cells,
// [patch][pixel]: exact, so no order to keep), and one atomic per pixel and
// patch goes to the accumulators; the selection counts are added by
// prepare_kernel.
// A pixel is selected at most once per row, so m values below `limit` at
// `scale` units stay inside a cell (nlh_lane_scale); a larger value goes to
// the accumulators directly. The launch bound (1024 threads for block 16)
// holds the kernel to 64 registers, which it fits without spilling.
template <bool Wiener, int Q>
__global__ void __launch_bounds__(1024, 1) filter_lane_kernel(NlhGroupArgs a, float limit, float scale) {
    extern __shared__ int sums[];
    const int m = a.block * a.block;
    const int g = blockIdx.x, k = threadIdx.x % Q, row = threadIdx.x / Q;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    const int pixel = a.indices[(static_cast<long long>(g) * m + row) * Q + k];
    const int offset = (pixel / a.block) * a.width + pixel % a.block;
    const long long unit = static_cast<long long>(kFixedScale / scale);  // accumulator units per shared unit
    for (int i = threadIdx.x; i < m * 16; i += blockDim.x) sums[i] = 0;
    __syncthreads();
    for (int c = 0; c < a.nch; ++c) {
        float v[16];
        lane_values<Wiener, Q>(a, match, n, offset, k, c, v);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j >= n) break;
            if (fabsf(v[j]) < limit) {
                atomicAdd(sums + j * m + pixel, __float2int_rn(v[j] * scale));
            } else {
                const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x + offset;
                fixed_add(a.num[c], match[j].t * static_cast<long long>(a.slice_step) + at, v[j]);
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < m * 16; i += blockDim.x) {
            const int sum = sums[i];
            sums[i] = 0;
            const int j = i / m, p = i % m;
            if (sum == 0 || j >= n) continue;
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x +
                                 (p / a.block) * a.width + p % a.block;
            atomicAdd(a.num[c] + match[j].t * static_cast<long long>(a.slice_step) + at,
                      static_cast<unsigned long long>(sum * unit));
        }
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
        shrink_row(a, a.wiener, match, idx, n, c, matrix, ref);
        for (int i = 0; i < q * n; ++i) a.coef[base + i] = matrix[i];
    } else {
        shrink_row(a, a.wiener, match, idx, n, c, a.coef + base, a.wiener ? a.ref_coef + base : nullptr);
    }
}

// Gather of the global-memory variant: one block per group, one thread per
// pixel, the selections inverted in shared memory.
__global__ void gather_lists_kernel(NlhGroupArgs a) {
    extern __shared__ unsigned cursor[];
    const int m = a.block * a.block;
    const int q = a.q;
    const int stride = q * a.group;
    auto* list = reinterpret_cast<unsigned short*>(cursor + m);
    const int g = blockIdx.x, pixel = threadIdx.x;
    const int n = group_columns(a.counts[g], a.group, a.pow2);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    int begin, count;
    invert_selection(a.indices + (static_cast<long long>(g) * m + pixel) * q, m, q, list, cursor, begin, count);
    for (int c = 0; c < a.nch; ++c) {
        float num[64];
        gather_list(list + begin, a.identity[c] ? 0 : count,
                    a.coef + (static_cast<long long>(g) * a.nch + c) * m * stride, stride, q, n, num);
        store_pixel(a, match, c, n, pixel, count, num);
    }
}

constexpr int kEstimateBlock = 8, kEstimateRows = 64, kEstimateGroup = 16;

// Stages rows[row][column] of a group of the estimate: thread r loads pixel r
// of every patch (the threads of a warp read consecutive words).
__device__ __forceinline__ void estimate_stage(const NlhGroupArgs& a, const float* const* planes, int stride, int c,
                                               const DeviceMatch* match, int n, float* rows) {
    const int r = threadIdx.x;
    const int offset = (r / kEstimateBlock) * a.width + r % kEstimateBlock;
#pragma unroll
    for (int j = 0; j < kEstimateGroup; ++j) {
        rows[r * kEstimateGroup + j] =
            j < n ? planes[match[j].t * stride + c][static_cast<long long>(match[j].y) * a.width + match[j].x + offset]
                  : 0.f;
    }
}

// Blind noise estimate of a batch (block 8, group 16, q 4): one block per
// group, one thread per pixel row. The guide group is staged in shared
// memory, every row selects its three nearest rows (nss::pixel_match without
// the row itself), and per channel the group's statistic is the mean over
// rows and partners of sqrt(sum_j (row_j - partner_j)^2 / n). The CPU sums
// in FP64; here differences, squares and sums are float pairs (FP64 is a
// small fraction of the FP32 rate on consumer devices) and one FP64 value
// per group and channel leaves the kernel.
__global__ void estimate_kernel(NlhGroupArgs a, double* partial) {
    __shared__ float rows[kEstimateRows * kEstimateGroup];
    __shared__ float row_hi[kEstimateRows], row_lo[kEstimateRows];
    const int g = blockIdx.x, r = threadIdx.x;
    const int n = group_columns(a.counts[g], kEstimateGroup, false);
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * kEstimateGroup;
    estimate_stage(a, a.guides, 1, 0, match, n, rows);
    __syncthreads();

    // The three nearest rows by (distance, index); candidates arrive in
    // index order. Columns beyond n are zero in every row.
    constexpr float kInf = 1.0e30f;
    float mine[kEstimateGroup];
#pragma unroll
    for (int c = 0; c < kEstimateGroup; ++c) mine[c] = rows[r * kEstimateGroup + c];
    float d1 = kInf, d2 = kInf, d3 = kInf;
    int i1 = r, i2 = r, i3 = r;
    for (int s = 0; s < kEstimateRows; ++s) {
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < kEstimateGroup; ++c) {
            const float e = rows[s * kEstimateGroup + c] - mine[c];
            d = fmaf(e, e, d);
        }
        const bool other = s != r;
        const bool l1 = other && (d < d1 || (d == d1 && s < i1));
        const bool l2 = other && (d < d2 || (d == d2 && s < i2));
        const bool l3 = other && (d < d3 || (d == d3 && s < i3));
        d3 = l2 ? d2 : (l3 ? d : d3);
        i3 = l2 ? i2 : (l3 ? s : i3);
        d2 = l1 ? d1 : (l2 ? d : d2);
        i2 = l1 ? i1 : (l2 ? s : i2);
        d1 = l1 ? d : d1;
        i1 = l1 ? s : i1;
    }
    const int partner[3]{i1, i2, i3};

    bool guide_staged = true;
    for (int c = 0; c < a.nch; ++c) {
        if (!(a.guided[c] && guide_staged)) {
            __syncthreads();
            estimate_stage(a, a.data, a.nch, c, match, n, rows);
            guide_staged = false;
            __syncthreads();
        }
        FloatPair sum{0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            FloatPair d2sum{0.f, 0.f};
#pragma unroll
            for (int j = 0; j < kEstimateGroup; ++j) {
                // (a - b)^2 as a pair: the difference with its error, the
                // square with its error (fmaf) and the cross term.
                const FloatPair d = two_sum(rows[r * kEstimateGroup + j], -rows[partner[k] * kEstimateGroup + j]);
                const float square = d.hi * d.hi;
                d2sum = pair_add(d2sum, {square, fmaf(d.hi, d.hi, -square) + 2.f * d.hi * d.lo});
            }
            sum = pair_add(sum, pair_root_mean(d2sum, static_cast<float>(n)));
        }
        row_hi[r] = sum.hi;
        row_lo[r] = sum.lo;
        __syncthreads();
        if (r == 0) {
            FloatPair total{0.f, 0.f};
            for (int i = 0; i < kEstimateRows; ++i) total = pair_add(total, {row_hi[i], row_lo[i]});
            partial[static_cast<long long>(g) * a.nch + c] =
                (static_cast<double>(total.hi) + total.lo) / (kEstimateRows * 3);
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

__global__ void finish_sums_kernel(const unsigned long long* num, const unsigned* den, float* out_num,
                                   float* out_den, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    constexpr float kUnit = 1.f / kFixedScale;
    out_num[i] = static_cast<float>(static_cast<long long>(num[i])) * kUnit;
    out_den[i] = static_cast<float>(den[i]);
}

unsigned blocks_for(long long items, int threads) { return static_cast<unsigned>((items + threads - 1) / threads); }

}  // namespace

void nlh_prepare_groups(const NlhGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block;
    if (nlh_prepare_shared(args.block, args.group)) {
        prepare_kernel<<<args.batch, m, m * args.group * sizeof(float), stream>>>(
            args, nlh_filter_lanes(args.block, args.group, args.q));
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    pack_kernel<<<blocks_for(static_cast<long long>(args.batch) * m, kThreads), kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
    pixel_match_kernel<<<blocks_for(static_cast<long long>(args.batch) * m, kThreads), kThreads, 0, stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_filter_groups(const NlhGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block;
    if (nlh_filter_lanes(args.block, args.group, args.q)) {
        const std::size_t bytes = static_cast<std::size_t>(m) * 16 * sizeof(int);
        const int threads = m * args.q;
        const float limit = nlh_lane_limit(args.block), scale = nlh_lane_scale(args.block);
        if (args.wiener && args.q == 4) {
            filter_lane_kernel<true, 4><<<args.batch, threads, bytes, stream>>>(args, limit, scale);
        } else if (args.wiener) {
            filter_lane_kernel<true, 2><<<args.batch, threads, bytes, stream>>>(args, limit, scale);
        } else if (args.q == 4) {
            filter_lane_kernel<false, 4><<<args.batch, threads, bytes, stream>>>(args, limit, scale);
        } else {
            filter_lane_kernel<false, 2><<<args.batch, threads, bytes, stream>>>(args, limit, scale);
        }
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    if (nlh_filter_fused(args.block, args.group, args.q)) {
        const std::size_t bytes = nlh_fused_bytes(args.block, args.group, args.q);
        const int rows = nlh_fused_rows(args.block, args.group, args.q);
        if (args.wiener) filter_fused_kernel<true><<<args.batch, m, bytes, stream>>>(args, rows);
        else filter_fused_kernel<false><<<args.batch, m, bytes, stream>>>(args, rows);
        NSS_CUDA_CHECK_LAUNCH();
        return;
    }
    const long long items = static_cast<long long>(args.batch) * args.nch * m;
    shrink_kernel<<<blocks_for(items, kThreads), kThreads, 0, stream>>>(args, nlh_local_matrix(args.group, args.q));
    NSS_CUDA_CHECK_LAUNCH();
    gather_lists_kernel<<<args.batch, m, nlh_lists_bytes(args.block, args.q), stream>>>(args);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_finish_sums(const unsigned long long* num, const unsigned* den, float* out_num, float* out_den,
                     std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    finish_sums_kernel<<<blocks_for(static_cast<long long>(count), 256), 256, 0, stream>>>(num, den, out_num, out_den,
                                                                                         count);
    NSS_CUDA_CHECK_LAUNCH();
}

void nlh_estimate_groups(const NlhGroupArgs& args, double* partial, double* sums, double* totals, cudaStream_t stream) {
    if (args.batch <= 0) return;
    estimate_kernel<<<args.batch, kEstimateRows, 0, stream>>>(args, partial);
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
