// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/lssc/kernels.hpp"
#include "cuda/runtime/error.hpp"


#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>

namespace nss_cuda {
namespace {

constexpr int kThreads = 256;

unsigned blocks_for(long long items, int threads = kThreads) {
    return static_cast<unsigned>((items + threads - 1) / threads);
}

// nss::normalize_atom.
__device__ void normalize_atom(float* d, int m) {
    float n2 = 0.f;
    for (int i = 0; i < m; ++i) n2 += d[i] * d[i];
    if (n2 < 1e-12f) {
        for (int i = 0; i < m; ++i) d[i] = i == 0 ? 1.f : 0.f;
        return;
    }
    const float inv = 1.f / sqrtf(n2);
    for (int i = 0; i < m; ++i) d[i] *= inv;
}

__global__ void pack_kernel(LsscWork w) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.m) return;
    const int j = static_cast<int>(id / w.m), i = static_cast<int>(id % w.m);
    w.patches[id] = w.plane[static_cast<long long>(w.grid.y(j) + i / w.block) * w.width + w.grid.x(j) + i % w.block];
}

__global__ void patch_atoms_kernel(LsscWork w, const int* sources, int first) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= w.atoms) return;
    float* atom = w.dict + a * w.m;
    if (a >= first) {
        const float* patch = w.patches + static_cast<long long>(sources[a - first]) * w.m;
        float mean = 0.f;
        for (int i = 0; i < w.m; ++i) mean += patch[i];
        mean /= static_cast<float>(w.m);
        for (int i = 0; i < w.m; ++i) atom[i] = patch[i] - mean;
        normalize_atom(atom, w.m);
    }
    normalize_atom(atom, w.m);
}

// K-SVD sample patches minus their means.
__global__ void sample_kernel(LsscWork w) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.samples) return;
    const float* patch = w.patches + (static_cast<long long>(id) * w.np / w.samples) * w.m;
    float mean = 0.f;
    for (int i = 0; i < w.m; ++i) mean += patch[i];
    mean /= static_cast<float>(w.m);
    for (int i = 0; i < w.m; ++i) w.sample_y[id * w.m + i] = patch[i] - mean;
}

// nss::lssc_omp (sparsity 8) in FP32 (the CPU takes the correlations,
// Cholesky solve and residual in FP64; FP32 measured the same against it on
// the references) for one sample per block: the atoms'
// correlations in parallel, the first strict maximum picked in atom order, the
// support's normal equations in parallel and their solve by one thread.
__global__ void __launch_bounds__(kThreads) omp_kernel(LsscWork w) {
    constexpr int kMax = 8;
    __shared__ float residual[256];
    __shared__ float magnitude[256];
    __shared__ unsigned char used[256];
    __shared__ int support[kMax];
    __shared__ float gram[kMax * kMax];
    __shared__ float rhs[kMax];
    __shared__ int state[2];  // support size, finished
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int j = blockIdx.x;
    const int m = w.m, atoms = w.atoms;
    const float* y = w.sample_y + j * m;
    float* a = w.sample_a + j * atoms;
    for (int k = lane; k < atoms; k += lanes) {
        used[k] = 0;
        a[k] = 0.f;
    }
    for (int row = lane; row < m; row += lanes) residual[row] = y[row];
    if (lane == 0) {
        state[0] = 0;
        state[1] = 0;
    }
    __syncthreads();
    const int kmax = min(kMax, atoms);
    for (int iteration = 0; iteration < kmax; ++iteration) {
        for (int k = lane; k < atoms; k += lanes) {
            float correlation = -1.f;
            if (!used[k]) {
                const float* column = w.dict + k * m;
                correlation = 0.f;
                for (int row = 0; row < m; ++row) correlation = fmaf(column[row], residual[row], correlation);
                correlation = fabsf(correlation);
            }
            magnitude[k] = correlation;
        }
        __syncthreads();
        if (lane == 0) {
            int best = -1;
            float largest = 0.f;
            for (int k = 0; k < atoms; ++k) {
                if (magnitude[k] > largest) {
                    largest = magnitude[k];
                    best = k;
                }
            }
            if (best < 0 || largest < 1e-12f) {
                state[1] = 1;
            } else {
                used[best] = 1;
                support[state[0]++] = best;
            }
        }
        __syncthreads();
        if (state[1]) break;
        const int count = state[0];
        for (int e = lane; e < count * count + count; e += lanes) {
            const float* dp = w.dict + support[e < count * count ? e / count : e - count * count] * m;
            float value = 0.f;
            if (e < count * count) {
                const float* dq = w.dict + support[e % count] * m;
                for (int row = 0; row < m; ++row) value = fmaf(dp[row], dq[row], value);
                gram[e] = value;
            } else {
                for (int row = 0; row < m; ++row) value = fmaf(dp[row], y[row], value);
                rhs[e - count * count] = value;
            }
        }
        __syncthreads();
        if (lane == 0) {
            // In-place Cholesky solve (nss solve_cholesky).
            for (int i = 0; i < count; ++i) {
                for (int q = 0; q <= i; ++q) {
                    float value = gram[i * count + q];
                    for (int p = 0; p < q; ++p) value = fmaf(-gram[i * count + p], gram[q * count + p], value);
                    gram[i * count + q] =
                        i == q ? sqrtf(fmaxf(value, 1e-12f)) : value / gram[q * count + q];
                }
            }
            for (int i = 0; i < count; ++i) {
                float value = rhs[i];
                for (int p = 0; p < i; ++p) value = fmaf(-gram[i * count + p], rhs[p], value);
                rhs[i] = value / gram[i * count + i];
            }
            for (int i = count - 1; i >= 0; --i) {
                float value = rhs[i];
                for (int p = i + 1; p < count; ++p) value = fmaf(-gram[p * count + i], rhs[p], value);
                rhs[i] = value / gram[i * count + i];
            }
            for (int p = 0; p < count; ++p) a[support[p]] = static_cast<float>(rhs[p]);
        }
        __syncthreads();
        for (int row = lane; row < m; row += lanes) {
            float value = y[row];
            for (int p = 0; p < count; ++p) value = fmaf(-rhs[p], w.dict[support[p] * m + row], value);
            residual[row] = value;
        }
        __syncthreads();
        if (lane == 0) {
            float norm_squared = 0.f;
            for (int row = 0; row < m; ++row) norm_squared = fmaf(residual[row], residual[row], norm_squared);
            if (norm_squared < 1e-12f) state[1] = 1;
        }
        __syncthreads();
        if (state[1]) break;
    }
}

// R = D A over the samples.
__global__ void sample_product_kernel(LsscWork w) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= w.samples * w.m) return;
    const int j = e / w.m, i = e % w.m;
    float sum = 0.f;
    for (int k = 0; k < w.atoms; ++k) sum = fmaf(w.dict[k * w.m + i], w.sample_a[j * w.atoms + k], sum);
    w.sample_r[e] = sum;
}

// The atom updates of one K-SVD round (nss ksvd_lite_workspace): each atom
// from the rank-one approximation of its restricted error matrix E (the first
// kLsscSupport samples that use it), whose top singular triplet comes from an
// FP32 power iteration on E^T E (the CPU takes it from an FP32 SVD of E).
//
// An atom reads and rewrites only the residuals of its support, so two atoms
// with disjoint supports commute: atom k waits only for the earlier atoms that
// share a sample with it, and a few blocks take the atoms in turn (block b the
// atoms b, b + kUpdateBlocks, ...), each atom with the arithmetic of the
// sequential loop, so the result is the same. A block waits only for lower
// atoms and takes its own atoms in ascending order, so the lowest unfinished
// atom can always run. Residuals written by another block are read through L2
// (__ldcg) after its done flag.
//
// Within an atom the support scan and the power iteration run on the first
// warp (a support has at most 32 samples): the scan by ballots over the
// samples in order, the iteration with one row a lane and the norm formed by
// every lane in the same order, so only __syncwarp separates the steps.
constexpr int kUpdateThreads = 256;
constexpr int kUpdateBlocks = 16;
constexpr int kPowerIterations = 400;
// The iteration stops when no component of the normalized vector moves by
// more than this (an FP32 eigenvalue cannot be stationary to the FP64 1e-14
// the earlier test used; the vector is what the atom is formed from).
constexpr float kPowerTolerance = 2e-6f;
constexpr int kScanChunks = kLsscSamples / 32;  // also the mask words of an atom
static_assert(kLsscSupport <= 32 && kUpdateThreads >= 32, "the support must fit the first warp");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// The first nsvd samples that use atom k, in sample order, for the calling
// warp: their indices and codes into support / previous and their bit mask
// into mask (each when not null). Returns their count.
__device__ int atom_users(const LsscWork& w, int k, int nsvd, int* support, float* previous, unsigned* mask) {
    const int lane = threadIdx.x & 31;
    float code[kScanChunks];
#pragma unroll
    for (int q = 0; q < kScanChunks; ++q) {
        const int j = q * 32 + lane;
        code[q] = j < w.samples ? w.sample_a[j * w.atoms + k] : 0.f;
    }
    int n = 0;
#pragma unroll
    for (int q = 0; q < kScanChunks; ++q) {
        const bool used = fabsf(code[q]) > 1e-8f;
        const unsigned users = __ballot_sync(0xffffffffu, used);
        const int slot = n + __popc(users & ((1u << lane) - 1u));
        const bool kept = used && slot < nsvd;
        if (support && kept) {
            support[slot] = q * 32 + lane;
            previous[slot] = code[q];
        }
        const unsigned kept_mask = __ballot_sync(0xffffffffu, kept);
        if (mask && lane == 0) mask[q] = kept_mask;
        n += __popc(users);
    }
    return min(n, nsvd);
}

// Support masks of every atom (one warp an atom) and cleared done flags.
__global__ void support_kernel(LsscWork w, unsigned* masks, int* done) {
    const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (k >= w.atoms) return;  // uniform over the warp
    atom_users(w, k, min(w.samples, kLsscSupport), nullptr, nullptr, masks + k * kScanChunks);
    if ((threadIdx.x & 31) == 0) done[k] = 0;
}

__global__ void __launch_bounds__(kUpdateThreads) ksvd_kernel(LsscWork w, const unsigned* masks, int* done) {
    extern __shared__ float scratch[];
    __shared__ float gram[kLsscSupport * kLsscSupport];
    __shared__ float vec[kLsscSupport];
    __shared__ float dominant;    // eigenvalue estimate
    __shared__ int support[kLsscSupport];
    __shared__ float previous[kLsscSupport];
    __shared__ int state[2];      // support size, update valid
    __shared__ float scalars[4];  // sign * s0, sign, 1 / norm (or 0 for the unit fallback)
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int m = w.m, atoms = w.atoms, ns = w.samples;
    const int stride = m + 1;                // E rows padded against bank conflicts in the Gram product
    float* errors = scratch;                 // E: column t at errors + t * stride
    float* old_atom = errors + kLsscSupport * stride;
    float* new_atom = old_atom + m;

    const int nsvd = min(ns, kLsscSupport);
    for (int k = blockIdx.x; k < atoms; k += gridDim.x) {
        float* atom = w.dict + k * m;
        // Every thread has left the previous atom before the first warp
        // rewrites the shared state.
        __syncthreads();
        if (lane < 32) {
            const int n = atom_users(w, k, nsvd, support, previous, nullptr);
            // Wait for the earlier atoms that share a sample.
            unsigned mine[kScanChunks];
#pragma unroll
            for (int q = 0; q < kScanChunks; ++q) mine[q] = masks[k * kScanChunks + q];
            for (int p = lane; p < k; p += 32) {
                unsigned common = 0;
#pragma unroll
                for (int q = 0; q < kScanChunks; ++q) common |= masks[p * kScanChunks + q] & mine[q];
                if (common) {
                    while (*static_cast<volatile const int*>(done + p) == 0) {
                    }
                }
            }
            __threadfence();
            if (lane == 0) state[0] = n;
        }
        __syncthreads();
        do {
            const int nsup = state[0];
            if (nsup < 1) break;  // uniform: every thread read the same count
            for (int i = lane; i < m; i += lanes) old_atom[i] = atom[i];
            for (int e = lane; e < nsup * m; e += lanes) {
                const int t = e / m, i = e % m, j = support[t];
                errors[t * stride + i] = w.sample_y[j * m + i] - __ldcg(w.sample_r + j * m + i) + atom[i] * previous[t];
            }
            __syncthreads();
            for (int e = lane; e < nsup * nsup; e += lanes) {
                const int t = e / nsup, u = e % nsup;
                if (u > t) continue;
                float sum = 0.f;
                for (int i = 0; i < m; ++i) sum = fmaf(errors[t * stride + i], errors[u * stride + i], sum);
                gram[t * nsup + u] = sum;
                gram[u * nsup + t] = sum;
            }
            __syncthreads();
            // Dominant eigenpair of E^T E by power iteration from the Gram
            // column of the largest diagonal entry.
            if (lane < 32) {
                int top = 0;
                for (int t = 1; t < nsup; ++t) {
                    if (gram[t * nsup + t] > gram[top * nsup + top]) top = t;
                }
                const bool valid = gram[top * nsup + top] > 0.f;
                float eigenvalue = 0.f;
                if (valid) {
                    const bool row = lane < nsup;
                    if (row) vec[lane] = gram[lane * nsup + top];
                    __syncwarp();
                    for (int iteration = 0; iteration < kPowerIterations; ++iteration) {
                        float sum = 0.f;
                        if (row) {
                            for (int u = 0; u < nsup; ++u) sum = fmaf(gram[lane * nsup + u], vec[u], sum);
                        }
                        const float norm = sqrtf(warp_sum(sum * sum));
                        eigenvalue = norm;
                        const float normalized = norm > 0.f ? sum / norm : 0.f;
                        // The first product starts from an unnormalized vector.
                        float change = row ? fabsf(normalized - vec[lane]) : 0.f;
#pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1) {
                            change = fmaxf(change, __shfl_xor_sync(0xffffffffu, change, offset));
                        }
                        __syncwarp();
                        if (row) vec[lane] = normalized;
                        __syncwarp();
                        if (iteration > 0 && change <= kPowerTolerance) break;  // uniform over the warp
                    }
                }
                if (lane == 0) {
                    state[1] = valid;
                    dominant = eigenvalue;
                }
            }
            __syncthreads();
            if (!state[1]) break;
            const float s0 = sqrtf(dominant);
            if (!(s0 > 0.f)) break;
            // Left singular vector u = E v / s0.
            for (int i = lane; i < m; i += lanes) {
                float sum = 0.f;
                for (int t = 0; t < nsup; ++t) sum = fmaf(errors[t * stride + i], vec[t], sum);
                new_atom[i] = sum / s0;
            }
            __syncthreads();
            if (lane < 32) {
                float ip = 0.f, n2 = 0.f;
                for (int i = lane; i < m; i += 32) {
                    ip = fmaf(old_atom[i], new_atom[i], ip);
                    n2 = fmaf(new_atom[i], new_atom[i], n2);
                }
                ip = warp_sum(ip);
                n2 = warp_sum(n2);
                if (lane == 0) {
                    const float sign = ip < 0.f ? -1.f : 1.f;
                    scalars[0] = sign * s0;
                    scalars[1] = sign;
                    scalars[2] = n2 < 1e-12f ? 0.f : 1.f / sqrtf(n2);
                }
            }
            __syncthreads();
            for (int i = lane; i < m; i += lanes) {
                atom[i] = scalars[2] == 0.f ? (i == 0 ? 1.f : 0.f) : scalars[1] * new_atom[i] * scalars[2];
            }
            __syncthreads();
            for (int e = lane; e < nsup * m; e += lanes) {
                const int t = e / m, i = e % m, j = support[t];
                const float updated = scalars[0] * vec[t];
                float* r = w.sample_r + j * m + i;
                *r = __ldcg(r) + (atom[i] * updated - old_atom[i] * previous[t]);
                if (i == 0) w.sample_a[j * atoms + k] = updated;
            }
        } while (false);
        // The atom's residuals and codes are written before its flag.
        __syncthreads();
        if (lane == 0) {
            __threadfence();
            atomicExch(done + k, 1);
        }
    }
}

// The coding dictionary (nss::lssc_omp_prepare): the pixel-major transpose,
// the Gram matrix (one fma chain over the pixels per entry) and the inverse
// atom norms (FP64 sums).
__global__ void prepare_kernel(LsscWork w) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pairs = static_cast<long long>(w.atoms) * w.atoms;
    if (id < pairs) {
        const int a = static_cast<int>(id / w.atoms), b = static_cast<int>(id % w.atoms);
        const float* da = w.dict + static_cast<long long>(a) * w.m;
        const float* db = w.dict + static_cast<long long>(b) * w.m;
        float acc = 0.f;
        for (int r = 0; r < w.m; ++r) acc = fmaf(da[r], db[r], acc);
        w.gram[id] = acc;
    }
    if (id < static_cast<long long>(w.atoms) * w.m) {
        const int a = static_cast<int>(id / w.m), r = static_cast<int>(id % w.m);
        w.transpose[static_cast<long long>(r) * w.atoms + a] = w.dict[id];
    }
    if (id < w.atoms) {
        const float* da = w.dict + id * w.m;
        double norm = 0.0;
        for (int r = 0; r < w.m; ++r) norm = fma(static_cast<double>(da[r]), static_cast<double>(da[r]), norm);
        w.inv_norm[id] = norm > 0.0 ? static_cast<float>(1.0 / sqrt(norm)) : 0.f;
    }
}

// One warp codes one patch (nss::lssc_omp_code, the same operations in the
// same order): atom a lives in lane a % 32, slot a / 32; support position t in
// lane t % 32, slot t / 32. The triangular solves run column by column, one
// shuffle per column: every entry still takes its terms in the CPU's order.
// The sums are the lane sums of nss/params/lssc.hpp.
constexpr int kCodeWarps = 1;
constexpr int kCodeWarpsPerSm = 16;  // resident coding warps an SM (their u scratch)
constexpr int kAtomSlots = 8;   // 256 atoms
constexpr int kSupportSlots = (nss::kLsscOmpMaxAtoms + 31) / 32;
constexpr int kPixelSlots = 8;  // 256 pixels
static_assert(nss::kLsscDefaultAtoms <= 32 * kAtomSlots, "atoms per lane");

__host__ __device__ inline int tri(int t) { return t * (t + 1) / 2; }

struct CodeShared {
    float* l;    // packed lower triangle of the factor, row t at tri(t)
    float* inv;  // its reciprocal diagonal
    float* z;
    float* x;
    int* support;
};

// The u rows stay in global memory: rows in shared memory (up to 4 of them)
// cost more in occupancy than they saved in L2 latency.
__host__ __device__ inline std::size_t code_warp_bytes(int cap) {
    const std::size_t c = static_cast<std::size_t>(cap);
    return (static_cast<std::size_t>(tri(cap)) + 3 * c) * sizeof(float) + c * sizeof(int);
}

template <class T>
__device__ __forceinline__ T lane_tree(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// x = L^-T z, column by column from the last.
__device__ void back_solve(const CodeShared& s, int k, int lane) {
    float r[kSupportSlots];
    for (int i = 0; i < kSupportSlots; ++i) {
        const int t = lane + 32 * i;
        r[i] = t < k ? s.z[t] : 0.f;
    }
    for (int q = k - 1; q >= 0; --q) {
        const int owner = q % 32, slot = q / 32;
        float xq = 0.f;
        for (int i = 0; i < kSupportSlots; ++i)
            if (i == slot && lane == owner) xq = r[i] * s.inv[q];
        xq = __shfl_sync(0xffffffffu, xq, owner);
        if (lane == owner) s.x[q] = xq;
        const float* lq = s.l + tri(q);
        for (int i = 0; i < kSupportSlots; ++i) {
            const int t = lane + 32 * i;
            if (t < q) r[i] = fmaf(-lq[t], xq, r[i]);
        }
    }
    __syncwarp();
}

// Centering (warp per patch): the mean and ||y||^2 as FP32 lane sums, y into
// values, a flag for patches with a value that is not finite (copied through).
constexpr int kCenterWarps = 8;
__global__ void __launch_bounds__(kCenterWarps * 32) center_kernel(LsscWork w) {
    const int lane = threadIdx.x % 32;
    const long long j = static_cast<long long>(blockIdx.x) * kCenterWarps + threadIdx.x / 32;
    if (j >= w.np) return;
    const int m = w.m;
    const float* patch = w.patches + j * m;
    float* y = w.values + j * m;
    if (lane == 0) w.meta[j] = AggregatePatch{w.grid.x(static_cast<int>(j)), w.grid.y(static_cast<int>(j)), 0, 1.f};
    float pix[kPixelSlots];
    bool finite = true;
    float acc = 0.f;
    for (int i = 0; i < kPixelSlots; ++i) {
        const int r = lane + 32 * i;
        pix[i] = r < m ? patch[r] : 0.f;
        if (r < m) {
            finite = finite && isfinite(pix[i]);
            acc = acc + pix[i];
        }
    }
    finite = __all_sync(0xffffffffu, finite);
    const float mean = lane_tree(acc) / static_cast<float>(m);
    acc = 0.f;
    for (int i = 0; i < kPixelSlots; ++i) {
        const int r = lane + 32 * i;
        if (r < m) {
            const float v = finite ? pix[i] - mean : 0.f;
            y[r] = v;
            acc = fmaf(v, v, acc);
        }
    }
    const float yy = lane_tree(acc);
    if (lane == 0) {
        w.mean[j] = mean;
        w.yy[j] = yy;
        w.flags[j] = finite ? 0 : 1;
    }
}

// alpha0 = D^T y for kAlphaTile patches of the batch per block: thread
// (ta, tp) of 64 x 4 takes the atoms ta + 64 u (u < 4) and the patches
// tp * 16 + v (v < 16) in registers; each entry one fma chain over the pixels
// in order (as the CPU). Shapes alpha_tiled_kernel does not take (m not a
// multiple of 4, or above 128).
constexpr int kAlphaTile = 64;
__global__ void __launch_bounds__(256) alpha_kernel(LsscWork w, int begin, int count, int tile) {
    extern __shared__ float tile_y[];  // tile * m
    const long long first = begin + static_cast<long long>(blockIdx.x) * tile;
    const int n = static_cast<int>(min(static_cast<long long>(tile), begin + static_cast<long long>(count) - first));
    const int m = w.m, atoms = w.atoms;
    for (int e = threadIdx.x; e < n * m; e += blockDim.x) tile_y[e] = w.values[first * m + e];
    for (int e = n * m + threadIdx.x; e < tile * m; e += blockDim.x) tile_y[e] = 0.f;
    __syncthreads();
    const int ta = threadIdx.x % 64, tp = threadIdx.x / 64;
    const int per = tile / 4;  // patches a thread (16, or 4 for m > 64)
    float acc[4][16];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 16; ++v) acc[u][v] = 0.f;
    for (int r = 0; r < m; ++r) {
        float d[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int a = ta + 64 * u;
            d[u] = a < atoms ? __ldg(w.transpose + static_cast<long long>(r) * atoms + a) : 0.f;
        }
#pragma unroll
        for (int v = 0; v < 16; ++v) {
            if (v >= per) break;
            const float y = tile_y[(tp * per + v) * m + r];
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[u][v] = fmaf(d[u], y, acc[u][v]);
        }
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int p = tp * per + v;
        if (v >= per || p >= n) break;
        float* out = w.alpha0 + (first - begin + p) * atoms;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int a = ta + 64 * u;
            if (a < atoms) out[a] = acc[u][v];
        }
    }
}

// alpha0 = D^T y register-tiled, for m a multiple of 4 up to 128: a block takes kAlphaTile patches and every
// atom (padded to 256), thread (lane, warp) the atoms 4 lane + i and 128 + 4 lane + i (i < 4) and the
// patches 4 warp + j and 32 + 4 warp + j (j < 4). D^T comes through shared memory kTiledRows pixels at a
// time (a float4 of atoms a load), y from the block's tile as float4 along the pixels: six shared loads per
// 64 fma where alpha_kernel issued twenty. Each entry is the same fma chain over the pixels in order, so the
// same bits as alpha_kernel.
constexpr int kTiledRows = 16, kTiledAtoms = 256;
constexpr std::size_t alpha_tiled_bytes(int m) {
    return (static_cast<std::size_t>(kAlphaTile) * m + kTiledRows * kTiledAtoms) * sizeof(float);
}
__global__ void __launch_bounds__(256) alpha_tiled_kernel(LsscWork w, int begin, int count) {
    extern __shared__ float4 alpha_shared[];
    const int m = w.m, atoms = w.atoms;
    float* const tile_y = reinterpret_cast<float*>(alpha_shared);  // kAlphaTile * m
    float* const dsh = tile_y + kAlphaTile * m;                     // kTiledRows * kTiledAtoms
    const float4* const dsh4 = reinterpret_cast<const float4*>(dsh);
    const long long first = begin + static_cast<long long>(blockIdx.x) * kAlphaTile;
    const int n = static_cast<int>(min(static_cast<long long>(kAlphaTile), begin + static_cast<long long>(count) - first));
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int e = tid; e < n * m; e += 256) tile_y[e] = w.values[first * m + e];
    for (int e = n * m + tid; e < kAlphaTile * m; e += 256) tile_y[e] = 0.f;
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < m; k0 += kTiledRows) {
        const int rows = min(kTiledRows, m - k0);
        __syncthreads();  // the tile is loaded; the chunk before is no longer read
        for (int e = tid; e < rows * kTiledAtoms; e += 256) {
            const int r = e / kTiledAtoms, a = e % kTiledAtoms;
            dsh[e] = a < atoms ? __ldg(w.transpose + static_cast<long long>(k0 + r) * atoms + a) : 0.f;
        }
        __syncthreads();
        for (int kk = 0; kk < rows; kk += 4) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float4 yv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    yv[j] = *reinterpret_cast<const float4*>(tile_y + (32 * h + 4 * warp + j) * m + k0 + kk);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float4 d0 = dsh4[(kk + q) * (kTiledAtoms / 4) + lane];
                    const float4 d1 = dsh4[(kk + q) * (kTiledAtoms / 4) + 32 + lane];
                    const float d[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float y = q == 0 ? yv[j].x : q == 1 ? yv[j].y : q == 2 ? yv[j].z : yv[j].w;
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[i][4 * h + j] = fmaf(d[i], y, acc[i][4 * h + j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int p = 32 * (j / 4) + 4 * warp + j % 4;
        if (p >= n) continue;
        float* out = w.alpha0 + (first - begin + p) * atoms;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int a = 128 * (i / 4) + 4 * lane + i % 4;
            if (a < atoms) out[a] = acc[i][j];
        }
    }
}

// Persistent warps: a warp takes patches of the batch from a counter and
// keeps the u rows of its current patch in its slice of w.ubuf.
__global__ void __launch_bounds__(kCodeWarps * 32) code_kernel(LsscWork w, float budget, int begin, int count,
                                                              int* counter) {
    extern __shared__ __align__(16) unsigned char code_shared[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int m = w.m, atoms = w.atoms, cap = w.cap;
    CodeShared s;
    s.l = reinterpret_cast<float*>(code_shared + warp * code_warp_bytes(cap));
    s.inv = s.l + tri(cap);
    s.z = s.inv + cap;
    s.x = s.z + cap;
    s.support = reinterpret_cast<int*>(s.x + cap);
    float* ubuf = w.ubuf + (static_cast<long long>(blockIdx.x) * kCodeWarps + warp) * cap * atoms;
    const auto urow = [&](int q) { return ubuf + static_cast<long long>(q) * atoms; };
    const bool full = atoms == 32 * kAtomSlots;
    for (;;) {
        int local = 0;
        if (lane == 0) local = atomicAdd(counter, 1);
        local = __shfl_sync(0xffffffffu, local, 0);
        if (local >= count) return;
        const long long j = begin + static_cast<long long>(local);
        const float* patch = w.patches + j * m;
        float* out = w.values + j * m;
        if (w.flags[j]) {
            for (int r = lane; r < m; r += 32) out[r] = patch[r];
            continue;
        }
        const float mean = w.mean[j];
        float energy = w.yy[j];
        float alpha[kAtomSlots], inv[kAtomSlots];
        unsigned available = 0;
        const float* a0 = w.alpha0 + static_cast<long long>(local) * atoms;
        for (int i = 0; i < kAtomSlots; ++i) {
            const int a = lane + 32 * i;
            alpha[i] = a < atoms ? a0[a] : 0.f;
            inv[i] = a < atoms ? w.inv_norm[a] : 0.f;
            if (a < atoms) available |= 1u << i;
        }
        // alpha0_best is read from the batch's alpha0 (alpha has moved on).
        int k = 0;
        while (energy > budget && k < cap) {
            // The largest score, the lowest atom on ties.
            float best_value = 0.f;
            int best = -1;
            for (int i = 0; i < kAtomSlots; ++i) {
                const float score = fabsf(alpha[i]) * inv[i] * ((available >> i) & 1u ? 1.f : 0.f);
                if (score > best_value) {
                    best_value = score;
                    best = lane + 32 * i;
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float v = __shfl_xor_sync(0xffffffffu, best_value, offset);
                const int b = __shfl_xor_sync(0xffffffffu, best, offset);
                if (v > best_value || (v == best_value && b >= 0 && (best < 0 || b < best))) {
                    best_value = v;
                    best = b;
                }
            }
            if (!(best_value > 0.f)) break;
            // Row k of the factor: L^-1 g, g_t = G[best][S_t], column by column;
            // lane t keeps its own L_kt (r0: t = lane, r1: t = lane + 32).
            const float* gb = w.gram + static_cast<long long>(best) * atoms;
            float r0 = lane < k ? gb[s.support[lane]] : 0.f;
            float r1 = lane + 32 < k ? gb[s.support[lane + 32]] : 0.f;
            for (int q = 0; q < k; ++q) {
                const bool low = q < 32;
                const int owner = q & 31;
                float wq = 0.f;
                if (lane == owner) {
                    wq = (low ? r0 : r1) * s.inv[q];
                    if (low) r0 = wq; else r1 = wq;
                }
                wq = __shfl_sync(0xffffffffu, wq, owner);
                if (lane > q && lane < k) r0 = fmaf(-s.l[tri(lane) + q], wq, r0);
                if (lane + 32 > q && lane + 32 < k) r1 = fmaf(-s.l[tri(lane + 32) + q], wq, r1);
            }
            float p = 0.f, v = 0.f;
            if (lane < k) {
                p = fmaf(r0, r0, p);
                v = fmaf(r0, s.z[lane], v);
            }
            if (lane + 32 < k) {
                p = fmaf(r1, r1, p);
                v = fmaf(r1, s.z[lane + 32], v);
            }
            const float gbb = gb[best];
            const float pivot = gbb - lane_tree(p);
            if (!(pivot > nss::kLsscOmpPivot * gbb)) break;
            const float a0b = __ldg(a0 + best);
            const float lkk = sqrtf(pivot);
            const float inv_kk = 1.f / lkk;
            const float zk = (a0b - lane_tree(v)) * inv_kk;
            float* lk = s.l + tri(k);
            if (lane < k) lk[lane] = r0;
            if (lane + 32 < k) lk[lane + 32] = r1;
            if (lane == 0) {
                lk[k] = lkk;
                s.inv[k] = inv_kk;
                s.z[k] = zk;
                s.support[k] = best;
            }
            energy = fmaf(-zk, zk, energy);
            if (lane == best % 32) available &= ~(1u << (best / 32));
            __syncwarp();
            // u_k = (G[best] - sum_q L_kq u_q) / L_kk and alpha -= z_k u_k,
            // unless this was the last atom.
            if (energy > budget && k + 1 < cap) {
                float u[kAtomSlots];
                if (full) {
#pragma unroll
                    for (int i = 0; i < kAtomSlots; ++i) u[i] = __ldg(gb + lane + 32 * i);
#pragma unroll 4
                    for (int q = 0; q < k; ++q) {
                        const float c = -lk[q];
                        const float* uq = urow(q) + lane;
#pragma unroll
                        for (int i = 0; i < kAtomSlots; ++i) u[i] = fmaf(c, uq[32 * i], u[i]);
                    }
                } else {
                    for (int i = 0; i < kAtomSlots; ++i) u[i] = lane + 32 * i < atoms ? gb[lane + 32 * i] : 0.f;
                    for (int q = 0; q < k; ++q) {
                        const float c = -lk[q];
                        const float* uq = urow(q) + lane;
                        for (int i = 0; i < kAtomSlots; ++i)
                            if (lane + 32 * i < atoms) u[i] = fmaf(c, uq[32 * i], u[i]);
                    }
                }
                float* uk = urow(k) + lane;
                for (int i = 0; i < kAtomSlots; ++i) {
                    if (lane + 32 * i >= atoms) continue;
                    u[i] = u[i] * inv_kk;
                    uk[32 * i] = u[i];
                    alpha[i] = fmaf(-zk, u[i], alpha[i]);
                }
                __syncwarp();
            }
            ++k;
        }
        if (k > 0) back_solve(s, k, lane);
        float rec[kPixelSlots];
        for (int i = 0; i < kPixelSlots; ++i) rec[i] = 0.f;
        if (m == 64) {
#pragma unroll 4
            for (int t = 0; t < k; ++t) {
                const float xt = s.x[t];
                const float* col = w.dict + static_cast<long long>(s.support[t]) * m + lane;
                rec[0] = fmaf(xt, __ldg(col), rec[0]);
                rec[1] = fmaf(xt, __ldg(col + 32), rec[1]);
            }
        } else {
            for (int t = 0; t < k; ++t) {
                const float xt = s.x[t];
                const float* col = w.dict + static_cast<long long>(s.support[t]) * m + lane;
                for (int i = 0; i < kPixelSlots; ++i)
                    if (lane + 32 * i < m) rec[i] = fmaf(xt, __ldg(col + 32 * i), rec[i]);
            }
        }
        bool valid = true;
        for (int i = 0; i < kPixelSlots; ++i) {
            if (lane + 32 * i >= m) continue;
            valid = valid && isfinite(rec[i]) && fabsf(rec[i]) <= 8.f;
            rec[i] = rec[i] + mean;
        }
        valid = __all_sync(0xffffffffu, valid);
        for (int i = 0; i < kPixelSlots; ++i) {
            const int r = lane + 32 * i;
            if (r < m) out[r] = valid ? rec[i] : patch[r];
        }
        __syncwarp();
    }
}

// out = (num + weight src) / (den + weight) (nss image term, then aggregate_finish).
__global__ void finish_kernel(const float* num, const float* den, const float* src, int width, int height, int pitch,
                              float weight, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const long long i = static_cast<long long>(y) * pitch + x;
    const float n = fmaf(weight, src[i], num[i]);
    const float d = den[i] + weight;
    out[i] = d > 1e-12f ? n / d : src[i];
}

}  // namespace

std::size_t lssc_work_bytes(int m, int np) {
    const std::size_t n = static_cast<std::size_t>(np), mm = static_cast<std::size_t>(m);
    const std::size_t atoms = np < 256 ? np : 256, samples = np < kLsscSamples ? np : kLsscSamples;
    // patches, values, dictionary + transpose, Gram, inverse norms, K-SVD
    // samples, means, the alpha0 batch; then ||y||^2 and flags.
    const std::size_t batch = std::min<std::size_t>(n, kLsscBatch);
    const std::size_t floats = 2 * n * mm + 2 * atoms * mm + atoms * atoms + atoms + 2 * samples * mm + atoms * samples + n +
                               batch * atoms + 2 * n;
    const std::size_t ints = atoms * (kLsscSamples / 32 + 1) + (atoms > mm ? atoms - mm : 0) + n;
    return floats * sizeof(float) + ints * sizeof(int) + n * sizeof(AggregatePatch) + 32 * 32;
}

void lssc_pack(const LsscWork& w, cudaStream_t stream) {
    pack_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_patch_atoms(const LsscWork& w, const int* sources, int first, cudaStream_t stream) {
    patch_atoms_kernel<<<blocks_for(w.atoms), kThreads, 0, stream>>>(w, sources, first);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_ksvd(const LsscWork& w, cudaStream_t stream) {
    sample_kernel<<<blocks_for(w.samples), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    omp_kernel<<<w.samples, kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    sample_product_kernel<<<blocks_for(static_cast<long long>(w.samples) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    unsigned* masks = reinterpret_cast<unsigned*>(w.ksvd_sync);
    int* done = w.ksvd_sync + w.atoms * kScanChunks;
    support_kernel<<<blocks_for(static_cast<long long>(w.atoms) * 32), kThreads, 0, stream>>>(w, masks, done);
    NSS_CUDA_CHECK_LAUNCH();
    const std::size_t shared = static_cast<std::size_t>(kLsscSupport * (w.m + 1) + 2 * w.m) * sizeof(float);
    ksvd_kernel<<<std::min(w.atoms, kUpdateBlocks), kUpdateThreads, shared, stream>>>(w, masks, done);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_prepare(const LsscWork& w, cudaStream_t stream) {
    const long long items = std::max<long long>(static_cast<long long>(w.atoms) * w.atoms, static_cast<long long>(w.atoms) * w.m);
    prepare_kernel<<<blocks_for(items), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_code(const LsscWork& w, float sigma, cudaStream_t stream) {
    const float budget = nss::lssc_omp_budget(w.m, sigma);
    center_kernel<<<blocks_for(w.np, kCenterWarps), kCenterWarps * 32, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    const bool tiled = w.m % 4 == 0 && alpha_tiled_bytes(w.m) <= 48 * 1024 && w.atoms <= kTiledAtoms;
    const int tile = w.m <= 64 ? kAlphaTile : 16;
    const std::size_t tile_bytes = static_cast<std::size_t>(tile) * w.m * sizeof(float);
    static std::atomic<std::size_t> opted{0};
    if (tile_bytes > 48 * 1024 && opted.load() < tile_bytes) {
        NSS_CUDA_CHECK(cudaFuncSetAttribute(alpha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tile_bytes)));
        opted.store(tile_bytes);
    }
    const std::size_t shared = kCodeWarps * code_warp_bytes(w.cap);
    const int warps = w.code_slots;
    NSS_CUDA_CHECK(cudaMemsetAsync(w.counters, 0, static_cast<std::size_t>((w.np + w.batch - 1) / w.batch) * sizeof(int), stream));
    for (int begin = 0, b = 0; begin < w.np; begin += w.batch, ++b) {
        const int count = std::min(w.batch, w.np - begin);
        if (tiled) {
            alpha_tiled_kernel<<<blocks_for(count, kAlphaTile), 256, alpha_tiled_bytes(w.m), stream>>>(w, begin, count);
        } else {
            alpha_kernel<<<blocks_for(count, tile), 256, tile_bytes, stream>>>(w, begin, count, tile);
        }
        NSS_CUDA_CHECK_LAUNCH();
        const int grid = std::min((count + kCodeWarps - 1) / kCodeWarps, warps / kCodeWarps);
        code_kernel<<<grid, kCodeWarps * 32, shared, stream>>>(w, budget, begin, count, w.counters + b);
        NSS_CUDA_CHECK_LAUNCH();
    }
}

int lssc_code_warps(int device) {
    // Per device: the resident blocks of code_kernel (registers, shared
    // memory) up to kCodeWarpsPerSm warps an SM.
    static std::atomic<int> cache[64];
    if (device < 0 || device >= 64) device = 0;
    int warps = cache[device].load();
    if (warps > 0) return warps;
    int current = 0, sms = 0, blocks = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) NSS_CUDA_CHECK(cudaSetDevice(device));
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const std::size_t shared = kCodeWarps * code_warp_bytes(nss::kLsscOmpMaxAtoms);
    NSS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, code_kernel, kCodeWarps * 32, shared));
    if (current != device) NSS_CUDA_CHECK(cudaSetDevice(current));
    warps = sms * std::max(1, std::min(blocks * kCodeWarps, kCodeWarpsPerSm));
    cache[device].store(warps);
    return warps;
}

int lssc_code_slots(int device, int np) { return std::max(1, std::min(lssc_code_warps(device), (np + 31) / 32)); }

std::size_t lssc_code_scratch_bytes(int slots, int m, int atoms) {
    return static_cast<std::size_t>(slots) * nss::lssc_omp_cap(m, atoms) * atoms * sizeof(float);
}

void lssc_finish(const float* num, const float* den, const float* src, int width, int height, int pitch, float weight,
                 float* out, cudaStream_t stream) {
    const dim3 threads(32, 8);
    const dim3 blocks((width + 31) / 32, (height + 7) / 8);
    finish_kernel<<<blocks, threads, 0, stream>>>(num, den, src, width, height, pitch, weight, out);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
