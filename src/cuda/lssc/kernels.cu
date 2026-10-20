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

__device__ __forceinline__ float ssd(const float* a, const float* b, int m) {
    float sum = 0.f;
    for (int i = 0; i < m; ++i) {
        const float e = a[i] - b[i];
        sum = fmaf(e, e, sum);
    }
    return sum;
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

__global__ void seed_kernel(LsscWork w) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.m) return;
    const int c = id / w.m, i = id % w.m;
    const long long source = static_cast<long long>(c) * w.np / w.clusters;
    w.centroids[id] = w.patches[source * w.m + i];
}

// Nearest centroid, the first on ties.
// state: changed, done (a round after one that changed nothing is skipped).
__global__ void assign_kernel(LsscWork w, bool first, int* state) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= w.np || state[1]) return;
    const float* patch = w.patches + static_cast<long long>(j) * w.m;
    int best = 0;
    float best_d = 0.f;
    for (int c = 0; c < w.clusters; ++c) {
        const float d = ssd(patch, w.centroids + c * w.m, w.m);
        if (c == 0 || d < best_d) {
            best_d = d;
            best = c;
        }
    }
    if (first || w.assign[j] != best) atomicOr(state, 1);
    w.assign[j] = best;
}

// Whether (d1, c1) wins over (d2, c2) in assign_kernel's scan over the
// clusters in order (first strict minimum): a NaN distance at cluster 0 keeps
// cluster 0, other NaN distances never win, equal distances go to the lower
// cluster.
__device__ __forceinline__ bool nearer(float d1, int c1, float d2, int c2) {
    const bool n1 = isnan(d1), n2 = isnan(d2);
    if (n1 && c1 == 0) return true;
    if (n2 && c2 == 0) return false;
    if (n1 != n2) return n2;
    if (d1 != d2) return d1 < d2;
    return c1 < c2;
}

// assign_kernel for m <= kAssignRows as a register tile: a block takes
// kAssignTile patches against every centroid, a thread 4 patches by 4
// clusters (patches 4 tp.., clusters 4 tc..), operands as 16-byte loads from
// shared memory (rows [pixel][patch or cluster], padded to 68 floats). Each
// distance is the same chain as ssd(); the nearest is chosen by nearer() over
// the thread's clusters and then over the 16 threads of a patch row. The
// block also counts its patches per cluster into w.counts (zeroed before).
constexpr int kAssignRows = 64;
constexpr int kAssignTile = 64;
constexpr int kAssignPitch = kAssignTile + 4;
constexpr int kClusters = 64;  // largest cluster count (nss::kLsscDefaultClusters)
static_assert(kClusters == kAssignTile && kThreads == 256, "16 x 16 threads, 4 x 4 pairs each");
__global__ void __launch_bounds__(kThreads) assign_tile_kernel(LsscWork w, bool first, int* state) {
    if (state[1]) return;
    __shared__ __align__(16) float cs[kAssignRows][kAssignPitch];  // centroids: cs[i][cluster]
    __shared__ __align__(16) float ps[kAssignRows][kAssignPitch];  // patches: ps[i][patch]
    __shared__ int histogram[kClusters];
    const int m = w.m, mshift = __ffs(m) - 1, clusters = w.clusters;
    const int base = blockIdx.x * kAssignTile;
    for (int e = threadIdx.x; e < clusters * m; e += blockDim.x) cs[e & (m - 1)][e >> mshift] = w.centroids[e];
    for (int e = threadIdx.x; e < kAssignTile * m; e += blockDim.x) {
        ps[e & (m - 1)][e >> mshift] = base + (e >> mshift) < w.np ? w.patches[static_cast<long long>(base) * m + e] : 0.f;
    }
    if (threadIdx.x < kClusters) histogram[threadIdx.x] = 0;
    __syncthreads();
    const int tp = threadIdx.x / 16, tc = threadIdx.x % 16;
    float d[4][4] = {};
    for (int i = 0; i < m; ++i) {
        const float4 x4 = *reinterpret_cast<const float4*>(&ps[i][4 * tp]);
        const float4 c4 = *reinterpret_cast<const float4*>(&cs[i][4 * tc]);
        const float x[4] = {x4.x, x4.y, x4.z, x4.w}, c[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float e = x[r] - c[q];
                d[r][q] = fmaf(e, e, d[r][q]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        int best = INT_MAX;  // no cluster yet: loses every comparison
        float best_d = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = 4 * tc + q;
            if (c < clusters && (best == INT_MAX || nearer(d[r][q], c, best_d, best))) {
                best_d = d[r][q];
                best = c;
            }
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const float other_d = __shfl_xor_sync(0xffffffffu, best_d, offset, 16);
            const int other = __shfl_xor_sync(0xffffffffu, best, offset, 16);
            if (other != INT_MAX && (best == INT_MAX || nearer(other_d, other, best_d, best))) {
                best_d = other_d;
                best = other;
            }
        }
        const int j = base + 4 * tp + r;
        if (tc == 0 && j < w.np) {
            if (first || w.assign[j] != best) atomicOr(state, 1);
            w.assign[j] = best;
            atomicAdd(&histogram[best], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x < clusters) {
        const int count = histogram[threadIdx.x];
        w.chunks[blockIdx.x * clusters + threadIdx.x] = count;
        if (count) atomicAdd(w.counts + threadIdx.x, count);
    }
}

// Member lists from the assignment tiles' counts (kAssignTile patches a
// chunk): for each cluster the members before each chunk, a warp a cluster
// scanning 32 chunks at a time (block `clusters`: the offsets) ...
__device__ __forceinline__ int warp_inclusive_scan(int v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(~0u, v, d);
        if (lane >= d) v += u;
    }
    return v;
}

__global__ void chunk_offsets_kernel(LsscWork w, const int* skip) {
    if (*skip) return;
    const int c = blockIdx.x, lane = threadIdx.x, clusters = w.clusters, chunks = (w.np + kAssignTile - 1) / kAssignTile;
    // Block `clusters` lays out the clusters, the others scan a cluster's chunks.
    const int n = c == clusters ? clusters : chunks;
    int before = 0;
    for (int base = 0; base < n; base += 32) {
        const int k = base + lane;
        int* at = c == clusters ? w.offsets + k : w.chunks + static_cast<long long>(k) * clusters + c;
        const int count = k < n ? (c == clusters ? w.counts[k] : *at) : 0;
        const int inclusive = warp_inclusive_scan(count);
        if (k < n) *at = before + inclusive - count;
        before += __shfl_sync(~0u, inclusive, 31);
    }
    if (c == clusters && lane == 0) w.offsets[clusters] = before;
}

// ... then every chunk's patches into place, in patch order within a cluster.
__global__ void __launch_bounds__(kAssignTile) chunk_members_kernel(LsscWork w, const int* skip) {
    __shared__ int cluster[kAssignTile];
    if (*skip) return;
    const int j = blockIdx.x * kAssignTile + threadIdx.x;
    const int c = j < w.np ? w.assign[j] : -1;
    cluster[threadIdx.x] = c;
    __syncthreads();
    if (c < 0) return;
    int rank = 0;
    for (int t = 0; t < threadIdx.x; ++t) rank += cluster[t] == c;
    w.members[w.offsets[c] + w.chunks[blockIdx.x * w.clusters + c] + rank] = j;
}

// Member counts, one block a cluster (nothing when *skip).
__global__ void count_kernel(LsscWork w, const int* skip) {
    __shared__ int total;
    if (skip && *skip) return;
    const int c = blockIdx.x;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    int count = 0;
    for (int j = threadIdx.x; j < w.np; j += blockDim.x) count += w.assign[j] == c;
    atomicAdd(&total, count);
    __syncthreads();
    if (threadIdx.x == 0) w.counts[c] = total;
}

// Member lists in (cluster, patch) order and their offsets, one block a
// cluster: each thread takes a run of patches, counts its members, and writes
// them after the members of the runs before it.
constexpr int kMemberThreads = 512;
__global__ void __launch_bounds__(kMemberThreads) members_kernel(LsscWork w, const int* skip) {
    __shared__ int warp_total[kMemberThreads / 32];
    __shared__ int start;
    if (skip && *skip) return;
    const int c = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int run = (w.np + kMemberThreads - 1) / kMemberThreads;
    const int first = threadIdx.x * run, last = min(first + run, w.np);
    if (threadIdx.x == 0) {
        int offset = 0;
        for (int b = 0; b < c; ++b) offset += w.counts[b];
        start = offset;
        w.offsets[c] = offset;
        if (c == w.clusters - 1) w.offsets[w.clusters] = offset + w.counts[c];
    }
    int count = 0;
    for (int j = first; j < last; ++j) count += w.assign[j] == c;
    int before = count;  // inclusive scan over the warp
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, before, offset);
        if (lane >= offset) before += other;
    }
    if (lane == 31) warp_total[warp] = before;
    __syncthreads();
    if (warp == 0) {
        int total = lane < kMemberThreads / 32 ? warp_total[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, total, offset);
            if (lane >= offset) total += other;
        }
        if (lane < kMemberThreads / 32) warp_total[lane] = total;
    }
    __syncthreads();
    int slot = start + before - count + (warp > 0 ? warp_total[warp - 1] : 0);
    for (int j = first; j < last; ++j) {
        if (w.assign[j] == c) w.members[slot++] = j;
    }
}

// One thread per (cluster, pixel): the member sum in ascending patch order
// and the mean.
__global__ void accumulate_kernel(LsscWork w, const int* skip) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.m || *skip) return;
    const int c = id / w.m, i = id % w.m;
    const int begin = w.offsets[c], end = w.offsets[c + 1];
    float sum = 0.f;
    // Eight loads issued ahead of their (ordered) additions.
    constexpr int kAhead = 8;
    int t = begin;
    for (; t + kAhead <= end; t += kAhead) {
        float value[kAhead];
#pragma unroll
        for (int u = 0; u < kAhead; ++u) value[u] = w.patches[static_cast<long long>(w.members[t + u]) * w.m + i];
#pragma unroll
        for (int u = 0; u < kAhead; ++u) sum += value[u];
    }
    for (; t < end; ++t) sum += w.patches[static_cast<long long>(w.members[t]) * w.m + i];
    const int count = end - begin;
    w.centroids[id] = count > 0 ? sum * (1.f / static_cast<float>(count)) : 0.f;
}

// The end of a k-means round (state: changed, done): every empty cluster, in
// order, takes the patch farthest from its centroid among clusters with more
// than one member (the first strict maximum over the patches,
// nss::lssc_cluster_farthest; patch c % np when there is none), then the round
// sets done when nothing changed after the first round. One block.
constexpr int kStealThreads = 1024;
__global__ void __launch_bounds__(kStealThreads) steal_kernel(LsscWork w, int round, int* state) {
    __shared__ float best_d[kStealThreads / 32];
    __shared__ int best_j[kStealThreads / 32];
    __shared__ int steal, changed;
    if (state[1]) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) changed = state[0];
    for (int c = 0; c < w.clusters; ++c) {
        __syncthreads();
        if (w.counts[c] > 0) continue;  // uniform: written by thread 0 before the barrier
        float d = -1.f;
        int j = -1;
        for (int t = threadIdx.x; t < w.np; t += blockDim.x) {
            const int a = w.assign[t];
            const float dt = w.counts[a] <= 1 ? -1.f : ssd(w.patches + static_cast<long long>(t) * w.m, w.centroids + a * w.m, w.m);
            if (dt > d) {
                d = dt;
                j = t;
            }
        }
        // The larger distance wins, the lower patch on ties (the first maximum).
        const auto merge = [&](float od, int oj) {
            if (oj >= 0 && (j < 0 || od > d || (od == d && oj < j))) {
                d = od;
                j = oj;
            }
        };
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float od = __shfl_xor_sync(0xffffffffu, d, offset);
            const int oj = __shfl_xor_sync(0xffffffffu, j, offset);
            merge(od, oj);
        }
        if (lane == 0) {
            best_d[warp] = d;
            best_j[warp] = j;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int v = 1; v < kStealThreads / 32; ++v) merge(best_d[v], best_j[v]);
            const int patch = j < 0 ? c % w.np : j;
            const int old = w.assign[patch];
            if (old != c && w.counts[old] > 0) --w.counts[old];
            w.assign[patch] = c;
            w.counts[c] = 1;
            steal = patch;
            changed = 1;
        }
        __syncthreads();
        for (int k = threadIdx.x; k < w.m; k += blockDim.x) {
            w.centroids[c * w.m + k] = w.patches[static_cast<long long>(steal) * w.m + k];
        }
    }
    if (threadIdx.x == 0) {
        if (!changed && round > 0) state[1] = 1;
        state[0] = 0;
    }
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

// nss::lssc_prepare_context's power iteration; work holds atoms + m floats.
// One block: D v one row a thread and D^T w one atom a thread, each sum in the
// sequential order; the two norms on thread 0 in index order.
constexpr int kLipschitzThreads = 256;
__global__ void __launch_bounds__(kLipschitzThreads) lipschitz_kernel(LsscWork w, float* work, float* out) {
    __shared__ float scalars[2];  // 1 / norm (0: stop), result
    float* pv = work;
    float* pw = work + w.atoms;
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int m = w.m, atoms = w.atoms;
    const float inva = 1.f / sqrtf(static_cast<float>(atoms));
    for (int a = lane; a < atoms; a += lanes) pv[a] = inva;
    if (lane == 0) scalars[1] = 1.f;
    __syncthreads();
    for (int pit = 0; pit < 8; ++pit) {
        for (int i = lane; i < m; i += lanes) {
            float sum = 0.f;
            for (int a = 0; a < atoms; ++a) sum = fmaf(w.dict[a * m + i], pv[a], sum);
            pw[i] = sum;
        }
        __syncthreads();
        if (lane == 0) {
            float w2 = 0.f;
            for (int i = 0; i < m; ++i) w2 += pw[i] * pw[i];
            scalars[1] = w2;
        }
        for (int a = lane; a < atoms; a += lanes) {
            float sum = 0.f;
            for (int i = 0; i < m; ++i) sum = fmaf(w.dict[a * m + i], pw[i], sum);
            pv[a] = sum;
        }
        __syncthreads();
        if (lane == 0) {
            float v2 = 0.f;
            for (int a = 0; a < atoms; ++a) v2 += pv[a] * pv[a];
            if (!(v2 > 1e-20f) || isinf(v2) || isnan(v2)) {
                scalars[1] = 1.f;
                scalars[0] = 0.f;
            } else {
                scalars[0] = 1.f / sqrtf(v2);
            }
        }
        __syncthreads();
        const float inv = scalars[0];
        if (inv == 0.f) break;  // uniform
        for (int a = lane; a < atoms; a += lanes) pv[a] *= inv;
        __syncthreads();
    }
    if (lane == 0) {
        const float lipschitz = scalars[1];
        *out = !(lipschitz > 1e-6f) || isinf(lipschitz) || isnan(lipschitz) ? 1.f : lipschitz;
    }
}

// The coding preconditioner, nss::lssc_prepare_context's in its order
// (src/cpu/lssc/reconstruct.cpp): S = D D^T (lower triangle, an entry a
// thread, one fma chain over ascending atoms), a right-looking Cholesky of
// S + delta I in place (one block; every entry's updates in ascending column
// order), then per atom the forward and back solves (a thread an atom, each
// sum one fma chain in ascending index order). w.factor + m * m holds the
// state: usable (1 or 0), delta, c. The lower triangle of w.factor is the
// factor after factor_kernel (whose shared copy is written back).
__global__ void gram_dict_kernel(LsscWork w) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = w.m;
    if (id >= m * m) return;
    const int i = id / m, j = id % m;
    if (j > i) return;
    float sum = 0.f;
    for (int a = 0; a < w.atoms; ++a) sum = fmaf(w.dict[a * m + i], w.dict[a * m + j], sum);
    w.factor[id] = sum;
}

// The factor (and the solves) work on the matrix in shared memory when it
// fits (m <= kFactorShared), in w.factor otherwise; same operations and order.
constexpr int kFactorThreads = 256;
constexpr int kFactorShared = 64;
constexpr int kFactorPitch = kFactorShared + 1;
template <bool Shared>
__global__ void __launch_bounds__(kFactorThreads) factor_kernel(LsscWork w, float* lipschitz) {
    __shared__ float tile[Shared ? kFactorShared * kFactorPitch : 1];
    __shared__ float root;
    __shared__ int usable;
    const int m = w.m;
    const int pitch = Shared ? kFactorPitch : m;
    float* f = Shared ? tile : w.factor;
    float* state = w.factor + m * m;
    if (Shared) {
        for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
            const int i = e / m, j = e % m;
            if (j <= i) tile[i * pitch + j] = w.factor[e];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float trace = 0.f;
        for (int i = 0; i < m; ++i) trace += f[i * pitch + i];
        const float mean = trace / static_cast<float>(m);
        usable = mean > 1e-12f && isfinite(mean);
        state[1] = nss::kLsscPrecondAlpha * mean;
        state[2] = mean + state[1];
    }
    __syncthreads();
    const float delta = state[1];
    if (usable) {
        for (int i = threadIdx.x; i < m; i += blockDim.x) f[i * pitch + i] += delta;
    }
    __syncthreads();
    for (int k = 0; k < m && usable; ++k) {
        if (threadIdx.x == 0) {
            const float pivot = f[k * pitch + k];
            if (!(pivot > 0.f) || !isfinite(pivot)) {
                usable = 0;
            } else {
                root = sqrtf(pivot);
                f[k * pitch + k] = root;
            }
        }
        __syncthreads();
        if (!usable) break;  // uniform
        for (int i = k + 1 + threadIdx.x; i < m; i += blockDim.x) f[i * pitch + k] /= root;
        __syncthreads();
        // Row i of the trailing triangle on one thread: f[i][j] for k < j <= i.
        for (int i = k + 1 + threadIdx.x; i < m; i += blockDim.x) {
            const float fik = f[i * pitch + k];
            for (int j = k + 1; j <= i; ++j) f[i * pitch + j] = fmaf(-fik, f[j * pitch + k], f[i * pitch + j]);
        }
        __syncthreads();
    }
    if (Shared && usable) {
        for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
            const int i = e / m, j = e % m;
            if (j <= i) w.factor[e] = tile[i * pitch + j];
        }
    }
    if (threadIdx.x == 0) {
        state[0] = usable ? 1.f : 0.f;
        // L' = c L / (L + delta) (the step bound of the preconditioned problem).
        if (usable) *lipschitz = state[2] * *lipschitz / (*lipschitz + delta);
    }
}

// An atom a thread. Shared: kSolveAtoms atoms a block, the factor and the
// block's solution vectors (one column an atom) in shared memory.
constexpr int kSolveAtoms = 64;
template <bool Shared>
__global__ void __launch_bounds__(kSolveAtoms) solve_kernel(LsscWork w) {
    __shared__ float tile[Shared ? kFactorShared * kFactorPitch : 1];
    __shared__ float ys[Shared ? kFactorShared * kSolveAtoms : 1];
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = w.m;
    const float* state = w.factor + m * m;
    const bool usable = state[0] != 0.f;
    const int pitch = Shared ? kFactorPitch : m;
    if (Shared && usable) {
        for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
            const int i = e / m, j = e % m;
            if (j <= i) tile[i * pitch + j] = w.factor[e];
        }
        __syncthreads();
    }
    if (a >= w.atoms) return;
    const float* f = Shared ? tile : w.factor;
    const float* d = w.dict + a * m;
    float* out = w.gradient + a * m;
    if (!usable) {  // no usable factor: D^T and unit factors
        for (int i = 0; i < m; ++i) out[i] = d[i];
        w.lambda_scale[a] = 1.f;
        return;
    }
    float* y = Shared ? ys + threadIdx.x : out;
    const int ystride = Shared ? kSolveAtoms : 1;
    for (int i = 0; i < m; ++i) {
        float sum = d[i];
        for (int p = 0; p < i; ++p) sum = fmaf(-f[i * pitch + p], y[p * ystride], sum);
        y[i * ystride] = sum / f[i * pitch + i];
    }
    for (int i = m - 1; i >= 0; --i) {
        float sum = y[i * ystride];
        for (int p = i + 1; p < m; ++p) sum = fmaf(-f[p * pitch + i], y[p * ystride], sum);
        y[i * ystride] = sum / f[i * pitch + i];
    }
    const float c = state[2];
    float norm2 = 0.f;
    for (int i = 0; i < m; ++i) {
        const float v = c * y[i * ystride];
        out[i] = v;
        norm2 = fmaf(v, v, norm2);
    }
    w.lambda_scale[a] = sqrtf(norm2);
}

__global__ void center_kernel(LsscWork w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= w.np) return;
    float* patch = w.patches + static_cast<long long>(j) * w.m;
    float sum = 0.f;
    for (int i = 0; i < w.m; ++i) sum += patch[i];
    const float mean = sum * (1.f / static_cast<float>(w.m));
    w.mean[j] = mean;
    for (int i = 0; i < w.m; ++i) patch[i] -= mean;
}

// The coefficients are stored before their group soft threshold: a reader
// takes coef * scale of the patch's cluster (the value the threshold would
// have stored), and a stopped cluster's coefficients as zero. A round codes
// from the extrapolated point x + beta (x - x_last) of the last two rounds'
// thresholded values (the CPU's MulAdd(beta, x - previous, x)).
__device__ __forceinline__ float extrapolate(float x, float x_last, float beta) { return fmaf(beta, x - x_last, x); }

// values = Y - D Z (residual) or D A (final product, beta = 0), one thread
// per element.
__global__ void product_kernel(LsscWork w, bool residual, float beta) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.m) return;
    const int j = static_cast<int>(id / w.m), i = static_cast<int>(id % w.m);
    const int c = w.assign[j];
    const bool stopped = w.flags[w.clusters + c];
    if (residual && stopped) return;
    const float* a = w.coef + static_cast<long long>(j) * w.atoms;
    const float* a_last = w.coef_last + static_cast<long long>(j) * w.atoms;
    const float* scale = w.scale + c * w.atoms;
    const float* scale_last = w.scale_last + c * w.atoms;
    float sum = 0.f;
    for (int k = 0; k < w.atoms; ++k) {
        float z = 0.f;
        if (!stopped) {
            z = a[k] * scale[k];
            if (beta != 0.f) z = extrapolate(z, a_last[k] * scale_last[k], beta);
        }
        sum = fmaf(w.dict[k * w.m + i], z, sum);
    }
    w.values[id] = residual ? w.patches[id] - sum : sum;
}

// The coding products for m <= kTileRows as register tiles: a block takes
// kTile patches by kTile atoms (gradient) or pixels (product), a thread 4 x 4
// of them (patches tp + 16 r, columns tc + 16 q), operands staged in shared
// memory with padded rows. Every output is the same fmaf chain in ascending
// order as in gradient_kernel / product_kernel. The step size is
// mu = 0.9 / L' from the device's preconditioned Lipschitz bound.
constexpr int kTile = 64;
constexpr int kTileRows = 64;   // largest m of the tiled kernels
constexpr int kTileChunk = 32;  // atoms staged at a time by the product
constexpr int kTileSide = 16;   // threads along each side of a tile
static_assert(kTileSide * kTileSide == kThreads && kTileSide * 4 == kTile, "4 x 4 outputs a thread");

// Tiles of the coding rounds (m <= kTileRows): up to kTile members of one
// cluster each, cluster by cluster in member order. w.tiles holds the count,
// then the tiles' first member positions and clusters, then each cluster's
// first tile (clusters + 1 entries), then the largest magnitude of each
// (cluster, atom) row as float bits (an integer maximum); w.partial holds the
// tiles' partial row norms (tile x atoms).
struct LsscTiles {
    int* count;
    int* start;
    int* cluster;
    int* first;
    int* largest;
    float* norm2;
};
__device__ __forceinline__ LsscTiles lssc_tiles(const LsscWork& w) {
    const int capacity = lssc_tile_capacity(w.np, w.clusters);
    LsscTiles t;
    t.count = w.tiles;
    t.start = w.tiles + 1;
    t.cluster = t.start + capacity;
    t.first = t.cluster + capacity;
    t.largest = t.first + w.clusters + 1;
    t.norm2 = w.partial;
    return t;
}

// Whether cluster c is in the coding batch (by where its members start).
__device__ __forceinline__ bool in_part(const LsscWork& w, int c) {
    if (w.parts == 1) return true;
    const long long at = static_cast<long long>(w.offsets[c]) * w.parts;
    return at >= static_cast<long long>(w.part) * w.np && at < static_cast<long long>(w.part + 1) * w.np;
}

// The tile list from the member offsets (one block).
__global__ void tiles_kernel(LsscWork w) {
    const LsscTiles t = lssc_tiles(w);
    if (threadIdx.x == 0) {
        int total = 0;
        for (int c = 0; c < w.clusters; ++c) {
            t.first[c] = total;
            total += (w.offsets[c + 1] - w.offsets[c] + kTile - 1) / kTile;
        }
        t.first[w.clusters] = total;
        *t.count = total;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < w.clusters; c += blockDim.x) {
        for (int i = t.first[c], start = w.offsets[c]; i < t.first[c + 1]; ++i, start += kTile) {
            t.start[i] = start;
            t.cluster[i] = c;
        }
    }
    for (int e = threadIdx.x; e < w.clusters * w.atoms; e += blockDim.x) t.largest[e] = 0;
}

// scale_kernel from the tiles' partial norms (in tile order) and the row
// maxima; the maxima are cleared for the next round.
__global__ void scale_tiles_kernel(LsscWork w, const float* lipschitz, float sigma) {
    const float mu = 0.9f / *lipschitz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.atoms) return;
    const int c = id / w.atoms, k = id % w.atoms;
    if (w.flags[w.clusters + c] || !in_part(w, c)) return;
    const LsscTiles t = lssc_tiles(w);
    float norm2 = 0.f;
    for (int i = t.first[c]; i < t.first[c + 1]; ++i) norm2 += t.norm2[static_cast<long long>(i) * w.atoms + k];
    const float largest = __int_as_float(t.largest[id]);
    t.largest[id] = 0;
    const float lambda = mu * (fabsf(sigma) * sqrtf(static_cast<float>(w.offsets[c + 1] - w.offsets[c]))) *
                         w.lambda_scale[k];
    const float norm = sqrtf(norm2);
    const float scale = norm <= lambda ? 0.f : 1.f - lambda / norm;
    w.scale_last[id] = scale;
    if (largest * scale > 1.0e4f) atomicOr(w.flags + c, 1);
}


// One proximal-gradient round over a tile of kTile patches (round > 0:
// residual R = Y - D Z in registers, then through shared memory; round 0:
// R = Y), then A = mu P R + Z for every atom tile, written to coef_last
// (Momentum: Z extrapolated from coef and coef_last, stored over coef_last by
// the product as it stages it, so the gradient step reads one array;
// otherwise Z = coef).
// Each product output and each gradient output is the same fmaf chain, in the
// same order, as in product_kernel and gradient_kernel; the residual never
// leaves the block.
// Operands are staged a chunk at a time, the next chunk's loads issued into
// registers before the current chunk is computed.
constexpr int kRoundChunk = 16;               // atoms of a product chunk
constexpr int kRoundRows = 2 * kRoundChunk;   // dictionary rows of a gradient chunk
constexpr int kCodeLoads = kTile * kRoundChunk / kThreads;
constexpr int kRowLoads = kRoundChunk * kTileRows / kThreads;
constexpr int kGradientLoads = kTile * kRoundRows / kThreads;
constexpr int kRoundPitch = kTile + 4;  // a row of a shared tile: 16-byte aligned, 4 banks apart
constexpr int kRoundShift = 4;               // log2 kRoundChunk
static_assert(1 << kRoundShift == kRoundChunk, "");

// A kTile x n operand (rows contiguous in memory: row r, column i) staged
// transposed into shared rows [i][r] of kRoundPitch floats: a thread takes
// column i of four rows 4 q to 4 q + 3 and stores them as one 16-byte store
// (a warp's stores in the fewest bank cycles), for q = q0, q0 + step, ...
// below kTile / 4; a warp's loads of one row are consecutive columns.
struct Quads {
    int i, q0, step;
};
__device__ __forceinline__ Quads quad_lanes(int n, int nshift) {
    return {static_cast<int>(threadIdx.x) & (n - 1), static_cast<int>(threadIdx.x) >> nshift, kThreads >> nshift};
}

template <bool Momentum>
__global__ void __launch_bounds__(kThreads) round_tile_kernel(LsscWork w, bool first, float beta, const float* lipschitz) {
    const LsscTiles tiles = lssc_tiles(w);
    const int tile = blockIdx.x;
    if (tile >= *tiles.count) return;
    const int c = tiles.cluster[tile];
    if (w.flags[w.clusters + c] || !in_part(w, c)) return;  // stopped or another batch (uniform over the block)
    const float mu = 0.9f / *lipschitz;
    // A thread takes patches 4 tp to 4 tp + 3 and columns 4 tc to 4 tc + 3,
    // its operands as one 16-byte load each.
    __shared__ __align__(16) float rs[kTileRows][kRoundPitch];      // residual rows: rs[i][patch]
    __shared__ __align__(16) float stage[kRoundRows][kRoundPitch];  // codes and dictionary rows, or dictionary rows
    __shared__ int member[kTile];                     // the tile's patches; -1 past the end
    const int m = w.m, atoms = w.atoms;
    const int mshift = __ffs(m) - 1;                  // m is a power of two
    const int start = tiles.start[tile], size = min(kTile, w.offsets[c + 1] - start);
    const float* scales = w.scale + c * atoms;
    const float* scales_last = w.scale_last + c * atoms;
    if (threadIdx.x < kTile) member[threadIdx.x] = threadIdx.x < size ? w.members[start + threadIdx.x] : -1;
    __syncthreads();
    // A warp takes 4 patch quads by 8 column quads: its 16-byte operand loads
    // of a step are one bank cycle each.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tp = (warp >> 1) * 4 + (lane >> 3), tc = (warp & 1) * 8 + (lane & 7);
    {  // R = Y; the product is subtracted below
        const Quads y = quad_lanes(m, mshift);
        for (int q = y.q0; q < kTile / 4; q += y.step) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = member[4 * q + u];
                v[u] = j >= 0 ? w.patches[static_cast<long long>(j) * m + y.i] : 0.f;
            }
            *reinterpret_cast<float4*>(&rs[y.i][4 * q]) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
    if (!first) {
        float acc[4][4] = {};
        float(*codes)[kRoundPitch] = stage;
        float(*rows)[kRoundPitch] = stage + kRoundChunk;
        static_assert(kCodeLoads == 4 && kThreads == kTile / 4 * kRoundChunk, "a quad of codes a thread");
        float code[kCodeLoads], scale, row[kRowLoads];
        float code_last[Momentum ? kCodeLoads : 1], scale_last = 0.f;
        const Quads lanes = quad_lanes(kRoundChunk, kRoundShift);
        const auto fetch = [&](int kb) {
            const int chunk = min(kRoundChunk, atoms - kb), k = lanes.i;
            scale = k < chunk ? scales[kb + k] : 0.f;
            if (Momentum) scale_last = k < chunk ? scales_last[kb + k] : 0.f;
#pragma unroll
            for (int u = 0; u < kCodeLoads; ++u) {
                const int j = member[4 * lanes.q0 + u];
                const long long at = static_cast<long long>(j) * atoms + kb + k;
                code[u] = j >= 0 && k < chunk ? w.coef[at] : 0.f;
                if (Momentum) code_last[u] = j >= 0 && k < chunk ? w.coef_last[at] : 0.f;
            }
#pragma unroll
            for (int u = 0; u < kRowLoads; ++u) {
                const int e = threadIdx.x + u * kThreads;
                row[u] = e < chunk * m ? w.dict[kb * m + e] : 0.f;
            }
        };
        fetch(0);
        for (int kb = 0; kb < atoms; kb += kRoundChunk) {
            const int chunk = min(kRoundChunk, atoms - kb);
            __syncthreads();
            float z[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                z[u] = code[u] * scale;
                if (Momentum) {
                    z[u] = extrapolate(z[u], code_last[u] * scale_last, beta);
                    // Z replaces the codes of the round before (read above): the
                    // gradient step below adds to it.
                    const int j = member[4 * lanes.q0 + u];
                    if (j >= 0 && lanes.i < chunk) w.coef_last[static_cast<long long>(j) * atoms + kb + lanes.i] = z[u];
                }
            }
            *reinterpret_cast<float4*>(&codes[lanes.i][4 * lanes.q0]) = make_float4(z[0], z[1], z[2], z[3]);
#pragma unroll
            for (int u = 0; u < kRowLoads; ++u) {
                const int e = threadIdx.x + u * kThreads;
                if (e < chunk * m) rows[e >> mshift][e & (m - 1)] = row[u];
            }
            __syncthreads();
            if (kb + kRoundChunk < atoms) fetch(kb + kRoundChunk);
            for (int k = 0; k < chunk; ++k) {
                const float4 a4 = *reinterpret_cast<const float4*>(&codes[k][4 * tp]);
                const float4 d4 = *reinterpret_cast<const float4*>(&rows[k][4 * tc]);
                // Columns at or past m are formed from stale rows and never stored.
                const float a[4] = {a4.x, a4.y, a4.z, a4.w};
                const float d[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[rr][q] = fmaf(d[q], a[rr], acc[rr][q]);
                }
            }
        }
        bool live[4];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) live[rr] = member[4 * tp + rr] >= 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (4 * tc + q >= m) continue;
            float4* cell = reinterpret_cast<float4*>(&rs[4 * tc + q][4 * tp]);
            const float4 y = *cell;
            *cell = make_float4(live[0] ? y.x - acc[0][q] : 0.f, live[1] ? y.y - acc[1][q] : 0.f,
                                live[2] ? y.z - acc[2][q] : 0.f, live[3] ? y.w - acc[3][q] : 0.f);
        }
    }
    // Gradient: chunks (atom tile, rows) in order, kRoundRows rows (or m) each.
    const int count = min(kRoundRows, m), cshift = __ffs(count) - 1;
    const bool vector = atoms % 4 == 0;  // rows of codes and scales 16-byte aligned
    const int per_tile = m / count, chunks = (atoms + kTile - 1) / kTile * per_tile;
    static_assert(kGradientLoads % 4 == 0, "quads of dictionary rows");
    float next[kGradientLoads];
    const Quads lanes = quad_lanes(count, cshift);
    const auto fetch_rows = [&](int s) {
        const int abase = s / per_tile * kTile, ib = s % per_tile * count;
#pragma unroll
        for (int u = 0; u < kGradientLoads; ++u) {
            const int a = 4 * (lanes.q0 + u / 4 * lanes.step) + u % 4;
            next[u] = a < kTile && abase + a < atoms ? w.gradient[(abase + a) * m + ib + lanes.i] : 0.f;
        }
    };
    fetch_rows(0);
    float acc[4][4];
    for (int s = 0; s < chunks; ++s) {
        const int abase = s / per_tile * kTile;
        if (s % per_tile == 0) {
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[rr][q] = 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kGradientLoads; u += 4) {
            const int q = lanes.q0 + u / 4 * lanes.step;
            if (q < kTile / 4) {
                *reinterpret_cast<float4*>(&stage[lanes.i][4 * q]) =
                    make_float4(next[u], next[u + 1], next[u + 2], next[u + 3]);
            }
        }
        __syncthreads();
        if (s + 1 < chunks) fetch_rows(s + 1);
        const int ib = s % per_tile * count;
        for (int i = 0; i < count; ++i) {
            const float4 r4 = *reinterpret_cast<const float4*>(&rs[ib + i][4 * tp]);
            const float4 d4 = *reinterpret_cast<const float4*>(&stage[i][4 * tc]);
            const float r[4] = {r4.x, r4.y, r4.z, r4.w};
            const float d[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[rr][q] = fmaf(d[q], r[rr], acc[rr][q]);
            }
        }
        if (s % per_tile != per_tile - 1) continue;
        // The new codes of this atom tile, then their squares summed over the
        // thread's rows (in order) and over the 16 row groups (in order) into
        // the tile's partial row norms; the largest magnitude by an integer
        // maximum (order-free); a value that is not finite stops the cluster.
        float square[4] = {}, largest[4] = {};
        bool invalid = false;
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int j = member[4 * tp + rr];
            if (j < 0) continue;
            const long long at = static_cast<long long>(j) * atoms + abase + 4 * tc;
            const float* codes = w.coef + at;
            float* out = w.coef_last + at;  // Momentum: holds Z (written by the product above)
            const float* scale = scales + abase + 4 * tc;
            float value[4];
            if (vector) {
                if (abase + 4 * tc >= atoms) continue;
                float z[4] = {};
                if (Momentum) {
                    const float4 point = *reinterpret_cast<const float4*>(out);
                    z[0] = point.x;
                    z[1] = point.y;
                    z[2] = point.z;
                    z[3] = point.w;
                } else if (!first) {
                    const float4 code = *reinterpret_cast<const float4*>(codes);
                    const float4 factor = *reinterpret_cast<const float4*>(scale);
                    z[0] = code.x * factor.x;
                    z[1] = code.y * factor.y;
                    z[2] = code.z * factor.z;
                    z[3] = code.w * factor.w;
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) value[q] = fmaf(mu, acc[rr][q], z[q]);
                *reinterpret_cast<float4*>(out) = make_float4(value[0], value[1], value[2], value[3]);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    value[q] = 0.f;
                    if (abase + 4 * tc + q >= atoms) continue;
                    const float z = Momentum ? out[q] : first ? 0.f : codes[q] * scale[q];
                    out[q] = value[q] = fmaf(mu, acc[rr][q], z);
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                square[q] = fmaf(value[q], value[q], square[q]);
                largest[q] = fmaxf(largest[q], fabsf(value[q]));
                invalid = invalid || !isfinite(value[q]);
            }
        }
        if (invalid) atomicOr(w.flags + c, 1);
        __syncthreads();  // every thread is done with the staged rows
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            stage[tp][4 * tc + q] = square[q];
            stage[kTileSide + tp][4 * tc + q] = largest[q];
        }
        __syncthreads();
        if (threadIdx.x < kTile && abase + threadIdx.x < atoms) {
            float norm2 = 0.f, top = 0.f;
            for (int t = 0; t < kTileSide; ++t) {
                norm2 += stage[t][threadIdx.x];
                top = fmaxf(top, stage[kTileSide + t][threadIdx.x]);
            }
            tiles.norm2[static_cast<long long>(tile) * atoms + abase + threadIdx.x] = norm2;
            atomicMax(tiles.largest + c * atoms + abase + threadIdx.x, __float_as_int(top));
        }
    }
}

// round_tile_kernel<Momentum> with both products on mma.m16n8k8 TF32 tensor
// cores (sm_80+, m == 64, atoms % 64 == 0). Staging, momentum and order of
// the round are those of round_tile_kernel; the products round their operands
// to TF32 (FP32 accumulation), so the codes move from the CPU's (103 to 112 dB
// against 137 to 141 on the references) at the same quality: the round 1.10x
// with momentum, 1.29x without, LSSC 1080p 1.05x (docs/cuda-r2-gaps-20261019.md).
// Warps 4 (patches) x 2 (32 columns); a lane holds patches 16 wm + lane / 4
// (and + 8) at columns 32 wn + 8 t + 2 (lane % 4) + {0, 1}. On TF32 the rate is
// the FMA pipe's: the gain is fewer instructions and shared loads per product.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned round_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
#endif
__device__ __forceinline__ void round_mma(float (&c)[4], const float (&a)[4], const float (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(round_tf32(a[0])), "r"(round_tf32(a[1])), "r"(round_tf32(a[2])), "r"(round_tf32(a[3])),
                   "r"(round_tf32(b[0])), "r"(round_tf32(b[1])));
#else
    __trap();
#endif
}

template <bool Momentum>
__global__ void __launch_bounds__(kThreads) round_tile_mma_kernel(LsscWork w, bool first, float beta, const float* lipschitz) {
    const LsscTiles tiles = lssc_tiles(w);
    const int tile = blockIdx.x;
    if (tile >= *tiles.count) return;
    const int c = tiles.cluster[tile];
    if (w.flags[w.clusters + c] || !in_part(w, c)) return;
    const float mu = 0.9f / *lipschitz;
    __shared__ __align__(16) float rs[kTileRows][kRoundPitch];
    __shared__ __align__(16) float stage[kRoundRows][kRoundPitch];
    __shared__ int member[kTile];
    constexpr int m = 64;
    const int atoms = w.atoms;
    const int start = tiles.start[tile], size = min(kTile, w.offsets[c + 1] - start);
    const float* scales = w.scale + c * atoms;
    const float* scales_last = w.scale_last + c * atoms;
    if (threadIdx.x < kTile) member[threadIdx.x] = threadIdx.x < size ? w.members[start + threadIdx.x] : -1;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp >> 1, wn = warp & 1, lr = lane >> 2, lc = lane & 3;
    const int pm = wm * 16, pn = wn * 32;
    {
        const Quads y = quad_lanes(m, 6);
        for (int q = y.q0; q < kTile / 4; q += y.step) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int j = member[4 * q + u];
                v[u] = j >= 0 ? w.patches[static_cast<long long>(j) * m + y.i] : 0.f;
            }
            *reinterpret_cast<float4*>(&rs[y.i][4 * q]) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
    if (!first) {
        float acc[4][4] = {};
        float(*codes)[kRoundPitch] = stage;
        float(*rows)[kRoundPitch] = stage + kRoundChunk;
        float code[kCodeLoads], scale, row[kRowLoads];
        float code_last[Momentum ? kCodeLoads : 1], scale_last = 0.f;
        const Quads lanes = quad_lanes(kRoundChunk, kRoundShift);
        const auto fetch = [&](int kb) {
            const int k = lanes.i;
            scale = scales[kb + k];
            if (Momentum) scale_last = scales_last[kb + k];
#pragma unroll
            for (int u = 0; u < kCodeLoads; ++u) {
                const int j = member[4 * lanes.q0 + u];
                const long long at = static_cast<long long>(j) * atoms + kb + k;
                code[u] = j >= 0 ? w.coef[at] : 0.f;
                if (Momentum) code_last[u] = j >= 0 ? w.coef_last[at] : 0.f;
            }
#pragma unroll
            for (int u = 0; u < kRowLoads; ++u) row[u] = w.dict[kb * m + threadIdx.x + u * kThreads];
        };
        fetch(0);
        for (int kb = 0; kb < atoms; kb += kRoundChunk) {
            __syncthreads();
            float z[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                z[u] = code[u] * scale;
                if (Momentum) {
                    z[u] = extrapolate(z[u], code_last[u] * scale_last, beta);
                    const int j = member[4 * lanes.q0 + u];
                    if (j >= 0) w.coef_last[static_cast<long long>(j) * atoms + kb + lanes.i] = z[u];
                }
            }
            *reinterpret_cast<float4*>(&codes[lanes.i][4 * lanes.q0]) = make_float4(z[0], z[1], z[2], z[3]);
#pragma unroll
            for (int u = 0; u < kRowLoads; ++u) {
                const int e = threadIdx.x + u * kThreads;
                rows[e >> 6][e & 63] = row[u];
            }
            __syncthreads();
            if (kb + kRoundChunk < atoms) fetch(kb + kRoundChunk);
#pragma unroll
            for (int ks = 0; ks < kRoundChunk; ks += 8) {
                const float a[4] = {codes[ks + lc][pm + lr], codes[ks + lc][pm + lr + 8], codes[ks + lc + 4][pm + lr],
                                    codes[ks + lc + 4][pm + lr + 8]};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float b[2] = {rows[ks + lc][pn + 8 * t + lr], rows[ks + lc + 4][pn + 8 * t + lr]};
                    round_mma(acc[t], a, b);
                }
            }
        }
        const bool live0 = member[pm + lr] >= 0, live1 = member[pm + lr + 8] >= 0;
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int x = pn + 8 * t + 2 * lc;
            rs[x][pm + lr] = live0 ? rs[x][pm + lr] - acc[t][0] : 0.f;
            rs[x + 1][pm + lr] = live0 ? rs[x + 1][pm + lr] - acc[t][1] : 0.f;
            rs[x][pm + lr + 8] = live1 ? rs[x][pm + lr + 8] - acc[t][2] : 0.f;
            rs[x + 1][pm + lr + 8] = live1 ? rs[x + 1][pm + lr + 8] - acc[t][3] : 0.f;
        }
    }
    constexpr int count = kRoundRows, per_tile = m / count;
    const int chunks = atoms / kTile * per_tile;
    float next[kGradientLoads];
    const Quads lanes = quad_lanes(count, 5);
    const auto fetch_rows = [&](int s) {
        const int abase = s / per_tile * kTile, ib = s % per_tile * count;
#pragma unroll
        for (int u = 0; u < kGradientLoads; ++u) {
            const int a = 4 * (lanes.q0 + u / 4 * lanes.step) + u % 4;
            next[u] = a < kTile ? w.gradient[(abase + a) * m + ib + lanes.i] : 0.f;
        }
    };
    // Z of the lane's outputs of an atom tile (patches h, columns t, e).
    const auto load_z = [&](int abase, float (&zr)[2][4][2]) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = member[pm + lr + 8 * h];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int a = abase + pn + 8 * t + 2 * lc;
                float2 z = make_float2(0.f, 0.f);
                if (j >= 0) {
                    const long long at = static_cast<long long>(j) * atoms + a;
                    if (Momentum) {
                        z = *reinterpret_cast<const float2*>(w.coef_last + at);
                    } else if (!first) {
                        const float2 code = *reinterpret_cast<const float2*>(w.coef + at);
                        const float2 factor = *reinterpret_cast<const float2*>(scales + a);
                        z = make_float2(code.x * factor.x, code.y * factor.y);
                    }
                }
                zr[h][t][0] = z.x, zr[h][t][1] = z.y;
            }
        }
    };
    fetch_rows(0);
    float acc[4][4];
    float zr[2][4][2];
    for (int s = 0; s < chunks; ++s) {
        const int abase = s / per_tile * kTile;
        if (s % per_tile == 0) {
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kGradientLoads; u += 4) {
            const int q = lanes.q0 + u / 4 * lanes.step;
            if (q < kTile / 4) *reinterpret_cast<float4*>(&stage[lanes.i][4 * q]) = make_float4(next[u], next[u + 1], next[u + 2], next[u + 3]);
        }
        __syncthreads();
        if (s + 1 < chunks) fetch_rows(s + 1);
        const int ib = s % per_tile * count;
#pragma unroll
        for (int ks = 0; ks < count; ks += 8) {
            const float a[4] = {rs[ib + ks + lc][pm + lr], rs[ib + ks + lc][pm + lr + 8], rs[ib + ks + lc + 4][pm + lr],
                                rs[ib + ks + lc + 4][pm + lr + 8]};
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float b[2] = {stage[ks + lc][pn + 8 * t + lr], stage[ks + lc + 4][pn + 8 * t + lr]};
                round_mma(acc[t], a, b);
            }
        }
        if (s % per_tile != per_tile - 1) continue;
        load_z(abase, zr);
        float square[4][2] = {}, largest[4][2] = {};
        bool invalid = false;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = member[pm + lr + 8 * h];
            if (j < 0) continue;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int a = abase + pn + 8 * t + 2 * lc;
                const float v0 = fmaf(mu, acc[t][2 * h], zr[h][t][0]);
                const float v1 = fmaf(mu, acc[t][2 * h + 1], zr[h][t][1]);
                *reinterpret_cast<float2*>(w.coef_last + static_cast<long long>(j) * atoms + a) = make_float2(v0, v1);
                square[t][0] = fmaf(v0, v0, square[t][0]);
                square[t][1] = fmaf(v1, v1, square[t][1]);
                largest[t][0] = fmaxf(largest[t][0], fabsf(v0));
                largest[t][1] = fmaxf(largest[t][1], fabsf(v1));
                invalid = invalid || !isfinite(v0) || !isfinite(v1);
            }
        }
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    square[t][e] += __shfl_xor_sync(~0u, square[t][e], x);
                    largest[t][e] = fmaxf(largest[t][e], __shfl_xor_sync(~0u, largest[t][e], x));
                }
            }
        }
        if (invalid) atomicOr(w.flags + c, 1);
        __syncthreads();
        if (lr == 0) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    stage[wm][pn + 8 * t + 2 * lc + e] = square[t][e];
                    stage[4 + wm][pn + 8 * t + 2 * lc + e] = largest[t][e];
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < kTile) {
            float norm2 = 0.f, top = 0.f;
            for (int t = 0; t < 4; ++t) {
                norm2 += stage[t][threadIdx.x];
                top = fmaxf(top, stage[4 + t][threadIdx.x]);
            }
            tiles.norm2[static_cast<long long>(tile) * atoms + abase + threadIdx.x] = norm2;
            atomicMax(tiles.largest + c * atoms + abase + threadIdx.x, __float_as_int(top));
        }
    }
}

// values = Y - D A (residual: the patches of active clusters) or D A (final).
__global__ void __launch_bounds__(kThreads) product_tile_kernel(LsscWork w, bool residual) {
    __shared__ float as[kTileChunk][kTile + 1];  // codes: as[atom][patch]
    __shared__ float ds[kTileChunk][kTile + 1];  // dictionary: ds[atom][i]
    __shared__ int cluster[kTile];  // -1: stopped or past the end
    __shared__ int active[kTile];
    const int m = w.m, atoms = w.atoms;
    const int pbase = blockIdx.x * kTile;
    if (threadIdx.x < kTile) {
        const int j = pbase + threadIdx.x;
        const int c = j < w.np ? w.assign[j] : -1;
        const bool stopped = c >= 0 && w.flags[w.clusters + c];
        cluster[threadIdx.x] = c >= 0 && !stopped ? c : -1;
        active[threadIdx.x] = c >= 0 && !(residual && stopped);
    }
    const int tp = threadIdx.x / kTileSide, tc = threadIdx.x % kTileSide;
    float acc[4][4] = {};
    for (int kb = 0; kb < atoms; kb += kTileChunk) {
        const int chunk = min(kTileChunk, atoms - kb);
        __syncthreads();
        for (int e = threadIdx.x; e < kTile * kTileChunk; e += blockDim.x) {
            const int p = e / kTileChunk, k = e % kTileChunk, c = cluster[p];
            as[k][p] = c >= 0 && k < chunk
                           ? w.coef[static_cast<long long>(pbase + p) * atoms + kb + k] * w.scale[c * atoms + kb + k]
                           : 0.f;
        }
        for (int e = threadIdx.x; e < chunk * m; e += blockDim.x) ds[e / m][e % m] = w.dict[kb * m + e];
        __syncthreads();
        for (int k = 0; k < chunk; ++k) {
            float a[4], d[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                a[q] = as[k][tp + kTileSide * q];
                d[q] = tc + kTileSide * q < m ? ds[k][tc + kTileSide * q] : 0.f;
            }
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[rr][q] = fmaf(d[q], a[rr], acc[rr][q]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int p = tp + kTileSide * rr;
        if (!active[p]) continue;
        const long long row = static_cast<long long>(pbase + p) * m;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int i = tc + kTileSide * q;
            if (i < m) w.values[row + i] = residual ? w.patches[row + i] - acc[rr][q] : acc[rr][q];
        }
    }
}

// coef_last = mu P R + Z, an element a thread (Z as in product_kernel).
__global__ void gradient_kernel(LsscWork w, bool first, float beta, const float* lipschitz) {
    const float mu = 0.9f / *lipschitz;
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.atoms) return;
    const int j = static_cast<int>(id / w.atoms), k = static_cast<int>(id % w.atoms);
    const int c = w.assign[j];
    if (w.flags[w.clusters + c]) return;
    const float* r = (first ? w.patches : w.values) + static_cast<long long>(j) * w.m;
    float sum = 0.f;
    for (int i = 0; i < w.m; ++i) sum = fmaf(w.gradient[k * w.m + i], r[i], sum);
    float z = 0.f;
    if (!first) {
        z = w.coef[id] * w.scale[c * w.atoms + k];
        if (beta != 0.f) z = extrapolate(z, w.coef_last[id] * w.scale_last[c * w.atoms + k], beta);
    }
    w.coef_last[id] = fmaf(mu, sum, z);
}

// Group soft-threshold scale of every (cluster, atom) row of the new codes
// (coef_last): the row norm over the cluster's members in ascending patch
// order, the threshold at the atom's penalty factor.
__global__ void scale_kernel(LsscWork w, const float* lipschitz, float sigma) {
    const float mu = 0.9f / *lipschitz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.atoms) return;
    const int c = id / w.atoms, k = id % w.atoms;
    if (w.flags[w.clusters + c]) return;
    const int begin = w.offsets[c], end = w.offsets[c + 1];
    float norm2 = 0.f, largest = 0.f;
    bool invalid = false;
    for (int t = begin; t < end; ++t) {
        const float value = w.coef_last[static_cast<long long>(w.members[t]) * w.atoms + k];
        norm2 = fmaf(value, value, norm2);
        largest = fmaxf(largest, fabsf(value));
        invalid = invalid || !isfinite(value);
    }
    const float lambda = mu * (fabsf(sigma) * sqrtf(static_cast<float>(end - begin))) * w.lambda_scale[k];
    const float norm = sqrtf(norm2);
    const float scale = norm <= lambda ? 0.f : 1.f - lambda / norm;
    w.scale_last[id] = scale;
    // The thresholded row explodes when a value is not finite or above 1e4;
    // with finite values and a finite scale, |value * scale| is largest for
    // the largest |value| (rounding is monotonic).
    if (invalid || largest * scale > 1.0e4f) atomicOr(w.flags + c, 1);
}

// A cluster whose coefficients left the valid range stops iterating; its
// coefficients count as zero from then on.
__global__ void stop_kernel(LsscWork w) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < w.clusters && w.flags[c]) w.flags[w.clusters + c] = 1;
}

// values = D A + mean with the CPU's two fallbacks: an element outside the
// valid range becomes the mean, and a patch that still holds an invalid value
// is replaced by its source.
__global__ void finish_kernel(LsscWork w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= w.np) return;
    float* out = w.values + static_cast<long long>(j) * w.m;
    const float mean = w.mean[j];
    bool invalid = false;
    for (int i = 0; i < w.m; ++i) {
        float value = out[i] + mean;
        if (isnan(value) || isinf(value) || fabsf(out[i]) > 8.f) value = mean;
        out[i] = value;
        invalid = invalid || isnan(value) || isinf(value) || fabsf(value) > 8.f;
    }
    if (invalid) {
        for (int i = 0; i < w.m; ++i) {
            out[i] = w.plane[static_cast<long long>(w.grid.y(j) + i / w.block) * w.width + w.grid.x(j) + i % w.block];
        }
    }
    w.meta[j] = AggregatePatch{w.grid.x(j), w.grid.y(j), 0, 1.f};
}

}  // namespace

std::size_t lssc_work_bytes(int m, int np) {
    const std::size_t n = static_cast<std::size_t>(np), mm = static_cast<std::size_t>(m);
    const std::size_t atoms = np < 256 ? np : 256, clusters = np < 64 ? np : 64, samples = np < kLsscSamples ? np : kLsscSamples;
    const std::size_t floats = 2 * n * mm + 2 * n * atoms + n + atoms * mm + clusters * mm + 2 * clusters * atoms +
                               2 * samples * mm + atoms * samples + atoms + mm + atoms * mm + atoms +
                               lssc_precondition_floats(m);
    const std::size_t tiles = mm <= 64 ? static_cast<std::size_t>(lssc_tile_capacity(np, static_cast<int>(clusters))) : 0;
    const std::size_t ints = 2 * n + 3 * clusters + 1 + 4 + (atoms > mm ? atoms - mm : 0) + atoms * (kLsscSamples / 32 + 1) +
                             (mm <= 64 ? n + 64 : 0) +
                             (tiles ? 1 + 2 * tiles + clusters + 1 + clusters * atoms : 0) + tiles * atoms;
    return floats * sizeof(float) + ints * sizeof(int) + n * sizeof(AggregatePatch) + 32 * 32;
}

void lssc_pack(const LsscWork& w, cudaStream_t stream) {
    pack_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_cluster(const LsscWork& w, int rounds, int* state, cudaStream_t stream) {
    const auto members = [&](const int* skip) {
        count_kernel<<<w.clusters, kThreads, 0, stream>>>(w, skip);
        NSS_CUDA_CHECK_LAUNCH();
        members_kernel<<<w.clusters, kMemberThreads, 0, stream>>>(w, skip);
        NSS_CUDA_CHECK_LAUNCH();
    };
    if (w.clusters == 1) {
        NSS_CUDA_CHECK(cudaMemsetAsync(w.assign, 0, static_cast<std::size_t>(w.np) * sizeof(int), stream));
        members(nullptr);
        return;
    }
    NSS_CUDA_CHECK(cudaMemsetAsync(state, 0, 2 * sizeof(int), stream));
    seed_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    for (int round = 0; round < rounds; ++round) {
        if (w.m <= kAssignRows) {
            // The tile counts the members (a skipped round leaves zeros; the
            // counts after the rounds are taken afresh).
            NSS_CUDA_CHECK(cudaMemsetAsync(w.counts, 0, static_cast<std::size_t>(w.clusters) * sizeof(int), stream));
            assign_tile_kernel<<<blocks_for(w.np, kAssignTile), kThreads, 0, stream>>>(w, round == 0, state);
            NSS_CUDA_CHECK_LAUNCH();
            chunk_offsets_kernel<<<w.clusters + 1, 32, 0, stream>>>(w, state + 1);
            NSS_CUDA_CHECK_LAUNCH();
            chunk_members_kernel<<<blocks_for(w.np, kAssignTile), kAssignTile, 0, stream>>>(w, state + 1);
            NSS_CUDA_CHECK_LAUNCH();
        } else {
            assign_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w, round == 0, state);
            NSS_CUDA_CHECK_LAUNCH();
            count_kernel<<<w.clusters, kThreads, 0, stream>>>(w, state + 1);
            NSS_CUDA_CHECK_LAUNCH();
            members_kernel<<<w.clusters, kMemberThreads, 0, stream>>>(w, state + 1);
            NSS_CUDA_CHECK_LAUNCH();
        }
        accumulate_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.m), kThreads, 0, stream>>>(w, state + 1);
        NSS_CUDA_CHECK_LAUNCH();
        steal_kernel<<<1, kStealThreads, 0, stream>>>(w, round, state);
        NSS_CUDA_CHECK_LAUNCH();
    }
    members(nullptr);
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

void lssc_lipschitz(const LsscWork& w, float* work, float* lipschitz, cudaStream_t stream) {
    lipschitz_kernel<<<1, kLipschitzThreads, 0, stream>>>(w, work, lipschitz);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_precondition(const LsscWork& w, float* lipschitz, cudaStream_t stream) {
    gram_dict_kernel<<<blocks_for(static_cast<long long>(w.m) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    const int solve_blocks = (w.atoms + kSolveAtoms - 1) / kSolveAtoms;
    if (w.m <= kFactorShared) {
        factor_kernel<true><<<1, kFactorThreads, 0, stream>>>(w, lipschitz);
        NSS_CUDA_CHECK_LAUNCH();
        solve_kernel<true><<<solve_blocks, kSolveAtoms, 0, stream>>>(w);
    } else {
        factor_kernel<false><<<1, kFactorThreads, 0, stream>>>(w, lipschitz);
        NSS_CUDA_CHECK_LAUNCH();
        solve_kernel<false><<<solve_blocks, kSolveAtoms, 0, stream>>>(w);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_center(const LsscWork& w, cudaStream_t stream) {
    center_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

// TF32 mma needs compute capability 8.0; the answer is kept per device.
static bool tensor_cores() {
    int device = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&device));
    static std::array<std::atomic<int>, 64> known{};  // 0 unknown, 1 no, 2 yes
    if (device < 64 && known[device]) return known[device] == 2;
    int major = 0;
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (device < 64) known[device] = major >= 8 ? 2 : 1;
    return major >= 8;
}

int lssc_coding_parts(const LsscWork& w) {
    if (w.m > kTileRows || w.np <= 0) return 1;
    int device = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&device));
    const bool mma = w.m == 64 && w.atoms % kTile == 0 && tensor_cores();
    // Per device and round kernel: L2 bytes, and round blocks it holds at once.
    static std::array<std::atomic<long long>, 128> known{};  // 0 unknown, else l2 << 20 | resident
    const int slot = 2 * device + (mma ? 1 : 0);
    long long room = slot < 128 ? known[slot].load() : 0;
    if (room == 0) {
        int bytes = 0, sms = 0, blocks = 0;
        NSS_CUDA_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device));
        NSS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        if (mma) {
            NSS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, round_tile_mma_kernel<true>, kThreads, 0));
        } else {
            NSS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, round_tile_kernel<true>, kThreads, 0));
        }
        room = static_cast<long long>(bytes) << 20 | std::min(sms * blocks, (1 << 20) - 1);
        if (slot < 128) known[slot] = room;
    }
    const long long l2 = room >> 20;
    const int resident = static_cast<int>(room & ((1 << 20) - 1));
    // RTX 5080 (64 MiB L2, 252 round blocks): 1080p 2 batches gray 1.29x,
    // RGB 1.24x, YUV420 1.11x, 4K 8 batches 1.32x; 4 batches at 1080p, half
    // a wave of tiles each, only 1.02x (docs/cuda-rebench-20261020.md).
    const double codes = 2.0 * w.np * w.atoms * sizeof(float);
    const int parts = static_cast<int>(std::ceil(codes / (0.5 * static_cast<double>(l2))));
    // A batch must keep about a wave of tiles (7/8 of the resident blocks).
    if (l2 <= 0 || parts <= 1 || 8LL * ((w.np + kTile - 1) / kTile) < 7LL * resident * parts) return 1;
    return parts;
}

void lssc_round(const LsscWork& w, int round, float beta, const float* lipschitz, float sigma, cudaStream_t stream) {
    if (round == 0) beta = 0.f;  // nothing to extrapolate from
    if (w.m <= kTileRows) {
        if (round == 0 && w.part == 0) {
            tiles_kernel<<<1, kThreads, 0, stream>>>(w);
            NSS_CUDA_CHECK_LAUNCH();
        }
        const int tiles = lssc_tile_capacity(w.np, w.clusters);
        if (w.m == 64 && w.atoms % kTile == 0 && tensor_cores()) {
            if (beta != 0.f) {
                round_tile_mma_kernel<true><<<tiles, kThreads, 0, stream>>>(w, false, beta, lipschitz);
            } else {
                round_tile_mma_kernel<false><<<tiles, kThreads, 0, stream>>>(w, round == 0, 0.f, lipschitz);
            }
        } else if (beta != 0.f) {
            round_tile_kernel<true><<<tiles, kThreads, 0, stream>>>(w, false, beta, lipschitz);
        } else {
            round_tile_kernel<false><<<tiles, kThreads, 0, stream>>>(w, round == 0, 0.f, lipschitz);
        }
        NSS_CUDA_CHECK_LAUNCH();
        scale_tiles_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.atoms), kThreads, 0, stream>>>(w, lipschitz, sigma);
    } else {
        if (round > 0) {
            product_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w, true, beta);
            NSS_CUDA_CHECK_LAUNCH();
        }
        gradient_kernel<<<blocks_for(static_cast<long long>(w.np) * w.atoms), kThreads, 0, stream>>>(w, round == 0, beta, lipschitz);
        NSS_CUDA_CHECK_LAUNCH();
        scale_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.atoms), kThreads, 0, stream>>>(w, lipschitz, sigma);
    }
    NSS_CUDA_CHECK_LAUNCH();
    stop_kernel<<<blocks_for(w.clusters), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_reconstruct(const LsscWork& w, cudaStream_t stream) {
    if (w.m <= kTileRows) {
        product_tile_kernel<<<blocks_for(w.np, kTile), kThreads, 0, stream>>>(w, false);
    } else {
        product_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w, false, 0.f);
    }
    NSS_CUDA_CHECK_LAUNCH();
    finish_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
