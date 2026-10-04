// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/lssc/kernels.hpp"
#include "cuda/runtime/error.hpp"

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
__global__ void assign_kernel(LsscWork w, bool first, int* changed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= w.np) return;
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
    if (first || w.assign[j] != best) atomicOr(changed, 1);
    w.assign[j] = best;
}

// One thread per (cluster, pixel): the member sum in ascending patch order,
// the member count and the mean.
__global__ void accumulate_kernel(LsscWork w) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.m) return;
    const int c = id / w.m, i = id % w.m;
    float sum = 0.f;
    int count = 0;
    for (int j = 0; j < w.np; ++j) {
        if (w.assign[j] != c) continue;
        sum += w.patches[static_cast<long long>(j) * w.m + i];
        ++count;
    }
    w.centroids[id] = count > 0 ? sum * (1.f / static_cast<float>(count)) : 0.f;
    if (i == 0) w.counts[c] = count;
}

__global__ void count_kernel(LsscWork w) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= w.clusters) return;
    int count = 0;
    for (int j = 0; j < w.np; ++j) count += w.assign[j] == c;
    w.counts[c] = count;
}

__global__ void member_distance_kernel(LsscWork w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= w.np) return;
    const int c = w.assign[j];
    w.distance[j] = w.counts[c] <= 1
                        ? -1.f
                        : ssd(w.patches + static_cast<long long>(j) * w.m, w.centroids + c * w.m, w.m);
}

// First strict maximum of distance over all patches (nss::lssc_cluster_farthest).
__global__ void farthest_kernel(LsscWork w, int* result) {
    int steal = -1;
    float steal_d = -1.f;
    for (int j = 0; j < w.np; ++j) {
        if (w.distance[j] > steal_d) {
            steal_d = w.distance[j];
            steal = j;
        }
    }
    *result = steal;
}

__global__ void steal_kernel(LsscWork w, int cluster, int patch) {
    const int i = threadIdx.x;
    if (i == 0) {
        const int old = w.assign[patch];
        if (old != cluster && w.counts[old] > 0) --w.counts[old];
        w.assign[patch] = cluster;
        w.counts[cluster] = 1;
    }
    for (int k = i; k < w.m; k += blockDim.x) {
        w.centroids[cluster * w.m + k] = w.patches[static_cast<long long>(patch) * w.m + k];
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

// nss::lssc_omp (sparsity 8; FP64 correlations, Cholesky solve and residual)
// for one sample per block: the atoms' correlations in parallel, the first
// strict maximum picked in atom order, the support's normal equations in
// parallel and their solve by one thread.
__global__ void __launch_bounds__(kThreads) omp_kernel(LsscWork w) {
    constexpr int kMax = 8;
    __shared__ double residual[256];
    __shared__ double magnitude[256];
    __shared__ unsigned char used[256];
    __shared__ int support[kMax];
    __shared__ double gram[kMax * kMax];
    __shared__ double rhs[kMax];
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
            double correlation = -1.0;
            if (!used[k]) {
                const float* column = w.dict + k * m;
                correlation = 0.0;
                for (int row = 0; row < m; ++row) correlation = fma(static_cast<double>(column[row]), residual[row], correlation);
                correlation = fabs(correlation);
            }
            magnitude[k] = correlation;
        }
        __syncthreads();
        if (lane == 0) {
            int best = -1;
            double largest = 0.0;
            for (int k = 0; k < atoms; ++k) {
                if (magnitude[k] > largest) {
                    largest = magnitude[k];
                    best = k;
                }
            }
            if (best < 0 || largest < static_cast<double>(1e-12f)) {
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
            double value = 0.0;
            if (e < count * count) {
                const float* dq = w.dict + support[e % count] * m;
                for (int row = 0; row < m; ++row) value = fma(static_cast<double>(dp[row]), static_cast<double>(dq[row]), value);
                gram[e] = value;
            } else {
                for (int row = 0; row < m; ++row) value = fma(static_cast<double>(dp[row]), static_cast<double>(y[row]), value);
                rhs[e - count * count] = value;
            }
        }
        __syncthreads();
        if (lane == 0) {
            // In-place Cholesky solve (nss solve_cholesky).
            for (int i = 0; i < count; ++i) {
                for (int q = 0; q <= i; ++q) {
                    double value = gram[i * count + q];
                    for (int p = 0; p < q; ++p) value = fma(-gram[i * count + p], gram[q * count + p], value);
                    gram[i * count + q] =
                        i == q ? sqrt(fmax(value, static_cast<double>(1e-12f))) : value / gram[q * count + q];
                }
            }
            for (int i = 0; i < count; ++i) {
                double value = rhs[i];
                for (int p = 0; p < i; ++p) value = fma(-gram[i * count + p], rhs[p], value);
                rhs[i] = value / gram[i * count + i];
            }
            for (int i = count - 1; i >= 0; --i) {
                double value = rhs[i];
                for (int p = i + 1; p < count; ++p) value = fma(-gram[p * count + i], rhs[p], value);
                rhs[i] = value / gram[i * count + i];
            }
            for (int p = 0; p < count; ++p) a[support[p]] = static_cast<float>(rhs[p]);
        }
        __syncthreads();
        for (int row = lane; row < m; row += lanes) {
            double value = y[row];
            for (int p = 0; p < count; ++p) value = fma(-rhs[p], static_cast<double>(w.dict[support[p] * m + row]), value);
            residual[row] = value;
        }
        __syncthreads();
        if (lane == 0) {
            double norm_squared = 0.0;
            for (int row = 0; row < m; ++row) norm_squared = fma(residual[row], residual[row], norm_squared);
            if (norm_squared < static_cast<double>(1e-12f)) state[1] = 1;
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

// The atom updates of one K-SVD round (nss ksvd_lite_workspace) by a single
// small block: the atoms in order, each from the rank-one approximation of
// its restricted error matrix E. The top singular triplet of E comes from an
// FP64 power iteration on E^T E.
constexpr int kUpdateThreads = 64;
constexpr int kPowerIterations = 400;
__global__ void __launch_bounds__(kUpdateThreads) ksvd_kernel(LsscWork w) {
    __shared__ double gram[kLsscSupport * kLsscSupport];
    __shared__ double vec[kLsscSupport], next[kLsscSupport];
    __shared__ double dominant[2];  // eigenvalue estimate, 1 / norm
    __shared__ int support[kLsscSupport];
    __shared__ float previous[kLsscSupport];
    __shared__ int state[4];      // stationary, support size, seed column, update valid
    __shared__ float scalars[4];  // sign * s0, sign, 1 / norm (or 0 for the unit fallback)
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int m = w.m, atoms = w.atoms, ns = w.samples;
    float* errors = w.update;                 // E: column t at errors + t * m
    float* old_atom = errors + kLsscSupport * m;
    float* new_atom = old_atom + m;

    const int nsvd = min(ns, kLsscSupport);
    for (int k = 0; k < atoms; ++k) {
        float* atom = w.dict + k * m;
        // Every thread has left the previous atom (also through its early
        // exits) before thread 0 rewrites the shared state.
        __syncthreads();
        if (lane == 0) {
            int n = 0;
            for (int j = 0; j < ns && n < nsvd; ++j) {
                if (fabsf(w.sample_a[j * atoms + k]) > 1e-8f) {
                    support[n] = j;
                    previous[n] = w.sample_a[j * atoms + k];
                    ++n;
                }
            }
            state[1] = n;
        }
        __syncthreads();
        const int nsup = state[1];
        if (nsup < 1) continue;  // uniform: every thread read the same count
        for (int i = lane; i < m; i += lanes) old_atom[i] = atom[i];
        for (int e = lane; e < nsup * m; e += lanes) {
            const int t = e / m, i = e % m, j = support[t];
            errors[e] = w.sample_y[j * m + i] - w.sample_r[j * m + i] + atom[i] * previous[t];
        }
        __syncthreads();
        for (int e = lane; e < nsup * nsup; e += lanes) {
            const int t = e / nsup, u = e % nsup;
            if (u > t) continue;
            double sum = 0.0;
            for (int i = 0; i < m; ++i) sum = fma(static_cast<double>(errors[t * m + i]), static_cast<double>(errors[u * m + i]), sum);
            gram[t * nsup + u] = sum;
            gram[u * nsup + t] = sum;
        }
        __syncthreads();
        // Dominant eigenpair of E^T E by power iteration from the Gram column
        // of the largest diagonal entry, until the eigenvalue estimate is
        // stationary to FP64 precision.
        if (lane == 0) {
            int top = 0;
            for (int t = 1; t < nsup; ++t) {
                if (gram[t * nsup + t] > gram[top * nsup + top]) top = t;
            }
            state[2] = top;
            state[3] = gram[top * nsup + top] > 0.0;
            dominant[0] = 0.0;
        }
        __syncthreads();
        if (!state[3]) continue;
        for (int t = lane; t < nsup; t += lanes) vec[t] = gram[t * nsup + state[2]];
        __syncthreads();
        for (int iteration = 0; iteration < kPowerIterations; ++iteration) {
            for (int t = lane; t < nsup; t += lanes) {
                double sum = 0.0;
                for (int u = 0; u < nsup; ++u) sum = fma(gram[t * nsup + u], vec[u], sum);
                next[t] = sum;
            }
            __syncthreads();
            if (lane == 0) {
                double norm2 = 0.0;
                for (int t = 0; t < nsup; ++t) norm2 = fma(next[t], next[t], norm2);
                const double norm = sqrt(norm2);
                // The first product starts from an unnormalized vector.
                state[0] = iteration > 0 && fabs(norm - dominant[0]) <= 1e-14 * norm;
                dominant[0] = norm;
                dominant[1] = norm > 0.0 ? 1.0 / norm : 0.0;
            }
            __syncthreads();
            for (int t = lane; t < nsup; t += lanes) vec[t] = next[t] * dominant[1];
            const int stationary = state[0];
            __syncthreads();
            if (stationary) break;
        }
        const double s0 = sqrt(dominant[0]);
        if (!(s0 > 0.0)) continue;
        // Left singular vector u = E v / s0.
        for (int i = lane; i < m; i += lanes) {
            double sum = 0.0;
            for (int t = 0; t < nsup; ++t) sum = fma(static_cast<double>(errors[t * m + i]), vec[t], sum);
            new_atom[i] = static_cast<float>(sum / s0);
        }
        __syncthreads();
        if (lane == 0) {
            float ip = 0.f, n2 = 0.f;
            for (int i = 0; i < m; ++i) {
                ip += old_atom[i] * new_atom[i];
                n2 += new_atom[i] * new_atom[i];
            }
            const float sign = ip < 0.f ? -1.f : 1.f;
            scalars[0] = sign * static_cast<float>(s0);
            scalars[1] = sign;
            scalars[2] = n2 < 1e-12f ? 0.f : 1.f / sqrtf(n2);
        }
        __syncthreads();
        for (int i = lane; i < m; i += lanes) {
            atom[i] = scalars[2] == 0.f ? (i == 0 ? 1.f : 0.f) : scalars[1] * new_atom[i] * scalars[2];
        }
        __syncthreads();
        for (int e = lane; e < nsup * m; e += lanes) {
            const int t = e / m, i = e % m, j = support[t];
            const float updated = scalars[0] * static_cast<float>(vec[t]);
            w.sample_r[j * m + i] += atom[i] * updated - old_atom[i] * previous[t];
            if (i == 0) w.sample_a[j * atoms + k] = updated;
        }
        __syncthreads();
    }
}

// nss::lssc_prepare_context's power iteration; work holds atoms + m floats.
__global__ void lipschitz_kernel(LsscWork w, float* work, float* out) {
    float* pv = work;
    float* pw = work + w.atoms;
    const float inva = 1.f / sqrtf(static_cast<float>(w.atoms));
    for (int a = 0; a < w.atoms; ++a) pv[a] = inva;
    float lipschitz = 1.f;
    for (int pit = 0; pit < 8; ++pit) {
        float w2 = 0.f;
        for (int i = 0; i < w.m; ++i) {
            float sum = 0.f;
            for (int a = 0; a < w.atoms; ++a) sum = fmaf(w.dict[a * w.m + i], pv[a], sum);
            pw[i] = sum;
            w2 += sum * sum;
        }
        lipschitz = w2;
        float v2 = 0.f;
        for (int a = 0; a < w.atoms; ++a) {
            float sum = 0.f;
            for (int i = 0; i < w.m; ++i) sum = fmaf(w.dict[a * w.m + i], pw[i], sum);
            pv[a] = sum;
            v2 += sum * sum;
        }
        if (!(v2 > 1e-20f) || isinf(v2) || isnan(v2)) {
            lipschitz = 1.f;
            break;
        }
        const float inv = 1.f / sqrtf(v2);
        for (int a = 0; a < w.atoms; ++a) pv[a] *= inv;
    }
    if (!(lipschitz > 1e-6f) || isinf(lipschitz) || isnan(lipschitz)) lipschitz = 1.f;
    *out = lipschitz;
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
    float* a = w.coef + static_cast<long long>(j) * w.atoms;
    for (int k = 0; k < w.atoms; ++k) a[k] = 0.f;
}

// values = Y - D A (residual) or D A (final product), one thread per element.
__global__ void product_kernel(LsscWork w, bool residual) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.m) return;
    const int j = static_cast<int>(id / w.m), i = static_cast<int>(id % w.m);
    if (residual && w.flags[w.clusters + w.assign[j]]) return;
    const float* a = w.coef + static_cast<long long>(j) * w.atoms;
    float sum = 0.f;
    for (int k = 0; k < w.atoms; ++k) sum = fmaf(w.dict[k * w.m + i], a[k], sum);
    w.values[id] = residual ? w.patches[id] - sum : sum;
}

// A += mu D^T R (R = the centered patches in the first round), tiled: one
// block per kGradientTile patches with their residuals in shared memory, one
// thread per atom holding its dictionary column.
constexpr int kGradientTile = 16;
constexpr int kGradientRows = 64;  // largest m of the tiled kernel
__global__ void __launch_bounds__(kThreads) gradient_tile_kernel(LsscWork w, bool first, float mu) {
    __shared__ float residual[kGradientTile * kGradientRows];
    __shared__ int active[kGradientTile];
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int base = blockIdx.x * kGradientTile;
    const int m = w.m;
    const float* source = first ? w.patches : w.values;
    for (int e = lane; e < kGradientTile * m; e += lanes) {
        const int j = base + e / m;
        residual[e] = j < w.np ? source[static_cast<long long>(j) * m + e % m] : 0.f;
    }
    if (lane < kGradientTile) {
        const int j = base + lane;
        active[lane] = j < w.np && !w.flags[w.clusters + w.assign[j]];
    }
    __syncthreads();
    for (int k = lane; k < w.atoms; k += lanes) {
        float column[kGradientRows];
        for (int i = 0; i < m; ++i) column[i] = w.dict[k * m + i];
        for (int t = 0; t < kGradientTile; ++t) {
            if (!active[t]) continue;
            float sum = 0.f;
            for (int i = 0; i < m; ++i) sum = fmaf(column[i], residual[t * m + i], sum);
            float* code = w.coef + static_cast<long long>(base + t) * w.atoms + k;
            *code = fmaf(mu, sum, *code);
        }
    }
}

__global__ void gradient_kernel(LsscWork w, bool first, float mu) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.atoms) return;
    const int j = static_cast<int>(id / w.atoms), k = static_cast<int>(id % w.atoms);
    if (w.flags[w.clusters + w.assign[j]]) return;
    const float* r = (first ? w.patches : w.values) + static_cast<long long>(j) * w.m;
    float sum = 0.f;
    for (int i = 0; i < w.m; ++i) sum = fmaf(w.dict[k * w.m + i], r[i], sum);
    w.coef[id] = fmaf(mu, sum, w.coef[id]);
}

// Group soft-threshold scale of every (cluster, atom) row: the row norm over
// the cluster's members in ascending patch order.
__global__ void scale_kernel(LsscWork w, float mu, float sigma) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w.clusters * w.atoms) return;
    const int c = id / w.atoms, k = id % w.atoms;
    if (w.flags[w.clusters + c]) return;
    const int begin = w.offsets[c], end = w.offsets[c + 1];
    float norm2 = 0.f;
    for (int t = begin; t < end; ++t) {
        const float value = w.coef[static_cast<long long>(w.members[t]) * w.atoms + k];
        norm2 = fmaf(value, value, norm2);
    }
    const float lambda = mu * (fabsf(sigma) * sqrtf(static_cast<float>(end - begin)));
    const float norm = sqrtf(norm2);
    w.scale[id] = norm <= lambda ? 0.f : 1.f - lambda / norm;
}

__global__ void shrink_kernel(LsscWork w) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.atoms) return;
    const int j = static_cast<int>(id / w.atoms), k = static_cast<int>(id % w.atoms);
    const int c = w.assign[j];
    if (w.flags[w.clusters + c]) return;
    const float value = w.coef[id] * w.scale[c * w.atoms + k];
    w.coef[id] = value;
    if (isnan(value) || isinf(value) || fabsf(value) > 1.0e4f) atomicOr(w.flags + c, 1);
}

// A cluster whose coefficients left the valid range restarts from zero and
// stops iterating.
__global__ void explode_kernel(LsscWork w) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(w.np) * w.atoms) return;
    const int c = w.assign[static_cast<int>(id / w.atoms)];
    if (w.flags[c] && !w.flags[w.clusters + c]) w.coef[id] = 0.f;
}
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
    const std::size_t floats = 2 * n * mm + n * atoms + 2 * n + atoms * mm + clusters * mm + clusters * atoms +
                               2 * samples * mm + atoms * samples + (kLsscSupport + 2) * mm + atoms + mm;
    const std::size_t ints = 2 * n + 3 * clusters + 1 + 4 + (atoms > mm ? atoms - mm : 0);
    return floats * sizeof(float) + ints * sizeof(int) + n * sizeof(AggregatePatch) + 32 * 32;
}

void lssc_pack(const LsscWork& w, cudaStream_t stream) {
    pack_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_seed_centroids(const LsscWork& w, cudaStream_t stream) {
    seed_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_assign(const LsscWork& w, bool first, int* changed, cudaStream_t stream) {
    assign_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w, first, changed);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_accumulate(const LsscWork& w, cudaStream_t stream) {
    accumulate_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.m), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_farthest(const LsscWork& w, int* result, cudaStream_t stream) {
    member_distance_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    farthest_kernel<<<1, 1, 0, stream>>>(w, result);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_steal(const LsscWork& w, int cluster, int patch, cudaStream_t stream) {
    steal_kernel<<<1, kThreads, 0, stream>>>(w, cluster, patch);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_count(const LsscWork& w, cudaStream_t stream) {
    count_kernel<<<blocks_for(w.clusters), kThreads, 0, stream>>>(w);
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
    ksvd_kernel<<<1, kUpdateThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_lipschitz(const LsscWork& w, float* work, float* lipschitz, cudaStream_t stream) {
    lipschitz_kernel<<<1, 1, 0, stream>>>(w, work, lipschitz);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_center(const LsscWork& w, cudaStream_t stream) {
    center_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_round(const LsscWork& w, int round, float mu, float sigma, cudaStream_t stream) {
    const long long elements = static_cast<long long>(w.np) * w.m, codes = static_cast<long long>(w.np) * w.atoms;
    if (round > 0) {
        product_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(w, true);
        NSS_CUDA_CHECK_LAUNCH();
    }
    if (w.m <= kGradientRows) {
        gradient_tile_kernel<<<blocks_for(w.np, kGradientTile), kThreads, 0, stream>>>(w, round == 0, mu);
    } else {
        gradient_kernel<<<blocks_for(codes), kThreads, 0, stream>>>(w, round == 0, mu);
    }
    NSS_CUDA_CHECK_LAUNCH();
    scale_kernel<<<blocks_for(static_cast<long long>(w.clusters) * w.atoms), kThreads, 0, stream>>>(w, mu, sigma);
    NSS_CUDA_CHECK_LAUNCH();
    shrink_kernel<<<blocks_for(codes), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    explode_kernel<<<blocks_for(codes), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
    stop_kernel<<<blocks_for(w.clusters), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

void lssc_reconstruct(const LsscWork& w, cudaStream_t stream) {
    product_kernel<<<blocks_for(static_cast<long long>(w.np) * w.m), kThreads, 0, stream>>>(w, false);
    NSS_CUDA_CHECK_LAUNCH();
    finish_kernel<<<blocks_for(w.np), kThreads, 0, stream>>>(w);
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
