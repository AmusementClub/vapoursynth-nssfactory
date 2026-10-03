// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/twsc/kernels.hpp"
#include "cuda/common/block_jacobi.cuh"
#include "cuda/runtime/error.hpp"

namespace nss_cuda {
namespace {

// Precision of the solver workspace (D15).
using TwscReal = float;

constexpr double kNoiseFloor = 1e-6;  // nss::kTwscNoiseFloor

// Cyclic Jacobi eigendecomposition of the symmetric k x k matrix g (row-major,
// full storage): eigenvalues on the diagonal, eigenvector i in column i of v.
// Fixed pair order; a pair is skipped once it is negligible against its
// diagonal entries or against the matrix scale (mean eigenvalue), and the
// sweeps stop when none rotated. Directions far below the scale are noise for
// a Gram matrix and are left unrefined.
template <class Real>
__device__ void jacobi(Real* g, Real* v, int k) {
    for (int i = 0; i < k * k; ++i) v[i] = (i / k == i % k) ? Real(1) : Real(0);
    const Real tolerance = sizeof(Real) == 8 ? Real(1e-15) : Real(1e-7);
    Real trace = 0;
    for (int i = 0; i < k; ++i) trace += fabs(g[i * k + i]);
    const Real floor = tolerance * trace / k;
    for (int sweep = 0; sweep < 30; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < k - 1; ++p) {
            for (int q = p + 1; q < k; ++q) {
                const Real apq = g[p * k + q];
                const Real app = g[p * k + p], aqq = g[q * k + q];
                if (!(fabs(apq) > tolerance * sqrt(fabs(app * aqq))) || !(fabs(apq) > floor)) continue;
                rotated = true;
                const Real theta = (aqq - app) / (Real(2) * apq);
                const Real t = copysign(Real(1), theta) / (fabs(theta) + sqrt(theta * theta + Real(1)));
                const Real c = Real(1) / sqrt(t * t + Real(1));
                const Real s = t * c;
                for (int i = 0; i < k; ++i) {
                    const Real gip = g[i * k + p], giq = g[i * k + q];
                    g[i * k + p] = c * gip - s * giq;
                    g[i * k + q] = s * gip + c * giq;
                }
                for (int i = 0; i < k; ++i) {
                    const Real gpi = g[p * k + i], gqi = g[q * k + i];
                    g[p * k + i] = c * gpi - s * gqi;
                    g[q * k + i] = s * gpi + c * gqi;
                }
                for (int i = 0; i < k; ++i) {
                    const Real vip = v[i * k + p], viq = v[i * k + q];
                    v[i * k + p] = c * vip - s * viq;
                    v[i * k + q] = s * vip + c * viq;
                }
            }
        }
        if (!rotated) break;
    }
}

// One thread per group; every step has a fixed order (run-to-run identical).
template <class Real>
__global__ void twsc_group_kernel(TwscGroupArgs a) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= a.batch) return;
    const int area = a.block * a.block;
    const int m = area * a.nch;
    const int n = min(a.counts[g], a.group);
    const int kd = min(m, n);
    const bool tall = n <= m;
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    float* yc = a.centered + static_cast<long long>(g) * m * a.group;  // column j at yc + j * m
    // Workspace split (capacities for a full group, see twsc_work_reals).
    const int cap = min(m, a.group);
    Real* work = static_cast<Real*>(a.work) + static_cast<long long>(g) * twsc_work_reals(m, a.group);
    Real* mean = work;                // m
    Real* column = mean + m;          // group: max(column noise, floor)
    Real* raw = column + a.group;     // group: column noise
    Real* spectrum = raw + a.group;   // cap: reduced singular values, descending
    Real* eigen = spectrum + cap;     // cap: Sylvester eigenvalues
    Real* vector = eigen + cap;       // cap + 2: one column in the eigenbasis
    Real* gram = vector + cap + 2;    // cap^2
    Real* basis = gram + cap * cap;   // cap^2: eigenvectors of gram
    Real* atoms = basis + cap * cap;  // m * cap: atom k at atoms + k * m
    Real* data = atoms + static_cast<long long>(m) * cap;  // cap * group each: [k * n + j]
    Real* coef = data + cap * a.group;
    Real* aux = coef + cap * a.group;
    Real* dual = aux + cap * a.group;
    Real* rhs = dual + cap * a.group;
    Real* weighted = rhs + cap * a.group;  // cap^2: weighted atom Gram (unequal row weights)
    Real* rotation = weighted + cap * cap; // cap^2: its eigenvectors

    // Group matrix and per-column noise.
    for (int j = 0; j < n; ++j) {
        double rms2 = 0.0, residual = 0.0;
        for (int c = 0; c < a.nch; ++c) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x;
            const float* estimate = a.estimate[match[j].t * a.nch + c] + at;
            const float* original = a.original[match[j].t * a.nch + c] + at;
            const double noise = a.sigma[match[j].t * a.nch + c];
            rms2 += noise * noise;
            for (int p = 0; p < area; ++p) {
                const int offset = (p / a.block) * a.width + p % a.block;
                const float value = estimate[offset];
                yc[j * m + c * area + p] = value;
                if (a.residual) {
                    const double d = static_cast<double>(original[offset]) - value;
                    residual += d * d;
                }
            }
        }
        const float noise = static_cast<float>(a.lambda2 * sqrt(fabs(rms2 / a.nch - residual / m)));
        raw[j] = noise;
        column[j] = fmax(static_cast<double>(noise), kNoiseFloor);
    }
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += static_cast<double>(yc[j * m + i]) / n;
        mean[i] = static_cast<Real>(sum);
        for (int j = 0; j < n; ++j) yc[j * m + i] = static_cast<float>(static_cast<double>(yc[j * m + i]) - sum);
    }

    // SVD through the Gram matrix of the smaller side.
    for (int p = 0; p < kd; ++p) {
        for (int q = 0; q <= p; ++q) {
            Real sum = 0;
            if (tall) {
                for (int i = 0; i < m; ++i) sum += static_cast<Real>(yc[p * m + i]) * yc[q * m + i];
            } else {
                for (int j = 0; j < n; ++j) sum += static_cast<Real>(yc[j * m + p]) * yc[j * m + q];
            }
            gram[p * kd + q] = sum;
            gram[q * kd + p] = sum;
        }
    }
    jacobi(gram, basis, kd);
    // Singular values in descending order (stable selection): order kept in
    // `vector` as indices.
    for (int i = 0; i < kd; ++i) vector[i] = static_cast<Real>(i);
    for (int i = 0; i < kd; ++i) {
        int best = i;
        for (int j = i + 1; j < kd; ++j) {
            if (gram[static_cast<int>(vector[j]) * kd + static_cast<int>(vector[j])] >
                gram[static_cast<int>(vector[best]) * kd + static_cast<int>(vector[best])]) {
                best = j;
            }
        }
        const Real chosen = vector[best];
        for (int j = best; j > i; --j) vector[j] = vector[j - 1];
        vector[i] = chosen;
    }
    const double noise = n * static_cast<double>(raw[0]) * static_cast<double>(raw[0]);
    int active = 0;
    for (int k = 0; k < kd; ++k) {
        const int e = static_cast<int>(vector[k]);
        const double lambda = fmax(static_cast<double>(gram[e * kd + e]), 0.0);
        const double reduced = sqrt(fmax(lambda - noise, 0.0));
        spectrum[k] = static_cast<Real>(reduced);
        if (reduced > 0.0) active = k + 1;
        // Atom k = U_k s'_k: U_k = Yc v_k / s_k (tall) or the eigenvector (wide).
        Real* atom = atoms + static_cast<long long>(k) * m;
        if (!(reduced > 0.0)) {
            for (int i = 0; i < m; ++i) atom[i] = 0;
        } else if (tall) {
            const Real scale = static_cast<Real>(reduced / sqrt(lambda));
            for (int i = 0; i < m; ++i) {
                Real sum = 0;
                for (int j = 0; j < n; ++j) sum += static_cast<Real>(yc[j * m + i]) * basis[j * kd + e];
                atom[i] = sum * scale;
            }
        } else {
            for (int i = 0; i < m; ++i) atom[i] = basis[i * kd + e] * static_cast<Real>(reduced);
        }
    }
    Real precision[3];
    bool uniform = true;
    for (int c = 0; c < a.nch; ++c) {
        precision[c] = static_cast<Real>(1.0 / fmax(static_cast<double>(a.row_sigma[c]), kNoiseFloor));
        uniform = uniform && a.row_sigma[c] == a.row_sigma[0];
    }
    const int r = active;

    bool converged = true;
    if (r > 0) {
        // data = atoms^T W Yc and the weighted atom Gram matrix.
        for (int k = 0; k < r; ++k) {
            const Real* atom = atoms + static_cast<long long>(k) * m;
            for (int j = 0; j < n; ++j) {
                Real sum = 0;
                for (int i = 0; i < m; ++i) sum += atom[i] * precision[i / area] * yc[j * m + i];
                data[k * n + j] = sum;
                coef[k * n + j] = 0;
                aux[k * n + j] = 0;
                dual[k * n + j] = 0;
            }
            for (int l = 0; l <= (uniform ? 0 : k); ++l) {
                const Real* other = atoms + static_cast<long long>(uniform ? k : l) * m;
                Real sum = 0;
                for (int i = 0; i < m; ++i) sum += atom[i] * precision[i / area] * other[i];
                if (uniform) {
                    eigen[k] = sum;
                } else {
                    weighted[k * r + l] = sum;
                    weighted[l * r + k] = sum;
                }
            }
        }
        if (!uniform) {
            jacobi(weighted, rotation, r);
            for (int k = 0; k < r; ++k) eigen[k] = fmax(weighted[k * r + k], Real(0));
        }
        Real rho = static_cast<Real>(a.rho);
        converged = false;
        for (int iteration = 0; iteration < a.iterations; ++iteration) {
            double primal = 0.0, dc = 0.0, dz = 0.0;
            for (int j = 0; j < n; ++j) {
                const Real shift = Real(0.5) * rho * column[j];
                for (int k = 0; k < r; ++k) {
                    rhs[k * n + j] = data[k * n + j] + Real(0.5) * (rho * aux[k * n + j] - dual[k * n + j]) * column[j];
                }
                if (!uniform) {
                    // Solve (A + shift I) c = rhs in A's eigenbasis.
                    for (int i = 0; i < r; ++i) {
                        Real sum = 0;
                        for (int k = 0; k < r; ++k) sum += rotation[k * r + i] * rhs[k * n + j];
                        vector[i] = sum / (eigen[i] + shift);
                    }
                }
                for (int k = 0; k < r; ++k) {
                    Real value;
                    if (uniform) {
                        value = rhs[k * n + j] / (eigen[k] + shift);
                    } else {
                        value = 0;
                        for (int i = 0; i < r; ++i) value += rotation[k * r + i] * vector[i];
                    }
                    const Real temp = value + dual[k * n + j] / rho;
                    const Real z = copysign(fmax(fabs(temp) - Real(1) / rho, Real(0)), temp);
                    const double p = value - z, cchange = value - coef[k * n + j], zchange = z - aux[k * n + j];
                    primal += p * p;
                    dc += cchange * cchange;
                    dz += zchange * zchange;
                    coef[k * n + j] = value;
                    aux[k * n + j] = z;
                }
            }
            converged = sqrt(primal) <= a.tolerance && sqrt(dc) <= a.tolerance && sqrt(dz) <= a.tolerance;
            if (converged) break;
            for (int i = 0; i < r * n; ++i) {
                const int k = i / n, j = i % n;
                dual[k * n + j] += rho * (coef[k * n + j] - aux[k * n + j]);
            }
            if (iteration + 1 < a.iterations) rho *= static_cast<Real>(a.mu);
        }
    }
    if (!converged) atomicAdd(a.stalled, 1);

    // Reconstruction atoms C + mean into the channel-major patch values.
    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Real sum = 0;
            for (int k = 0; k < r; ++k) sum += atoms[static_cast<long long>(k) * m + i] * coef[k * n + j];
            a.values[(i / area) * channel_values + (static_cast<long long>(g) * a.group + j) * area + i % area] =
                static_cast<float>(sum + mean[i]);
        }
    }
    for (int j = 0; j < a.group; ++j) {
        a.patches[static_cast<long long>(g) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t,
                                   static_cast<float>(1.0 / fmax(static_cast<double>(raw[j]), kNoiseFloor))}
                  : AggregatePatch{0, 0, -1, 0.f};
    }
}

// Large groups: one block per group; every step is a parallel loop over
// matrix elements with fixed per-element operation order, and the norms of
// the convergence test are reduced in thread order, so the result does not
// depend on scheduling. The Gram matrix and its eigenvectors live in shared
// memory when they fit; the eigensolver is common/block_jacobi.cuh.
constexpr int kBlockThreads = 256;
constexpr int kSharedMatrix = 64;  // largest k with both k x k matrices in shared memory
constexpr int kSerialLimit = 16;   // groups up to this min(m, n) run one thread per group

template <class Real>
__global__ void __launch_bounds__(kBlockThreads) twsc_block_kernel(TwscGroupArgs a) {
    extern __shared__ unsigned char shared_bytes[];
    __shared__ int order[256];
    __shared__ int state[4];          // jacobi flag, active count, converged, unused
    __shared__ double partial[3 * kBlockThreads];
    __shared__ Real rho_shared;
    const int lane = threadIdx.x, lanes = blockDim.x;
    const int g = blockIdx.x;
    const int area = a.block * a.block;
    const int m = area * a.nch;
    const int n = min(a.counts[g], a.group);
    const int kd = min(m, n);
    const bool tall = n <= m;
    const DeviceMatch* match = a.matches + static_cast<long long>(g) * a.group;
    float* yc = a.centered + static_cast<long long>(g) * m * a.group;
    const int cap = min(m, a.group);
    Real* work = static_cast<Real*>(a.work) + static_cast<long long>(g) * twsc_work_reals(m, a.group);
    Real* mean = work;
    Real* column = mean + m;
    Real* raw = column + a.group;
    Real* spectrum = raw + a.group;
    Real* eigen = spectrum + cap;
    Real* cs = eigen + cap;           // cap + 2: the (c, s) of a Jacobi round's k / 2 pairs
    Real* gram = cs + cap + 2;
    Real* basis = gram + cap * cap;
    Real* atoms = basis + cap * cap;
    Real* data = atoms + static_cast<long long>(m) * cap;
    Real* coef = data + cap * a.group;
    Real* aux = coef + cap * a.group;
    Real* dual = aux + cap * a.group;
    Real* rhs = dual + cap * a.group;
    Real* weighted = rhs + cap * a.group;
    Real* rotation = weighted + cap * cap;
    Real* projected = rotation + cap * cap;  // cap * group
    if (cap <= kSharedMatrix) {
        gram = reinterpret_cast<Real*>(shared_bytes);
        basis = gram + cap * cap;
    }

    // Group matrix and per-column noise: one patch per loop item.
    for (int j = lane; j < n; j += lanes) {
        double rms2 = 0.0, residual = 0.0;
        for (int c = 0; c < a.nch; ++c) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x;
            const float* estimate = a.estimate[match[j].t * a.nch + c] + at;
            const float* original = a.original[match[j].t * a.nch + c] + at;
            const double noise = a.sigma[match[j].t * a.nch + c];
            rms2 += noise * noise;
            for (int p = 0; p < area; ++p) {
                const int offset = (p / a.block) * a.width + p % a.block;
                const float value = estimate[offset];
                yc[j * m + c * area + p] = value;
                if (a.residual) {
                    const double d = static_cast<double>(original[offset]) - value;
                    residual += d * d;
                }
            }
        }
        const float noise = static_cast<float>(a.lambda2 * sqrt(fabs(rms2 / a.nch - residual / m)));
        raw[j] = noise;
        column[j] = fmax(static_cast<double>(noise), kNoiseFloor);
    }
    __syncthreads();
    for (int i = lane; i < m; i += lanes) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += static_cast<double>(yc[j * m + i]) / n;
        mean[i] = static_cast<Real>(sum);
        for (int j = 0; j < n; ++j) yc[j * m + i] = static_cast<float>(static_cast<double>(yc[j * m + i]) - sum);
    }
    __syncthreads();

    // Gram matrix of the smaller side, then its eigendecomposition.
    for (int e = lane; e < kd * kd; e += lanes) {
        const int p = e / kd, q = e % kd;
        if (q > p) continue;
        Real sum = 0;
        if (tall) {
            for (int i = 0; i < m; ++i) sum += static_cast<Real>(yc[p * m + i]) * yc[q * m + i];
        } else {
            for (int j = 0; j < n; ++j) sum += static_cast<Real>(yc[j * m + p]) * yc[j * m + q];
        }
        gram[p * kd + q] = sum;
        gram[q * kd + p] = sum;
    }
    __syncthreads();
    block_jacobi(gram, basis, kd, cs, &state[0]);

    // Descending order by rank (ties by index), reduced spectrum, active count.
    for (int e = lane; e < kd; e += lanes) {
        const Real mine = gram[e * kd + e];
        int rank = 0;
        for (int f = 0; f < kd; ++f) {
            const Real other = gram[f * kd + f];
            rank += other > mine || (other == mine && f < e);
        }
        order[rank] = e;
    }
    __syncthreads();
    const double noise = n * static_cast<double>(raw[0]) * static_cast<double>(raw[0]);
    for (int k = lane; k < kd; k += lanes) {
        const int e = order[k];
        const double lambda = fmax(static_cast<double>(gram[e * kd + e]), 0.0);
        spectrum[k] = static_cast<Real>(sqrt(fmax(lambda - noise, 0.0)));
    }
    __syncthreads();
    if (lane == 0) {
        int active = 0;
        for (int k = 0; k < kd; ++k) {
            if (spectrum[k] > Real(0)) active = k + 1;
        }
        state[1] = active;
    }
    __syncthreads();
    const int r = state[1];

    // Atoms U_k s'_k.
    for (long long e = lane; e < static_cast<long long>(r) * m; e += lanes) {
        const int k = static_cast<int>(e / m), i = static_cast<int>(e % m);
        const int col = order[k];
        Real value = 0;
        if (spectrum[k] > Real(0)) {
            if (tall) {
                const double lambda = fmax(static_cast<double>(gram[col * kd + col]), 0.0);
                Real sum = 0;
                for (int j = 0; j < n; ++j) sum += static_cast<Real>(yc[j * m + i]) * basis[j * kd + col];
                value = sum * static_cast<Real>(static_cast<double>(spectrum[k]) / sqrt(lambda));
            } else {
                value = basis[i * kd + col] * spectrum[k];
            }
        }
        atoms[e] = value;
    }
    Real precision[3];
    bool uniform = true;
    for (int c = 0; c < a.nch; ++c) {
        precision[c] = static_cast<Real>(1.0 / fmax(static_cast<double>(a.row_sigma[c]), kNoiseFloor));
        uniform = uniform && a.row_sigma[c] == a.row_sigma[0];
    }
    __syncthreads();

    // data = atoms^T W Yc; Sylvester eigenvalues (diagonal with equal row
    // weights, otherwise through the weighted atom Gram matrix, which reuses
    // the Gram/basis storage).
    for (int e = lane; e < r * n; e += lanes) {
        const int k = e / n, j = e % n;
        const Real* atom = atoms + static_cast<long long>(k) * m;
        Real sum = 0;
        for (int i = 0; i < m; ++i) sum += atom[i] * precision[i / area] * yc[j * m + i];
        data[e] = sum;
        coef[e] = 0;
        aux[e] = 0;
        dual[e] = 0;
    }
    if (uniform) {
        for (int k = lane; k < r; k += lanes) {
            const Real* atom = atoms + static_cast<long long>(k) * m;
            Real sum = 0;
            for (int i = 0; i < m; ++i) sum += atom[i] * precision[i / area] * atom[i];
            eigen[k] = sum;
        }
        __syncthreads();
    } else {
        __syncthreads();
        weighted = cap <= kSharedMatrix ? gram : weighted;
        rotation = cap <= kSharedMatrix ? basis : rotation;
        for (int e = lane; e < r * r; e += lanes) {
            const int k = e / r, l = e % r;
            if (l > k) continue;
            const Real* atom = atoms + static_cast<long long>(k) * m;
            const Real* other = atoms + static_cast<long long>(l) * m;
            Real sum = 0;
            for (int i = 0; i < m; ++i) sum += atom[i] * precision[i / area] * other[i];
            weighted[k * r + l] = sum;
            weighted[l * r + k] = sum;
        }
        __syncthreads();
        if (r > 0) block_jacobi(weighted, rotation, r, cs, &state[0]);
        for (int k = lane; k < r; k += lanes) eigen[k] = fmax(weighted[k * r + k], Real(0));
        __syncthreads();
    }

    if (lane == 0) {
        rho_shared = static_cast<Real>(a.rho);
        state[2] = r > 0 ? 0 : 1;
    }
    __syncthreads();
    for (int iteration = 0; iteration < a.iterations && r > 0; ++iteration) {
        const Real rho = rho_shared;
        if (!uniform) {
            // (A + shift_j I) c_j = rhs_j in A's eigenbasis: project and divide.
            for (int e = lane; e < r * n; e += lanes) {
                rhs[e] = data[e] + Real(0.5) * (rho * aux[e] - dual[e]) * column[e % n];
            }
            __syncthreads();
            for (int e = lane; e < r * n; e += lanes) {
                const int i = e / n, j = e % n;
                Real sum = 0;
                for (int k = 0; k < r; ++k) sum += rotation[k * r + i] * rhs[k * n + j];
                projected[e] = sum / (eigen[i] + Real(0.5) * rho * column[j]);
            }
            __syncthreads();
        }
        double primal = 0.0, dc = 0.0, dz = 0.0;
        for (int e = lane; e < r * n; e += lanes) {
            const int k = e / n, j = e % n;
            Real value;
            if (uniform) {
                const Real right = data[e] + Real(0.5) * (rho * aux[e] - dual[e]) * column[j];
                value = right / (eigen[k] + Real(0.5) * rho * column[j]);
            } else {
                value = 0;
                for (int i = 0; i < r; ++i) value += rotation[k * r + i] * projected[i * n + j];
            }
            const Real temp = value + dual[e] / rho;
            const Real z = copysign(fmax(fabs(temp) - Real(1) / rho, Real(0)), temp);
            const double p = value - z, cchange = value - coef[e], zchange = z - aux[e];
            primal += p * p;
            dc += cchange * cchange;
            dz += zchange * zchange;
            coef[e] = value;
            aux[e] = z;
        }
        partial[lane] = primal;
        partial[kBlockThreads + lane] = dc;
        partial[2 * kBlockThreads + lane] = dz;
        __syncthreads();
        if (lane == 0) {
            double sp = 0.0, sc = 0.0, sz = 0.0;
            for (int t = 0; t < lanes; ++t) {
                sp += partial[t];
                sc += partial[kBlockThreads + t];
                sz += partial[2 * kBlockThreads + t];
            }
            state[2] = sqrt(sp) <= a.tolerance && sqrt(sc) <= a.tolerance && sqrt(sz) <= a.tolerance;
        }
        __syncthreads();
        if (state[2]) break;
        for (int e = lane; e < r * n; e += lanes) dual[e] += rho * (coef[e] - aux[e]);
        if (lane == 0 && iteration + 1 < a.iterations) rho_shared = rho * static_cast<Real>(a.mu);
        __syncthreads();
    }
    if (lane == 0 && !state[2]) atomicAdd(a.stalled, 1);

    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    for (int e = lane; e < m * n; e += lanes) {
        const int j = e / m, i = e % m;
        Real sum = 0;
        for (int k = 0; k < r; ++k) sum += atoms[static_cast<long long>(k) * m + i] * coef[k * n + j];
        a.values[(i / area) * channel_values + (static_cast<long long>(g) * a.group + j) * area + i % area] =
            static_cast<float>(sum + mean[i]);
    }
    for (int j = lane; j < a.group; j += lanes) {
        a.patches[static_cast<long long>(g) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t,
                                   static_cast<float>(1.0 / fmax(static_cast<double>(raw[j]), kNoiseFloor))}
                  : AggregatePatch{0, 0, -1, 0.f};
    }
}

}  // namespace

std::size_t twsc_real_bytes() { return sizeof(TwscReal); }

void twsc_filter_groups(const TwscGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block * args.nch;
    const int cap = m < args.group ? m : args.group;
    if (cap > kSerialLimit) {
        const std::size_t shared = cap <= kSharedMatrix ? 2 * static_cast<std::size_t>(cap) * cap * sizeof(TwscReal) : 0;
        twsc_block_kernel<TwscReal><<<args.batch, kBlockThreads, shared, stream>>>(args);
    } else {
        twsc_group_kernel<TwscReal><<<(args.batch + 63) / 64, 64, 0, stream>>>(args);
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
