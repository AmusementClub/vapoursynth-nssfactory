// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/twsc/kernels.hpp"
#include "cuda/common/block_jacobi.cuh"
#include "cuda/common/block_tridiag.cuh"
#include "cuda/runtime/error.hpp"

#include <algorithm>

namespace nss_cuda {
namespace {

// Precision of the solver workspace (D15).
using TwscReal = float;

constexpr double kNoiseFloor = 1e-6;  // nss::kTwscNoiseFloor

// FP64 is a small fraction of the device's throughput: the sums that the CPU
// model takes in double are float sums with their rounding errors carried
// along (two_sum), converted once at the end.
struct FloatSum {
    float hi = 0.f, lo = 0.f;
    __device__ __forceinline__ void add(float x) {
        const float s = hi + x;
        const float v = s - hi;
        lo += (hi - (s - v)) + (x - v);
        hi = s;
    }
    __device__ __forceinline__ double value() const { return static_cast<double>(hi) + lo; }
};

// Sum of x over the warp, in a fixed order (every lane returns it).
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) x += __shfl_xor_sync(0xffffffffu, x, step);
    return x;
}

// kWays sums of c_i v_i over i < n that share c, each added in index order.
// The block kernel's sums are bound by the loads from the group's arrays in
// global memory: a factor that neighbouring elements share is loaded once
// for them, the other factors are neighbours in memory, and a thread has
// kWays chains of additions in flight instead of one.
constexpr int kWays = 4;

// ADMM elements a thread keeps in registers, and pixels of a patch loaded
// at a time (twsc_block_kernel).
constexpr int kHeld = 16;
constexpr int kRun = 8;

// common(i) is c_i, each(way, i) the v_i of sum `way`.
template <class Real, class Common, class Each>
__device__ __forceinline__ void ordered_dots(int n, Real* sum, Common common, Each each) {
#pragma unroll
    for (int w = 0; w < kWays; ++w) sum[w] = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        Real c[4], v[kWays][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            c[t] = common(i + t);
#pragma unroll
            for (int w = 0; w < kWays; ++w) v[w][t] = each(w, i + t);
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int w = 0; w < kWays; ++w) sum[w] += c[t] * v[w][t];
        }
    }
    for (; i < n; ++i) {
        const Real c = common(i);
#pragma unroll
        for (int w = 0; w < kWays; ++w) sum[w] += c * each(w, i);
    }
}

// The elements e = outer * inners + inner of an outers x inners array from
// e = threadIdx.x by steps of blockDim.x: body(outer, inner). The pair is
// advanced by counters instead of a division per element.
template <class Body>
__device__ __forceinline__ void for_elements(int outers, int inners, Body body) {
    int outer = threadIdx.x / inners, inner = threadIdx.x % inners;
    const int outer_step = blockDim.x / inners, inner_step = blockDim.x % inners;
    while (outer < outers) {
        body(outer, inner);
        outer += outer_step;
        inner += inner_step;
        if (inner >= inners) inner -= inners, ++outer;
    }
}

// The elements (outer, inner) of an outers x inners array, a thread taking
// runs of up to kWays inner neighbours: body(outer, first inner, count).
template <class Body>
__device__ __forceinline__ void for_runs(int outers, int inners, Body body) {
    const int runs = (inners + kWays - 1) / kWays;
    if (runs == 0) return;
    for_elements(outers, runs, [&](int outer, int run) {
        const int first = run * kWays;
        body(outer, first, min(kWays, inners - first));
    });
}

// The eigendecomposition of a group's k x k matrix (block_eigen's contract).
// Caps 12 to 16 (one warp; block 4 in gray or on cfast's planes, joint RGB
// with a group up to 16) take the warp Jacobi, in registers: at 16 about 20k
// warp instructions a group against 35k for block_eigen, whose QL chain runs
// on one thread. Padded to 16 it is slower below 12 (block 4 / step 4 /
// window 20 at 1080p against block_eigen: group 10 0.92, 12 1.02, 14 1.12,
// 16 1.22; joint RGB group 12 0.99, 14 1.05; light, group 8, 0.95), and at
// cap 32 on four warps it took 220k instructions a group against 148k (RTX
// 5080, 2026-10-14). Caps above 16 (kDc instances) take divide and conquer
// on the tridiagonal matrix (block_eigen_dc) with the group's ADMM arrays,
// idle then, as its scratch; the QL chain it replaces was a quarter of a
// 64 x 64 block's samples (default call, 128 x 128: gray 1.42, RGB 1.37,
// joint RGB 1.31; block 8 / group 32 1.17). The second eigen step of
// unequal row weights keeps QL. The kDc = false instance (caps up to 16)
// does not carry it: in one instance the code cost the cap 16 path 2%.
constexpr int kWarpEigenMin = 12;
template <class Real, bool kDc>
__device__ void group_eigen(Real* a, Real* reflectors, int k, int cap, Real* fast, Real* values, Real* dc = nullptr) {
    if (!kDc && cap >= kWarpEigenMin && cap <= kWarpEigen) {
        warp_eigen(a, k, values);
    } else {
        block_eigen<Real, kDc>(a, reflectors, k, fast, values, k > kWarpEigen ? dc : nullptr);
    }
}

// One block per group; every step is a parallel loop over matrix elements
// with fixed per-element operation order, and the norms of the convergence
// test are reduced in a fixed order (within a warp, then over the warps), so
// the result does not depend on scheduling. The eigensolver is group_eigen;
// the Gram matrix, which its eigenvectors replace,
// lives in shared memory when the device has room for it
// (twsc_filter_groups).
//
// Shared scratch ahead of the matrix: the eigensolver's (its slow part too
// with a.shared_values), later three norms per warp.
template <class Real>
__host__ __device__ constexpr std::size_t block_scratch_bytes(int cap, int threads, bool values) {
    const std::size_t eigen =
        static_cast<std::size_t>(block_eigen_fast_reals(cap) + (values ? block_eigen_slow_reals(cap) : 0)) * sizeof(Real);
    const std::size_t norms = 3 * (threads / 32) * sizeof(double);
    return ((eigen > norms ? eigen : norms) + 7) & ~std::size_t{7};
}

// kThreads (64 to 256) bounds the block size; smaller blocks may be launched.
// The register bound (768 threads to an SM) is not what holds the blocks of
// an SM back at 64 x 64 (five by shared memory) or 90 x 90 (three).
template <class Real, int kThreads, bool kDc>
__global__ void __launch_bounds__(kThreads, 768 / kThreads) twsc_block_kernel(TwscGroupArgs a) {
    extern __shared__ unsigned char shared_bytes[];  // scratch, then the matrix with a.shared_matrices
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
    Real* work = static_cast<Real*>(a.work) + g * a.work_reals;
    Real* mean = work;
    Real* column = mean + m;
    Real* raw = column + a.group;
    Real* spectrum = raw + a.group;
    Real* eigen = spectrum + cap;
    int* order = reinterpret_cast<int*>(eigen + cap);  // cap: eigenvalue index by rank
    int* state = order + cap;         // [1]: active count
    Real* gram = eigen + cap + cap + 2;  // the Gram matrix, then its eigenvectors (one per row)
    Real* reflectors = gram + cap * cap;
    Real* atoms = reflectors + cap * cap;  // with unequal row weights only
    Real* data = atoms + static_cast<long long>(m) * cap;
    Real* coef = data + cap * a.group;
    Real* aux = coef + cap * a.group;
    Real* dual = aux + cap * a.group;
    Real* rhs = dual + cap * a.group;
    Real* projected = rhs + cap * a.group;
    Real* values = projected + cap * a.group;  // block_eigen's slow scratch: the eigenvalues first
    // The eigensolver's scratch, and after the eigen steps the per-warp norms
    // of the convergence test.
    Real* scratch = reinterpret_cast<Real*>(shared_bytes);
    double* partial = reinterpret_cast<double*>(shared_bytes);
    if (a.shared_matrices) gram = reinterpret_cast<Real*>(shared_bytes + block_scratch_bytes<Real>(cap, kThreads, a.shared_values));
    if (a.shared_values) values = scratch + block_eigen_fast_reals(cap);

    // Group matrix and per-column noise: one patch per loop item.
    for (int j = lane; j < n; j += lanes) {
        double rms2 = 0.0;
        FloatSum residual;
        for (int c = 0; c < a.nch; ++c) {
            const long long at = static_cast<long long>(match[j].y) * a.width + match[j].x;
            const float* estimate = a.estimate[match[j].t * a.nch + c] + at;
            const float* original = a.original[match[j].t * a.nch + c] + at;
            const double noise = a.sigma[match[j].t * a.nch + c];
            rms2 += noise * noise;
            // kRun pixels' loads ahead of their stores (yc may alias the
            // frames as far as the compiler knows), in pixel order.
            float* out = yc + j * m + c * area;
            for (int p = 0, y = 0, x = 0; p < area; p += kRun) {
                float value[kRun], given[kRun];
#pragma unroll
                for (int t = 0; t < kRun; ++t) {
                    if (p + t < area) {
                        const int offset = y * a.width + x;
                        value[t] = estimate[offset];
                        if (a.residual) given[t] = original[offset];
                        if (++x == a.block) x = 0, ++y;
                    }
                }
#pragma unroll
                for (int t = 0; t < kRun; ++t) {
                    if (p + t < area) {
                        out[p + t] = value[t];
                        if (a.residual) {
                            const float d = given[t] - value[t];
                            residual.add(d * d);
                        }
                    }
                }
            }
        }
        const float noise = static_cast<float>(a.lambda2 * sqrt(fabs(rms2 / a.nch - residual.value() / m)));
        raw[j] = noise;
        column[j] = fmax(static_cast<double>(noise), kNoiseFloor);
    }
    __syncthreads();
    for (int i = lane; i < m; i += lanes) {
        FloatSum sum;
        for (int j = 0; j < n; ++j) sum.add(yc[j * m + i]);
        const float center = (sum.hi + sum.lo) / static_cast<float>(n);
        mean[i] = center;
        for (int j = 0; j < n; ++j) yc[j * m + i] -= center;
    }
    __syncthreads();

    // Gram matrix of the smaller side, then its eigendecomposition.
    for_runs(kd, kd, [=](int p, int q, int count) {
        if (q > p) return;
        Real sum[kWays];
        if (tall) {
            ordered_dots<Real>(m, sum, [=](int i) { return yc[p * m + i]; },
                               [=](int w, int i) { return yc[min(q + w, p) * m + i]; });
        } else {
            ordered_dots<Real>(n, sum, [=](int j) { return yc[j * m + p]; },
                               [=](int w, int j) { return yc[j * m + min(q + w, p)]; });
        }
        for (int w = 0; w < count && q + w <= p; ++w) {
            gram[p * kd + q + w] = sum[w];
            gram[(q + w) * kd + p] = sum[w];
        }
    });
    __syncthreads();
    group_eigen<Real, kDc>(gram, reflectors, kd, cap, scratch, values, data);

    // Descending order by rank (ties by index), reduced spectrum, active count.
    for (int e = lane; e < kd; e += lanes) {
        const Real mine = values[e];
        int rank = 0;
        for (int f = 0; f < kd; ++f) {
            const Real other = values[f];
            rank += other > mine || (other == mine && f < e);
        }
        order[rank] = e;
    }
    __syncthreads();
    Real precision[3];
    bool uniform = true;
    for (int c = 0; c < a.nch; ++c) {
        precision[c] = static_cast<Real>(1.0 / fmax(static_cast<double>(a.row_sigma[c]), kNoiseFloor));
        uniform = uniform && a.row_sigma[c] == a.row_sigma[0];
    }
    Real* scale = projected;  // s'_k / s_k: a tall group's atom k is Yc v_k scale_k
    const double noise = n * static_cast<double>(raw[0]) * static_cast<double>(raw[0]);
    for (int k = lane; k < kd; k += lanes) {
        const int e = order[k];
        const double lambda = fmax(static_cast<double>(values[e]), 0.0);
        const Real reduced = static_cast<Real>(sqrt(fmax(lambda - noise, 0.0)));
        spectrum[k] = reduced;
        scale[k] = static_cast<Real>(static_cast<double>(reduced) / sqrt(lambda));
        eigen[k] = precision[0] * reduced * reduced;  // the Sylvester eigenvalue with equal row weights
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

    // data = atoms^T W Yc and the Sylvester eigenvalues, the atoms being
    // U_k s'_k. With equal row weights the atoms are orthogonal under W and
    // are not formed: a tall group's data is the eigenvector itself (G v =
    // lambda v), a wide group's the projection on it.
    if (uniform) {
        for_runs(n, r, [=](int j, int k, int count) {
            Real value[kWays];
            if (!tall) {
                // k + w < kd: an index of `order` even beyond the run.
                ordered_dots<Real>(m, value, [=](int i) { return yc[j * m + i]; },
                                   [=](int w, int i) { return gram[order[min(k + w, r - 1)] * kd + i]; });
            }
            for (int w = 0; w < count; ++w) {
                const int e = (k + w) * n + j;
                data[e] = tall ? eigen[k + w] / scale[k + w] * gram[order[k + w] * kd + j]
                               : precision[0] * spectrum[k + w] * value[w];
                coef[e] = 0;
                aux[e] = 0;
                dual[e] = 0;
            }
        });
        __syncthreads();
    } else {
        for (long long e = lane; e < static_cast<long long>(r) * m; e += lanes) {
            const int k = static_cast<int>(e / m), i = static_cast<int>(e % m);
            const int col = order[k];
            Real value;
            if (tall) {
                Real sum = 0;
                for (int j = 0; j < n; ++j) sum += static_cast<Real>(yc[j * m + i]) * gram[col * kd + j];
                value = sum * scale[k];
            } else {
                value = gram[col * kd + i] * spectrum[k];
            }
            atoms[e] = value;
        }
        __syncthreads();
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
        // The weighted atom Gram matrix takes the place of the group's.
        Real* weighted = gram;
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
        if (r > 0) group_eigen<Real, kDc>(weighted, reflectors, r, cap, scratch, values);
        for (int k = lane; k < r; k += lanes) eigen[k] = fmax(values[k], Real(0));
        __syncthreads();
    }

    bool converged = r == 0;
    // With up to kHeld elements a thread (cap 16 and a group up to 32 on one
    // warp, for one) the dual and auxiliary variables stay in registers: the
    // arrays of an SM's resident groups are more than its L1 holds, and the
    // ADMM pass waited on them (coef too: it spills).
    const bool held = uniform && r * n <= kHeld * lanes;
    Real held_dual[kHeld], held_aux[kHeld];
#pragma unroll
    for (int t = 0; t < kHeld; ++t) held_dual[t] = held_aux[t] = 0;
    // The dual update of an iteration is taken by the next one's first pass.
    Real rho = static_cast<Real>(a.rho), before = 0;
    for (int iteration = 0; iteration < a.iterations && r > 0; ++iteration) {
        if (!uniform) {
            // (A + shift_j I) c_j = rhs_j in A's eigenbasis: project and divide.
            for (int e = lane; e < r * n; e += lanes) {
                Real y = dual[e];
                if (iteration > 0) dual[e] = y += before * (coef[e] - aux[e]);
                rhs[e] = data[e] + Real(0.5) * (rho * aux[e] - y) * column[e % n];
            }
            __syncthreads();
            for (int e = lane; e < r * n; e += lanes) {
                const int i = e / n, j = e % n;
                Real sum = 0;
                for (int k = 0; k < r; ++k) sum += gram[i * r + k] * rhs[k * n + j];
                projected[e] = sum / (eigen[i] + Real(0.5) * rho * column[j]);
            }
            __syncthreads();
        }
        float primal = 0.f, dc = 0.f, dz = 0.f;
        // The shrinkage of one element, and its terms of the norms in the
        // thread's element order.
        const Real inverse_rho = Real(1) / rho;
        const auto shrink = [&](Real value, Real y, Real coef_was, Real aux_was) {
            const Real temp = value + y * inverse_rho;
            const Real z = copysign(fmax(fabs(temp) - inverse_rho, Real(0)), temp);
            const float p = value - z, cchange = value - coef_was, zchange = z - aux_was;
            primal += p * p;
            dc += cchange * cchange;
            dz += zchange * zchange;
            return z;
        };
        if (held) {
            // The thread's elements in for_elements order, unrolled so that
            // the registers are named.
            int k = lane / n, j = lane % n;
#pragma unroll
            for (int t = 0; t < kHeld; ++t) {
                if (k < r) {
                    const int e = k * n + j;
                    const Real coef_was = coef[e];
                    Real y = held_dual[t];
                    if (iteration > 0) held_dual[t] = y += before * (coef_was - held_aux[t]);
                    const Real right = data[e] + Real(0.5) * (rho * held_aux[t] - y) * column[j];
                    const Real value = __fdividef(right, eigen[k] + Real(0.5) * rho * column[j]);
                    held_aux[t] = shrink(value, y, coef_was, held_aux[t]);
                    coef[e] = value;
                }
                k += lanes / n;
                j += lanes % n;
                if (j >= n) j -= n, ++k;
            }
        } else {
            for_elements(r, n, [&](int k, int j) {
                const int e = k * n + j;
                Real value, y = dual[e];
                if (uniform) {
                    if (iteration > 0) dual[e] = y += before * (coef[e] - aux[e]);
                    const Real right = data[e] + Real(0.5) * (rho * aux[e] - y) * column[j];
                    value = __fdividef(right, eigen[k] + Real(0.5) * rho * column[j]);
                } else {
                    value = 0;
                    for (int i = 0; i < r; ++i) value += gram[i * r + k] * projected[i * n + j];
                }
                aux[e] = shrink(value, y, coef[e], aux[e]);
                coef[e] = value;
            });
        }
        primal = warp_sum(primal);
        dc = warp_sum(dc);
        dz = warp_sum(dz);
        if (lane % 32 == 0) {
            partial[3 * (lane / 32)] = primal;
            partial[3 * (lane / 32) + 1] = dc;
            partial[3 * (lane / 32) + 2] = dz;
        }
        __syncthreads();
        if (lane == 0) {
            double sp = 0.0, sc = 0.0, sz = 0.0;
            for (int w = 0; w < lanes / 32; ++w) {
                sp += partial[3 * w];
                sc += partial[3 * w + 1];
                sz += partial[3 * w + 2];
            }
            const double limit = a.tolerance * a.tolerance;  // on the squared norms
            partial[0] = sp <= limit && sc <= limit && sz <= limit ? 1.0 : 0.0;
        }
        __syncthreads();
        converged = partial[0] != 0.0;
        __syncthreads();  // the next iteration's sums overwrite the verdict
        if (converged) break;
        before = rho;
        if (iteration + 1 < a.iterations) rho *= static_cast<Real>(a.mu);
    }
    if (lane == 0 && !converged) atomicAdd(a.stalled, 1);

    const long long channel_values = static_cast<long long>(a.batch) * a.group * area;
    // atoms C + mean. Without the atoms: Yc (V diag(scale) C) for a tall
    // group, U (diag(s') C) for a wide one.
    Real* mix = rhs;
    if (uniform) {
        for_elements(r, n, [&](int k, int j) { coef[k * n + j] *= tall ? scale[k] : spectrum[k]; });
        __syncthreads();
        for_runs(tall ? n : 0, n, [=](int j, int l, int count) {
            Real sum[kWays];
            ordered_dots<Real>(r, sum, [=](int k) { return coef[k * n + j]; },
                               [=](int w, int k) { return gram[order[k] * kd + min(l + w, n - 1)]; });
            for (int w = 0; w < count; ++w) mix[(l + w) * n + j] = sum[w];
        });
        __syncthreads();
    }
    for_runs(n, m, [=](int j, int i, int count) {
        Real sum[kWays];
        if (!uniform) {
            ordered_dots<Real>(r, sum, [=](int k) { return coef[k * n + j]; },
                               [=](int w, int k) { return atoms[static_cast<long long>(k) * m + min(i + w, m - 1)]; });
        } else if (tall) {
            ordered_dots<Real>(n, sum, [=](int l) { return mix[l * n + j]; },
                               [=](int w, int l) { return static_cast<Real>(yc[l * m + min(i + w, m - 1)]); });
        } else {
            ordered_dots<Real>(r, sum, [=](int k) { return coef[k * n + j]; },
                               [=](int w, int k) { return gram[order[k] * kd + min(i + w, m - 1)]; });
        }
        for (int w = 0; w < count; ++w) {
            const int at = i + w;
            a.values[(at / area) * channel_values + (static_cast<long long>(g) * a.group + j) * area + at % area] =
                static_cast<float>(sum[w] + mean[at]);
        }
    });
    for (int j = lane; j < a.group; j += lanes) {
        a.patches[static_cast<long long>(g) * a.group + j] =
            j < n ? AggregatePatch{match[j].x, match[j].y, match[j].t,
                                   static_cast<float>(1.0 / fmax(static_cast<double>(raw[j]), kNoiseFloor))}
                  : AggregatePatch{0, 0, -1, 0.f};
    }
}

}  // namespace

std::size_t twsc_real_bytes() { return sizeof(TwscReal); }

namespace {

// Where a group's block keeps its state, from the device's limits.
struct Placement {
    int threads;         // per block
    int bound;           // the kernel instance
    bool divide;         // divide and conquer for the eigen step (cap above kWarpEigen)
    bool shared_matrices;
    bool shared_values;
    std::size_t shared;  // bytes per block
    int resident;        // blocks the device runs at a time
};

// About two threads per matrix row: blocks per SM count for more than
// threads per block (measured on sm_120, 2026-10-12 with QL; with divide and
// conquer, 128 threads at cap 32 and 256 at cap 64 measured 0.85 to 0.87,
// 2026-10-14; none of this enters the output). The matrix goes to
// shared memory when a block may hold it. An SM then runs sm / (block +
// 1 KiB) blocks (the driver's share, measured; the size in units of 256
// bytes), up to the kernel's register bound; the solver's slow scratch joins
// the matrix unless that costs a block.
Placement place(int cap) {
    int device = 0, per_block = 0, per_sm = 0, sms = 0;
    NSS_CUDA_CHECK(cudaGetDevice(&device));
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    NSS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    Placement p{};
    p.threads = cap <= 16 ? 32 : cap <= 32 ? 64 : cap <= 64 ? 128 : 256;
    p.bound = p.threads <= 64 ? 64 : p.threads <= 128 ? 128 : 256;
    p.divide = cap > kWarpEigen;
    const std::size_t matrix = static_cast<std::size_t>(cap) * cap * sizeof(TwscReal);
    p.shared_matrices = matrix + block_scratch_bytes<TwscReal>(cap, p.bound, false) <= static_cast<std::size_t>(per_block);
    const auto bytes = [&](bool values) { return block_scratch_bytes<TwscReal>(cap, p.bound, values) + (p.shared_matrices ? matrix : 0); };
    const auto blocks = [&](std::size_t size) {
        return std::min<std::size_t>(static_cast<std::size_t>(per_sm) / (((size + 255) & ~std::size_t{255}) + 1024), 768 / p.bound);
    };
    p.shared_values = bytes(true) <= static_cast<std::size_t>(per_block) && blocks(bytes(true)) == blocks(bytes(false));
    p.shared = bytes(p.shared_values);
    p.resident = sms * static_cast<int>(blocks(p.shared));
    return p;
}

__global__ void twsc_opponent_kernel(float* a, float* b, float* c, std::size_t count, bool inverse) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // nss::twsc_opponent_forward / twsc_opponent_inverse.
    const float x = a[i], y = b[i], z = c[i];
    if (inverse) {
        const float base = x * nss::kTwscOpponentA;
        a[i] = base + y * nss::kTwscOpponentB + z * nss::kTwscOpponentC;
        b[i] = base - 2 * (z * nss::kTwscOpponentC);
        c[i] = base - y * nss::kTwscOpponentB + z * nss::kTwscOpponentC;
    } else {
        a[i] = (x + y + z) * nss::kTwscOpponentA;
        b[i] = (x - z) * nss::kTwscOpponentB;
        c[i] = (x - 2 * y + z) * nss::kTwscOpponentC;
    }
}

}  // namespace

void twsc_opponent(float* a, float* b, float* c, std::size_t count, bool inverse, cudaStream_t stream) {
    if (count == 0) return;
    twsc_opponent_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(a, b, c, count, inverse);
    NSS_CUDA_CHECK_LAUNCH();
}

int twsc_resident_groups(int block, int group, int nch) {
    const int m = block * block * nch;
    return place(m < group ? m : group).resident;
}

void twsc_filter_groups(const TwscGroupArgs& args, cudaStream_t stream) {
    if (args.batch <= 0) return;
    const int m = args.block * args.block * args.nch;
    const Placement p = place(m < args.group ? m : args.group);
    TwscGroupArgs a = args;
    a.shared_matrices = p.shared_matrices;
    a.shared_values = p.shared_values;
    a.work_reals = static_cast<long long>(twsc_work_reals(m, a.group));
    const auto launch = [&](auto kernel) {
        if (p.shared > (48u << 10)) {
            NSS_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(p.shared)));
        }
        kernel<<<a.batch, p.threads, p.shared, stream>>>(a);
    };
    switch (p.bound) {
        case 64:
            if (p.divide) launch(twsc_block_kernel<TwscReal, 64, true>);
            else launch(twsc_block_kernel<TwscReal, 64, false>);
            break;
        case 128: launch(twsc_block_kernel<TwscReal, 128, true>); break;
        default: launch(twsc_block_kernel<TwscReal, 256, true>); break;
    }
    NSS_CUDA_CHECK_LAUNCH();
}

}  // namespace nss_cuda
